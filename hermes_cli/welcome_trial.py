"""The desktop's welcome trial: free inference on ``nous/welcome`` for a first install, once per OS user.

Only the backend the desktop spawned on this machine can claim it, only when this OS user has never
used Hermes here (``install_state``), and only with no provider and no Nous identity yet. The claim
mints a guest stamped with ``WELCOME_CLAIM_FLAG`` in the mint's own write and writes the OS user's
marker, so the trial is never claimed twice for that user. Another OS account on the same computer is
a separate first install. A claim that is lost later (the guest is reaped, ``auth.json`` is
deleted, the user signs in) is never made again: that user lands on provider setup and sign-in like
anyone else.

Guests that ran on the free model before the claim existed keep it (:func:`grandfather_existing_guest`).
Enforcement is client-side; the account service's own entitlement check follows separately.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from hermes_cli import anon_auth

logger = logging.getLogger("hermes_cli.auth")


def _has_nous_identity() -> bool:
    from hermes_cli.auth_nous import _read_shared_nous_state

    return anon_auth.current_nous_state() is not None or _read_shared_nous_state() is not None


def provider_configured() -> bool:
    """Something besides the welcome trial carries inference: a key, a config pin, a sign-in, a host
    credential. The resolver itself answers, with the free-tier rung hidden."""
    from hermes_cli.auth import resolve_provider
    from hermes_cli.auth_constants import AuthError

    try:
        provider = resolve_provider("auto", skip_free_tier=True)
    except AuthError:  # no_provider_configured
        return False
    return provider != "nous" or not anon_auth.has_guest()


def inference_available(*, assume_fresh: bool = False) -> bool:
    """Something can carry a first chat on this backend: the trial (claimed, or claimable now) or a
    configured provider. One walk of the provider ladder answers both halves."""
    if anon_auth.free_inference_active():
        return True
    configured = provider_configured()
    return configured or claim_blocker(assume_fresh=assume_fresh, configured=configured) is None


def _machine_is_used(assume_fresh: bool) -> bool:
    from hermes_cli.install_state import probe, trial_reserved

    return not assume_fresh and probe().existing and not trial_reserved()


@dataclass(frozen=True, slots=True)
class _Ask:
    assume_fresh: bool
    configured: Optional[bool]  # the caller's provider_configured() answer, when it has one


# Cheapest first; the last one reads the disk. Each is a reason the trial cannot be claimed.
_CLAIM_BLOCKERS: tuple[tuple[str, Callable[[_Ask], bool]], ...] = (
    ("not the desktop's local backend", lambda _ask: not anon_auth.on_local_desktop_backend()),
    ("guests are switched off", lambda _ask: not anon_auth.guest_allowed()),
    ("a Nous identity exists", lambda _ask: _has_nous_identity()),
    ("a provider is configured",
     lambda ask: provider_configured() if ask.configured is None else ask.configured),
    ("Hermes was used on this machine", lambda ask: _machine_is_used(ask.assume_fresh)),
)


def claim_blocker(*, assume_fresh: bool = False, configured: Optional[bool] = None) -> Optional[str]:
    """Why the trial cannot be claimed here, or None when it can. *assume_fresh* is the unpackaged
    desktop's ``--fresh-install`` switch: it skips only the machine check. *configured* is a
    ``provider_configured()`` answer the caller already has, so the ladder is not walked twice."""
    ask = _Ask(assume_fresh, configured)
    return next((reason for reason, blocks in _CLAIM_BLOCKERS if blocks(ask)), None)


def claim_desktop_free_tier(*, assume_fresh: bool = False, force: bool = False) -> Optional[dict[str, Any]]:
    """Claim the trial when nothing blocks it: mint the guest, stamp the claim, mark the machine.
    Returns the claimed state, or None when the trial is not available here. A failed mint raises
    the coded ``AuthError`` from :func:`anon_auth.ensure_portal_identity` (memoised with its cooldown;
    *force* is the user's own retry). Held under the OS user's claim lock from the checks to the
    marker write, so two installs starting together claim once."""
    from hermes_cli import install_state

    with install_state.claim_lock():
        install_state.probe.cache_clear()  # another install may have claimed since this process looked
        settle_interrupted_claim()
        if reason := claim_blocker(assume_fresh=assume_fresh):
            logger.debug("welcome trial not claimed: %s", reason)
            return None
        state = anon_auth.ensure_portal_identity(explicit=True, force=force, welcome_claim=True)
        if state is None:  # still inside the cooldown of the last failure
            failure = anon_auth.last_mint_failure() or {}
            raise anon_auth._anon_err(
                str(failure.get("error") or anon_auth.ANON_FAILURE_COPY[anon_auth.ANON_SERVER_ERROR]),
                str(failure.get("error_code") or anon_auth.ANON_SERVER_ERROR),
                retry_after=failure.get("retry_after"))
        install_state.mark_machine_used()
    logger.info("Nous free tier: welcome trial claimed on this machine")
    return state


def settle_interrupted_claim() -> bool:
    """A claim cut off between its guest write and its marker write leaves the reservation behind.
    The stamped guest is the claim, so the reservation goes: kept, it would let a second trial through
    once that guest is reaped or deleted. Runs before every claim and at serve boot; True when it did."""
    from hermes_cli.auth_nous import _read_shared_nous_state
    from hermes_cli.install_state import mark_machine_used, trial_reserved

    if not trial_reserved():
        return False
    stores = (anon_auth.current_nous_state(), _read_shared_nous_state())
    if not any(anon_auth.is_guest_state(s) and s.get(anon_auth.WELCOME_CLAIM_FLAG) is True for s in stores):
        return False
    mark_machine_used()
    logger.info("Nous free tier: an interrupted welcome-trial claim was settled")
    return True


def grandfather_existing_guest() -> bool:
    """A guest from before the claim existed keeps the free model when this home already chatted on
    it. Only a guest with no ``welcome_claim`` key qualifies: every guest minted since carries the
    key, so a connector guest (``False``) never does. Records the claim once; True when it did."""
    if not anon_auth.on_local_desktop_backend():
        return False
    state = anon_auth.current_nous_state()
    if not anon_auth.is_guest_state(state) or anon_auth.WELCOME_CLAIM_FLAG in state:
        return False
    from hermes_constants import get_default_hermes_root, get_hermes_home

    homes = dict.fromkeys((get_hermes_home(), get_default_hermes_root()))
    if not any(_chatted_on_welcome(home / "state.db") for home in homes):
        return False
    from hermes_cli.install_state import mark_machine_used

    if anon_auth.stamp_legacy_guest_claim() is None:  # re-checked under the locks, against shared auth
        return False
    mark_machine_used()
    logger.info("Nous free tier: existing free-model guest kept its trial")
    return True


def _chatted_on_welcome(db: Path) -> bool:
    """A session on the free model with at least one message. Read-only; a missing or unreadable
    database is no."""
    if not db.is_file():
        return False
    try:
        with contextlib.closing(sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True, timeout=1)) as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions s WHERE s.model = ? "
                "AND EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id) LIMIT 1",
                (anon_auth.GUEST_MODEL,)).fetchone()
    except sqlite3.Error:
        return False
    return row is not None

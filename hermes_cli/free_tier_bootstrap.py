"""Serve-start bootstrap for the Nous free tier: the ONE place a free-tier identity is created.

Every Hermes process that may need the free tier runs this once at boot (``hermes serve`` on a
daemon thread beside the other background boots; the CLI first-run guard synchronously). It
inventories credentials cheap-first, creates the identity only when the launch gate is open
(:func:`hermes_cli.anon_auth.guest_enabled`), resolves which provider carries inference, records
the answer in process memory, and tells every connected client with one ``setup.ready`` event.

Nothing else mints. ``free_tier.status`` and ``setup.status`` read the record; provider resolution
never reaches the portal; a dead credential is replaced by the explicit re-mint in
``auth_nous.resolve_nous_runtime_credentials``. Ruling: NS-845 Q1.2 (recorded on NS-847).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes_cli.auth")

# The desktop's first ``setup.status`` waits this long for the record before falling back to a live
# probe. The mint budget is 5 s (``GUEST_MINT_TIMEOUT_SECONDS``); the rest covers the inventory.
SETUP_READY_WAIT_SECONDS = 8.0
SETUP_READY_EVENT = "setup.ready"


@dataclass(frozen=True)
class SetupRecord:
    """What the bootstrap found. One shape for every reader; no version field (renderer and backend
    ship together)."""

    provider_configured: bool      # some provider can carry inference (free tier included)
    inference_provider: str        # ``resolve_provider("auto")``'s answer, "" when nothing resolves
    free_tier: bool                # the identity that exists is the free tier AND the tier is on
    has_identity: bool             # a Nous identity (free tier or account) is on disk
    other_providers: bool          # the inventory found something usable BESIDES the free tier
    error: str = ""                # why the mint did not happen, when it did not; "" otherwise
    finished_at: float = field(default_factory=time.time)

    def as_payload(self) -> Dict[str, Any]:
        return asdict(self)


_lock = threading.Lock()
_record: Optional[SetupRecord] = None
_done = threading.Event()
_started = False


def current_record() -> Optional[SetupRecord]:
    """The record, or None until the first bootstrap finishes."""
    return _record


def wait_for_record(timeout: float = SETUP_READY_WAIT_SECONDS) -> Optional[SetupRecord]:
    """Block up to ``timeout`` seconds for a bootstrap that is IN FLIGHT, then return whatever it
    produced. Returns None at once when no bootstrap ever started in this process (a bare
    ``tui_gateway`` under test, an old serve without the boot hook): the caller falls back to its
    live probe instead of paying the wait for nothing."""
    if not _started:
        return None
    _done.wait(timeout)
    return _record


def reset_for_tests() -> None:
    global _record, _started
    with _lock:
        _record = None
        _started = False
        _done.clear()


def _inventory_other_providers() -> bool:
    """Is anything usable configured BESIDES the free tier? Asks the resolver ladder itself (the
    thing that picks the provider for a turn) with the free-tier rung hidden: an explicit key, a
    config pin, a sign-in or a host credential answers; nothing else falls through to
    ``no_provider_configured``. Not ``_has_any_provider_configured``: that first-run guard counts
    keyless catalog providers as "configured" and is True on a blank machine."""
    from hermes_cli.auth import resolve_provider
    try:
        return resolve_provider("auto", skip_free_tier=True) != "nous"
    except Exception as exc:
        logger.debug("free tier bootstrap: nothing else carries inference (%s)", exc)
        return False


def _resolve_inference() -> str:
    from hermes_cli.auth import resolve_provider
    try:
        return str(resolve_provider("auto") or "")
    except Exception:
        return ""


def run_bootstrap(*, announce: bool = True) -> SetupRecord:
    """Inventory -> ensure identity (gate permitting) -> resolve inference -> record -> broadcast.

    Runs every boot; only the mint is gated. Idempotent per process: a second call returns the
    existing record without touching the portal. Never raises. ``announce=False`` skips the
    ``setup.ready`` event: the plain CLI has no client to tell and its stdout is the user's terminal.
    """
    global _record, _started
    with _lock:
        if _record is not None:
            return _record
        if _started:
            _done.wait(SETUP_READY_WAIT_SECONDS)
            if _record is not None:
                return _record
        _started = True

    from hermes_cli import anon_auth

    other = _inventory_other_providers()
    error = ""
    state: Optional[Dict[str, Any]] = anon_auth.current_nous_state()
    if anon_auth.guest_enabled():
        try:
            # ``other`` decides whether the mint may also claim ``active_provider`` (NS-845 Q1.3).
            state = anon_auth.ensure_portal_identity(explicit=True, carries_inference=not other)
        except Exception as exc:
            error = str(exc)
            logger.info("Nous free tier not set up at boot: %s", exc)
    free_tier = bool(state) and anon_auth.is_guest_state(state) and anon_auth.guest_enabled()

    if not other and not free_tier:
        from hermes_cli.main import _has_any_provider_configured
        if _has_any_provider_configured(strict_profile_scope=False):
            other = True

    record = SetupRecord(
        provider_configured=other or free_tier or (bool(state) and not anon_auth.is_guest_state(state)),
        inference_provider=_resolve_inference(),
        free_tier=free_tier,
        has_identity=bool(state),
        other_providers=other,
        error=error,
    )
    with _lock:
        _record = record
        _done.set()
    if announce:
        _broadcast(record)
    return record


def _broadcast(record: SetupRecord) -> None:
    try:
        from tui_gateway.server import _broadcast_global_event
        _broadcast_global_event(SETUP_READY_EVENT, record.as_payload())
    except Exception as exc:  # no serve process (plain CLI): nobody to tell
        logger.debug("setup.ready not broadcast: %s", exc)


def start_background_bootstrap() -> threading.Thread:
    """``hermes serve`` entry: run on a daemon thread so a slow portal never delays the socket."""
    thread = threading.Thread(target=run_bootstrap, daemon=True, name="free-tier-bootstrap")
    thread.start()
    return thread

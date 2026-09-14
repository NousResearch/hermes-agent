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
    # The structured form of ``error`` (``anon_auth.ANON_*`` codes): what a renderer keys its copy
    # and its doors on. ``retryable`` says whether a later attempt can succeed; ``retry_after`` is
    # the seconds still to wait before one may (0 when none is owed, or the code is terminal).
    error_code: str = ""
    retryable: bool = False
    retry_after: int = 0
    finished_at: float = field(default_factory=time.time)

    def as_payload(self) -> Dict[str, Any]:
        return asdict(self)

    def failure_fields(self) -> Dict[str, Any]:
        """The additive ``{error, error_code, retryable, retry_after}`` block every status RPC
        carries when the mint did not happen; ``{}`` otherwise."""
        if not self.error_code:
            return {}
        return {"error": self.error, "error_code": self.error_code, "retryable": self.retryable,
                "retry_after": self.retry_after}


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


def _build_record(*, other: bool, force: bool) -> SetupRecord:
    """One inventory-then-mint pass into a record. ``force`` is the user's own retry: it makes one
    attempt even inside the mint memo's cooldown (``anon_auth.ensure_portal_identity``)."""
    from hermes_cli import anon_auth

    error = ""
    failure: Dict[str, Any] = {}
    state: Optional[Dict[str, Any]] = anon_auth.current_nous_state()
    if anon_auth.guest_enabled():
        try:
            # ``other`` decides whether the mint may also claim ``active_provider`` (NS-845 Q1.3).
            state = anon_auth.ensure_portal_identity(explicit=True, carries_inference=not other, force=force)
        except Exception as exc:
            error = str(exc)
            logger.info("Nous free tier not set up at boot: %s", exc)
        if state is None:
            # Either this attempt failed (the memo now holds why) or an earlier one did and its
            # cooldown still runs: the record carries that verdict either way.
            failure = anon_auth.last_mint_failure() or {}
            error = error or str(failure.get("error") or "")
    free_tier = bool(state) and anon_auth.is_guest_state(state) and anon_auth.guest_enabled()
    return SetupRecord(
        provider_configured=other or free_tier or (bool(state) and not anon_auth.is_guest_state(state)),
        inference_provider=_resolve_inference(),
        free_tier=free_tier,
        has_identity=bool(state),
        other_providers=other,
        error=error,
        error_code=str(failure.get("error_code") or ""),
        retryable=bool(failure.get("retryable", False)),
        retry_after=int(failure.get("retry_after") or 0),
    )


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

    record = _build_record(other=_inventory_other_providers(), force=False)
    with _lock:
        _record = record
        _done.set()
    if announce:
        _broadcast(record)
    return record


# Background retries after a boot-time mint failure: the memo's cooldown decides WHEN (a server
# ``Retry-After``, the ops-breaker floor, or the unreachable ladder), this decides HOW MANY before
# the process stops trying on its own (the user's retry button, ``free_tier.provision``, is not
# counted). A terminal code (gate closed, proof of work, locked) is never retried.
BOOTSTRAP_RETRY_ATTEMPTS = 3
_sleep = time.sleep      # seam for tests


def retry_bootstrap_mint(*, force: bool = False, announce: bool = True) -> SetupRecord:
    """Re-run the mint once (``force`` bypasses the cooldown), replace the record and announce it.

    The desktop's ``free_tier.provision`` calls this with ``force=True``; the background loop calls
    it as each cooldown passes. Returns the existing record untouched when no bootstrap ran yet
    (nothing to replace) or when an identity already exists."""
    global _record
    current = _record
    if current is None:
        return run_bootstrap(announce=announce)
    if current.has_identity:
        return current
    record = _build_record(other=current.other_providers, force=force)
    with _lock:
        _record = record
    if announce:
        _broadcast(record)
    return record


def _retry_until_settled() -> None:
    """The background loop behind ``start_background_bootstrap``: wait out each cooldown and try
    again, up to ``BOOTSTRAP_RETRY_ATTEMPTS``, while the record says a later attempt can succeed."""
    for _ in range(BOOTSTRAP_RETRY_ATTEMPTS):
        record = _record
        if record is None or record.has_identity or not record.error_code or not record.retryable:
            return
        _sleep(max(1, int(record.retry_after or 0)))
        record = retry_bootstrap_mint(force=False)
        if record.has_identity:
            logger.info("Nous free tier set up after a boot-time retry")
            return


def _bootstrap_then_retry() -> None:
    run_bootstrap()
    try:
        _retry_until_settled()
    except Exception as exc:  # the loop is best effort; the record already says what happened
        logger.debug("free tier bootstrap retry loop stopped: %s", exc)


def _broadcast(record: SetupRecord) -> None:
    try:
        from tui_gateway.server import _broadcast_global_event
        _broadcast_global_event(SETUP_READY_EVENT, record.as_payload())
    except Exception as exc:  # no serve process (plain CLI): nobody to tell
        logger.debug("setup.ready not broadcast: %s", exc)


def start_background_bootstrap() -> threading.Thread:
    """``hermes serve`` entry: run on a daemon thread so a slow portal never delays the socket."""
    thread = threading.Thread(target=_bootstrap_then_retry, daemon=True, name="free-tier-bootstrap")
    thread.start()
    return thread

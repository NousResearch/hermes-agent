"""Record of who carries inference, for ``setup.status`` and ``setup.ready``.

It runs once per process on a daemon thread: at ``hermes serve`` boot, and on the gateway when the
local Desktop first connects (``tui_gateway.methods_onboarding.begin_local_desktop_boot``). It never
creates an identity: the desktop's welcome-trial claim (``hermes_cli.welcome_trial``) and the first
hosted connector action are the only creators. For the desktop's local backend it first records a
fresh install's onboarding decision (``onboarding_run.latch_first_install``) and settles a claim cut
off before its marker write (``welcome_trial.settle_interrupted_claim``). Then it inventories
credentials cheap-first, resolves which provider carries inference, records the answer in process
memory, and tells every connected client with one ``setup.ready`` event. ``free_tier.status`` and
``setup.status`` read the record; provider resolution never reaches the portal.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes_cli.auth")

# The desktop's first ``setup.status`` waits this long for the record before falling back to a live probe.
SETUP_READY_WAIT_SECONDS = 8.0
SETUP_READY_EVENT = "setup.ready"


@dataclass(frozen=True)
class SetupRecord:
    """What the bootstrap found. One shape for every reader; no version field (renderer and backend
    ship together)."""

    provider_configured: bool      # some provider can carry inference (free tier included)
    inference_provider: str        # ``resolve_provider("auto")``'s answer, "" when nothing resolves
    free_tier_account: bool        # the identity that exists is a guest that carries inference here
    has_identity: bool             # a Nous identity (free tier or account) is on disk
    other_providers: bool          # the inventory found something usable BESIDES the free tier
    error: str = ""                # why the last trial claim failed, when it did; "" otherwise
    # The mint memo's verdict, verbatim (``anon_auth.MintFailure.as_payload``):
    # ``{error, error_code, retryable, retry_after}`` after a failed claim, else ``{}``.
    # One wire shape: every status RPC spreads it as is.
    failure: dict[str, Any] = field(default_factory=dict)
    finished_at: float = field(default_factory=time.time)

    @property
    def free_tier_route(self) -> bool:
        return self.free_tier_account and self.inference_provider == "nous"

    def as_payload(self) -> dict[str, Any]:
        # The broadcast carries the failure block flat, the same shape ``setup.status`` spreads,
        # so a client keys on ``error_code`` identically whichever surface it read.
        payload = asdict(self)
        payload.update(payload.pop("failure"))
        payload["free_tier_route"] = self.free_tier_route
        return payload

    def failure_fields(self) -> dict[str, Any]:
        return dict(self.failure)


_lock = threading.Lock()
_record: Optional[SetupRecord] = None
_done = threading.Event()
_started = False
_settled = False  # the bootstrap recorded the install and settled the claim (``_settle_install``)
_resettle_lock = threading.Lock()
# ``(mtime_ns, size)`` of the files the inventory reads, taken by the inventory that built the
# current record; ``reconcile_record`` re-inventories only when they moved.
_inventory_stamp: Optional[tuple] = None
_INVENTORY_FILES = ("config.yaml", ".env", "auth.json")


def current_record() -> Optional[SetupRecord]:
    """The record, or None until the first bootstrap finishes."""
    return _record


def wait_for_record(timeout: float = SETUP_READY_WAIT_SECONDS) -> Optional[SetupRecord]:
    """Block up to ``timeout`` seconds for a bootstrap that is IN FLIGHT, then return whatever it
    produced, reconciled with any provider configured since (:func:`reconcile_record`). Returns
    None at once when no bootstrap ever started in this process (a bare ``tui_gateway`` under
    test, an old serve without the boot hook): the caller falls back to its live probe instead of
    paying the wait for nothing."""
    if not _started:
        return None
    _done.wait(timeout)
    return reconcile_record()


def reconcile_record() -> Optional[SetupRecord]:
    """Let a provider configured AFTER boot count: a record that says ``provider_configured:
    false`` is re-inventoried once ``config.yaml`` / ``.env`` / ``auth.json`` moved since the
    inventory that built it, and replaced (+ ``setup.ready``) when something now carries
    inference. The identity and failure block are kept as is. A record that already says ``True`` is never re-probed, so the answer
    only moves false -> true here. Every write path that assigns the main model (the Models page,
    a picker key save) calls this for the immediate broadcast; ``setup.status`` calls it for
    writes this process never saw (``hermes setup`` / ``hermes model`` from a shell, a hand edit).
    The record is the LAUNCH profile's: a call scoped to another profile's home (a dashboard
    write with ``?profile=B``) leaves it alone, or B's providers would open the launch gate."""
    global _record
    record = _record
    if record is None or record.provider_configured:
        return record
    from hermes_constants import get_process_hermes_home, hermes_home_key
    if hermes_home_key() != hermes_home_key(get_process_hermes_home()) or _inventory_stamp == _config_stamp():
        return record
    if not _inventory_other_providers():
        return _record
    refreshed = replace(record, provider_configured=True, other_providers=True,
                        inference_provider=_resolve_inference(), finished_at=time.time())
    with _lock:
        if _record is not record:  # a retry replaced it meanwhile; its inventory is newer
            return _record
        _record = refreshed
    _due_changed()
    _broadcast(refreshed)
    return refreshed


def reset_for_tests() -> None:
    global _record, _started, _settled, _inventory_stamp
    with _lock:
        _record = None
        _started = False
        _settled = False
        _inventory_stamp = None
        _done.clear()


def _config_stamp() -> tuple:
    from hermes_cli.config import get_hermes_home
    from utils import file_cache_version
    home = get_hermes_home()
    stamp = []
    for name in _INVENTORY_FILES:
        try:
            stamp.append(file_cache_version(home / name, always_digest=True))
        except OSError:
            stamp.append(None)
    return tuple(stamp)


def _inventory_other_providers() -> bool:
    """Is anything usable configured BESIDES the free tier? Asks the resolver ladder itself (the
    thing that picks the provider for a turn) with the free-tier rung hidden: an explicit key, a
    config pin, a sign-in or a host credential answers; nothing else falls through to
    ``no_provider_configured``. Not ``_has_any_provider_configured``: that first-run guard also
    counts host credentials (gh auth, Claude Code) and a config pin, and it does not hide the
    free-tier rung.

    Stamps the config files BEFORE reading them, so a write that lands during the inventory is
    seen by the next :func:`reconcile_record`.
    """
    global _inventory_stamp
    from hermes_cli.auth import resolve_provider
    _inventory_stamp = _config_stamp()
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


def _build_record(*, other: bool) -> SetupRecord:
    """One inventory pass into a record. Reads only: no identity is created here."""
    from hermes_cli import anon_auth

    state: Optional[dict[str, Any]] = anon_auth.current_nous_state()
    failure: dict[str, Any] = {} if state else (anon_auth.last_mint_failure() or {})
    free_tier_account = anon_auth.free_inference_active(state) if state else False
    return SetupRecord(
        provider_configured=other or free_tier_account or (bool(state) and not anon_auth.is_guest_state(state)),
        inference_provider=_resolve_inference(),
        free_tier_account=free_tier_account,
        has_identity=bool(state),
        other_providers=other,
        error=str(failure.get("error") or ""),
        failure=failure,
    )


def run_bootstrap(*, announce: bool = True) -> SetupRecord:
    """Settle an interrupted claim -> inventory -> resolve inference -> record ->
    broadcast. Idempotent per process: a second call returns the existing record. Never raises.
    ``announce=False`` skips the ``setup.ready`` event."""
    global _record, _started, _settled
    with _lock:
        if _record is not None:
            return _record
        if _started:
            _done.wait(SETUP_READY_WAIT_SECONDS)
            if _record is not None:
                return _record
        _started = True

    settled = _settle_install()
    record = _build_record(other=_inventory_other_providers())
    with _lock:
        _record, _settled = record, settled
        _done.set()
    if announce:
        _broadcast(record)
    return record


def _settle_install() -> bool:
    """Record a fresh install and settle the guest's claim; True when neither step failed. A failure is
    logged: onboarding.state still answers live, and the record still says who carries inference."""
    settled = True
    try:
        from hermes_cli import anon_auth
        from hermes_cli.onboarding_run import latch_first_install
        if anon_auth.on_local_desktop_backend():
            latch_first_install()
    except Exception:
        settled = False
        logger.warning("onboarding: could not record the first-install decision", exc_info=True)
    try:
        from hermes_cli.welcome_trial import settle_interrupted_claim
        settle_interrupted_claim()
    except Exception:
        settled = False
        logger.warning("free tier: could not settle an interrupted claim", exc_info=True)
    return settled


def resettle_after_failed_boot() -> None:
    """The local Desktop connected again to a process whose bootstrap could not settle the install:
    try those steps again and rebuild the record when they pass. Nothing before the bootstrap
    finished or once it settled."""
    global _settled
    with _lock:
        unsettled = _record is not None and not _settled
    if not unsettled or not _resettle_lock.acquire(blocking=False):  # one retry at a time
        return
    try:
        if _settle_install():
            with _lock:
                _settled = True
            refresh_record()
    finally:
        _resettle_lock.release()


def refresh_record(*, announce: bool = True) -> Optional[SetupRecord]:
    """Rebuild the launch profile's record after the identity changed (a trial claim), so
    ``setup.status`` and the ``setup.ready`` listeners move with it. None before the first bootstrap."""
    global _record
    if _record is None:
        return None
    record = _build_record(other=_inventory_other_providers())
    with _lock:
        _record = record
    _due_changed()
    if announce:
        _broadcast(record)
    return record


def _due_changed() -> None:
    from hermes_cli import onboarding_due

    onboarding_due.invalidate()


def _broadcast(record: SetupRecord) -> None:
    try:
        from tui_gateway.server import _broadcast_global_event
        _broadcast_global_event(SETUP_READY_EVENT, record.as_payload())
    except Exception as exc:  # no serve process (plain CLI): nobody to tell
        logger.debug("setup.ready not broadcast: %s", exc)


def start_background_bootstrap() -> threading.Thread:
    """Run on a daemon thread so the inventory never delays the socket."""
    from agent.memory_provider import spawn_context_thread

    thread = spawn_context_thread(run_bootstrap, name="free-tier-bootstrap")
    thread.start()
    return thread

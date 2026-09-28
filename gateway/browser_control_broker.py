"""Transport-neutral browser-control broker core: binds an identity-scoped *controller*
(the party driving a browser) to *callers* on any transport. Tickets are short-lived,
single-use, identity-bound and consumed exactly once; ``select`` matches every stable
identity field plus the current capability set; ``complete`` is single-shot; ``detach``
fails pending work closed while ``disconnect`` only marks the transport offline. State
changes happen under one RLock; the send callback runs *outside* it so a controller may
``complete`` from inside its own send."""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)
_OWNER_UNSET = object()

#: Default lifetime of a minted registration ticket, in clock seconds.
DEFAULT_TICKET_TTL = 30.0
#: Default wall time a dispatch waits for the controller to complete.
DEFAULT_COMMAND_TIMEOUT = 30.0
#: Maximum cancel frames retained while a same-identity controller is offline.
MAX_DEFERRED_CANCELS = 512
#: Current wire protocol version; registration requires this exact int (bools rejected).
BROWSER_CONTROL_PROTOCOL_VERSION = 1

#: Exact controller capability allowlist shared by every transport (raw CDP/eval/console stay out).
BROWSER_CONTROL_CAPABILITIES = frozenset({
    "controller.noop", "browser_back", "browser_click", "browser_navigate", "browser_press", "browser_screenshot",
    "browser_scroll", "browser_snapshot", "browser_tab_activate", "browser_tabs", "browser_type",
})
#: Privileged capabilities: fail-closed unless Developer Mode is on AND explicitly negotiated.
BROWSER_CONTROL_DEVELOPER_CAPABILITIES = frozenset({"browser_cdp", "browser_evaluate"})
#: Artifact transport capabilities; non-developer because only a store-validated ``artifact_id`` travels.
BROWSER_CONTROL_ARTIFACT_CAPABILITIES = frozenset({"browser_artifact_download", "browser_artifact_upload"})

#: Wire method names for controller frames; transports carry them verbatim.
FRAME_COMMAND = "browser.controller.command"
FRAME_CANCEL = "browser.controller.cancel"


def browser_control_protocol_supported(value: Any) -> bool:
    """Return whether ``value`` names the exact supported wire version."""
    return type(value) is int and value == BROWSER_CONTROL_PROTOCOL_VERSION


def _extension_control_flag(config: Optional[dict], key: str) -> bool:
    """Read ``browser.extension_control.<key>`` as a literal ``True`` (default off)."""
    if config is None:
        try:
            # Hot path (every browser tool call): the read-only loader skips load_config()'s deepcopy.
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except Exception:
            return False
    browser = config.get("browser") if isinstance(config, dict) else None
    extension_control = browser.get("extension_control") if isinstance(browser, dict) else None
    return isinstance(extension_control, dict) and extension_control.get(key, False) is True


def browser_control_developer_mode(config: Optional[dict] = None) -> bool:
    """Explicit Developer Mode flag; gates ``browser_evaluate``/raw CDP only."""
    return _extension_control_flag(config, "developer_mode")


def browser_control_enabled(config: Optional[dict] = None) -> bool:
    """Return the explicit browser-control feature flag (disabled by default)."""
    return _extension_control_flag(config, "enabled")


def filter_browser_control_capabilities(value: Any, *, developer_mode: Optional[bool] = None) -> frozenset:
    """Permitted subset of a capability list (non-list -> empty); developer caps only in Developer Mode."""
    if not isinstance(value, list):
        return frozenset()
    allowed = BROWSER_CONTROL_CAPABILITIES | BROWSER_CONTROL_ARTIFACT_CAPABILITIES
    if (browser_control_developer_mode() if developer_mode is None else developer_mode) is True:
        allowed |= BROWSER_CONTROL_DEVELOPER_CAPABILITIES
    return frozenset(c for c in value if isinstance(c, str) and c in allowed)


class BrowserControlError(Exception):
    """Base class for broker contract failures."""


class ControllerTicketInvalid(BrowserControlError):
    """A registration ticket is unknown, already consumed, or expired."""


class ControllerUnavailable(BrowserControlError):
    """No attached controller exactly matches the requested scope/capability."""


class ControllerCancelled(BrowserControlError):
    """A pending command was cancelled (explicitly or by detach)."""


class ControllerTimeout(BrowserControlError):
    """The controller did not complete the command before the timeout."""


class ControllerRejected(BrowserControlError):
    """The controller completed the command with ``ok=False``."""


@dataclass(frozen=True)
class ControllerScope:
    """Exact controller identity plus capability set; equality is over all fields."""
    principal_id: Optional[str] = None
    profile_id: Optional[str] = None
    session_id: Optional[str] = None
    controller_id: Optional[str] = None
    browser_profile_id: Optional[str] = None
    transport_family: Optional[str] = None
    capabilities: frozenset = frozenset()


#: Stable identity fields; negotiated ``capabilities`` are deliberately excluded.
_IDENTITY_FIELDS = ("principal_id", "profile_id", "session_id", "controller_id", "browser_profile_id", "transport_family")


def _same_scope_identity(first: ControllerScope, second: ControllerScope) -> bool:
    return all(getattr(first, name) == getattr(second, name) for name in _IDENTITY_FIELDS)


@dataclass(frozen=True)
class Ticket:
    """Opaque, single-use registration credential."""
    value: str
    expires_at: float


@dataclass
class _TicketRecord:
    scope: ControllerScope
    expires_at: float
    consumed: bool = False


@dataclass
class _Controller:
    scope: ControllerScope
    send: Callable[[dict], None]
    owner: Any = None
    connected: bool = True
    deferred_cancels: list[dict] = field(default_factory=list)
    # Serializes command/cancel writes with detach or replacement; never held with broker state.
    send_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _PendingCommand:
    scope: ControllerScope
    command_id: str
    tool_call_id: Optional[str]
    event: threading.Event = field(default_factory=threading.Event)
    done: bool = False
    cancelled: bool = False
    ok: bool = False
    result: Any = None


def _cancel_frame(pending: _PendingCommand) -> dict:
    return {"method": FRAME_CANCEL, "params": {"command_id": pending.command_id, "tool_call_id": pending.tool_call_id}}


class BrowserControlBroker:
    """Thread-safe broker core; ``clock`` is injectable (default ``time.monotonic``)."""
    def __init__(self, *, ticket_ttl: float = DEFAULT_TICKET_TTL, command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
                 clock: Optional[Callable[[], float]] = None, developer_mode: Optional[bool] = None) -> None:
        self._ticket_ttl = ticket_ttl
        self._command_timeout = command_timeout
        self._clock = clock if clock is not None else time.monotonic
        self._lock = threading.RLock()
        self._tickets: Dict[str, _TicketRecord] = {}
        self._controllers: Dict[ControllerScope, _Controller] = {}
        self._pending: Dict[str, _PendingCommand] = {}
        # None defers to live config on every selection (so flipping developer_mode off REVOKES
        # raw CDP/eval from attached controllers without restart); a bool pins the gate.
        self._developer_mode_pinned: Optional[bool] = None if developer_mode is None else developer_mode is True
        # Artifact stores keyed by profile id; ``None`` is the default slot.
        self._artifact_stores: Dict[Optional[str], Any] = {}

    @property
    def developer_mode(self) -> bool:
        """Whether privileged capabilities may be selected/dispatched (live config unless pinned)."""
        if self._developer_mode_pinned is not None:
            return self._developer_mode_pinned
        try:
            return browser_control_developer_mode()
        except Exception:
            return False

    def attach_artifact_store(self, store: Any, *, profile_id: Optional[str] = None) -> None:
        """Attach a store exposing ``validate(artifact_id, *, scope) -> receipt`` for one profile
        (``None`` = default slot); ``store=None`` clears the slot. Artifact actions fail closed without one."""
        if store is None:
            self._artifact_stores.pop(profile_id, None)
        else:
            self._artifact_stores[profile_id] = store

    def _artifact_store_for_scope(self, scope: "ControllerScope") -> Any:
        store = self._artifact_stores.get(getattr(scope, "profile_id", None) or None)
        return store if store is not None else self._artifact_stores.get(None)

    def mint_ticket(self, scope: ControllerScope) -> Ticket:
        """Mint a short-lived, single-use ticket bound to ``scope``."""
        now = self._clock()
        with self._lock:
            self._tickets = {v: rec for v, rec in self._tickets.items() if rec.expires_at > now}
            value = secrets.token_urlsafe(32)
            self._tickets[value] = record = _TicketRecord(scope=scope, expires_at=now + self._ticket_ttl)
        return Ticket(value=value, expires_at=record.expires_at)

    def consume_ticket(self, value: str) -> ControllerScope:
        """Exchange a ticket for its scope exactly once; unknown/consumed/expired -> ControllerTicketInvalid."""
        now = self._clock()
        with self._lock:
            record = self._tickets.get(value)
            if record is None:
                raise ControllerTicketInvalid("unknown ticket")
            if record.consumed:
                raise ControllerTicketInvalid("ticket already consumed")
            if now > record.expires_at:
                raise ControllerTicketInvalid("ticket expired")
            record.consumed = True
            return record.scope

    def _controller_for_identity_locked(self, scope: ControllerScope) -> Optional[_Controller]:
        """Attached controller sharing ``scope``'s stable identity (any capabilities)."""
        return next((c for c in self._controllers.values() if _same_scope_identity(c.scope, scope)), None)

    def _live_controller(self, scope: ControllerScope) -> Optional[_Controller]:
        with self._lock:
            controller = self._controller_for_identity_locked(scope)
        return controller if controller is not None and controller.connected else None

    def attach(self, scope: ControllerScope, send: Callable[[dict], None], *, owner: Any = None) -> None:
        """Attach or refresh the controller for one stable identity: a same-identity reconnect refreshes send
        and capabilities without cancelling pending work; a different controller/browser profile in the same
        authenticated session lane hard-replaces it."""
        while True:
            with self._lock:
                existing = self._controller_for_identity_locked(scope)
                lane_scopes = [
                    c for c in self._controllers if not _same_scope_identity(c, scope)
                    and (c.principal_id, c.profile_id, c.session_id, c.transport_family)
                    == (scope.principal_id, scope.profile_id, scope.session_id, scope.transport_family)
                ]
                if existing is None and not lane_scopes:
                    self._controllers[scope] = _Controller(scope=scope, send=send, owner=owner)
                    return

            # Hard replacement, not a recoverable reconnect: terminalize the
            # lane siblings before inserting so session lookup stays unique.
            if lane_scopes:
                for lane_scope in lane_scopes:
                    self.detach(lane_scope, notify_controller=False)
                continue

            with existing.send_lock:
                with self._lock:
                    if self._controllers.get(existing.scope) is not existing:
                        continue
                    self._controllers.pop(existing.scope, None)
                    existing.scope, existing.send, existing.owner, existing.connected = scope, send, owner, False
                    for pending in self._pending_for_scope_locked(scope):
                        pending.scope = scope
                    deferred, existing.deferred_cancels = existing.deferred_cancels, []
                    self._controllers[scope] = existing

                for index, frame in enumerate(deferred):
                    try:
                        send(frame)
                        continue
                    except Exception:
                        logger.exception("failed to flush deferred browser-controller cancel")
                    with self._lock:
                        if self._controllers.get(scope) is existing:
                            existing.deferred_cancels = deferred[index:][-MAX_DEFERRED_CANCELS:]
                    raise ConnectionError("browser controller reconnect could not flush deferred cancels")
                with self._lock:
                    if self._controllers.get(scope) is existing:
                        existing.connected = True
                return

    def select(self, scope: ControllerScope, capability: str) -> Optional[_Controller]:
        """Connected controller matching identity whose *current* negotiated set holds ``capability`` (the
        caller's set is not authoritative); developer capabilities are also gated on LIVE Developer Mode."""
        if capability in BROWSER_CONTROL_DEVELOPER_CAPABILITIES and not self.developer_mode:
            return None
        controller = self._live_controller(scope)
        return controller if controller is not None and capability in controller.scope.capabilities else None

    def is_owner(self, scope: ControllerScope, owner: Any) -> bool:
        """Whether ``owner`` is the exact live transport for ``scope`` (capability-independent)."""
        controller = self._live_controller(scope)
        return controller is not None and controller.owner is owner

    def disconnect(self, scope: ControllerScope, *, owner: Any = _OWNER_UNSET) -> bool:
        """Mark one exact controller transport offline without cancelling work."""
        with self._lock:
            controller = self._controller_for_identity_locked(scope)
        if controller is None:
            return False
        with controller.send_lock:
            with self._lock:
                owned = owner is _OWNER_UNSET or controller.owner is owner
                if self._controllers.get(controller.scope) is not controller or not owned:
                    return False
                controller.connected, controller.owner = False, None
        return True

    def detach(self, scope: ControllerScope, *, owner: Any = _OWNER_UNSET, notify_controller: bool = True) -> None:
        """Remove the controller for ``scope`` and fail its pending work closed (ControllerCancelled)."""
        with self._lock:
            controller = self._controllers.get(scope)
        if controller is None or (owner is not _OWNER_UNSET and controller.owner != owner):
            return
        with controller.send_lock:
            with self._lock:
                if self._controllers.get(scope) is not controller or (owner is not _OWNER_UNSET and controller.owner != owner):
                    return
                self._controllers.pop(scope, None)
                pendings = self._pending_for_scope_locked(scope)
                for pending in pendings:
                    self._resolve_pending(pending, cancelled=True)
            # Hold the old generation's send lock through cancellation so a
            # command frame can never overtake its terminal cancel frame.
            if notify_controller:
                self._emit_cancel_frames(controller, pendings)

    def dispatch(
        self, scope: ControllerScope, *, action: str, arguments: Optional[dict] = None, tool_call_id: Optional[str] = None,
    ) -> Any:
        """Send one controller command and block for completion; raises ControllerUnavailable/Cancelled/Timeout/
        Rejected. Artifact actions also need an attached store and an approved ``artifact_id`` (only the id travels)."""
        controller = self.select(scope, action)
        if controller is None:
            raise ControllerUnavailable(f"no controller for scope {scope!r} with capability {action!r}")
        arguments = dict(arguments or {})
        if action in BROWSER_CONTROL_ARTIFACT_CAPABILITIES:
            self._validate_artifact_reference(scope, action, arguments)
        command_id = secrets.token_hex(16)
        frame = {"method": FRAME_COMMAND, "params": {
            "command_id": command_id, "action": action, "arguments": arguments, "controller_id": scope.controller_id,
            "browser_profile_id": scope.browser_profile_id, "tool_call_id": tool_call_id,
        }}
        pending = _PendingCommand(scope=controller.scope, command_id=command_id, tool_call_id=tool_call_id)
        with controller.send_lock:
            with self._lock:
                # select() ran outside the send lock; revalidate the live
                # controller so disconnect/replacement can't strand a command.
                if self._controller_for_identity_locked(scope) is not controller or not controller.connected:
                    raise ControllerUnavailable(f"controller for scope {scope!r} detached before dispatch")
                pending.scope = controller.scope
                self._pending[command_id] = pending
            try:
                controller.send(frame)
            except Exception:
                # Never left the building: unreserve the id, surface the error.
                with self._lock:
                    self._pending.pop(command_id, None)
                raise

        if not pending.event.wait(timeout=self._command_timeout):
            with self._lock:
                # Event.wait() may return False at the exact boundary where a
                # completion already won and removed the pending command.
                timed_out = not pending.done and self._pending.get(command_id) is pending
                if timed_out:
                    pending.done = True
                    del self._pending[command_id]
            if timed_out:
                with controller.send_lock:
                    with self._lock:
                        active = self._controller_for_identity_locked(scope) or controller
                        if not active.connected:
                            self._defer_cancel_locked(active, pending)
                            active = None
                    if active is not None:
                        self._emit_cancel_frames(active, [pending])
                raise ControllerTimeout(f"controller did not complete command {command_id!r} within {self._command_timeout}s")
        if pending.cancelled:
            raise ControllerCancelled(f"command {command_id!r} was cancelled")
        if not pending.ok:
            raise ControllerRejected(f"controller rejected command {command_id!r}: {pending.result!r}")
        return pending.result

    def complete(self, command_id: str, *, scope: Optional[ControllerScope] = None, ok: bool, result: Any = None) -> bool:
        """Resolve a pending command by id; ``False`` when none is pending. Safe from inside the send callback."""
        with self._lock:
            pending = self._pending.get(command_id)
            if pending is None or pending.done or (scope is not None and pending.scope != scope):
                return False
            pending.done, pending.ok, pending.result = True, ok is True, result
            del self._pending[command_id]
            pending.event.set()
        return True

    def cancel(self, scope: ControllerScope, *, tool_call_id: Optional[str]) -> bool:
        """Cancel the pending command matching ``scope`` + tool_call_id (one cancel frame); ``False`` if none."""
        controller = self._live_controller(scope)
        if controller is None:
            return False
        with controller.send_lock:
            with self._lock:
                if self._controller_for_identity_locked(scope) is not controller or not controller.connected:
                    return False
                target = next((p for p in self._pending_for_scope_locked(scope)
                               if p.tool_call_id == tool_call_id and not p.done), None)
                if target is None:
                    return False
                self._resolve_pending(target, cancelled=True)
            self._emit_cancel_frames(controller, [target])
            return True

    def _resolve_pending(self, pending: _PendingCommand, *, cancelled: bool) -> None:
        pending.cancelled, pending.done = cancelled, True
        del self._pending[pending.command_id]
        pending.event.set()

    def _validate_artifact_reference(self, scope: ControllerScope, action: str, arguments: dict) -> None:
        """Fail closed unless ``arguments`` carries a store-approved artifact id (failures -> ControllerRejected)."""
        store = self._artifact_store_for_scope(scope)
        if store is None:
            raise ControllerRejected(f"{action} requires an attached artifact store")
        artifact_id = arguments.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ControllerRejected(f"{action} requires a non-empty artifact_id")
        try:
            store.validate(artifact_id.strip(), scope=scope)
        except ControllerRejected:
            raise
        except Exception as exc:
            raise ControllerRejected(f"{action} rejected artifact reference {artifact_id!r}: {exc}") from exc

    def _defer_cancel_locked(self, controller: _Controller, pending: _PendingCommand) -> None:
        controller.deferred_cancels.append(_cancel_frame(pending))
        if len(controller.deferred_cancels) > MAX_DEFERRED_CANCELS:
            del controller.deferred_cancels[:-MAX_DEFERRED_CANCELS]

    def _pending_for_scope_locked(self, scope: ControllerScope) -> list[_PendingCommand]:
        return [p for p in list(self._pending.values()) if _same_scope_identity(p.scope, scope)]

    def _emit_cancel_frames(self, controller: _Controller, pendings: list[_PendingCommand]) -> None:
        """Send cancel frames (caller holds ``send_lock``, never the broker lock)."""
        for pending in pendings:
            try:
                controller.send(_cancel_frame(pending))
            except Exception:
                logger.exception("failed to emit cancel frame for command %r", pending.command_id)

    def _lane_scopes(self, session_id, task_id, principal_id, transport_family) -> list[ControllerScope]:
        """Attached scopes bound to one session lane (session + principal + transport)."""
        key = tuple(str(v or "").strip() for v in (session_id or task_id, principal_id, transport_family))
        if not all(key):
            return []
        with self._lock:
            return [s for s in self._controllers if (s.session_id, s.principal_id, s.transport_family) == key]

    def scope_for_session(self, *, session_id: Optional[str] = None, task_id: Optional[str] = None,
                          principal_id: Optional[str] = None, transport_family: Optional[str] = None) -> Optional[ControllerScope]:
        """One unambiguous attached scope for a server-owned session (session id is only a hint; the caller
        supplies its server-derived principal + transport family). Missing/ambiguous identity fails closed."""
        matches = self._lane_scopes(session_id, task_id, principal_id, transport_family)
        return matches[0] if len(matches) == 1 else None

    def lane_registered(self, *, session_id: Optional[str] = None, task_id: Optional[str] = None,
                        principal_id: Optional[str] = None, transport_family: Optional[str] = None) -> bool:
        """Whether ANY controller (even offline) registered for this lane: "bound but unavailable" fails closed
        vs "never registered" (caller keeps the legacy backend). Ambiguous lanes report True."""
        return bool(self._lane_scopes(session_id, task_id, principal_id, transport_family))

    def disconnect_owner(self, owner: Any) -> int:
        """Mark every controller owned by one lost transport offline."""
        with self._lock:
            scopes = [s for s, c in self._controllers.items() if c.owner is owner]
        return sum(int(self.disconnect(scope, owner=owner)) for scope in scopes)

    def reset(self) -> None:
        """Fail all live work closed and clear tickets (tests/shutdown)."""
        with self._lock:
            scopes = list(self._controllers)
        for scope in scopes:
            self.detach(scope)
        with self._lock:
            self._tickets.clear()
            # Pending entries whose controller a concurrent teardown removed.
            for pending in list(self._pending.values()):
                self._resolve_pending(pending, cancelled=True)

    @property
    def ticket_ttl_seconds(self) -> float:
        """Configured lifetime for newly minted one-shot tickets."""
        return self._ticket_ttl

    @property
    def pending_count(self) -> int:
        """Number of commands awaiting completion (diagnostics/tests)."""
        with self._lock:
            return len(self._pending)


_GLOBAL_BROKER = BrowserControlBroker()


def get_browser_control_broker() -> BrowserControlBroker:
    """Process-local broker shared by API and dashboard Gateway transports."""
    return _GLOBAL_BROKER


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

BROWSER_CONTROL_ALL_CAPABILITIES = frozenset(
    BROWSER_CONTROL_CAPABILITIES
    | BROWSER_CONTROL_ARTIFACT_CAPABILITIES
    | BROWSER_CONTROL_DEVELOPER_CAPABILITIES
)
# ---- END PLUGIN-COMPAT ----

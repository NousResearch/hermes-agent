"""Transport-neutral browser-control broker core.

This module is the in-process heart of the browser-control feature: it binds
an *identity-scoped controller* (the party that physically drives a browser)
to *callers* (agents talking to that browser over any transport) without the
broker itself knowing anything about HTTP, WebSocket, or any wire format. The
transport layers built in later phases wrap this core; nothing here routes
traffic.

Why a broker at all: the browser is a stateful, single-owner resource and the
agent side is multi-tenant (many principals, profiles, sessions) and
multi-transport (local API, remote API, …). A controller must never be
addressable by a caller that merely resembles the right identity, and a
command must never be completable twice, cancellable by a stranger, or
observable after its owner has gone away. Every rule below exists to make one
of those violations structurally impossible rather than merely discouraged.

Contract (each rule is exercised by tests/gateway/test_browser_control_broker.py):

- **Registration tickets are short-lived, single-use, identity-bound, and
  cryptographically random.** ``mint_ticket`` returns an opaque value
  (``secrets``-derived, >= 32 chars) plus an expiry derived from the injected
  clock; ``consume_ticket`` exchanges it exactly once for the
  :class:`ControllerScope` it was minted for, raising
  :class:`ControllerTicketInvalid` for unknown, already-consumed, or expired values.
  The ticket is the only cross-transport credential minted here; transports
  decide how to carry it.

- **Exact identity and capability selection.** ``attach`` registers a send
  callback under a :class:`ControllerScope`; ``select`` returns a controller
  only when the caller's scope matches on *every stable identity field* —
  principal, profile, session, controller id, browser profile id, and
  transport family — and the requested capability is present in the
  controller's current negotiated capability set. Partial matches return
  ``None``. A same-identity reconnect may renegotiate capabilities without
  creating an ambiguous second controller.

- **One pending command per command id; single-shot completion.** Each
  ``dispatch`` mints a fresh command id, emits one
  ``browser.controller.command`` frame, and parks a waiter keyed by that id.
  ``complete`` resolves a command exactly once and returns ``False`` for any
  later attempt (late completion after cancellation or detach is ignored).

- **Scoped cancellation.** ``cancel`` aborts only the pending command whose
  scope and tool_call_id match, emits a ``browser.controller.cancel`` frame
  for it, and returns ``False`` when nothing matched.

- **Detach fails closed.** ``detach`` removes the controller and cancels every
  pending command of that scope; waiting dispatchers observe
  :class:`ControllerCancelled` rather than hanging or racing a detached
  controller's late ``complete``.

- **Unexpected disconnects are recoverable.** ``disconnect`` marks an exact
  transport owner offline without accepting new dispatches or cancelling
  already-running work. Re-attaching the same stable identity refreshes the
  transport callback and can deliver a terminal result for the original
  command. Explicit detach, cancel, identity replacement, and timeout remain
  terminal boundaries.

Thread-safety: all public state transitions happen under a single reentrant
lock; the send callback is invoked *outside* the lock so a controller may
synchronously ``complete`` from inside its own send (the no-op round trip),
and waiters are parked on per-command events, not on the broker lock.
"""

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

#: Current wire protocol version. Registration requires this exact integer;
#: booleans are rejected even though ``bool`` subclasses ``int`` in Python.
BROWSER_CONTROL_PROTOCOL_VERSION = 1

#: Exact controller capability contract shared by every transport. The broker
#: never accepts arbitrary browser methods: raw CDP, script evaluation, console
#: access, uploads, and other privileged surfaces remain outside this allowlist.
BROWSER_CONTROL_CAPABILITIES = frozenset(
    {
        "controller.noop",
        "browser_back",
        "browser_click",
        "browser_navigate",
        "browser_press",
        "browser_screenshot",
        "browser_scroll",
        "browser_snapshot",
        "browser_tab_activate",
        "browser_tabs",
        "browser_type",
    }
)

#: Privileged capabilities (Phase 8 Task 30) that are never negotiable through
#: the base allowlist. ``browser_evaluate`` executes JavaScript in the page
#: context; ``browser_cdp`` is raw CDP. Both are fail-closed unless the broker
#: runs in Developer Mode (``browser.extension_control.developer_mode``) AND
#: the controller explicitly negotiated the capability.
BROWSER_CONTROL_DEVELOPER_CAPABILITIES = frozenset(
    {
        "browser_cdp",
        "browser_evaluate",
    }
)

#: Artifact-transport capabilities (Phase 8 Task 29). These are regular
#: (non-developer) capabilities because upload/download of bounded, validated
#: artifacts is a safe surface; the payloads are never carried in controller
#: frames. Artifact actions are dispatched only after the broker validates the
#: referenced artifact id against the attached store ("approved artifact id
#: only").
BROWSER_CONTROL_ARTIFACT_CAPABILITIES = frozenset(
    {
        "browser_artifact_download",
        "browser_artifact_upload",
    }
)

#: The complete set a controller may negotiate: base + artifact. Developer
#: capabilities are admitted by :func:`filter_browser_control_capabilities`
#: only when Developer Mode is enabled.
BROWSER_CONTROL_ALL_CAPABILITIES = frozenset(
    BROWSER_CONTROL_CAPABILITIES
    | BROWSER_CONTROL_ARTIFACT_CAPABILITIES
    | BROWSER_CONTROL_DEVELOPER_CAPABILITIES
)


def browser_control_protocol_supported(value: Any) -> bool:
    """Return whether ``value`` names the exact supported wire version."""
    return type(value) is int and value == BROWSER_CONTROL_PROTOCOL_VERSION


def browser_control_developer_mode(config: Optional[dict] = None) -> bool:
    """Return the explicit Developer Mode flag (disabled by default).

    Reads ``browser.extension_control.developer_mode`` from the global
    config. Developer Mode is the *additional* gate for ``browser_evaluate``
    and raw CDP; it never widens the base action allowlist on its own.
    """
    if config is None:
        try:
            # Read-only flag probe on every browser tool call / check_fn
            # evaluation: skip load_config()'s defensive deepcopy (~135us);
            # this function only reads nested dicts and never mutates.
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            return False
    if not isinstance(config, dict):
        return False
    browser = config.get("browser")
    if not isinstance(browser, dict):
        return False
    extension_control = browser.get("extension_control")
    if not isinstance(extension_control, dict):
        return False
    return extension_control.get("developer_mode", False) is True


def filter_browser_control_capabilities(
    value: Any,
    *,
    developer_mode: Optional[bool] = None,
) -> frozenset:
    """Return the permitted subset of a JSON/RPC capability list.

    A malformed non-list value has no capabilities. Unknown or non-string
    entries are ignored; registration rejects an empty returned set.

    Base and artifact capabilities always pass. Developer capabilities
    (``browser_evaluate``, ``browser_cdp``) pass only when Developer Mode
    is explicitly enabled — either passed in or read from the live config.
    """
    if not isinstance(value, list):
        return frozenset()
    allowed = frozenset(BROWSER_CONTROL_CAPABILITIES | BROWSER_CONTROL_ARTIFACT_CAPABILITIES)
    if developer_mode is None:
        developer_mode = browser_control_developer_mode()
    if developer_mode is True:
        allowed = frozenset(allowed | BROWSER_CONTROL_DEVELOPER_CAPABILITIES)
    return frozenset(
        capability
        for capability in value
        if isinstance(capability, str) and capability in allowed
    )

#: Wire method names for controller frames. Transport-neutral by contract:
#: transports carry these envelopes verbatim.
FRAME_COMMAND = "browser.controller.command"
FRAME_CANCEL = "browser.controller.cancel"


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
    """Exact identity of a browser controller plus its capability set.

    Equality is structural over *all* fields, so two scopes differing in any
    single field (including ``transport_family``) never match — this is the
    "exact identity" contract.
    """

    principal_id: Optional[str] = None
    profile_id: Optional[str] = None
    session_id: Optional[str] = None
    controller_id: Optional[str] = None
    browser_profile_id: Optional[str] = None
    transport_family: Optional[str] = None
    capabilities: frozenset = frozenset()


def _scope_identity(scope: ControllerScope) -> tuple:
    """Return stable controller identity, excluding negotiated capabilities."""
    return (
        scope.principal_id,
        scope.profile_id,
        scope.session_id,
        scope.controller_id,
        scope.browser_profile_id,
        scope.transport_family,
    )


def _same_scope_identity(first: ControllerScope, second: ControllerScope) -> bool:
    return _scope_identity(first) == _scope_identity(second)


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
    # Serialize command/cancel writes with detach or replacement.  Broker state
    # is never held while waiting for this lock, so a transport callback may
    # synchronously call complete() without deadlocking the broker.
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


class BrowserControlBroker:
    """Thread-safe broker core binding controllers to callers.

    Parameters
    ----------
    ticket_ttl:
        Lifetime of minted tickets in clock seconds.
    command_timeout:
        Seconds a ``dispatch`` waits for completion before raising
        :class:`ControllerTimeout`.
    clock:
        Injectable time source (defaults to ``time.monotonic``); tests pin it
        to make expiry deterministic.
    """

    def __init__(
        self,
        *,
        ticket_ttl: float = DEFAULT_TICKET_TTL,
        command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
        clock: Optional[Callable[[], float]] = None,
        developer_mode: Optional[bool] = None,
    ) -> None:
        self._ticket_ttl = ticket_ttl
        self._command_timeout = command_timeout
        self._clock = clock if clock is not None else time.monotonic
        self._lock = threading.RLock()
        self._tickets: Dict[str, _TicketRecord] = {}
        self._controllers: Dict[ControllerScope, _Controller] = {}
        self._pending: Dict[str, _PendingCommand] = {}
        # Developer Mode gates privileged capabilities (browser_evaluate,
        # browser_cdp). None defers to the live config on every selection so
        # a mid-process config change is honored without restart — including
        # REVOKING raw CDP/eval from already-attached controllers; an
        # explicit bool pins the gate for tests and multi-tenant hosts.
        self._developer_mode_pinned: Optional[bool] = (
            None if developer_mode is None else developer_mode is True
        )
        # Artifact stores keyed by resolved profile id; ``None`` is the
        # default/unscoped store (tests, single-profile hosts). A multiplex
        # listener attaches one store per profile so profile A touching the
        # artifact route first can never pin profile B to A's physical root.
        self._artifact_stores: Dict[Optional[str], Any] = {}

    def _developer_mode_now(self) -> bool:
        """Current Developer Mode authority (live config unless pinned)."""
        if self._developer_mode_pinned is not None:
            return self._developer_mode_pinned
        try:
            return browser_control_developer_mode()
        except Exception:
            return False

    def attach_artifact_store(
        self, store: Any, *, profile_id: Optional[str] = None
    ) -> None:
        """Attach an artifact store for "approved artifact id only".

        ``store`` must expose ``validate(artifact_id, *, scope) -> receipt``
        raising the artifacts module's :class:`ArtifactError` subclasses.
        ``profile_id`` scopes the store to one profile on multiplex hosts;
        ``None`` registers the default store. ``store=None`` clears that
        slot; dispatching an artifact action without a resolvable store
        fails closed.
        """
        if store is None:
            self._artifact_stores.pop(profile_id, None)
            return
        self._artifact_stores[profile_id] = store

    def _artifact_store_for_scope(self, scope: "ControllerScope") -> Any:
        """Select the artifact store for one controller scope.

        Prefers the exact profile-scoped store, falling back to the default
        (``None``) slot so single-profile hosts and existing tests keep the
        historical one-store behaviour.
        """
        profile = getattr(scope, "profile_id", None) or None
        store = self._artifact_stores.get(profile)
        if store is not None:
            return store
        return self._artifact_stores.get(None)

    @property
    def developer_mode(self) -> bool:
        """Whether privileged capabilities may be selected/dispatched."""
        return self._developer_mode_now()

    # ------------------------------------------------------------------
    # Registration tickets
    # ------------------------------------------------------------------

    def mint_ticket(self, scope: ControllerScope) -> Ticket:
        """Mint a short-lived, single-use ticket bound to ``scope``."""
        now = self._clock()
        with self._lock:
            self._prune_tickets(now)
            value = secrets.token_urlsafe(32)
            record = _TicketRecord(scope=scope, expires_at=now + self._ticket_ttl)
            self._tickets[value] = record
        return Ticket(value=value, expires_at=record.expires_at)

    def consume_ticket(self, value: str) -> ControllerScope:
        """Exchange a ticket for its scope, exactly once.

        Raises :class:`ControllerTicketInvalid` for unknown, already-consumed, or
        expired tickets. The expiry check happens against the live clock at
        consume time, so a ticket that outlived its TTL can never be used.
        """
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

    def _prune_tickets(self, now: float) -> None:
        """Drop expired tickets (caller must hold the lock)."""
        expired = [value for value, rec in self._tickets.items() if rec.expires_at <= now]
        for value in expired:
            del self._tickets[value]

    # ------------------------------------------------------------------
    # Controller registration / selection
    # ------------------------------------------------------------------

    def attach(
        self,
        scope: ControllerScope,
        send: Callable[[dict], None],
        *,
        owner: Any = None,
    ) -> None:
        """Attach or refresh the controller for one stable identity.

        A reconnect with the same principal/profile/session/controller/browser
        profile/transport identity refreshes the send callback and negotiated
        capabilities without cancelling pending work. Capabilities are not an
        identity field. A different controller or browser profile in the same
        authenticated session lane hard-replaces the previous identity.
        """
        while True:
            with self._lock:
                existing_entry = next(
                    (
                        (candidate_scope, controller)
                        for candidate_scope, controller in self._controllers.items()
                        if _same_scope_identity(candidate_scope, scope)
                    ),
                    None,
                )
                lane_scopes = [
                    candidate_scope
                    for candidate_scope in self._controllers
                    if candidate_scope.principal_id == scope.principal_id
                    and candidate_scope.profile_id == scope.profile_id
                    and candidate_scope.session_id == scope.session_id
                    and candidate_scope.transport_family == scope.transport_family
                    and not _same_scope_identity(candidate_scope, scope)
                ]
                if existing_entry is None and not lane_scopes:
                    self._controllers[scope] = _Controller(
                        scope=scope,
                        send=send,
                        owner=owner,
                    )
                    return

            # A different identity in the same authenticated session lane is a
            # hard replacement, not a recoverable reconnect. Terminalize it
            # before inserting the successor so session lookup stays unique.
            if lane_scopes:
                for lane_scope in lane_scopes:
                    self.detach(lane_scope, notify_controller=False)
                continue

            if existing_entry is not None:
                existing_scope, existing = existing_entry

            with existing.send_lock:
                with self._lock:
                    if self._controllers.get(existing_scope) is not existing:
                        continue
                    self._controllers.pop(existing_scope, None)
                    existing.scope = scope
                    existing.send = send
                    existing.owner = owner
                    existing.connected = False
                    for pending in self._pending.values():
                        if _same_scope_identity(pending.scope, scope):
                            pending.scope = scope
                    deferred = list(existing.deferred_cancels)
                    existing.deferred_cancels.clear()
                    self._controllers[scope] = existing

                unsent: list[dict] = []
                for index, frame in enumerate(deferred):
                    try:
                        send(frame)
                    except Exception:
                        logger.exception(
                            "failed to flush deferred browser-controller cancel"
                        )
                        unsent = deferred[index:]
                        break
                if unsent:
                    with self._lock:
                        if self._controllers.get(scope) is existing:
                            existing.deferred_cancels = unsent[
                                -MAX_DEFERRED_CANCELS:
                            ]
                    raise ConnectionError(
                        "browser controller reconnect could not flush deferred cancels"
                    )
                with self._lock:
                    if self._controllers.get(scope) is existing:
                        existing.connected = True
                return

    def select(self, scope: ControllerScope, capability: str) -> Optional[_Controller]:
        """Return the connected controller matching identity and capability.

        The caller's capabilities are not authoritative on reconnect. Selection
        matches the stable identity fields, then checks the attached
        controller's current negotiated capability set. Offline controllers
        preserve old pending work but never accept new dispatches.

        Privileged capabilities (``browser_evaluate``, ``browser_cdp``) are
        additionally gated on Developer Mode: with the gate off they are
        never selectable, even when a controller somehow negotiated them.
        The gate consults the LIVE flag on every selection (unless pinned at
        construction), so flipping ``developer_mode`` off in config revokes
        raw CDP/eval from already-attached controllers without a restart.
        """
        if (
            capability in BROWSER_CONTROL_DEVELOPER_CAPABILITIES
            and not self._developer_mode_now()
        ):
            return None
        with self._lock:
            matches = [
                controller
                for controller in self._controllers.values()
                if _same_scope_identity(controller.scope, scope)
                and controller.connected
                and capability in controller.scope.capabilities
            ]
        return matches[0] if len(matches) == 1 else None

    def is_owner(self, scope: ControllerScope, owner: Any) -> bool:
        """Return whether ``owner`` is the exact live transport for ``scope``.

        Ownership is independent of capabilities. Transport handlers use this
        for heartbeat and result admission so a least-privilege controller does
        not need to request ``controller.noop`` merely to complete a real action.
        """
        with self._lock:
            matches = [
                controller
                for controller in self._controllers.values()
                if _same_scope_identity(controller.scope, scope)
                and controller.connected
                and controller.owner is owner
            ]
        return len(matches) == 1

    def disconnect(
        self,
        scope: ControllerScope,
        *,
        owner: Any = _OWNER_UNSET,
    ) -> bool:
        """Mark one exact controller transport offline without cancelling work."""
        with self._lock:
            entry = next(
                (
                    (candidate_scope, controller)
                    for candidate_scope, controller in self._controllers.items()
                    if _same_scope_identity(candidate_scope, scope)
                ),
                None,
            )
        if entry is None:
            return False
        candidate_scope, controller = entry
        with controller.send_lock:
            with self._lock:
                if self._controllers.get(candidate_scope) is not controller:
                    return False
                if owner is not _OWNER_UNSET and controller.owner is not owner:
                    return False
                controller.connected = False
                controller.owner = None
        return True

    def detach(
        self,
        scope: ControllerScope,
        *,
        owner: Any = _OWNER_UNSET,
        notify_controller: bool = True,
    ) -> None:
        """Remove the controller for ``scope`` and fail its pending work closed.

        Every pending command of the scope is marked cancelled and resolved,
        so waiting dispatchers raise :class:`ControllerCancelled`; a late
        ``complete`` for any of them returns ``False`` (the command id is no
        longer pending).
        """
        with self._lock:
            controller = self._controllers.get(scope)
        if controller is None:
            return
        if owner is not _OWNER_UNSET and controller.owner != owner:
            return
        with controller.send_lock:
            with self._lock:
                if self._controllers.get(scope) is not controller:
                    return
                if owner is not _OWNER_UNSET and controller.owner != owner:
                    return
                self._controllers.pop(scope, None)
                pendings = self._pending_for_scope_locked(scope)
                for pending in pendings:
                    self._resolve_pending(pending, cancelled=True)
            # Keep the old generation's send lock through cancellation so a
            # command frame can never overtake its terminal cancel frame.
            if notify_controller:
                self._emit_cancel_frames(controller, pendings)

    # ------------------------------------------------------------------
    # Command lifecycle
    # ------------------------------------------------------------------

    def dispatch(
        self,
        scope: ControllerScope,
        *,
        action: str,
        arguments: Optional[dict] = None,
        tool_call_id: Optional[str] = None,
    ) -> Any:
        """Send one controller command and block for its completion.

        Emits a ``browser.controller.command`` frame carrying a fresh command
        id, then waits up to ``command_timeout`` seconds. Returns the
        controller's completion result, or raises:

        - :class:`ControllerUnavailable` — no exact scope/capability match;
        - :class:`ControllerCancelled` — cancelled via ``cancel``/``detach``;
        - :class:`ControllerTimeout` — no completion within the timeout;
        - :class:`ControllerRejected` — completed with ``ok=False``.

        Exactly one pending command exists per command id; ``complete`` is
        single-shot, so a command can never resolve twice.

        Artifact actions (``browser_artifact_upload`` /
        ``browser_artifact_download``) additionally require an attached
        artifact store and a live, scope-bound artifact reference: the
        ``arguments`` mapping must carry an ``artifact_id`` whose validation
        passes against the store ("approved artifact id only").  The payload
        is never carried in the frame — only the id travels to the
        controller.
        """
        controller = self.select(scope, action)
        if controller is None:
            raise ControllerUnavailable(
                f"no controller for scope {scope!r} with capability {action!r}"
            )

        arguments = dict(arguments or {})
        if action in BROWSER_CONTROL_ARTIFACT_CAPABILITIES:
            self._validate_artifact_reference(scope, action, arguments)

        command_id = secrets.token_hex(16)
        frame = {
            "method": FRAME_COMMAND,
            "params": {
                "command_id": command_id,
                "action": action,
                "arguments": arguments,
                "controller_id": scope.controller_id,
                "browser_profile_id": scope.browser_profile_id,
                "tool_call_id": tool_call_id,
            },
        }
        pending = _PendingCommand(
            scope=controller.scope,
            command_id=command_id,
            tool_call_id=tool_call_id,
        )
        with controller.send_lock:
            with self._lock:
                # select() intentionally runs outside the send lock. Revalidate
                # the exact live controller after acquiring it so disconnect or
                # identity replacement cannot leave a stale command waiting.
                attached = next(
                    (
                        candidate
                        for candidate in self._controllers.values()
                        if _same_scope_identity(candidate.scope, scope)
                    ),
                    None,
                )
                if attached is not controller or not controller.connected:
                    raise ControllerUnavailable(
                        f"controller for scope {scope!r} detached before dispatch"
                    )
                pending.scope = controller.scope
                self._pending[command_id] = pending

            try:
                controller.send(frame)
            except Exception:
                # The command never left the building; unreserve the id and
                # surface the transport failure to the caller.
                with self._lock:
                    self._pending.pop(command_id, None)
                raise

        if not pending.event.wait(timeout=self._command_timeout):
            timed_out = False
            with self._lock:
                # Event.wait() may return False at the exact boundary where a
                # completion already won and removed the pending command.
                if not pending.done and self._pending.get(command_id) is pending:
                    pending.done = True
                    del self._pending[command_id]
                    timed_out = True
            if timed_out:
                with controller.send_lock:
                    with self._lock:
                        active = next(
                            (
                                candidate
                                for candidate in self._controllers.values()
                                if _same_scope_identity(candidate.scope, scope)
                            ),
                            None,
                        )
                        if active is None:
                            active = controller
                        if not active.connected:
                            self._defer_cancel_locked(active, pending)
                            active = None
                    if active is not None:
                        self._emit_cancel_frames(active, [pending])
                raise ControllerTimeout(
                    f"controller did not complete command {command_id!r} "
                    f"within {self._command_timeout}s"
                )

        if pending.cancelled:
            raise ControllerCancelled(f"command {command_id!r} was cancelled")
        if not pending.ok:
            raise ControllerRejected(
                f"controller rejected command {command_id!r}: {pending.result!r}"
            )
        return pending.result

    def complete(
        self,
        command_id: str,
        *,
        scope: Optional[ControllerScope] = None,
        ok: bool,
        result: Any = None,
    ) -> bool:
        """Resolve a pending command by id; ``False`` when none is pending.

        Safe to call from inside the controller's own ``send`` callback (the
        broker never holds its lock across a send). Late completions — after
        ``cancel`` or ``detach`` already resolved the command — are ignored
        and report ``False``.
        """
        with self._lock:
            pending = self._pending.get(command_id)
            if pending is None or pending.done:
                return False
            if scope is not None and pending.scope != scope:
                return False
            pending.done = True
            pending.ok = ok is True
            pending.result = result
            del self._pending[command_id]
            pending.event.set()
        return True

    def cancel(self, scope: ControllerScope, *, tool_call_id: Optional[str]) -> bool:
        """Cancel exactly the pending command matching ``scope`` + tool_call_id.

        Emits one ``browser.controller.cancel`` frame naming the cancelled
        command's id. Returns ``True`` when a command was cancelled and
        ``False`` when nothing matched (so transports can answer idempotently
        without inventing state).
        """
        with self._lock:
            controller = next(
                (
                    candidate
                    for candidate in self._controllers.values()
                    if _same_scope_identity(candidate.scope, scope)
                ),
                None,
            )
        if controller is None or not controller.connected:
            return False
        with controller.send_lock:
            with self._lock:
                attached = next(
                    (
                        candidate
                        for candidate in self._controllers.values()
                        if _same_scope_identity(candidate.scope, scope)
                    ),
                    None,
                )
                if attached is not controller or not controller.connected:
                    return False
                target = None
                for pending in self._pending.values():
                    if (
                        _same_scope_identity(pending.scope, scope)
                        and pending.tool_call_id == tool_call_id
                        and not pending.done
                    ):
                        target = pending
                        break
                if target is None:
                    return False
                self._resolve_pending(target, cancelled=True)
            self._emit_cancel_frames(controller, [target])
            return True

    # ------------------------------------------------------------------
    # Internals (all callers must hold the lock)
    # ------------------------------------------------------------------

    def _resolve_pending(self, pending: _PendingCommand, *, cancelled: bool) -> None:
        """Mark ``pending`` resolved and drop it from the registry."""
        pending.cancelled = cancelled
        pending.done = True
        del self._pending[pending.command_id]
        pending.event.set()

    def _validate_artifact_reference(
        self,
        scope: ControllerScope,
        action: str,
        arguments: dict,
    ) -> None:
        """Fail closed unless ``arguments`` carries an approved artifact id.

        The store is consulted through the duck-typed ``validate`` contract
        (raises :class:`ArtifactError` subclasses on any problem), so the
        broker never guesses at artifact validity: missing store, missing id,
        traversal, expiry, checksum, or scope mismatch all surface as
        :class:`ControllerRejected` before any frame is emitted.
        """
        store = self._artifact_store_for_scope(scope)
        if store is None:
            raise ControllerRejected(
                f"{action} requires an attached artifact store"
            )
        artifact_id = arguments.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ControllerRejected(f"{action} requires a non-empty artifact_id")
        try:
            store.validate(artifact_id.strip(), scope=scope)
        except ControllerRejected:
            raise
        except Exception as exc:
            raise ControllerRejected(
                f"{action} rejected artifact reference {artifact_id!r}: {exc}"
            ) from exc

    @staticmethod
    def _cancel_frame(pending: _PendingCommand) -> dict:
        return {
            "method": FRAME_CANCEL,
            "params": {
                "command_id": pending.command_id,
                "tool_call_id": pending.tool_call_id,
            },
        }

    def _defer_cancel_locked(
        self,
        controller: _Controller,
        pending: _PendingCommand,
    ) -> None:
        controller.deferred_cancels.append(self._cancel_frame(pending))
        if len(controller.deferred_cancels) > MAX_DEFERRED_CANCELS:
            del controller.deferred_cancels[:-MAX_DEFERRED_CANCELS]

    def _pending_for_scope_locked(self, scope: ControllerScope) -> list[_PendingCommand]:
        return [
            pending
            for pending in list(self._pending.values())
            if _same_scope_identity(pending.scope, scope)
        ]

    def _emit_cancel_frames(
        self, controller: _Controller, pendings: list[_PendingCommand]
    ) -> None:
        for pending in pendings:
            frame = self._cancel_frame(pending)
            try:
                controller.send(frame)
            except Exception:
                logger.exception(
                    "failed to emit cancel frame for command %r", pending.command_id
                )

    def scope_for_session(
        self,
        *,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        principal_id: Optional[str] = None,
        transport_family: Optional[str] = None,
    ) -> Optional[ControllerScope]:
        """Return one unambiguous attached scope for a server-owned session.

        A public session id is only a lookup hint.  The caller must also supply
        its server-derived principal and transport family; missing identity,
        no match, or multiple matches fail closed rather than selecting by
        insertion order.
        """
        target = str(session_id or task_id or "").strip()
        principal = str(principal_id or "").strip()
        family = str(transport_family or "").strip()
        if not target or not principal or not family:
            return None
        with self._lock:
            matches = [
                scope
                for scope in self._controllers
                if scope.session_id == target
                and scope.principal_id == principal
                and scope.transport_family == family
            ]
        return matches[0] if len(matches) == 1 else None

    def lane_registered(
        self,
        *,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        principal_id: Optional[str] = None,
        transport_family: Optional[str] = None,
    ) -> bool:
        """Return whether ANY controller (even offline) registered for this lane.

        Distinguishes "a controller bound this session lane and is currently
        unavailable" (fail closed — the extension lane stays authoritative)
        from "no controller ever registered here" (the caller keeps the
        legacy browser backend). Ambiguous lanes report True so the caller
        still fails closed rather than silently switching browsers.
        """
        target = str(session_id or task_id or "").strip()
        principal = str(principal_id or "").strip()
        family = str(transport_family or "").strip()
        if not target or not principal or not family:
            return False
        with self._lock:
            return any(
                scope.session_id == target
                and scope.principal_id == principal
                and scope.transport_family == family
                for scope in self._controllers
            )

    def disconnect_owner(self, owner: Any) -> int:
        """Mark every controller owned by one lost transport offline."""
        with self._lock:
            scopes = [
                scope
                for scope, controller in self._controllers.items()
                if controller.owner is owner
            ]
        disconnected = 0
        for scope in scopes:
            disconnected += int(self.disconnect(scope, owner=owner))
        return disconnected

    def detach_owner(self, owner: Any, *, notify_controller: bool = True) -> int:
        """Hard-detach every controller owned by one transport connection."""
        with self._lock:
            scopes = [
                scope
                for scope, controller in self._controllers.items()
                if controller.owner is owner
            ]
        for scope in scopes:
            self.detach(
                scope,
                owner=owner,
                notify_controller=notify_controller,
            )
        return len(scopes)

    def reset(self) -> None:
        """Fail all live work closed and clear tickets (tests/shutdown)."""
        with self._lock:
            scopes = list(self._controllers)
        for scope in scopes:
            self.detach(scope)
        with self._lock:
            self._tickets.clear()
            # Defensive cleanup for any pending entry whose controller was
            # concurrently removed by a transport teardown.
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


def browser_control_enabled(config: Optional[dict] = None) -> bool:
    """Return the explicit browser-control feature flag (disabled by default)."""
    if config is None:
        try:
            # Read-only flag probe on every browser tool call / check_fn
            # evaluation: skip load_config()'s defensive deepcopy (~135us);
            # this function only reads nested dicts and never mutates.
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            return False
    if not isinstance(config, dict):
        return False
    browser = config.get("browser")
    if not isinstance(browser, dict):
        return False
    extension_control = browser.get("extension_control")
    if not isinstance(extension_control, dict):
        return False
    return extension_control.get("enabled", False) is True

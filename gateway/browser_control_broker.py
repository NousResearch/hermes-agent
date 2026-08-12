"""Transport-neutral browser-control broker core (Phase 4).

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
  :class:`TicketInvalid` for unknown, already-consumed, or expired values.
  The ticket is the only cross-transport credential minted here; transports
  decide how to carry it.

- **Exact identity and capability selection.** ``attach`` registers a send
  callback under a :class:`ControllerScope`; ``select`` returns a controller
  only when the caller's scope matches on *every* identity field —
  principal, profile, session, controller id, browser profile id, and
  transport family — and the requested capability is present in the
  controller's capability set. Partial matches return ``None``.

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

#: Wire method names for controller frames. Transport-neutral by contract:
#: transports carry these envelopes verbatim.
FRAME_COMMAND = "browser.controller.command"
FRAME_CANCEL = "browser.controller.cancel"


class BrowserControlError(Exception):
    """Base class for broker contract failures."""


class TicketInvalid(BrowserControlError):
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
    ) -> None:
        self._ticket_ttl = ticket_ttl
        self._command_timeout = command_timeout
        self._clock = clock if clock is not None else time.monotonic
        self._lock = threading.RLock()
        self._tickets: Dict[str, _TicketRecord] = {}
        self._controllers: Dict[ControllerScope, _Controller] = {}
        self._pending: Dict[str, _PendingCommand] = {}

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

        Raises :class:`TicketInvalid` for unknown, already-consumed, or
        expired tickets. The expiry check happens against the live clock at
        consume time, so a ticket that outlived its TTL can never be used.
        """
        now = self._clock()
        with self._lock:
            record = self._tickets.get(value)
            if record is None:
                raise TicketInvalid("unknown ticket")
            if record.consumed:
                raise TicketInvalid("ticket already consumed")
            if now > record.expires_at:
                raise TicketInvalid("ticket expired")
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
        """Register the controller owning ``scope`` with frame callback ``send``.

        Re-attaching an already-attached scope replaces the prior controller
        (logged); a controller that wants to go away must call ``detach``.
        """
        replacement = _Controller(scope=scope, send=send, owner=owner)
        with self._lock:
            existing = self._controllers.get(scope)
        if existing is None:
            with self._lock:
                # A concurrent attach may have won after the optimistic read;
                # retry through the replacement path rather than overwriting it.
                existing = self._controllers.get(scope)
                if existing is None:
                    self._controllers[scope] = replacement
                    return

        assert existing is not None
        logger.warning(
            "browser controller re-attached for scope %r; replacing prior controller",
            scope,
        )
        with existing.send_lock:
            with self._lock:
                if self._controllers.get(scope) is not existing:
                    # Another replacement won while this caller waited. Re-run
                    # against the new generation so its pending work is not
                    # orphaned by an unconditional overwrite.
                    retry = True
                    pendings = []
                else:
                    retry = False
                    pendings = self._pending_for_scope_locked(scope)
                    for pending in pendings:
                        self._resolve_pending(pending, cancelled=True)
                    self._controllers[scope] = replacement
            if not retry:
                self._emit_cancel_frames(existing, pendings)
                return
        self.attach(scope, send, owner=owner)

    def select(self, scope: ControllerScope, capability: str) -> Optional[_Controller]:
        """Return the controller exactly matching ``scope`` and ``capability``.

        ``None`` when any identity field differs or the capability is not in
        the controller's capability set. The controller's own scope is the
        authority on capabilities.
        """
        with self._lock:
            controller = self._controllers.get(scope)
            if controller is None:
                return None
            if capability not in controller.scope.capabilities:
                return None
            return controller

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
        """
        controller = self.select(scope, action)
        if controller is None:
            raise ControllerUnavailable(
                f"no controller for scope {scope!r} with capability {action!r}"
            )

        command_id = secrets.token_hex(16)
        frame = {
            "method": FRAME_COMMAND,
            "params": {
                "command_id": command_id,
                "action": action,
                "arguments": dict(arguments or {}),
                "controller_id": scope.controller_id,
                "browser_profile_id": scope.browser_profile_id,
                "tool_call_id": tool_call_id,
            },
        }
        pending = _PendingCommand(
            scope=scope,
            command_id=command_id,
            tool_call_id=tool_call_id,
        )
        with controller.send_lock:
            with self._lock:
                # select() intentionally runs outside the send lock. Revalidate
                # the exact controller generation after acquiring it so detach
                # or replacement cannot leave a stale command waiting forever.
                if self._controllers.get(scope) is not controller:
                    raise ControllerUnavailable(
                        f"controller for scope {scope!r} detached before dispatch"
                    )
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
                        still_attached = self._controllers.get(scope) is controller
                    if still_attached:
                        self._emit_cancel_frames(controller, [pending])
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
            controller = self._controllers.get(scope)
        if controller is None:
            return False
        with controller.send_lock:
            with self._lock:
                if self._controllers.get(scope) is not controller:
                    return False
                target = None
                for pending in self._pending.values():
                    if (
                        pending.scope == scope
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

    def _pending_for_scope_locked(self, scope: ControllerScope) -> list[_PendingCommand]:
        return [
            pending
            for pending in list(self._pending.values())
            if pending.scope == scope
        ]

    def _emit_cancel_frames(
        self, controller: _Controller, pendings: list[_PendingCommand]
    ) -> None:
        for pending in pendings:
            frame = {
                "method": FRAME_CANCEL,
                "params": {
                    "command_id": pending.command_id,
                    "tool_call_id": pending.tool_call_id,
                },
            }
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

    def detach_owner(self, owner: Any, *, notify_controller: bool = True) -> int:
        """Detach every controller owned by one transport connection."""
        with self._lock:
            scopes = [
                scope
                for scope, controller in self._controllers.items()
                if controller.owner == owner
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
    """Return the explicit Phase 4 feature flag (disabled by default)."""
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
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

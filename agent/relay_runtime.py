"""Profile-scoped NeMo Relay runtimes owned by the Hermes agent core."""

from __future__ import annotations

import atexit
import asyncio
import contextvars
import importlib
import inspect
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

SESSION_SCOPE = "hermes.session"
TURN_SCOPE = "hermes.turn"
LOGICAL_LLM_SCOPE = "hermes.logical_llm_call"
RUNTIME_SCHEMA_KEY = "hermes.relay.schema_version"
RUNTIME_SCHEMA_VERSION = "hermes.relay.runtime.v1"
RUNTIME_INSTANCE_KEY = "hermes.relay.runtime_instance"
_PROFILE_KEY_CACHE: dict[str, str] = {}
# Native ScopeHandle equality is identity-based across Python wrappers; compare
# by uuid when unwinding nested scopes after a LIFO mismatch.
_SCOPE_STACK_MISMATCH_MARKERS = (
    "scope handle is not at the top of the stack",
    "root scope cannot be removed",
)
_MAX_SCOPE_UNWIND_POPS = 64


@dataclass
class RelaySession:
    """One isolated Relay scope stack owned by a Hermes session."""

    session_id: str
    parent_session_id: str = ""
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    closing: bool = False
    handle: Any = None
    context: contextvars.Context | None = None
    # Set when a scope-stack mismatch forced a rebuild. Drain paths may use
    # this as a signal that waiting for graceful turn completion is futile.
    scope_stack_healed: bool = False


class RelayRuntime:
    """Own Relay session scopes independently of any exporter or plugin."""

    def __init__(self, relay: Any = None, *, profile_key: str | None = None) -> None:
        self.relay = relay or _load_nemo_relay()
        self.profile_key = profile_key or current_profile_key()
        self.runtime_id = uuid.uuid4().hex
        self._sessions_lock = threading.RLock()
        self._sessions: dict[str, RelaySession] = {}
        self._subagent_parents: dict[str, str] = {}
        self._subagent_parent_handles: dict[str, Any] = {}
        self._execution_consumers_lock = threading.RLock()
        self._execution_consumers: set[str] = set()
        self._shutdown_registered = True
        atexit.register(self.shutdown)

    def retain_managed_execution(self, consumer: str) -> None:
        """Keep managed LLM and tool execution active for one consumer."""
        if not consumer:
            raise ValueError("Relay managed-execution consumer must not be empty")
        with self._execution_consumers_lock:
            self._execution_consumers.add(consumer)

    def release_managed_execution(self, consumer: str) -> None:
        """Release a consumer's managed-execution requirement."""
        with self._execution_consumers_lock:
            self._execution_consumers.discard(consumer)

    def managed_execution_enabled(self) -> bool:
        """Return whether a Hermes-managed consumer needs the Relay pipeline."""
        with self._execution_consumers_lock:
            return bool(self._execution_consumers)

    def ensure_session(
        self,
        event: dict[str, Any],
        *,
        data: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> RelaySession | None:
        """Return the existing session scope or create it once."""
        session_id = _session_id(event)
        if not session_id:
            return None
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                parent_session_id = self._subagent_parents.get(session_id, "")
                session = RelaySession(
                    session_id=session_id,
                    parent_session_id=parent_session_id,
                )
                self._sessions[session_id] = session
        with session.lock:
            if session.closing:
                return None
            if session.handle is None:
                parent_handle = None
                scope_metadata = {
                    **(metadata or {}),
                    RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
                    RUNTIME_INSTANCE_KEY: self.runtime_id,
                }
                if session.parent_session_id:
                    with self._sessions_lock:
                        parent_handle = self._subagent_parent_handles.get(session_id)
                    if parent_handle is None:
                        parent = self.ensure_session({
                            "session_id": session.parent_session_id
                        })
                        if parent is not None:
                            parent_handle = parent.handle
                    scope_metadata["nemo_relay_scope_role"] = "subagent"
                context = contextvars.Context()
                try:
                    session.handle = context.run(
                        self.relay.scope.push,
                        SESSION_SCOPE,
                        self.relay.ScopeType.Agent,
                        handle=parent_handle,
                        data=data,
                        input={},
                        metadata=scope_metadata,
                    )
                except Exception:
                    session.context = None
                    raise
                session.context = context
        return session

    def register_subagent(
        self,
        event: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RelaySession | None:
        """Open a child Agent scope under its spawning turn when available."""
        parent_session_id = str(event.get("parent_session_id") or "")
        child_session_id = str(event.get("child_session_id") or "")
        if (
            not parent_session_id
            or not child_session_id
            or parent_session_id == child_session_id
        ):
            return None
        parent = self.ensure_session({"session_id": parent_session_id})
        parent_handle = None if parent is None else parent.handle
        turn = active_turn(parent_session_id)
        if (
            turn is not None
            and not turn.closed
            and turn.handle is not None
            and turn.lease.host is self
            and turn.lease.session is not None
            and turn.lease.session.session_id == parent_session_id
        ):
            parent_handle = turn.handle
        with self._sessions_lock:
            self._subagent_parents[child_session_id] = parent_session_id
            if parent_handle is not None:
                self._subagent_parent_handles[child_session_id] = parent_handle
        return self.ensure_session(
            {"session_id": child_session_id},
            metadata=metadata,
        )

    def unregister_subagent(self, event: dict[str, Any]) -> None:
        """Close a delegated session and forget its parent relationship."""
        child_session_id = str(event.get("child_session_id") or "")
        if not child_session_id:
            return
        self.close_session({"session_id": child_session_id})
        with self._sessions_lock:
            self._subagent_parents.pop(child_session_id, None)
            self._subagent_parent_handles.pop(child_session_id, None)

    def get_session(self, session_id: str) -> RelaySession | None:
        """Return an active Hermes Relay session without creating one."""
        with self._sessions_lock:
            session = self._sessions.get(str(session_id or ""))
        if session is None:
            return None
        with session.lock:
            return None if session.closing else session

    def get_session_handle(self, session_id: str) -> Any:
        """Return the Relay parent handle for a Hermes session, if active."""
        session = self.get_session(session_id)
        return None if session is None else session.handle

    def run_in_session(
        self,
        session: RelaySession,
        callback: Callable[..., Any],
        *args: Any,
        allow_closing: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run a Relay operation against a session's isolated scope stack."""
        with session.lock:
            if session.closing and not allow_closing:
                raise RuntimeError("Hermes Relay session is closing")
            if session.context is None or session.handle is None:
                raise RuntimeError("Hermes Relay session context is unavailable")
            relay_context = session.context.copy()

        context = contextvars.copy_context()
        for variable, value in relay_context.items():
            context.run(variable.set, value)

        def invoke() -> Any:
            self.relay.get_scope_stack()
            return callback(*args, **kwargs)

        # A copy permits a helper called by an existing Relay callback to
        # re-enter the same logical session without re-entering Context.
        return context.run(invoke)

    async def run_in_session_async(
        self,
        session: RelaySession,
        callback: Callable[..., Any],
        *args: Any,
        allow_closing: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Create and await an operation inside the session's saved context."""
        with session.lock:
            if session.closing and not allow_closing:
                raise RuntimeError("Hermes Relay session is closing")
            if session.context is None or session.handle is None:
                raise RuntimeError("Hermes Relay session context is unavailable")
            relay_context = session.context.copy()

        context = contextvars.copy_context()
        for variable, value in relay_context.items():
            context.run(variable.set, value)

        async def invoke() -> Any:
            self.relay.get_scope_stack()
            result = callback(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        task = context.run(asyncio.create_task, invoke())
        return await task

    def emit_mark(
        self,
        name: str,
        event: dict[str, Any],
        *,
        data: Any = None,
        metadata: Any = None,
    ) -> bool:
        """Emit a mark parented to the Hermes session identified by ``event``."""
        session = self.ensure_session(event)
        if session is None:
            return False
        self.run_in_session(
            session,
            self.relay.scope.event,
            name,
            handle=session.handle,
            data=data,
            metadata=metadata,
        )
        return True

    def apply_tool_request_intercepts(
        self,
        *,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply Relay request rewriting before Hermes authorizes a tool call."""
        if not self.managed_execution_enabled():
            return args
        request_intercepts = getattr(
            getattr(self.relay, "tools", None),
            "request_intercepts",
            None,
        )
        if not callable(request_intercepts):
            return args
        session = self.ensure_session({"session_id": session_id})
        if session is None:
            return args
        result = self.run_in_session(
            session,
            request_intercepts,
            tool_name,
            args,
        )
        return result if isinstance(result, dict) else args

    def close_session(self, event: dict[str, Any]) -> None:
        """Close one session scope and remove it from the core registry."""
        session_id = _session_id(event)
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            with self._sessions_lock:
                self._subagent_parents.pop(session_id, None)
                self._subagent_parent_handles.pop(session_id, None)
            return
        failures: list[str] = []
        with session.lock:
            if session.closing:
                return
            session.closing = True
            if session.handle is not None:
                try:
                    self.pop_scope_resilient(
                        session,
                        session.handle,
                        output={},
                        metadata={
                            RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
                            RUNTIME_INSTANCE_KEY: self.runtime_id,
                        },
                        allow_closing=True,
                        rebuild_on_failure=False,
                    )
                except Exception as exc:
                    # Closing abandons the native stack with the session
                    # object; a failed pop must not block registry cleanup.
                    failures.append(f"session scope close failed: {exc}")
                session.handle = None
                session.context = None
        try:
            self.relay.subscribers.flush()
        except Exception as exc:
            failures.append(f"subscriber flush failed: {exc}")
        with self._sessions_lock:
            if self._sessions.get(session_id) is session:
                self._sessions.pop(session_id, None)
            self._subagent_parents.pop(session_id, None)
            self._subagent_parent_handles.pop(session_id, None)
        if failures:
            logger.warning(
                "Hermes Relay session %s closed with errors: %s",
                session_id,
                "; ".join(failures),
            )

    def pop_scope_resilient(
        self,
        session: RelaySession,
        handle: Any,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        allow_closing: bool = False,
        rebuild_on_failure: bool = True,
        reason: str = "scope_pop",
    ) -> bool:
        """Pop ``handle``, unwinding nested orphans on LIFO mismatch.

        NeMo Relay scopes are strictly stack-ordered. Interrupted compaction,
        shared-metrics task scopes, or logical-LLM children left above a turn
        make ``scope.pop(turn_handle)`` raise::

            RuntimeError: invalid argument: scope handle is not at the top of the stack

        Catching that without healing leaves the session stack corrupted so
        every future ``end_turn`` / ``close_session`` logs the same error.
        This helper first tries a direct pop, then abandons nested scopes
        until ``handle`` is top (matched by uuid), then pops it. If the stack
        is still unusable, optionally rebuild a fresh session root so later
        turns start clean.
        """
        if handle is None:
            return True
        meta = metadata or {
            RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
            RUNTIME_INSTANCE_KEY: self.runtime_id,
        }
        try:
            self.run_in_session(
                session,
                self.relay.scope.pop,
                handle,
                output=output,
                metadata=meta,
                allow_closing=allow_closing,
            )
            return True
        except Exception as exc:
            if not _is_scope_stack_mismatch(exc):
                raise
            logger.warning(
                "Hermes Relay scope pop mismatched for session %s (%s); "
                "unwinding nested scopes",
                session.session_id,
                reason,
                exc_info=True,
            )

        target_uuid = _scope_handle_uuid(handle)
        session_uuid = _scope_handle_uuid(session.handle)
        abandoned = 0
        get_handle = getattr(self.relay.scope, "get_handle", None)
        if not callable(get_handle):
            if rebuild_on_failure and not session.closing:
                self.rebuild_session_scope(session, reason=reason)
                return False
            raise RuntimeError(
                f"Hermes Relay could not pop scope during {reason} for session "
                f"{session.session_id} (no get_handle to unwind)"
            )
        for _ in range(_MAX_SCOPE_UNWIND_POPS):
            try:
                top = self.run_in_session(
                    session,
                    get_handle,
                    allow_closing=allow_closing,
                )
            except Exception:
                break
            top_uuid = _scope_handle_uuid(top)
            top_name = str(getattr(top, "name", "") or "")
            if not top_uuid:
                break
            if target_uuid and top_uuid == target_uuid:
                try:
                    self.run_in_session(
                        session,
                        self.relay.scope.pop,
                        handle,
                        output=output,
                        metadata=meta,
                        allow_closing=allow_closing,
                    )
                    return True
                except Exception as exc:
                    if not _is_scope_stack_mismatch(exc):
                        raise
                    break
            if session_uuid and top_uuid == session_uuid:
                # Target is already gone; session root is current top.
                return False
            if top_name == "root":
                break
            try:
                self.run_in_session(
                    session,
                    self.relay.scope.pop,
                    top,
                    output={"outcome": "abandoned", "reason": reason},
                    metadata={
                        **meta,
                        "hermes.relay.abandoned_scope": True,
                    },
                    allow_closing=allow_closing,
                )
                abandoned += 1
            except Exception as exc:
                if _is_scope_stack_mismatch(exc) and "root scope" in str(exc).lower():
                    break
                logger.warning(
                    "Hermes Relay nested scope abandon failed for session %s",
                    session.session_id,
                    exc_info=True,
                )
                break

        if abandoned:
            logger.warning(
                "Hermes Relay abandoned %d nested scope(s) on session %s during %s",
                abandoned,
                session.session_id,
                reason,
            )

        if rebuild_on_failure and not session.closing:
            self.rebuild_session_scope(session, reason=reason)
            return False
        raise RuntimeError(
            f"Hermes Relay could not pop scope during {reason} for session "
            f"{session.session_id}"
        )

    def rebuild_session_scope(
        self,
        session: RelaySession,
        *,
        reason: str,
    ) -> None:
        """Replace a corrupted per-session scope stack with a fresh root.

        The previous Context/ScopeStack is abandoned (not shared with other
        sessions). Callers that still hold child handles on the old stack must
        treat those handles as invalid after this returns.
        """
        with session.lock:
            if session.closing:
                return
            parent_session_id = session.parent_session_id
            session_id = session.session_id

        # Resolve parent outside session.lock so we never nest
        # session.lock -> _sessions_lock (ensure_session takes the opposite order).
        parent_handle = None
        if parent_session_id:
            with self._sessions_lock:
                parent_handle = self._subagent_parent_handles.get(session_id)
            if parent_handle is None:
                parent = self.get_session(parent_session_id)
                if parent is not None:
                    parent_handle = parent.handle

        scope_metadata = {
            RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
            RUNTIME_INSTANCE_KEY: self.runtime_id,
            "hermes.relay.rebuilt": True,
            "hermes.relay.rebuild_reason": reason,
        }
        if parent_session_id:
            scope_metadata["nemo_relay_scope_role"] = "subagent"
        context = contextvars.Context()
        try:
            new_handle = context.run(
                self.relay.scope.push,
                SESSION_SCOPE,
                self.relay.ScopeType.Agent,
                handle=parent_handle,
                input={},
                metadata=scope_metadata,
            )
        except Exception:
            logger.warning(
                "Hermes Relay session %s rebuild failed after %s",
                session_id,
                reason,
                exc_info=True,
            )
            raise

        with session.lock:
            if session.closing:
                return
            session.handle = new_handle
            session.context = context
            session.scope_stack_healed = True
            logger.warning(
                "Hermes Relay rebuilt scope stack for session %s after %s",
                session_id,
                reason,
            )

    def shutdown(self) -> None:
        """Close all core-owned Relay session scopes."""
        with self._sessions_lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self._safe(self.close_session, {"session_id": session_id})
        if self._shutdown_registered:
            try:
                atexit.unregister(self.shutdown)
            except Exception:
                pass
            self._shutdown_registered = False

    @staticmethod
    def _safe(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return callback(*args, **kwargs)
        except Exception:
            logger.warning("Hermes Relay runtime operation failed", exc_info=True)
            return None


@dataclass(frozen=True)
class NoopRelayRuntime:
    """Explicit reduced-capability host for platforms without Relay wheels."""

    profile_key: str
    reason: str

    @property
    def available(self) -> bool:
        return False

    def apply_tool_request_intercepts(
        self,
        *,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        del session_id, tool_name
        return args

    @staticmethod
    def retain_managed_execution(consumer: str) -> None:
        del consumer

    @staticmethod
    def release_managed_execution(consumer: str) -> None:
        del consumer

    @staticmethod
    def managed_execution_enabled() -> bool:
        return False

    def shutdown(self) -> None:
        """No resources are allocated on unsupported platforms."""


RelayHost = RelayRuntime | NoopRelayRuntime


class RelayHostRegistry:
    """Own exactly one Relay host for each canonical Hermes profile."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._hosts: dict[str, RelayHost] = {}

    def for_profile(
        self,
        profile_key: str | None = None,
        *,
        create: bool = True,
    ) -> RelayHost | None:
        key = profile_key or current_profile_key()
        host = self._hosts.get(key)
        if host is not None or not create:
            return host
        with self._lock:
            host = self._hosts.get(key)
            if host is not None or not create:
                return host
            try:
                host = RelayRuntime(profile_key=key)
            except Exception as exc:
                logger.warning(
                    "Hermes Relay runtime initialization failed", exc_info=True
                )
                host = NoopRelayRuntime(profile_key=key, reason=str(exc))
            self._hosts[key] = host
            return host

    def shutdown_profile(self, profile_key: str) -> None:
        with self._lock:
            host = self._hosts.pop(profile_key, None)
        if host is not None:
            host.shutdown()

    def shutdown_all(self) -> None:
        with self._lock:
            hosts = list(self._hosts.values())
            self._hosts.clear()
        for host in hosts:
            host.shutdown()


HOST_REGISTRY = RelayHostRegistry()


@dataclass
class ConversationLease:
    """A resumable reference to one profile-scoped conversation scope."""

    profile_key: str
    session_id: str
    platform: str
    host: RelayHost
    session: RelaySession | None
    parent_session_id: str = ""
    released: bool = False


@dataclass
class RelayTurnContext:
    """Runtime-only context for one Hermes turn or top-level task."""

    lease: ConversationLease
    turn_id: str
    task_id: str
    handle: Any = None
    logical_llm_calls: dict[str, Any] = field(default_factory=dict, repr=False)
    logical_llm_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )
    finalize_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )
    _previous_turn: RelayTurnContext | None = field(default=None, repr=False)
    _active_registered: bool = field(default=False, repr=False)
    relay_enabled: bool = True
    closed: bool = False


_CURRENT_TURN: contextvars.ContextVar[RelayTurnContext | None] = contextvars.ContextVar(
    "hermes_relay_turn", default=None
)


class RelaySessionCoordinator:
    """Own semantic conversation and turn lifetimes for Hermes core."""

    def __init__(self, registry: RelayHostRegistry = HOST_REGISTRY) -> None:
        self.registry = registry
        self._initializer_lock = threading.RLock()
        self._session_initializers: dict[
            str,
            Callable[[RelayRuntime, dict[str, Any]], None],
        ] = {}
        self._active_turns_lock = threading.RLock()
        self._active_turns: dict[tuple[str, str], set[int]] = {}

    def register_session_initializer(
        self,
        name: str,
        callback: Callable[[RelayRuntime, dict[str, Any]], None],
    ) -> None:
        """Register idempotent profile/session preparation before scope creation."""
        with self._initializer_lock:
            self._session_initializers[name] = callback

    def unregister_session_initializer(self, name: str) -> None:
        """Remove a previously registered session initializer."""
        with self._initializer_lock:
            self._session_initializers.pop(name, None)

    def _prepare_session(
        self,
        host: RelayRuntime,
        context: dict[str, Any],
    ) -> None:
        with self._initializer_lock:
            initializers = list(self._session_initializers.items())
        for name, callback in initializers:
            try:
                callback(host, context)
            except Exception:
                logger.warning(
                    "Hermes Relay session initializer failed: %s",
                    name,
                    exc_info=True,
                )

    def acquire_conversation(
        self,
        *,
        profile_key: str,
        session_id: str,
        platform: str,
        parent_session_id: str = "",
        model: str = "",
    ) -> ConversationLease:
        host = self.registry.for_profile(profile_key)
        if host is None:
            host = NoopRelayRuntime(profile_key, "Relay host creation was disabled")
        session = None
        if isinstance(host, RelayRuntime):
            try:
                session_context = {
                    "profile_key": profile_key,
                    "session_id": session_id,
                    "platform": platform,
                    "parent_session_id": parent_session_id,
                    "model": model,
                }
                self._prepare_session(host, session_context)
                metadata = {"hermes.execution_surface": platform or "unknown"}
                if parent_session_id and parent_session_id != session_id:
                    session = host.register_subagent(
                        {
                            "parent_session_id": parent_session_id,
                            "child_session_id": session_id,
                        },
                        metadata=metadata,
                    )
                else:
                    session = host.ensure_session(
                        {"session_id": session_id},
                        metadata=metadata,
                    )
            except Exception:
                logger.warning(
                    "Hermes Relay conversation initialization failed",
                    exc_info=True,
                )
        return ConversationLease(
            profile_key=profile_key,
            session_id=session_id,
            platform=platform,
            host=host,
            session=session,
            parent_session_id=parent_session_id,
        )

    def begin_turn(
        self,
        lease: ConversationLease,
        *,
        turn_id: str,
        task_id: str,
    ) -> RelayTurnContext:
        if lease.released:
            raise RuntimeError("Hermes Relay conversation lease is released")
        turn = RelayTurnContext(lease=lease, turn_id=turn_id, task_id=task_id)
        key = (lease.profile_key, lease.session_id)
        with self._active_turns_lock:
            active = self._active_turns.get(key)
            if active:
                # A Relay session owns one physical scope stack. Concurrent
                # Hermes turns would create sibling scopes on that stack, but
                # their completion order is not guaranteed to be LIFO.
                turn.relay_enabled = False
                logger.warning(
                    "Skipping Relay instrumentation for concurrent Hermes turn "
                    "%s in session %s",
                    turn_id,
                    lease.session_id,
                )
            else:
                self._active_turns[key] = {id(turn)}
                turn._active_registered = True
        if (
            turn.relay_enabled
            and isinstance(lease.host, RelayRuntime)
            and lease.session is not None
        ):
            try:
                turn.handle = lease.host.run_in_session(
                    lease.session,
                    lease.host.relay.scope.push,
                    TURN_SCOPE,
                    lease.host.relay.ScopeType.Function,
                    handle=lease.session.handle,
                    input={},
                    metadata={
                        RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
                        RUNTIME_INSTANCE_KEY: lease.host.runtime_id,
                        "hermes.execution_surface": lease.platform or "unknown",
                    },
                )
            except Exception:
                logger.warning("Hermes Relay turn initialization failed", exc_info=True)
        turn._previous_turn = _CURRENT_TURN.get()
        _CURRENT_TURN.set(turn)
        return turn

    def end_turn(
        self,
        turn: RelayTurnContext,
        *,
        outcome: str,
    ) -> None:
        with turn.finalize_lock:
            if turn.closed:
                self._reset_turn_context(turn)
                return
            turn.closed = True
            lease = turn.lease
            try:
                if isinstance(lease.host, RelayRuntime) and lease.session is not None:
                    self._finish_logical_calls(turn, outcome=outcome)
                    if turn.handle is not None:
                        try:
                            lease.host.pop_scope_resilient(
                                lease.session,
                                turn.handle,
                                output={"outcome": outcome},
                                metadata={
                                    RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
                                    RUNTIME_INSTANCE_KEY: lease.host.runtime_id,
                                },
                                reason="turn_finalization",
                            )
                        except Exception:
                            logger.warning(
                                "Hermes Relay turn finalization failed", exc_info=True
                            )
                        finally:
                            # Handles on a rebuilt stack are invalid; clear so
                            # a double end_turn cannot re-pop a stale uuid.
                            turn.handle = None
                            with turn.logical_llm_lock:
                                turn.logical_llm_calls.clear()
            finally:
                try:
                    # Delegated agents own one turn. Close their conversation
                    # while the active-turn guard is still held so a parent
                    # timeout fallback cannot race this terminal boundary.
                    if (
                        lease.parent_session_id
                        and isinstance(lease.host, RelayRuntime)
                    ):
                        lease.host.unregister_subagent({
                            "child_session_id": lease.session_id
                        })
                except Exception:
                    logger.warning(
                        "Hermes Relay child conversation finalization failed",
                        exc_info=True,
                    )
                finally:
                    self._unregister_active_turn(turn)
                    self._reset_turn_context(turn)

    def has_active_turn(self, *, profile_key: str, session_id: str) -> bool:
        """Return whether a turn is still running for one profile/session."""
        key = (profile_key, session_id)
        with self._active_turns_lock:
            return bool(self._active_turns.get(key))

    def _unregister_active_turn(self, turn: RelayTurnContext) -> None:
        if not turn._active_registered:
            return
        key = (turn.lease.profile_key, turn.lease.session_id)
        with self._active_turns_lock:
            active = self._active_turns.get(key)
            if active is not None:
                active.discard(id(turn))
                if not active:
                    self._active_turns.pop(key, None)
            turn._active_registered = False

    def _reset_active_turns_for_tests(self) -> None:
        with self._active_turns_lock:
            self._active_turns.clear()

    def finish_logical_calls(
        self,
        turn: RelayTurnContext,
        *,
        outcome: str,
    ) -> None:
        """Close logical LLM children before sibling task aggregation scopes."""
        with turn.finalize_lock:
            if turn.closed:
                return
            self._finish_logical_calls(turn, outcome=outcome)

    @staticmethod
    def _finish_logical_calls(
        turn: RelayTurnContext,
        *,
        outcome: str,
    ) -> None:
        lease = turn.lease
        if not isinstance(lease.host, RelayRuntime) or lease.session is None:
            return
        with turn.logical_llm_lock:
            logical_calls = list(turn.logical_llm_calls.items())
            turn.logical_llm_calls.clear()
        for index in range(len(logical_calls) - 1, -1, -1):
            request_id, logical_handle = logical_calls[index]
            try:
                lease.host.pop_scope_resilient(
                    lease.session,
                    logical_handle,
                    output={"outcome": outcome},
                    metadata={
                        RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
                        RUNTIME_INSTANCE_KEY: lease.host.runtime_id,
                    },
                    # Leave rebuild to turn finalization so one heal covers
                    # the whole orphaned prefix above the turn handle.
                    rebuild_on_failure=False,
                    reason="logical_llm_finalization",
                )
            except Exception:
                with turn.logical_llm_lock:
                    # Relay scopes are stack-owned. If the newest remaining
                    # handle cannot close, older handles cannot close safely
                    # either, so retain the unclosed prefix for diagnostics.
                    for pending_request_id, pending_handle in logical_calls[
                        : index + 1
                    ]:
                        turn.logical_llm_calls.setdefault(
                            pending_request_id,
                            pending_handle,
                        )
                logger.warning(
                    "Hermes Relay logical LLM finalization failed",
                    exc_info=True,
                )
                break

    @staticmethod
    def _reset_turn_context(turn: RelayTurnContext) -> None:
        """Unwind ``turn`` without disturbing a newer context-local turn."""
        if _CURRENT_TURN.get() is not turn:
            return
        previous = turn._previous_turn
        seen = {id(turn)}
        while previous is not None and previous.closed:
            if id(previous) in seen:
                previous = None
                break
            seen.add(id(previous))
            previous = previous._previous_turn
        _CURRENT_TURN.set(previous)

    @staticmethod
    def release_conversation(lease: ConversationLease) -> None:
        """Release a caller lease without closing a resumable conversation."""
        lease.released = True

    def finalize_conversation(
        self,
        *,
        profile_key: str,
        session_id: str,
    ) -> None:
        host = self.registry.for_profile(profile_key, create=False)
        if isinstance(host, RelayRuntime):
            host.close_session({"session_id": session_id})

    def shutdown_profile(self, profile_key: str) -> None:
        self.registry.shutdown_profile(profile_key)


SESSION_COORDINATOR = RelaySessionCoordinator()


def current_turn() -> RelayTurnContext | None:
    """Return the turn context inherited by current async and thread work."""
    return _CURRENT_TURN.get()


def relay_instrumentation_enabled() -> bool:
    """Return whether this inherited turn may create Relay instrumentation."""
    turn = current_turn()
    return turn is None or (turn.relay_enabled and not turn.closed)


def active_turn(session_id: str | None = None) -> RelayTurnContext | None:
    """Return a live turn only when it belongs to the active profile/session."""
    turn = current_turn()
    if (
        turn is None
        or not turn.relay_enabled
        or turn.closed
        or turn.lease.released
    ):
        return None
    if turn.lease.profile_key != current_profile_key():
        return None
    if session_id is not None and turn.lease.session_id != session_id:
        return None
    if isinstance(turn.lease.host, RelayRuntime):
        if turn.lease.session is None:
            return None
        if turn.lease.host.get_session(turn.lease.session_id) is not turn.lease.session:
            return None
    return turn


def resolve_execution_context(
    session_id: str,
) -> tuple[RelayRuntime | None, RelaySession | None, Any]:
    """Resolve one active turn/session parent for managed Relay execution."""
    inherited_turn = current_turn()
    if inherited_turn is not None and (
        not inherited_turn.relay_enabled or inherited_turn.closed
    ):
        return None, None, None
    turn = active_turn(session_id)
    if (
        turn is not None
        and isinstance(turn.lease.host, RelayRuntime)
        and turn.lease.session is not None
    ):
        session = turn.lease.session
        return turn.lease.host, session, turn.handle or session.handle
    # Managed-execution consumers create and retain the profile host before
    # reaching an out-of-turn adapter. Do not initialize Relay for the default
    # no-consumer path.
    runtime = get_runtime(create=False)
    if runtime is None:
        return None, None, None
    if not runtime.managed_execution_enabled():
        return None, None, None
    session = runtime.get_session(session_id)
    if session is None:
        session = runtime.ensure_session({"session_id": session_id})
    return runtime, session, None if session is None else session.handle


def emit_mark(
    name: str,
    *,
    session_id: str,
    data: Any = None,
    metadata: Any = None,
) -> bool:
    """Emit a fail-open Relay mark under a Hermes session."""
    runtime = get_runtime(create=False)
    if runtime is None:
        return False
    try:
        return runtime.emit_mark(
            name,
            {"session_id": session_id},
            data=data,
            metadata=metadata,
        )
    except Exception:
        logger.warning("Hermes Relay mark failed: %s", name, exc_info=True)
        return False


def apply_tool_request_intercepts(
    *,
    session_id: str,
    tool_name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Return Relay-rewritten arguments at Hermes's authorization boundary."""
    if not session_id:
        return args
    runtime = get_runtime(create=False)
    if runtime is None:
        return args
    return runtime.apply_tool_request_intercepts(
        session_id=session_id,
        tool_name=tool_name,
        args=args,
    )


def ensure_session(*, session_id: str, **context: Any) -> RelaySession | None:
    """Create or return the shared Relay session used by Hermes core."""
    runtime = get_runtime()
    if runtime is None:
        return None
    try:
        return runtime.ensure_session({"session_id": session_id, **context})
    except Exception:
        logger.warning("Hermes Relay session initialization failed", exc_info=True)
        return None


def run_in_session(
    session_id: str,
    callback: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a scope, LLM, or tool API against a shared Hermes session."""
    runtime = get_runtime()
    if runtime is None:
        raise RuntimeError("Hermes Relay runtime is unavailable")
    session = runtime.get_session(session_id)
    if session is None:
        session = runtime.ensure_session({"session_id": session_id})
    if session is None:
        raise RuntimeError("Hermes Relay session is unavailable")
    return runtime.run_in_session(session, callback, *args, **kwargs)


async def run_in_session_async(
    session_id: str,
    callback: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Await a Relay operation inside a shared Hermes session context."""
    runtime = get_runtime()
    if runtime is None:
        raise RuntimeError("Hermes Relay runtime is unavailable")
    session = runtime.get_session(session_id)
    if session is None:
        session = runtime.ensure_session({"session_id": session_id})
    if session is None:
        raise RuntimeError("Hermes Relay session is unavailable")
    return await runtime.run_in_session_async(session, callback, *args, **kwargs)


def get_session_handle(session_id: str) -> Any:
    """Return the shared Relay handle for direct core instrumentation."""
    runtime = get_runtime(create=False)
    return None if runtime is None else runtime.get_session_handle(session_id)


def _is_relay_wrapped_callback_error(
    relay_error: BaseException,
    callback_error: BaseException,
) -> bool:
    """Match Relay's native callback wrapper without masking policy errors."""
    if relay_error is callback_error:
        return True
    if not isinstance(relay_error, RuntimeError):
        return False
    callback_type = callback_error.__class__
    type_names = {
        callback_type.__name__,
        callback_type.__qualname__,
        f"{callback_type.__module__}.{callback_type.__qualname__}",
    }
    message = str(relay_error)
    return any(
        message.startswith(f"internal error: {type_name}: {callback_error}")
        for type_name in type_names
    )


def get_runtime(
    *,
    create: bool = True,
    profile_key: str | None = None,
) -> RelayRuntime | None:
    """Return the Relay host for the active Hermes profile."""
    host = HOST_REGISTRY.for_profile(profile_key, create=create)
    return host if isinstance(host, RelayRuntime) else None


def get_host(
    *,
    create: bool = True,
    profile_key: str | None = None,
) -> RelayHost | None:
    """Return the explicit real or reduced-capability host for a profile."""
    return HOST_REGISTRY.for_profile(profile_key, create=create)


def current_profile_key() -> str:
    """Return the canonical profile identity used for runtime isolation."""
    home = get_hermes_home().expanduser()
    if not home.is_absolute():
        return str(home.resolve())
    raw = str(home)
    cached = _PROFILE_KEY_CACHE.get(raw)
    if cached is not None:
        return cached
    resolved = str(home.resolve())
    return _PROFILE_KEY_CACHE.setdefault(raw, resolved)


def _load_nemo_relay() -> Any:
    """Load the binding only when a producer or consumer needs Relay."""
    return importlib.import_module("nemo_relay")


def _session_id(event: dict[str, Any]) -> str:
    return str(event.get("session_id") or "")


def _scope_handle_uuid(handle: Any) -> str:
    """Return the stable native uuid for a ScopeHandle-like object."""
    if handle is None:
        return ""
    uuid_value = getattr(handle, "uuid", None)
    if uuid_value is not None:
        return str(uuid_value)
    if isinstance(handle, tuple) and len(handle) >= 3:
        # Fake handles used by unit tests: ("scope", name, serial)
        return str(handle[2])
    return str(handle)


def _is_scope_stack_mismatch(exc: BaseException) -> bool:
    """Detect native LIFO / root-protection errors from nemo_relay.scope.pop."""
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _SCOPE_STACK_MISMATCH_MARKERS)


def _reset_for_tests() -> None:
    """Reset all profile-scoped Relay hosts for isolated tests."""
    SESSION_COORDINATOR._reset_active_turns_for_tests()
    HOST_REGISTRY.shutdown_all()
    _PROFILE_KEY_CACHE.clear()

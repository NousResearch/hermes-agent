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
from concurrent.futures import TimeoutError as FuturesTimeoutError
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

# Bound for native scope lifecycle operations (push/pop/flush) that gate
# turn/session completion.  Healthy operations complete in microseconds;
# only a wedged native pipeline breaches this, and the correct trade there
# is one lost span, never a blocked agent (2026-08-10 delegation stall).
_SCOPE_OP_TIMEOUT = 10.0

_SCOPE_OP_EXECUTOR: Any = None
_SCOPE_OP_EXECUTOR_LOCK = threading.Lock()


def _scope_op_executor():
    """Shared daemon executor for bounded native scope operations.

    Daemon workers (tools.daemon_pool) so a wedged native call abandoned at
    timeout cannot block interpreter exit.  Sized generously: workers are
    only consumed for the duration of healthy (microsecond) operations plus
    any wedged calls, and ``Future.result(timeout=...)`` bounds callers even
    when every worker is consumed by wedged calls — an unstarted future
    still honors the result timeout, so exhaustion degrades to fast
    timeouts, never a new hang.
    """
    global _SCOPE_OP_EXECUTOR
    if _SCOPE_OP_EXECUTOR is None:
        with _SCOPE_OP_EXECUTOR_LOCK:
            if _SCOPE_OP_EXECUTOR is None:
                from tools.daemon_pool import DaemonThreadPoolExecutor

                _SCOPE_OP_EXECUTOR = DaemonThreadPoolExecutor(
                    max_workers=8,
                    thread_name_prefix="relay-scope-op",
                )
    return _SCOPE_OP_EXECUTOR


@dataclass
class RelaySession:
    """One isolated Relay scope stack owned by a Hermes session."""

    session_id: str
    parent_session_id: str = ""
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    closing: bool = False
    handle: Any = None
    context: contextvars.Context | None = None


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
                    try:
                        session.handle = _scope_op_executor().submit(
                            context.run,
                            self.relay.scope.push,
                            SESSION_SCOPE,
                            self.relay.ScopeType.Agent,
                            handle=parent_handle,
                            data=data,
                            input={},
                            metadata=scope_metadata,
                        ).result(timeout=_SCOPE_OP_TIMEOUT)
                    except RuntimeError:
                        # Interpreter shutdown: executor refuses new futures;
                        # push synchronously (no agent turn waits at exit).
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
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run a Relay operation against a session's isolated scope stack.

        ``timeout`` (seconds) bounds the native call by running it on a
        shared daemon executor; ``TimeoutError`` propagates to the caller's
        existing exception handling on breach.  ``None`` (default) preserves
        the historical synchronous behavior.  Scope lifecycle operations
        that gate turn/session completion pass ``_SCOPE_OP_TIMEOUT``: the
        native binding's ``scope.pop`` "returns after the scope is closed
        successfully" — unbounded — and a wedged native pipeline (proven
        live 2026-08-10 in the delegation topology) must cost at most one
        span, never the agent.  The abandoned daemon worker cannot block
        process exit (tools.daemon_pool contract).
        """
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
        if timeout is None:
            return context.run(invoke)
        try:
            future = _scope_op_executor().submit(context.run, invoke)
        except RuntimeError:
            # Interpreter shutdown: the executor refuses new futures, but
            # the atexit close path must still flush cleanly.  No agent
            # turn is waiting at shutdown, so the unbounded call is safe.
            return context.run(invoke)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            raise TimeoutError(
                f"Relay scope operation exceeded {timeout}s "
                f"(session={session.session_id}); abandoning the native call "
                "so the agent can continue — the span for this scope is lost"
            ) from exc

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
                    self.run_in_session(
                        session,
                        self.relay.scope.pop,
                        session.handle,
                        output={},
                        metadata={
                            RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
                            RUNTIME_INSTANCE_KEY: self.runtime_id,
                        },
                        allow_closing=True,
                        timeout=_SCOPE_OP_TIMEOUT,
                    )
                except Exception as exc:
                    failures.append(f"session scope close failed: {exc}")
        try:
            try:
                _scope_op_executor().submit(
                    self.relay.subscribers.flush
                ).result(timeout=_SCOPE_OP_TIMEOUT)
            except RuntimeError:
                # Interpreter shutdown: executor refuses new futures; flush
                # synchronously so the atexit close path still exports.
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

# Depth of managed Relay callbacks executing on the current logical call path.
# Set >0 while the native Relay pipeline is mid-dispatch of a Hermes callback
# (tool or LLM). Nested managed execution inside that window is structurally
# broken — the native pipeline binds its Futures to the outer, blocked event
# loop — so resolve_execution_context() bypasses Relay while the flag is set.
# ContextVar so the marker follows contextvars.copy_context() into the worker
# threads / per-thread loops that tools use for their internal async work.
_MANAGED_CALLBACK_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "hermes_relay_managed_callback_depth", default=0
)


class managed_callback_guard:
    """Mark the current context as inside a managed Relay callback.

    Synchronous context manager used by the relay adapters around the
    ``invoke()`` callbacks they hand to the native pipeline. Everything the
    callback transitively calls (including work it forwards to worker threads
    via ``contextvars.copy_context()``) sees the marker and runs unmanaged.
    """

    def __enter__(self) -> "managed_callback_guard":
        self._token = _MANAGED_CALLBACK_DEPTH.set(_MANAGED_CALLBACK_DEPTH.get() + 1)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        _MANAGED_CALLBACK_DEPTH.reset(self._token)


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
                    timeout=_SCOPE_OP_TIMEOUT,
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
                            lease.host.run_in_session(
                                lease.session,
                                lease.host.relay.scope.pop,
                                turn.handle,
                                output={"outcome": outcome},
                                metadata={
                                    RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
                                    RUNTIME_INSTANCE_KEY: lease.host.runtime_id,
                                },
                                timeout=_SCOPE_OP_TIMEOUT,
                            )
                        except Exception:
                            logger.warning(
                                "Hermes Relay turn finalization failed", exc_info=True
                            )
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
                lease.host.run_in_session(
                    lease.session,
                    lease.host.relay.scope.pop,
                    logical_handle,
                    output={"outcome": outcome},
                    metadata={
                        RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
                        RUNTIME_INSTANCE_KEY: lease.host.runtime_id,
                    },
                    timeout=_SCOPE_OP_TIMEOUT,
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
    if _MANAGED_CALLBACK_DEPTH.get() > 0:
        # A managed Relay callback is already executing on this logical call
        # path (e.g. the native ``tools.execute`` pipeline is mid-dispatch of
        # a Hermes tool). Nested managed execution here is structurally
        # impossible: the native pipeline binds its Futures to the OUTER
        # call's event loop, which is blocked inside the synchronous tool
        # callback until the tool returns. A nested managed LLM call (the
        # vision_analyze auxiliary path) therefore awaits a foreign-loop
        # Future that can never complete — "attached to a different loop"
        # at best, deadlock at worst, and "Event loop is closed" during
        # shutdown when the orphaned Future is completed late (#77244).
        # Run nested calls unmanaged; the outer tool scope still records
        # the tool-level event for observability.
        return None, None, None
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


def _reset_for_tests() -> None:
    """Reset all profile-scoped Relay hosts for isolated tests."""
    SESSION_COORDINATOR._reset_active_turns_for_tests()
    HOST_REGISTRY.shutdown_all()
    _PROFILE_KEY_CACHE.clear()

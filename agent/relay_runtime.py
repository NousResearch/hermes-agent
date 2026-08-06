"""Profile-scoped NeMo Relay runtimes owned by the Hermes agent core."""

from __future__ import annotations

import atexit
import asyncio
import contextvars
import importlib
import inspect
import logging
import os
import threading
import tomllib
import uuid
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

SESSION_SCOPE = "hermes.session"
TURN_SCOPE = "hermes.turn"
LOGICAL_LLM_SCOPE = "hermes.logical_llm_call"
RUNTIME_SCHEMA_KEY = "hermes.relay.schema_version"
RUNTIME_SCHEMA_VERSION = "hermes.relay.runtime.v1"
RUNTIME_INSTANCE_KEY = "hermes.relay.runtime_instance"
RELAY_PLUGINS_CONFIG_ENV = "HERMES_NEMO_RELAY_PLUGINS_TOML"
RELAY_PLUGINS_EXECUTION_CONSUMER = "hermes.nemo_relay.plugins"
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


def _run_bounded_on_exit_thread(fn: Callable[[], Any], timeout: float) -> Any:
    """Bounded fallback lane for interpreter shutdown.

    When the shared executor refuses new futures (interpreter shutdown),
    the operation still must not run unbounded on the calling thread: a
    wedged native call would block process exit forever — the same defect
    class this module exists to prevent, on the exit lane.  Run it on a
    fresh daemon thread with a bounded join; on breach the daemon worker
    is abandoned exactly like the executor lane abandons its worker.
    """
    result: list[Any] = []
    error: list[BaseException] = []

    def _target() -> None:
        try:
            result.append(fn())
        except BaseException as exc:  # noqa: BLE001 - propagated below
            error.append(exc)

    worker = threading.Thread(
        target=_target, daemon=True, name="relay-scope-op-exit"
    )
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(
            f"Relay scope operation exceeded {timeout}s during interpreter "
            "shutdown; abandoning the native call so process exit can proceed"
        )
    if error:
        raise error[0]
    return result[0] if result else None


def pop_relay_scope(
    relay: Any,
    handle: Any,
    *,
    output: Any = None,
    metadata: Any = None,
    timestamp: Any = None,
) -> Any:
    """Pop a Relay scope without passing kwargs the binding rejects.

    NeMo Relay ``scope.pop`` gained ``metadata`` in 0.4+. Older wheels (e.g.
    0.3.x) raise ``TypeError: pop() got an unexpected keyword argument
    'metadata'`` when Hermes finalization forwards runtime metadata. Filter to
    parameters the live binding accepts so turn/session close can complete.
    """
    pop = relay.scope.pop
    kwargs: dict[str, Any] = {}
    if output is not None:
        kwargs["output"] = output
    if metadata is not None:
        kwargs["metadata"] = metadata
    if timestamp is not None:
        kwargs["timestamp"] = timestamp
    try:
        params = inspect.signature(pop).parameters
    except (TypeError, ValueError):
        params = {}
    if params and not any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    ):
        kwargs = {key: value for key, value in kwargs.items() if key in params}
    return pop(handle, **kwargs)


@dataclass
class RelaySession:
    """One isolated Relay scope stack owned by a Hermes session."""

    session_id: str
    parent_session_id: str = ""
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    closing: bool = False
    handle: Any = None
    context: contextvars.Context | None = None
    # --- session-span segmentation (continuous sessions) ---
    # Segment index of the CURRENT session scope (0 = first). Rotation
    # closes the current scope and pushes segment N+1 at a turn boundary.
    segment: int = 0
    # Turns completed within the current segment (max_turns accounting).
    segment_turns: int = 0
    # Set by compaction notification; consumed at the next begin_turn.
    rotate_pending: bool = False
    # Rotating compaction landed while a turn was live on THIS session:
    # closing now would pop the session scope under a live turn scope
    # (LIFO violation). end_turn consumes this and closes the session.
    close_pending: bool = False


# ---------------------------------------------------------------------------
# Session-span segmentation config (gateway.telemetry.session_segments).
# Cached at first read; both defaults OFF => rotation never fires and the
# scope lifecycle is byte-identical to the pre-segmentation behavior.
# ---------------------------------------------------------------------------

_SEGMENTS_CONFIG: dict[str, Any] | None = None
_SEGMENTS_CONFIG_LOCK = threading.Lock()


def _segments_config() -> dict[str, Any]:
    """Resolve session-segmentation settings; inert defaults when unset."""
    global _SEGMENTS_CONFIG
    if _SEGMENTS_CONFIG is None:
        with _SEGMENTS_CONFIG_LOCK:
            if _SEGMENTS_CONFIG is None:
                on_compaction = False
                max_turns = 0
                try:
                    from gateway.run import _load_gateway_config  # late import

                    telemetry = (
                        (_load_gateway_config().get("gateway") or {}).get(
                            "telemetry"
                        )
                        or {}
                    )
                    segments = telemetry.get("session_segments") or {}
                    on_compaction = bool(segments.get("on_compaction", False))
                    try:
                        max_turns = max(0, int(segments.get("max_turns", 0) or 0))
                    except (TypeError, ValueError):
                        max_turns = 0
                except Exception:  # noqa: BLE001 - config absence must not crash
                    pass
                _SEGMENTS_CONFIG = {
                    "on_compaction": on_compaction,
                    "max_turns": max_turns,
                }
    return _SEGMENTS_CONFIG


def _reset_segments_config_for_tests() -> None:
    global _SEGMENTS_CONFIG
    _SEGMENTS_CONFIG = None


class RelayOperationLease:
    """Keep process-wide Relay plugins alive across a deferred operation."""

    def __init__(self, runtime: "RelayRuntime") -> None:
        self._lock = threading.Lock()
        self._runtime: RelayRuntime | None = runtime

    def run_in_session(
        self,
        session: RelaySession,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run cleanup while this lease still owns the runtime lifetime."""
        with self._lock:
            runtime = self._runtime
            if runtime is None:
                raise RuntimeError("Hermes Relay operation lease is released")
            return runtime._run_in_session_untracked(
                session,
                callback,
                *args,
                **kwargs,
            )

    def release(self) -> None:
        """Release this lease exactly once."""
        with self._lock:
            runtime = self._runtime
            self._runtime = None
        if runtime is not None:
            runtime._end_operation()


class _ProcessRelayPluginConfiguration:
    """Own one Relay plugin configuration across profile-scoped hosts."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._owners: set[int] = set()
        self._active = False
        self._relay: Any = None
        self._activation: Any = None

    def acquire(self, owner: Any, relay: Any) -> bool:
        """Join the process configuration, initializing it for the first host."""
        owner_id = id(owner)
        with self._lock:
            if owner_id in self._owners:
                return self._active
            if self._owners:
                self._owners.add(owner_id)
                return self._active
            if self._active and not self._clear_active():
                logger.warning(
                    "Hermes Relay plugin cleanup is still pending; refusing to "
                    "replace the process-global configuration"
                )
                return False

            try:
                plugin_mod = getattr(relay, "plugin", None)
                plugin_config, dynamic_plugins = _configured_plugin_inputs(relay)
                if dynamic_plugins:
                    initialize_dynamic = getattr(
                        plugin_mod,
                        "initialize_with_dynamic_plugins",
                        None,
                    )
                    if callable(initialize_dynamic):
                        try:
                            activation = _resolve_plugin_awaitable(
                                initialize_dynamic(plugin_config, dynamic_plugins)
                            )
                            if activation is None:
                                raise RuntimeError(
                                    "NeMo Relay dynamic plugin initialization "
                                    "returned no activation handle"
                                )
                            self._activation = activation
                        except Exception:
                            logger.warning(
                                "Hermes Relay dynamic plugin activation failed; "
                                "continuing with configured and discovered "
                                "static plugins",
                                exc_info=True,
                            )
                    else:
                        logger.warning(
                            "Hermes Relay dynamic plugins require a binding that "
                            "exposes plugin.initialize_with_dynamic_plugins; "
                            "continuing with configured and discovered static "
                            "plugins"
                        )

                if self._activation is None:
                    initialize = getattr(plugin_mod, "initialize", None)
                    if not callable(initialize):
                        raise RuntimeError(
                            "installed NeMo Relay binding does not expose "
                            "plugin.initialize"
                        )
                    # Relay owns ambient file discovery and precedence. An explicit
                    # Hermes file, when configured, is supplied as the final overlay.
                    _resolve_plugin_awaitable(initialize(plugin_config))
            except Exception as exc:
                self._activation = None
                logger.warning(
                    "Hermes Relay plugin initialization failed: %s",
                    exc,
                    exc_info=True,
                )
                return False

            self._owners.add(owner_id)
            self._active = True
            self._relay = relay
            return True

    def release(self, owner: Any) -> None:
        """Release one host and clear Relay after the final host exits."""
        owner_id = id(owner)
        with self._lock:
            if owner_id not in self._owners:
                return
            self._owners.remove(owner_id)
            if self._owners:
                return
            self._clear_active()

    def reset_for_tests(self) -> None:
        """Clear process-global state left by directly constructed test hosts."""
        with self._lock:
            self._owners.clear()
            self._clear_active()

    def retry_pending_cleanup(self) -> None:
        """Retry a failed final cleanup without disrupting live owners."""
        with self._lock:
            if not self._owners:
                self._clear_active()

    def _clear_active(self) -> bool:
        relay = self._relay
        activation = self._activation
        active = self._active
        if not active or relay is None:
            return True
        try:
            _flush_relay_subscribers(relay)
        except Exception:
            logger.warning(
                "Hermes Relay plugin subscriber flush failed",
                exc_info=True,
            )
            return False
        try:
            if activation is not None:
                close = getattr(activation, "close", None)
                if not callable(close):
                    raise RuntimeError(
                        "NeMo Relay dynamic plugin activation has no close method"
                    )
                _resolve_plugin_awaitable(close())
            else:
                _clear_relay_plugins(relay)
        except Exception:
            logger.warning(
                "Hermes Relay plugin configuration cleanup failed",
                exc_info=True,
            )
            return False
        self._active = False
        self._relay = None
        self._activation = None
        return True


_PLUGIN_CONFIGURATION = _ProcessRelayPluginConfiguration()
atexit.register(_PLUGIN_CONFIGURATION.retry_pending_cleanup)


class RelayRuntime:
    """Own Relay session scopes and optional process plugin configuration."""

    def __init__(self, relay: Any = None, *, profile_key: str | None = None) -> None:
        self.relay = relay or _load_nemo_relay()
        self.profile_key = profile_key or current_profile_key()
        self.runtime_id = uuid.uuid4().hex
        self._sessions_lock = threading.RLock()
        self._sessions: dict[str, RelaySession] = {}
        self._subagent_parents: dict[str, str] = {}
        self._subagent_parent_handles: dict[str, Any] = {}
        self._closing = False
        self._shutdown_started = False
        self._shutdown_complete = threading.Event()
        self._operations_idle = threading.Event()
        self._operations_idle.set()
        self._active_operations = 0
        self._execution_consumers_lock = threading.RLock()
        self._execution_consumers: set[str] = set()
        self._plugin_configuration_registered = _PLUGIN_CONFIGURATION.acquire(
            self,
            self.relay,
        )
        if self._plugin_configuration_registered:
            self.retain_managed_execution(RELAY_PLUGINS_EXECUTION_CONSUMER)
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
            if self._closing:
                return None
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

    def rotate_session_scope(self, session: RelaySession, *, reason: str) -> None:
        """Close the current session scope and open the next segment.

        Called ONLY at a turn boundary (before the turn scope pushes), never
        mid-turn: the scope stack is LIFO and rotating under a live child
        would close a parent out of order. Both native calls are bounded by
        ``_SCOPE_OP_TIMEOUT`` — a wedged rotation costs one segment span,
        never the agent. Segment bookkeeping advances even when a native
        call fails, so a degraded rotation cannot retry on every turn.
        """
        with session.lock:
            if session.closing or session.handle is None:
                return
            old_handle = session.handle
            # Advance bookkeeping FIRST: a failed native call must not leave
            # rotate_pending set (tight rotation loop on every turn).
            session.segment += 1
            session.segment_turns = 0
            session.rotate_pending = False
            try:
                self.run_in_session(
                    session,
                    self.relay.scope.pop,
                    old_handle,
                    output={"hermes.session.segment_reason": reason},
                    metadata={
                        RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
                        RUNTIME_INSTANCE_KEY: self.runtime_id,
                    },
                    timeout=_SCOPE_OP_TIMEOUT,
                )
            except Exception:
                logger.warning(
                    "Hermes Relay segment close failed (session=%s segment=%d); "
                    "abandoning the old segment span",
                    session.session_id,
                    session.segment - 1,
                    exc_info=True,
                )
            scope_metadata = {
                RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
                RUNTIME_INSTANCE_KEY: self.runtime_id,
                "hermes.session.segment": session.segment,
                "hermes.session.segment_reason": reason,
            }
            parent_handle = None
            if session.parent_session_id:
                with self._sessions_lock:
                    parent_handle = self._subagent_parent_handles.get(
                        session.session_id
                    )
                scope_metadata["nemo_relay_scope_role"] = "subagent"
            context = contextvars.Context()
            try:
                session.handle = _scope_op_executor().submit(
                    context.run,
                    self.relay.scope.push,
                    SESSION_SCOPE,
                    self.relay.ScopeType.Agent,
                    handle=parent_handle,
                    input={},
                    metadata=scope_metadata,
                ).result(timeout=_SCOPE_OP_TIMEOUT)
                session.context = context
            except Exception:
                logger.warning(
                    "Hermes Relay segment open failed (session=%s segment=%d); "
                    "keeping the prior scope handle",
                    session.session_id,
                    session.segment,
                    exc_info=True,
                )

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
            if self._closing:
                return None
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
            if self._closing:
                return None
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
        self._begin_operation()
        try:
            return self._run_in_session_untracked(
                session,
                callback,
                *args,
                allow_closing=allow_closing,
                timeout=timeout,
                **kwargs,
            )
        finally:
            self._end_operation()

    def _run_in_session_untracked(
        self,
        session: RelaySession,
        callback: Callable[..., Any],
        *args: Any,
        allow_closing: bool = False,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run inside a session whose host-level lifetime is already held."""
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
            # the atexit close path must still flush cleanly.  Still
            # bounded — a wedged native call must not block process exit
            # (the CI runner hang, 2026-08-12: 6 tests passed in 4s, then
            # the file-timeout SIGKILL'd a process stuck in this lane).
            return _run_bounded_on_exit_thread(
                lambda: context.run(invoke), timeout
            )
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
        self._begin_operation()
        try:
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
        finally:
            self._end_operation()

    def _begin_operation(self) -> None:
        """Admit one Relay call while keeping process plugins alive."""
        with self._sessions_lock:
            if self._closing:
                raise RuntimeError("Hermes Relay runtime is shutting down")
            self._active_operations += 1
            self._operations_idle.clear()

    def _end_operation(self) -> None:
        with self._sessions_lock:
            self._active_operations -= 1
            if self._active_operations == 0:
                self._operations_idle.set()

    def acquire_operation_lease(self) -> RelayOperationLease:
        """Retain plugin lifetime for work that outlives one Relay await."""
        self._begin_operation()
        return RelayOperationLease(self)

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

    def _close_scope_handle(
        self,
        session: RelaySession,
        handle: Any,
        *,
        output: dict[str, Any] | None = None,
        allow_closing: bool = False,
        failure_label: str = "scope close failed",
        drain_limit: int = 32,
    ) -> str | None:
        """Pop ``handle``, draining orphaned children in the same session context.

        Relay scopes are strict LIFO. Empty-stream retries + interrupt can
        abandon a physical LLM scope above TURN/SESSION (#81521). Drain and
        close must run inside one ``run_in_session`` callback so ContextVar
        stack views stay consistent across pops.
        """
        if handle is None:
            return None
        metadata = {
            RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
            RUNTIME_INSTANCE_KEY: self.runtime_id,
        }
        close_output = output or {}
        session_root = session.handle
        drained_holder = {"count": 0}
        error_holder: dict[str, BaseException] = {}

        def close_with_drain() -> None:
            def current_top() -> Any:
                # Version-correct accessor first: the pinned nemo-relay
                # binding exposes ``scope.get_handle()`` returning the
                # current top-of-stack ScopeHandle.  Its
                # ``get_scope_stack()`` returns a native ScopeStack object
                # that ``scope.pop`` rejects with TypeError, so it must
                # never be treated as a handle (#81601 review).
                get_handle = getattr(
                    getattr(self.relay, "scope", None), "get_handle", None
                )
                if callable(get_handle):
                    try:
                        return get_handle()
                    except Exception:
                        pass
                top = self.relay.get_scope_stack()
                # Some Relay builds return the live stack (list). Others
                # return the top handle directly — including tuple handles
                # like ("scope", name, serial) from the test fake. Only
                # unwrap real list stacks; never index a handle tuple.
                if isinstance(top, list):
                    return top[-1] if top else None
                return top

            def same_handle(a: Any, b: Any) -> bool:
                # Native ScopeHandle instances do not implement __eq__ by
                # value — two handles for the same scope compare unequal —
                # so compare by uuid when both sides expose one.
                if a is None or b is None:
                    return a is b
                if a is b or a == b:
                    return True
                a_uuid = getattr(a, "uuid", None)
                b_uuid = getattr(b, "uuid", None)
                return a_uuid is not None and a_uuid == b_uuid

            try:
                pop_relay_scope(
                    self.relay,
                    handle,
                    output=close_output,
                    metadata=metadata,
                )
                return
            except Exception as first_exc:
                error_holder["first"] = first_exc

            for _ in range(drain_limit):
                top = current_top()
                if top is None or same_handle(top, handle):
                    break
                # Never pop the session root while draining for a nested handle.
                if (
                    session_root is not None
                    and same_handle(top, session_root)
                    and handle is not session_root
                ):
                    break
                try:
                    pop_relay_scope(
                        self.relay,
                        top,
                        output={
                            "outcome": "cancelled",
                            "hermes.orphan_drain": True,
                        },
                        metadata=metadata,
                    )
                    drained_holder["count"] += 1
                except Exception as drain_exc:
                    error_holder["drain"] = drain_exc
                    logger.warning(
                        "Hermes Relay orphaned scope drain failed",
                        exc_info=True,
                    )
                    break

            if drained_holder["count"]:
                logger.warning(
                    "Hermes Relay drained %d orphaned scope(s) before closing %s",
                    drained_holder["count"],
                    handle,
                )
            try:
                pop_relay_scope(
                    self.relay,
                    handle,
                    output=close_output,
                    metadata=metadata,
                )
                error_holder.pop("first", None)
                error_holder.pop("drain", None)
            except Exception as retry_exc:
                error_holder["retry"] = retry_exc

        try:
            self.run_in_session(
                session,
                close_with_drain,
                allow_closing=allow_closing,
                # Bound the whole drain+close like the direct pops it
                # replaced: a wedged native pipeline must cost at most one
                # span, never block turn/session completion (see
                # tests/agent/test_relay_runtime_bounded_scope_ops.py).
                timeout=_SCOPE_OP_TIMEOUT,
            )
        except Exception as exc:
            return f"{failure_label}: {exc}"
        retry_exc = error_holder.get("retry") or error_holder.get("first")
        if retry_exc is not None:
            return f"{failure_label}: {retry_exc}"
        return None

    def close_session(self, event: dict[str, Any]) -> None:
        """Close one session scope and remove it from the core registry."""
        try:
            self._begin_operation()
        except RuntimeError:
            return
        try:
            self._close_session(event)
        finally:
            self._end_operation()

    def _close_session(self, event: dict[str, Any]) -> None:
        """Close one session already admitted by the host lifecycle gate."""
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
                failure = self._close_scope_handle(
                    session,
                    session.handle,
                    output={},
                    allow_closing=True,
                    failure_label="session scope close failed",
                )
                if failure:
                    failures.append(failure)
        # Subscriber flushing is process-wide and may wait for publications
        # owned by other sessions. Final plugin teardown flushes once after all
        # tracked operations drain; doing it here can deadlock an asyncio loop.
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
        """Close core scopes and release process plugin configuration."""
        with self._sessions_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True
            self._closing = True
            has_active_operations = self._active_operations > 0
        if has_active_operations:
            thread = threading.Thread(
                target=self._finish_shutdown_after_operations,
                name=f"hermes-nemo-relay-shutdown-{self.runtime_id[:8]}",
                daemon=True,
            )
            try:
                thread.start()
            except Exception:
                with self._sessions_lock:
                    self._shutdown_started = False
                logger.warning(
                    "Hermes Relay deferred shutdown could not start",
                    exc_info=True,
                )
            return
        self._finish_shutdown()

    def _finish_shutdown_after_operations(self) -> None:
        self._operations_idle.wait()
        self._finish_shutdown()

    def _finish_shutdown(self) -> None:
        try:
            with self._sessions_lock:
                session_ids = list(self._sessions)
            for session_id in session_ids:
                self._safe(self._close_session, {"session_id": session_id})
            if self._plugin_configuration_registered:
                self.release_managed_execution(RELAY_PLUGINS_EXECUTION_CONSUMER)
                _PLUGIN_CONFIGURATION.release(self)
                self._plugin_configuration_registered = False
            if self._shutdown_registered:
                try:
                    atexit.unregister(self.shutdown)
                except Exception:
                    pass
                self._shutdown_registered = False
        except Exception:
            with self._sessions_lock:
                self._shutdown_started = False
            logger.warning("Hermes Relay shutdown failed", exc_info=True)
            return
        with self._sessions_lock:
            self._shutdown_complete.set()

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
            # Session-span segmentation: consume a pending rotation (set by
            # compaction) or the max_turns cap HERE — the only point where
            # no turn scope is live on this session's stack, so the session
            # scope can close/reopen without violating LIFO order.
            try:
                config = _segments_config()
                session = lease.session
                cap = config["max_turns"]
                if (config["on_compaction"] and session.rotate_pending) or (
                    cap > 0 and session.segment_turns >= cap
                ):
                    reason = (
                        "compaction"
                        if config["on_compaction"] and session.rotate_pending
                        else "max_turns"
                    )
                    lease.host.rotate_session_scope(session, reason=reason)
            except Exception:
                logger.warning(
                    "Hermes Relay segment rotation failed", exc_info=True
                )
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
                        failure = lease.host._close_scope_handle(
                            lease.session,
                            turn.handle,
                            output={"outcome": outcome},
                            failure_label="turn scope close failed",
                        )
                        if failure:
                            logger.warning(
                                "Hermes Relay turn finalization failed: %s",
                                failure,
                            )
            finally:
                try:
                    # Segment turn accounting (max_turns rotation trigger).
                    if (
                        turn._active_registered
                        and isinstance(lease.host, RelayRuntime)
                        and lease.session is not None
                    ):
                        with lease.session.lock:
                            lease.session.segment_turns += 1
                except Exception:  # noqa: BLE001 - accounting must never block
                    pass
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
                self._consume_deferred_close(lease)

    def _consume_deferred_close(self, lease: Any) -> None:
        """Close a session whose rotating-compaction close was deferred.

        ``notify_session_compacted`` sets ``close_pending`` instead of
        closing when the old session still has a live turn (closing then
        would pop the session scope under the live turn scope — LIFO
        violation). The turn that was live consumes the flag here, after
        its own turn scope popped and it unregistered from the
        active-turn table. Skips when another turn is still live on the
        same session; that turn's end_turn will consume it instead.
        """
        try:
            if not (
                isinstance(lease.host, RelayRuntime) and lease.session is not None
            ):
                return
            session = lease.session
            with session.lock:
                pending = session.close_pending and not session.closing
            if not pending:
                return
            if self.has_active_turn(
                profile_key=lease.profile_key, session_id=lease.session_id
            ):
                return
            lease.host.close_session({"session_id": lease.session_id})
        except Exception:  # noqa: BLE001 - telemetry must never block end_turn
            logger.warning(
                "Hermes Relay deferred session close failed", exc_info=True
            )

    def notify_session_compacted(
        self,
        *,
        profile_key: str,
        session_id: str,
        old_session_id: str = "",
    ) -> None:
        """React to a completed compaction, per compaction mode.

        In-place compaction (``old_session_id`` empty or equal to
        ``session_id``): flag the session for segment rotation at its next
        turn boundary. Never rotates immediately — a compaction can
        complete while a turn is live, and rotation under a live turn
        scope would violate the scope stack's LIFO order; ``begin_turn``
        consumes the flag.

        Legacy rotating compaction (``old_session_id`` differs): the next
        turn acquires a fresh Relay session under the new id on its own,
        but the OLD session's scope would stay open forever — an
        unexported orphan. Close it now so the pre-compaction segment
        exports.

        Unknown sessions and disabled config are silent no-ops; this
        method must never add work or failure modes to the compaction
        critical path.
        """
        try:
            if not _segments_config()["on_compaction"]:
                return
            host = self.registry.for_profile(profile_key)
            if not isinstance(host, RelayRuntime):
                return
            if old_session_id and old_session_id != session_id:
                # Rotating compaction: export the orphaned pre-compaction
                # session scope (close_session is already bounded). If a
                # turn is still LIVE on the old session, closing now would
                # pop the session scope under the live turn scope (LIFO
                # violation) — defer to that turn's end_turn instead.
                with host._sessions_lock:
                    old_session = host._sessions.get(old_session_id)
                if old_session is not None and self.has_active_turn(
                    profile_key=profile_key, session_id=old_session_id
                ):
                    with old_session.lock:
                        if not old_session.closing:
                            old_session.close_pending = True
                    return
                host.close_session({"session_id": old_session_id})
                return
            with host._sessions_lock:
                session = host._sessions.get(session_id)
            if session is None:
                return
            with session.lock:
                if not session.closing:
                    session.rotate_pending = True
        except Exception:  # noqa: BLE001 - telemetry must never block compaction
            logger.warning(
                "Hermes Relay compaction notification failed", exc_info=True
            )

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
            failure = lease.host._close_scope_handle(
                lease.session,
                logical_handle,
                output={"outcome": outcome},
                failure_label="logical LLM scope close failed",
            )
            if failure is None:
                continue
            with turn.logical_llm_lock:
                # Relay scopes are stack-owned. If the newest remaining
                # handle cannot close even after orphan drain, older
                # handles cannot close safely either — retain the
                # unclosed prefix for diagnostics (#81521).
                for pending_request_id, pending_handle in logical_calls[
                    : index + 1
                ]:
                    turn.logical_llm_calls.setdefault(
                        pending_request_id,
                        pending_handle,
                    )
            logger.warning("Hermes Relay logical LLM finalization failed: %s", failure)
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


def _configured_plugin_inputs(relay: Any) -> tuple[dict[str, Any], list[Any]]:
    """Load one explicit static overlay and dynamic host specs."""
    configured = os.environ.get(RELAY_PLUGINS_CONFIG_ENV, "").strip()
    if not configured:
        return {}, []

    config_path = Path(configured).expanduser()
    try:
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)
        legacy_dynamic_plugins = _dynamic_plugin_specs(config, config_path)
        plugins_section = config.get("plugins")
        if plugins_section is not None and legacy_dynamic_plugins:
            raise ValueError(
                "configure either Relay [[plugins.dynamic]] records or Hermes "
                "[[dynamic_plugins]] activation specs, not both"
            )

        dynamic_plugins = legacy_dynamic_plugins
        if plugins_section:
            plugin_mod = getattr(relay, "plugin", None)
            load_dynamic_plugins = getattr(
                plugin_mod,
                "load_dynamic_plugin_activation_specs",
                None,
            )
            if not callable(load_dynamic_plugins):
                raise RuntimeError(
                    "installed NeMo Relay binding does not expose "
                    "plugin.load_dynamic_plugin_activation_specs"
                )
            dynamic_plugins = load_dynamic_plugins(config_path)
        plugin_config = dict(config)
        plugin_config.pop("dynamic_plugins", None)
        plugin_config.pop("plugins", None)
        return plugin_config, dynamic_plugins
    except Exception:
        logger.warning(
            "Hermes Relay plugin configuration could not be loaded from %s; "
            "continuing with discovered static plugins",
            config_path,
            exc_info=True,
        )
        return {}, []


def _dynamic_plugin_specs(
    config: dict[str, Any],
    config_path: Path,
) -> list[dict[str, Any]]:
    """Validate Hermes worker activation specs pending Relay lifecycle loading."""
    plugins_section = config.get("plugins")
    if plugins_section is not None and not isinstance(plugins_section, dict):
        raise ValueError("[plugins] must be a table")

    raw_specs = config.get("dynamic_plugins")
    if raw_specs is None:
        return []
    if not isinstance(raw_specs, list):
        raise ValueError("dynamic_plugins must be an array of tables")

    specs: list[dict[str, Any]] = []
    for index, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, dict):
            raise ValueError(f"dynamic_plugins[{index}] must be a table")
        plugin_id = raw_spec.get("plugin_id")
        kind = raw_spec.get("kind")
        manifest_ref = raw_spec.get("manifest_ref")
        environment_ref = raw_spec.get("environment_ref")
        plugin_config = raw_spec.get("config", {})
        if not isinstance(plugin_id, str) or not plugin_id.strip():
            raise ValueError(f"dynamic_plugins[{index}].plugin_id is required")
        if kind not in {"rust_dynamic", "worker"}:
            raise ValueError(
                f"dynamic_plugins[{index}].kind must be rust_dynamic or worker"
            )
        if not isinstance(manifest_ref, str) or not manifest_ref.strip():
            raise ValueError(f"dynamic_plugins[{index}].manifest_ref is required")
        if not isinstance(plugin_config, dict):
            raise ValueError(f"dynamic_plugins[{index}].config must be a table")
        if environment_ref is not None and (
            not isinstance(environment_ref, str) or not environment_ref.strip()
        ):
            raise ValueError(
                f"dynamic_plugins[{index}].environment_ref must be a non-empty string"
            )

        spec: dict[str, Any] = {
            "plugin_id": plugin_id.strip(),
            "kind": kind,
            "manifest_ref": _config_relative_path(
                manifest_ref.strip(),
                config_path,
            ),
            "config": plugin_config,
        }
        if environment_ref is not None:
            spec["environment_ref"] = _config_relative_path(
                environment_ref.strip(),
                config_path,
            )
        specs.append(spec)
    return specs


def _config_relative_path(value: str, config_path: Path) -> str:
    """Resolve one activation path relative to its physical TOML file."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((config_path.resolve().parent / path).resolve())


def _flush_relay_subscribers(relay: Any) -> None:
    """Flush Relay without blocking a 0.7 asyncio event-loop thread."""
    subscribers = getattr(relay, "subscribers", None)
    flush_async = getattr(subscribers, "flush_async", None)
    if callable(flush_async):
        _resolve_plugin_awaitable(flush_async())
        return
    flush = getattr(subscribers, "flush", None)
    if callable(flush):
        _resolve_plugin_awaitable(flush())


def _clear_relay_plugins(relay: Any) -> None:
    """Clear Relay plugins through the newest available binding API."""
    plugin_mod = getattr(relay, "plugin", None)
    clear_async = getattr(plugin_mod, "clear_async", None)
    if callable(clear_async):
        _resolve_plugin_awaitable(clear_async())
        return
    clear = getattr(plugin_mod, "clear", None)
    if callable(clear):
        _resolve_plugin_awaitable(clear())


def _resolve_plugin_awaitable(value: Any) -> Any:
    """Resolve Relay's async plugin API from synchronous host construction."""
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(value)
        except BaseException as exc:  # pragma: no cover - re-raised below
            error["exc"] = exc

    thread = threading.Thread(
        target=_runner,
        name="hermes-nemo-relay-plugin-lifecycle",
        daemon=True,
    )
    thread.start()
    thread.join()
    if "exc" in error:
        raise error["exc"]
    return result.get("value")


def _session_id(event: dict[str, Any]) -> str:
    return str(event.get("session_id") or "")


def _reset_for_tests() -> None:
    """Reset all profile-scoped Relay hosts for isolated tests."""
    SESSION_COORDINATOR._reset_active_turns_for_tests()
    HOST_REGISTRY.shutdown_all()
    _PLUGIN_CONFIGURATION.reset_for_tests()
    _PROFILE_KEY_CACHE.clear()

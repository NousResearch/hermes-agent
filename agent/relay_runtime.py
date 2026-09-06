"""Profile-scoped NeMo Relay runtimes owned by the Hermes agent core."""

from __future__ import annotations

import atexit
import asyncio
import contextlib
import contextvars
import functools
import importlib
import inspect
import logging
import os
import threading
import tomllib
import uuid
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home
from hermes_cli.relay_plugin_cutover import (RELAY_PLUGINS_CONFIG_ENV, configured_legacy_relay_env_vars)

logger = logging.getLogger(__name__)

SESSION_SCOPE = "hermes.session"
TURN_SCOPE = "hermes.turn"
LOGICAL_LLM_SCOPE = "hermes.logical_llm_call"
RUNTIME_SCHEMA_KEY = "hermes.relay.schema_version"
RUNTIME_SCHEMA_VERSION = "hermes.relay.runtime.v1"
RUNTIME_INSTANCE_KEY = "hermes.relay.runtime_instance"
RELAY_PLUGINS_EXECUTION_CONSUMER = "hermes.nemo_relay.plugins"
_PROFILE_KEY_CACHE: dict[str, str] = {}

# Bound for native scope ops gating turn/session completion: a wedged pipeline costs one
# lost span, never a blocked agent.
_SCOPE_OP_TIMEOUT = 10.0



class _Lazy:
    """Double-checked, thread-safe once-only factory; ``reset()`` re-arms it (tests)."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory, self._value, self._lock = factory, None, threading.Lock()

    def get(self) -> Any:
        if self._value is None:
            with self._lock:
                if self._value is None:
                    self._value = self._factory()
        return self._value

    def reset(self) -> None:
        self._value = None


def _new_scope_op_executor() -> Any:
    # Daemon workers: a wedged call abandoned at timeout cannot block interpreter exit;
    # ``Future.result(timeout=...)`` still bounds callers when every worker is wedged.
    from tools.daemon_pool import DaemonThreadPoolExecutor
    return DaemonThreadPoolExecutor(max_workers=8, thread_name_prefix="relay-scope-op")


_SCOPE_OP_EXECUTOR = _Lazy(_new_scope_op_executor)
_scope_op_executor = _SCOPE_OP_EXECUTOR.get


def runtime_metadata(runtime_id: str, **extra: Any) -> dict[str, Any]:
    """Return the scope metadata that stamps every Hermes-owned Relay scope."""
    return {RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION, RUNTIME_INSTANCE_KEY: runtime_id, **extra}


def _run_on_daemon_thread(
    fn: Callable[[], Any], *, name: str, timeout: float | None = None, timeout_message: str = ""
) -> Any:
    """Run ``fn`` on a fresh daemon thread; re-raise its error or return its result.
    With ``timeout`` a still-running worker is abandoned with ``TimeoutError`` (daemon: cannot block exit)."""
    outcome: dict[str, Any] = {}

    def _target() -> None:
        try:
            outcome["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - propagated below
            outcome["error"] = exc

    worker = threading.Thread(target=_target, daemon=True, name=name)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(timeout_message)
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")


def pop_relay_scope(relay: Any, handle: Any, *, output: Any = None, metadata: Any = None, timestamp: Any = None) -> Any:
    """Pop a Relay scope, forwarding only the kwargs the live binding accepts.
    ``scope.pop`` gained ``metadata`` in nemo-relay 0.4+; older wheels raise TypeError."""
    pop = relay.scope.pop
    kwargs = {k: v for k, v in (("output", output), ("metadata", metadata), ("timestamp", timestamp)) if v is not None}
    try:
        params = inspect.signature(pop).parameters
    except (TypeError, ValueError):
        params = {}
    if params and not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        kwargs = {key: value for key, value in kwargs.items() if key in params}
    return pop(handle, **kwargs)


def _current_top(relay: Any) -> Any:
    """Return the current top-of-stack scope handle, or None."""
    # Prefer scope.get_handle(): get_scope_stack() may return a native ScopeStack that scope.pop rejects.
    get_handle = getattr(getattr(relay, "scope", None), "get_handle", None)
    if callable(get_handle):
        with contextlib.suppress(Exception):
            return get_handle()
    top = relay.get_scope_stack()
    # Some builds return the live stack (list), others the top handle: only unwrap real lists.
    return (top[-1] if top else None) if isinstance(top, list) else top


def _same_handle(a: Any, b: Any) -> bool:
    # Native ScopeHandle has no value __eq__; compare by uuid when both expose one.
    if a is None or b is None:
        return a is b
    a_uuid = getattr(a, "uuid", None)
    return a is b or a == b or (a_uuid is not None and a_uuid == getattr(b, "uuid", None))


# Process-wide plugin-configuration result shared by every currently hosted profile.
_RelayPluginConfigurationState = Enum("_RelayPluginConfigurationState", "UNINITIALIZED DISABLED ACTIVE FOREIGN FAILED")


class _RelayPluginConfigurationLoadError(RuntimeError):
    """An explicitly selected Relay plugin configuration could not be loaded."""


@dataclass
class RelaySession:
    """One isolated Relay scope stack owned by a Hermes session."""

    session_id: str
    parent_session_id: str = ""
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    closing: bool = False
    handle: Any = None
    context: contextvars.Context | None = None
    # Segmentation: rotation closes the session scope and pushes segment N+1 at a turn
    # boundary (the only LIFO-safe point). Flags are set by compaction, consumed at that boundary.
    segment: int = 0  # index of the CURRENT session scope (0 = first)
    segment_turns: int = 0  # turns completed within the current segment
    rotate_pending: bool = False  # consumed at next begin_turn
    close_pending: bool = False  # rotating compaction hit a live turn; end_turn consumes it


def _load_segments_config() -> dict[str, Any]:
    """gateway.telemetry.session_segments; both defaults OFF => rotation never fires."""
    on_compaction = False
    max_turns = 0
    try:
        from gateway.run import _load_gateway_config  # late import
        telemetry = (_load_gateway_config().get("gateway") or {}).get("telemetry") or {}
        segments = telemetry.get("session_segments") or {}
        on_compaction = bool(segments.get("on_compaction", False))
        try:
            max_turns = max(0, int(segments.get("max_turns", 0) or 0))
        except (TypeError, ValueError):
            max_turns = 0
    except Exception:  # noqa: BLE001 - config absence (or a malformed section) must not crash
        pass
    return {"on_compaction": on_compaction, "max_turns": max_turns}


_SEGMENTS_CONFIG = _Lazy(_load_segments_config)  # cached at first read
_segments_config = _SEGMENTS_CONFIG.get
_reset_segments_config_for_tests = _SEGMENTS_CONFIG.reset


class RelayOperationLease:
    """Keep process-wide Relay plugins alive across a deferred operation."""

    def __init__(self, runtime: "RelayRuntime") -> None:
        self._lock, self._runtime = threading.Lock(), runtime

    def run_in_session(self, session: RelaySession, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run cleanup while this lease still owns the runtime lifetime."""
        with self._lock:
            if self._runtime is None:
                raise RuntimeError("Hermes Relay operation lease is released")
            return self._runtime._run_in_session_untracked(session, callback, *args, **kwargs)

    def release(self) -> None:
        """Release this lease exactly once."""
        with self._lock:
            runtime, self._runtime = self._runtime, None
        if runtime is not None:
            runtime._end_operation()


class _ProcessRelayPluginConfiguration:
    """Own one Relay plugin configuration across profile-scoped hosts."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._owners: set[int] = set()
        self._state = _RelayPluginConfigurationState.UNINITIALIZED
        self._relay: Any = None  # set while a Hermes-owned configuration is active
        self._activation: Any = None

    def acquire(self, owner: Any, relay: Any) -> _RelayPluginConfigurationState:
        """Join the process configuration, initializing it for the first host."""
        with self._lock:
            if not self._owners:
                # First host decides for the whole process; later hosts just join.
                self._state = self._preflight(relay) or self._activate(relay)
                if self._state is _RelayPluginConfigurationState.ACTIVE:
                    logger.info(
                        "Relay plugins are active process-wide and apply to all profiles hosted by this Hermes process."
                    )
            self._owners.add(id(owner))
            return self._state

    def _activate(self, relay: Any) -> _RelayPluginConfigurationState:
        try:
            if not self._initialize(relay):
                return _RelayPluginConfigurationState.DISABLED
        except Exception as exc:
            self._activation = None
            logger.warning("Hermes Relay plugin initialization failed: %s", exc, exc_info=True)
            return _RelayPluginConfigurationState.FAILED
        self._relay = relay
        return _RelayPluginConfigurationState.ACTIVE

    def _preflight(self, relay: Any) -> _RelayPluginConfigurationState | None:
        """Return a terminal state when the process cannot take ownership; None to proceed."""
        if self._relay is not None and not self._clear_active():
            logger.warning(
                "Hermes Relay plugin cleanup is still pending; refusing to replace the process-global configuration"
            )
            return _RelayPluginConfigurationState.FAILED
        try:
            existing_report = relay.plugin.report()
        except Exception:
            logger.warning(
                "Hermes could not determine whether a process-global Relay plugin configuration is already "
                "active; refusing to replace it", exc_info=True,
            )
            return _RelayPluginConfigurationState.FAILED
        if existing_report is not None:
            logger.warning(
                "A process-global Relay plugin configuration is already active outside Hermes native ownership; "
                "leaving it unchanged and disabling Hermes-managed Relay middleware for this process"
            )
            return _RelayPluginConfigurationState.FOREIGN
        return None

    def _initialize(self, relay: Any) -> bool:
        """Initialize Relay from the selected plugins.toml; False when none is selected."""
        configured_inputs = _configured_plugin_inputs(relay)
        if configured_inputs is None:
            return False
        plugin_config, dynamic_plugins = configured_inputs
        if dynamic_plugins:
            try:
                initialize = relay.plugin.initialize_with_dynamic_plugins
                activation = _resolve_plugin_awaitable(initialize(plugin_config, dynamic_plugins))
                if activation is None:
                    raise RuntimeError("NeMo Relay dynamic plugin initialization returned no activation handle")
                self._activation = activation
            except Exception as exc:
                raise RuntimeError("Hermes Relay dynamic plugin activation failed") from exc
        if self._activation is None:
            # Reached only after explicit opt-in. Relay 0.8 no longer layers repository-local
            # configuration onto this explicitly selected payload.
            _resolve_plugin_awaitable(relay.plugin.initialize(plugin_config))
        return True

    def release(self, owner: Any) -> None:
        """Release one host and clear Relay after the final host exits."""
        with self._lock:
            if id(owner) in self._owners:
                self._owners.remove(id(owner))
                self.retry_pending_cleanup()

    def reset_for_tests(self) -> None:
        """Clear process-global state left by directly constructed test hosts."""
        with self._lock:
            self._owners.clear()
            self.retry_pending_cleanup()

    def retry_pending_cleanup(self) -> None:
        """Retry a failed final cleanup without disrupting live owners."""
        with self._lock:
            if not self._owners and self._clear_active():
                self._state = _RelayPluginConfigurationState.UNINITIALIZED

    def _clear_active(self) -> bool:
        relay, activation = self._relay, self._activation
        if relay is None:
            return True

        def close_configuration() -> Any:
            if activation is None:
                return _resolve_plugin_awaitable(relay.plugin.clear_async())
            if callable(close := getattr(activation, "close", None)):
                return _resolve_plugin_awaitable(close())
            raise RuntimeError("NeMo Relay dynamic plugin activation has no close method")

        for what, step in (
            ("subscriber flush", lambda: _resolve_plugin_awaitable(relay.subscribers.flush_async())),
            ("configuration cleanup", close_configuration),
        ):
            try:
                step()
            except Exception:
                logger.warning("Hermes Relay plugin %s failed", what, exc_info=True)
                return False
        self._relay = self._activation = None
        return True


_PLUGIN_CONFIGURATION = _ProcessRelayPluginConfiguration()
atexit.register(_PLUGIN_CONFIGURATION.retry_pending_cleanup)


class RelayRuntime:
    """Own Relay session scopes and optional process plugin configuration."""

    def __init__(self, relay: Any = None, *, profile_key: str | None = None) -> None:
        self.relay = relay or _load_nemo_relay()
        self.profile_key = profile_key or current_profile_key()
        self.runtime_id = uuid.uuid4().hex
        self._sessions_lock, self._execution_consumers_lock = threading.RLock(), threading.RLock()
        self._sessions: dict[str, RelaySession] = {}
        self._subagent_parents: dict[str, str] = {}
        self._subagent_parent_handles: dict[str, Any] = {}
        self._execution_consumers: set[str] = set()
        self._closing = self._shutdown_started = False
        self._shutdown_complete, self._operations_idle = threading.Event(), threading.Event()
        self._operations_idle.set()
        self._active_operations = 0
        self._plugin_configuration_state = _PLUGIN_CONFIGURATION.acquire(self, self.relay)
        # Cleared (with the atexit hook) by the first successful _finish_shutdown.
        self._plugin_configuration_registered = True
        if self._plugins_active():
            self.retain_managed_execution(RELAY_PLUGINS_EXECUTION_CONSUMER)
        atexit.register(self.shutdown)

    def _plugins_active(self) -> bool:
        return self._plugin_configuration_state is _RelayPluginConfigurationState.ACTIVE

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

    def _open_session_scope(
        self, session: RelaySession, scope_metadata: dict[str, Any], *, resolve_parent: bool,
        exit_fallback: bool = False, **push_kwargs: Any,
    ) -> None:
        """Push a fresh SESSION_SCOPE for ``session`` (bounded by ``_SCOPE_OP_TIMEOUT``); record handle + context.
        Subagents parent under their spawning turn/session handle (``resolve_parent`` creates the parent when
        unknown). ``exit_fallback``: at interpreter shutdown the executor refuses futures; push synchronously."""
        parent_handle = None
        if session.parent_session_id:
            with self._sessions_lock:
                parent_handle = self._subagent_parent_handles.get(session.session_id)
            if parent_handle is None and resolve_parent:
                parent = self.ensure_session({"session_id": session.parent_session_id})
                if parent is not None:
                    parent_handle = parent.handle
            scope_metadata["nemo_relay_scope_role"] = "subagent"
        context = contextvars.Context()
        args = (self.relay.scope.push, SESSION_SCOPE, self.relay.ScopeType.Agent)
        push_kwargs.update(handle=parent_handle, metadata=scope_metadata, input={})
        try:
            future = _scope_op_executor().submit(context.run, *args, **push_kwargs)
            session.handle = future.result(timeout=_SCOPE_OP_TIMEOUT)
        except RuntimeError:
            if not exit_fallback:
                raise
            session.handle = context.run(*args, **push_kwargs)
        session.context = context

    def ensure_session(
        self, event: dict[str, Any], *, data: Any = None, metadata: dict[str, Any] | None = None
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
                session = self._sessions[session_id] = RelaySession(
                    session_id=session_id, parent_session_id=self._subagent_parents.get(session_id, "")
                )
        with session.lock:
            if session.closing:
                return None
            if session.handle is None:
                self._open_session_scope(
                    session, {**(metadata or {}), **runtime_metadata(self.runtime_id)},
                    resolve_parent=True, data=data, exit_fallback=True,
                )
        return session

    def rotate_session_scope(self, session: RelaySession, *, reason: str) -> None:
        """Close the current session scope and open the next segment (turn boundary ONLY: LIFO).
        Bookkeeping advances even when a native call fails so a degraded rotation cannot retry every turn."""
        with session.lock:
            if session.closing or session.handle is None:
                return
            old_handle = session.handle
            # Bookkeeping FIRST: a failed native call must not leave rotate_pending set.
            session.segment += 1
            session.segment_turns = 0
            session.rotate_pending = False
            try:
                self.run_in_session(
                    session, self.relay.scope.pop, old_handle, output={"hermes.session.segment_reason": reason},
                    metadata=runtime_metadata(self.runtime_id), timeout=_SCOPE_OP_TIMEOUT,
                )
            except Exception:
                logger.warning(
                    "Hermes Relay segment close failed (session=%s segment=%d); abandoning the old segment span",
                    session.session_id, session.segment - 1, exc_info=True,
                )
            scope_metadata = runtime_metadata(
                self.runtime_id, **{"hermes.session.segment": session.segment, "hermes.session.segment_reason": reason},
            )
            try:
                self._open_session_scope(session, scope_metadata, resolve_parent=False)
            except Exception:
                logger.warning(
                    "Hermes Relay segment open failed (session=%s segment=%d); keeping the prior scope handle",
                    session.session_id, session.segment, exc_info=True,
                )

    def register_subagent(
        self, event: dict[str, Any], *, metadata: dict[str, Any] | None = None
    ) -> RelaySession | None:
        """Open a child Agent scope under its spawning turn when available."""
        parent_session_id = str(event.get("parent_session_id") or "")
        child_session_id = str(event.get("child_session_id") or "")
        if not parent_session_id or not child_session_id or parent_session_id == child_session_id:
            return None
        parent = self.ensure_session({"session_id": parent_session_id})
        parent_handle = None if parent is None else parent.handle
        turn = active_turn(parent_session_id)  # already proved liveness and (RelayRuntime host) an open session
        if turn is not None and turn.handle is not None and turn.lease.host is self:
            parent_handle = turn.handle
        with self._sessions_lock:
            if self._closing:
                return None
            self._subagent_parents[child_session_id] = parent_session_id
            if parent_handle is not None:
                self._subagent_parent_handles[child_session_id] = parent_handle
        return self.ensure_session({"session_id": child_session_id}, metadata=metadata)

    def unregister_subagent(self, event: dict[str, Any]) -> None:
        """Close a delegated session and forget its parent relationship."""
        child_session_id = str(event.get("child_session_id") or "")
        if child_session_id:
            self.close_session({"session_id": child_session_id})
            self._forget_subagent(child_session_id)

    def _forget_subagent(self, session_id: str) -> None:
        with self._sessions_lock:
            self._subagent_parents.pop(session_id, None)
            self._subagent_parent_handles.pop(session_id, None)

    def _lookup(self, session_id: str) -> RelaySession | None:
        """Registry lookup (closing sessions included) without creating one."""
        with self._sessions_lock:
            return self._sessions.get(session_id)

    def get_session(self, session_id: str) -> RelaySession | None:
        """Return an active Hermes Relay session without creating one."""
        with self._sessions_lock:
            session = None if self._closing else self._sessions.get(str(session_id or ""))
        if session is None:
            return None
        with session.lock:
            return None if session.closing else session

    def _session_context(self, session: RelaySession, *, allow_closing: bool) -> contextvars.Context:
        """Copy the current context and overlay the session's saved Relay vars (a copy: re-entrant from callbacks)."""
        with session.lock:
            if session.closing and not allow_closing:
                raise RuntimeError("Hermes Relay session is closing")
            if session.context is None or session.handle is None:
                raise RuntimeError("Hermes Relay session context is unavailable")
            relay_context = session.context.copy()
        context = contextvars.copy_context()
        for variable, value in relay_context.items():
            context.run(variable.set, value)
        return context

    def run_in_session(
        self, session: RelaySession, callback: Callable[..., Any], *args: Any,
        allow_closing: bool = False, timeout: float | None = None, **kwargs: Any,
    ) -> Any:
        """Run a Relay operation against a session's isolated scope stack.
        ``timeout`` bounds the native call on the daemon executor (``TimeoutError`` on breach); ``None``
        runs synchronously. Lifecycle ops pass ``_SCOPE_OP_TIMEOUT``: a wedged pipeline costs one span."""
        with self._operation():
            return self._run_in_session_untracked(
                session, callback, *args, allow_closing=allow_closing, timeout=timeout, **kwargs
            )

    def _run_in_session_untracked(
        self, session: RelaySession, callback: Callable[..., Any], *args: Any,
        allow_closing: bool = False, timeout: float | None = None, **kwargs: Any,
    ) -> Any:
        """Run inside a session whose host-level lifetime is already held."""
        context = self._session_context(session, allow_closing=allow_closing)

        def invoke() -> Any:
            self.relay.get_scope_stack()
            return callback(*args, **kwargs)

        if timeout is None:
            return context.run(invoke)
        exceeded = f"Relay scope operation exceeded {timeout}s"
        try:
            future = _scope_op_executor().submit(context.run, invoke)
        except RuntimeError:
            # Interpreter shutdown: the executor refuses futures but the atexit close path must
            # still flush — still bounded so a wedged call cannot block exit.
            return _run_on_daemon_thread(
                lambda: context.run(invoke), name="relay-scope-op-exit", timeout=timeout,
                timeout_message=f"{exceeded} during interpreter shutdown; abandoning the native "
                "call so process exit can proceed",
            )
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            raise TimeoutError(
                f"{exceeded} (session={session.session_id}); abandoning the native call "
                "so the agent can continue — the span for this scope is lost"
            ) from exc

    async def run_in_session_async(
        self, session: RelaySession, callback: Callable[..., Any], *args: Any,
        allow_closing: bool = False, **kwargs: Any,
    ) -> Any:
        """Create and await an operation inside the session's saved context."""
        with self._operation():
            context = self._session_context(session, allow_closing=allow_closing)

            async def invoke() -> Any:
                self.relay.get_scope_stack()
                result = callback(*args, **kwargs)
                return await result if inspect.isawaitable(result) else result

            return await context.run(asyncio.create_task, invoke())

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

    @contextlib.contextmanager
    def _operation(self):
        """``_begin_operation`` / ``_end_operation`` around one tracked Relay call."""
        self._begin_operation()
        try:
            yield
        finally:
            self._end_operation()

    def acquire_operation_lease(self) -> RelayOperationLease:
        """Retain plugin lifetime for work that outlives one Relay await."""
        self._begin_operation()
        return RelayOperationLease(self)

    def apply_tool_request_intercepts(self, *, session_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Apply Relay request rewriting before Hermes authorizes a tool call."""
        request_intercepts = getattr(getattr(self.relay, "tools", None), "request_intercepts", None)
        managed = self.managed_execution_enabled() and callable(request_intercepts)
        session = self.ensure_session({"session_id": session_id}) if managed else None
        if session is None:
            return args
        result = self.run_in_session(session, request_intercepts, tool_name, args)
        return result if isinstance(result, dict) else args

    def _pop_with_drain(
        self, handle: Any, *, output: dict[str, Any], metadata: dict[str, Any], session_root: Any, drain_limit: int,
    ) -> BaseException | None:
        """Pop ``handle``; if that fails, drain orphans above it and retry once; return the retry's error.
        Must run inside ONE ``run_in_session`` callback so ContextVar stack views stay consistent."""
        with contextlib.suppress(Exception):
            pop_relay_scope(self.relay, handle, output=output, metadata=metadata)
            return None
        drained = 0
        for _ in range(drain_limit):
            top = _current_top(self.relay)
            if top is None or _same_handle(top, handle):
                break
            # Never pop the session root while draining for a nested handle.
            if session_root is not None and _same_handle(top, session_root) and handle is not session_root:
                break
            try:
                orphan_output = {"outcome": "cancelled", "hermes.orphan_drain": True}
                pop_relay_scope(self.relay, top, output=orphan_output, metadata=metadata)
                drained += 1
            except Exception:
                logger.warning("Hermes Relay orphaned scope drain failed", exc_info=True)
                break
        if drained:
            logger.warning("Hermes Relay drained %d orphaned scope(s) before closing %s", drained, handle)
        try:
            pop_relay_scope(self.relay, handle, output=output, metadata=metadata)
            return None
        except Exception as retry_exc:
            return retry_exc

    def _close_scope_handle(
        self, session: RelaySession, handle: Any, *, output: dict[str, Any] | None = None, allow_closing: bool = False,
        failure_label: str = "scope close failed", drain_limit: int = 32, operation_already_held: bool = False,
    ) -> str | None:
        """Pop ``handle``, draining orphaned children in the same session context; failure string or None.
        Relay scopes are strict LIFO; empty-stream retries + interrupt can abandon a physical LLM scope
        above TURN/SESSION. Drain+close is bounded so a wedged pipeline never blocks completion."""
        if handle is None:
            return None
        run_in_session = (self._run_in_session_untracked if operation_already_held else self.run_in_session)
        try:
            failure = run_in_session(
                session, self._pop_with_drain, handle, output=output or {},
                metadata=runtime_metadata(self.runtime_id), session_root=session.handle,
                drain_limit=drain_limit, allow_closing=allow_closing, timeout=_SCOPE_OP_TIMEOUT,
            )
        except Exception as exc:
            failure = exc
        return None if failure is None else f"{failure_label}: {failure}"

    def close_session(self, event: dict[str, Any]) -> None:
        """Close one session scope and remove it from the core registry (no-op once shutting down)."""
        with contextlib.suppress(RuntimeError), self._operation():  # _close_session itself never raises
            self._close_session(event)

    def _close_session(self, event: dict[str, Any]) -> None:
        """Close one session already admitted by the host lifecycle gate."""
        session_id = _session_id(event)
        session = self._lookup(session_id)
        if session is None:
            self._forget_subagent(session_id)
            return
        failure = None
        with session.lock:
            if session.closing:
                return
            session.closing = True
            if session.handle is not None:
                failure = self._close_scope_handle(
                    session, session.handle, output={}, allow_closing=True,
                    failure_label="session scope close failed", operation_already_held=True,
                )
        # No subscriber flush here: process-wide, may wait on other sessions and deadlock an asyncio
        # loop; final plugin teardown flushes once.
        with self._sessions_lock:
            if self._sessions.get(session_id) is session:
                del self._sessions[session_id]
            self._forget_subagent(session_id)
        if failure:
            logger.warning("Hermes Relay session %s closed with errors: %s", session_id, failure)

    def shutdown(self) -> None:
        """Close core scopes and release process plugin configuration."""
        with self._sessions_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = self._closing = True
            has_active_operations = self._active_operations > 0
        if not has_active_operations:
            self._finish_shutdown()
            return
        thread = threading.Thread(
            target=lambda: (self._operations_idle.wait(), self._finish_shutdown()),
            name=f"hermes-nemo-relay-shutdown-{self.runtime_id[:8]}", daemon=True,
        )
        try:
            thread.start()
        except Exception:
            with self._sessions_lock:
                self._shutdown_started = False
            logger.warning("Hermes Relay deferred shutdown could not start", exc_info=True)

    def _finish_shutdown(self) -> None:
        try:
            with self._sessions_lock:
                session_ids = list(self._sessions)
            for session_id in session_ids:
                _warn_on_error("runtime operation", self._close_session, {"session_id": session_id})
            if self._plugin_configuration_registered:
                if self._plugins_active():
                    self.release_managed_execution(RELAY_PLUGINS_EXECUTION_CONSUMER)
                _PLUGIN_CONFIGURATION.release(self)
                self._plugin_configuration_registered = False
                with contextlib.suppress(Exception):
                    atexit.unregister(self.shutdown)
        except Exception:
            with self._sessions_lock:
                self._shutdown_started = False
            logger.warning("Hermes Relay shutdown failed", exc_info=True)
            return
        with self._sessions_lock:
            self._shutdown_complete.set()


@dataclass(frozen=True)
class NoopRelayRuntime:
    """Explicit reduced-capability host for platforms without Relay wheels."""

    profile_key: str
    reason: str

    def apply_tool_request_intercepts(self, *, session_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        return args

    @staticmethod
    def retain_managed_execution(consumer: str) -> None:
        pass

    release_managed_execution = retain_managed_execution
    managed_execution_enabled = staticmethod(lambda: False)
    shutdown = staticmethod(lambda: None)  # no resources are allocated on unsupported platforms


RelayHost = RelayRuntime | NoopRelayRuntime


class RelayHostRegistry:
    """Own exactly one Relay host for each canonical Hermes profile."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._hosts: dict[str, RelayHost] = {}

    def for_profile(self, profile_key: str | None = None, *, create: bool = True) -> RelayHost | None:
        key = profile_key or current_profile_key()
        with self._lock:
            host = self._hosts.get(key)
            if host is not None or not create:
                return host
            try:
                host = RelayRuntime(profile_key=key)
            except Exception as exc:
                logger.warning("Hermes Relay runtime initialization failed", exc_info=True)
                host = NoopRelayRuntime(profile_key=key, reason=str(exc))
            self._hosts[key] = host
            return host

    def shutdown_all(self) -> None:
        with self._lock:
            hosts, self._hosts = list(self._hosts.values()), {}
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

    def live_runtime(self) -> RelayRuntime | None:
        """Return the real Relay host when this lease owns an open session."""
        return self.host if isinstance(self.host, RelayRuntime) and self.session is not None else None


@dataclass
class RelayTurnContext:
    """Runtime-only context for one Hermes turn or top-level task."""

    lease: ConversationLease
    turn_id: str
    task_id: str
    handle: Any = None
    logical_llm_calls: dict[str, Any] = field(default_factory=dict, repr=False)
    logical_llm_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    finalize_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _previous_turn: RelayTurnContext | None = field(default=None, repr=False)
    _active_registered: bool = field(default=False, repr=False)
    relay_enabled: bool = True
    closed: bool = False


_CURRENT_TURN: contextvars.ContextVar[RelayTurnContext | None] = contextvars.ContextVar(
    "hermes_relay_turn", default=None
)

# >0 while the native pipeline is mid-dispatch of a Hermes tool/LLM callback. Nested managed
# execution there is structurally broken (the pipeline binds its Futures to the OUTER call's
# loop, blocked inside the synchronous callback), so resolve_execution_context() bypasses Relay.
# A ContextVar so the marker follows copy_context() into worker threads / per-thread loops.
_MANAGED_CALLBACK_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "hermes_relay_managed_callback_depth", default=0
)


@contextlib.contextmanager
def managed_callback_guard():
    """Mark the current context as inside a managed Relay callback: everything the wrapped ``invoke()``
    transitively calls (incl. work forwarded via copy_context()) runs unmanaged."""
    token = _MANAGED_CALLBACK_DEPTH.set(_MANAGED_CALLBACK_DEPTH.get() + 1)
    try:
        yield
    finally:
        _MANAGED_CALLBACK_DEPTH.reset(token)


def _warn_on_error(what: str, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run fail-open telemetry work: log ``Hermes Relay <what> failed`` and return None on error."""
    try:
        return callback(*args, **kwargs)
    except Exception:
        logger.warning("Hermes Relay %s failed", what, exc_info=True)
        return None


def _fail_open(what: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator form of ``_warn_on_error`` for telemetry hooks that must never block the caller."""
    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            return _warn_on_error(what, fn, *args, **kwargs)
        return guarded
    return wrap


def _flag_open_session(session: RelaySession, flag: str) -> None:
    """Set a pending-rotation/close flag unless the session is already closing."""
    with session.lock:
        if not session.closing:
            setattr(session, flag, True)


class RelaySessionCoordinator:
    """Own semantic conversation and turn lifetimes for Hermes core."""

    def __init__(self, registry: RelayHostRegistry = HOST_REGISTRY) -> None:
        self.registry = registry
        self._initializer_lock = threading.RLock()
        self._session_initializers: dict[str, Callable[[RelayRuntime, dict[str, Any]], None]] = {}
        self._active_turns_lock = threading.RLock()
        self._active_turns: dict[tuple[str, str], set[int]] = {}

    def register_session_initializer(self, name: str, callback: Callable[[RelayRuntime, dict[str, Any]], None]) -> None:
        """Register idempotent profile/session preparation before scope creation."""
        with self._initializer_lock:
            self._session_initializers[name] = callback

    def _prepare_session(self, host: RelayRuntime, context: dict[str, Any]) -> None:
        with self._initializer_lock:
            initializers = list(self._session_initializers.items())
        for name, callback in initializers:
            try:
                callback(host, context)
            except Exception:
                logger.warning("Hermes Relay session initializer failed: %s", name, exc_info=True)

    def acquire_conversation(
        self, *, profile_key: str, session_id: str, platform: str, parent_session_id: str = "", model: str = "",
    ) -> ConversationLease:
        host = self.registry.for_profile(profile_key) or NoopRelayRuntime(profile_key, "Relay host creation was disabled")
        session = None
        if isinstance(host, RelayRuntime):
            context = {
                "profile_key": profile_key, "session_id": session_id, "platform": platform,
                "parent_session_id": parent_session_id, "model": model,
            }
            session = _warn_on_error("conversation initialization", self._open_conversation_session, host, context)
        return ConversationLease(
            profile_key=profile_key, session_id=session_id, platform=platform, host=host,
            session=session, parent_session_id=parent_session_id,
        )

    def _open_conversation_session(self, host: RelayRuntime, context: dict[str, Any]) -> RelaySession | None:
        self._prepare_session(host, context)
        session_id, parent_session_id = context["session_id"], context["parent_session_id"]
        metadata = {"hermes.execution_surface": context["platform"] or "unknown"}
        if parent_session_id and parent_session_id != session_id:
            event = {"parent_session_id": parent_session_id, "child_session_id": session_id}
            return host.register_subagent(event, metadata=metadata)
        return host.ensure_session({"session_id": session_id}, metadata=metadata)

    def begin_turn(self, lease: ConversationLease, *, turn_id: str, task_id: str) -> RelayTurnContext:
        if lease.released:
            raise RuntimeError("Hermes Relay conversation lease is released")
        turn = RelayTurnContext(lease=lease, turn_id=turn_id, task_id=task_id)
        key = (lease.profile_key, lease.session_id)
        with self._active_turns_lock:
            if self._active_turns.get(key):
                # One physical scope stack per session; concurrent turns' sibling scopes would not close LIFO.
                turn.relay_enabled = False
                logger.warning(
                    "Skipping Relay instrumentation for concurrent Hermes turn %s in session %s",
                    turn_id, lease.session_id,
                )
            else:
                self._active_turns[key] = {id(turn)}
                turn._active_registered = True
        host = lease.live_runtime() if turn.relay_enabled else None
        if host is not None:
            # Rotation happens HERE: no live turn scope on the stack, so the session scope can close/reopen LIFO.
            _warn_on_error("segment rotation", self._maybe_rotate_segment, host, lease.session)
            turn.handle = _warn_on_error(
                "turn initialization", host.run_in_session, lease.session, host.relay.scope.push,
                TURN_SCOPE, host.relay.ScopeType.Function, handle=lease.session.handle, input={},
                metadata=runtime_metadata(host.runtime_id, **{"hermes.execution_surface": lease.platform or "unknown"}),
                timeout=_SCOPE_OP_TIMEOUT,
            )
        turn._previous_turn = _CURRENT_TURN.get()
        _CURRENT_TURN.set(turn)
        return turn

    @staticmethod
    def _maybe_rotate_segment(host: RelayRuntime, session: RelaySession) -> None:
        """Rotate the session scope when compaction flagged it or the turn cap is hit."""
        config = _segments_config()
        cap = config["max_turns"]
        compaction = config["on_compaction"] and session.rotate_pending
        if compaction or (cap > 0 and session.segment_turns >= cap):
            host.rotate_session_scope(session, reason="compaction" if compaction else "max_turns")

    def end_turn(self, turn: RelayTurnContext, *, outcome: str) -> None:
        with turn.finalize_lock:
            if turn.closed:
                self._reset_turn_context(turn)
                return
            turn.closed = True
            lease = turn.lease
            host = lease.live_runtime()
            try:
                if host is not None:
                    self._close_turn_scope(host, turn, outcome=outcome)
            finally:
                if turn._active_registered and host is not None:
                    with contextlib.suppress(Exception), lease.session.lock:  # accounting never blocks
                        lease.session.segment_turns += 1  # max_turns rotation trigger
                try:
                    # Delegated agents own one turn: close their conversation while the active-turn
                    # guard is held so a parent timeout fallback cannot race it.
                    if lease.parent_session_id and isinstance(lease.host, RelayRuntime):
                        _warn_on_error(
                            "child conversation finalization", lease.host.unregister_subagent,
                            {"child_session_id": lease.session_id},
                        )
                finally:
                    self._unregister_active_turn(turn)
                    self._reset_turn_context(turn)
                self._consume_deferred_close(lease)

    def _close_turn_scope(self, host: RelayRuntime, turn: RelayTurnContext, *, outcome: str) -> None:
        """Pop the turn's logical LLM children, then the turn scope itself (LIFO)."""
        self._finish_logical_calls(turn, outcome=outcome)
        failure = host._close_scope_handle(
            turn.lease.session, turn.handle, output={"outcome": outcome}, failure_label="turn scope close failed",
        )
        if failure:
            logger.warning("Hermes Relay turn finalization failed: %s", failure)

    @_fail_open("deferred session close")
    def _consume_deferred_close(self, lease: ConversationLease) -> None:
        """Close a session whose rotating-compaction close was deferred (``close_pending``).
        The last live turn consumes it after its own scope popped and it left the active-turn table."""
        host = lease.live_runtime()
        if host is None:
            return
        with lease.session.lock:
            pending = lease.session.close_pending and not lease.session.closing
        if pending and not self.has_active_turn(profile_key=lease.profile_key, session_id=lease.session_id):
            host.close_session({"session_id": lease.session_id})

    @_fail_open("compaction notification")
    def notify_session_compacted(self, *, profile_key: str, session_id: str, old_session_id: str = "") -> None:
        """React to a completed compaction, per compaction mode; unknown sessions / disabled config are no-ops.
        In-place (``old_session_id`` empty/equal): flag rotation for the next turn boundary — never rotate
        immediately, a live turn under it breaks LIFO. Rotating (ids differ): the next turn gets a fresh
        session under the new id, so close the OLD session now or its scope stays an unexported orphan."""
        if not _segments_config()["on_compaction"]:
            return
        host = self.registry.for_profile(profile_key)
        if not isinstance(host, RelayRuntime):
            return
        if old_session_id and old_session_id != session_id:
            # A LIVE turn on the old session: closing now would pop under it (LIFO).
            old_session = host._lookup(old_session_id)
            if old_session is not None and self.has_active_turn(profile_key=profile_key, session_id=old_session_id):
                _flag_open_session(old_session, "close_pending")
            else:
                host.close_session({"session_id": old_session_id})
            return
        session = host._lookup(session_id)
        if session is not None:
            _flag_open_session(session, "rotate_pending")

    def has_active_turn(self, *, profile_key: str, session_id: str) -> bool:
        """Return whether a turn is still running for one profile/session."""
        with self._active_turns_lock:
            return bool(self._active_turns.get((profile_key, session_id)))

    def _unregister_active_turn(self, turn: RelayTurnContext) -> None:
        if not turn._active_registered:
            return
        key = (turn.lease.profile_key, turn.lease.session_id)
        with self._active_turns_lock:
            active = self._active_turns.get(key)
            if active is not None:
                active.discard(id(turn))
                if not active:
                    del self._active_turns[key]
            turn._active_registered = False

    def finish_logical_calls(self, turn: RelayTurnContext, *, outcome: str) -> None:
        """Close logical LLM children before sibling task aggregation scopes."""
        with turn.finalize_lock:
            if not turn.closed:
                self._finish_logical_calls(turn, outcome=outcome)

    @staticmethod
    def _finish_logical_calls(turn: RelayTurnContext, *, outcome: str) -> None:
        lease = turn.lease
        host = lease.live_runtime()
        if host is None:
            return
        with turn.logical_llm_lock:
            logical_calls = list(turn.logical_llm_calls.items())
            turn.logical_llm_calls.clear()
        while logical_calls:
            _request_id, logical_handle = logical_calls[-1]
            failure = host._close_scope_handle(
                lease.session, logical_handle, output={"outcome": outcome},
                failure_label="logical LLM scope close failed",
            )
            if failure is None:
                logical_calls.pop()
                continue
            with turn.logical_llm_lock:
                # Stack-owned: if the newest handle cannot close even after drain, older ones cannot either.
                for pending_request_id, pending_handle in logical_calls:
                    turn.logical_llm_calls.setdefault(pending_request_id, pending_handle)
            logger.warning("Hermes Relay logical LLM finalization failed: %s", failure)
            break

    @staticmethod
    def _reset_turn_context(turn: RelayTurnContext) -> None:
        """Unwind ``turn`` without disturbing a newer context-local turn."""
        if _CURRENT_TURN.get() is not turn:
            return
        previous, seen = turn._previous_turn, {id(turn)}
        while previous is not None and previous.closed and id(previous) not in seen:
            seen.add(id(previous))
            previous = previous._previous_turn
        if previous is not None and previous.closed:  # cycle: no live ancestor
            previous = None
        _CURRENT_TURN.set(previous)

    @staticmethod
    def release_conversation(lease: ConversationLease) -> None:
        """Release a caller lease without closing a resumable conversation."""
        lease.released = True

    def finalize_conversation(self, *, profile_key: str, session_id: str) -> None:
        host = self.registry.for_profile(profile_key, create=False)
        if isinstance(host, RelayRuntime):
            host.close_session({"session_id": session_id})


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
    if turn is None or not turn.relay_enabled or turn.closed or turn.lease.released:
        return None
    lease = turn.lease
    if lease.profile_key != current_profile_key() or (session_id is not None and lease.session_id != session_id):
        return None
    if isinstance(lease.host, RelayRuntime) and (
        lease.session is None or lease.host.get_session(lease.session_id) is not lease.session
    ):
        return None
    return turn


def resolve_execution_context(session_id: str) -> tuple[RelayRuntime | None, RelaySession | None, Any]:
    """Resolve one active turn/session parent for managed Relay execution."""
    # Nested managed execution is impossible (see _MANAGED_CALLBACK_DEPTH); the outer scope
    # still records the tool-level event.
    if _MANAGED_CALLBACK_DEPTH.get() > 0 or not relay_instrumentation_enabled():
        # A managed Relay callback is already executing on this logical call path (e.g. the native
        # ``tools.execute`` pipeline is mid-dispatch of a Hermes tool). Nested managed execution here is
        # structurally impossible: the native pipeline binds its Futures to the OUTER call's event loop,
        # which is blocked inside the synchronous tool callback until the tool returns. A nested managed LLM
        # call (the vision_analyze auxiliary path) therefore awaits a foreign-loop Future that can never
        # complete — "attached to a different loop" at best, deadlock at worst, and "Event loop is closed"
        # during shutdown when the orphaned Future is completed late (#77244).
        return None, None, None
    turn = active_turn(session_id)
    host = turn.lease.live_runtime() if turn is not None else None
    if host is not None:
        session = turn.lease.session
        return host, session, turn.handle or session.handle
    # Consumers retain the profile host before reaching an out-of-turn adapter; never
    # initialize Relay for the default no-consumer path.
    runtime = get_runtime(create=False)
    if runtime is None or not runtime.managed_execution_enabled():
        return None, None, None
    session = runtime.get_session(session_id) or runtime.ensure_session({"session_id": session_id})
    return runtime, session, None if session is None else session.handle


def apply_tool_request_intercepts(*, session_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Return Relay-rewritten arguments at Hermes's authorization boundary."""
    runtime = get_runtime(create=False) if session_id else None
    if runtime is None:
        return args
    return runtime.apply_tool_request_intercepts(session_id=session_id, tool_name=tool_name, args=args)


def _is_relay_wrapped_callback_error(relay_error: BaseException, callback_error: BaseException) -> bool:
    """Match Relay's native callback wrapper without masking policy errors."""
    if relay_error is callback_error:
        return True
    if not isinstance(relay_error, RuntimeError):
        return False
    kind = type(callback_error)
    type_names = {kind.__name__, kind.__qualname__, f"{kind.__module__}.{kind.__qualname__}"}
    return any(str(relay_error).startswith(f"internal error: {name}: {callback_error}") for name in type_names)


def get_runtime(*, create: bool = True, profile_key: str | None = None) -> RelayRuntime | None:
    """Return the Relay host for the active Hermes profile."""
    host = HOST_REGISTRY.for_profile(profile_key, create=create)
    return host if isinstance(host, RelayRuntime) else None


def current_profile_key() -> str:
    """Return the canonical profile identity used for runtime isolation."""
    home = get_hermes_home().expanduser()
    if not home.is_absolute():
        return str(home.resolve())
    return _PROFILE_KEY_CACHE.get(str(home)) or _PROFILE_KEY_CACHE.setdefault(str(home), str(home.resolve()))


def _load_nemo_relay() -> Any:
    """Load the binding only when a producer or consumer needs Relay."""
    return importlib.import_module("nemo_relay")


def _configured_plugin_inputs(relay: Any) -> tuple[dict[str, Any], list[Any]] | None:
    """Load selected plugin inputs, or return ``None`` when none were selected."""
    configured = os.environ.get(RELAY_PLUGINS_CONFIG_ENV, "").strip()
    if not configured:
        if legacy_vars := configured_legacy_relay_env_vars(os.environ):
            logger.warning(
                "Legacy NeMo Relay exporter variables are set but no %s was provided. %s no longer activate "
                "Relay exporters; migrate the exporter configuration to a Relay plugins.toml file.",
                RELAY_PLUGINS_CONFIG_ENV, ", ".join(legacy_vars),
            )
        return None
    config_path = Path(configured).expanduser()
    try:
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)
        if "dynamic_plugins" in config:
            raise ValueError("Hermes [[dynamic_plugins]] records are unsupported; use Relay [[plugins.dynamic]] records")
        dynamic_plugins = relay.plugin.load_dynamic_plugin_activation_specs(config_path) if "plugins" in config else []
        return {k: v for k, v in config.items() if k != "plugins"}, dynamic_plugins
    except Exception as exc:
        raise _RelayPluginConfigurationLoadError(
            f"Hermes Relay plugin configuration could not be loaded from {config_path}; continuing without Relay plugins"
        ) from exc


def _resolve_plugin_awaitable(value: Any) -> Any:
    """Resolve Relay's async plugin API from synchronous host construction."""
    if not inspect.isawaitable(value):
        return value
    # Only the "no running loop" probe is guarded: a RuntimeError raised by the awaitable itself
    # (re-raised from the daemon thread) must propagate, not fall through to a second asyncio.run.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    return _run_on_daemon_thread(lambda: asyncio.run(value), name="hermes-nemo-relay-plugin-lifecycle")


def _session_id(event: dict[str, Any]) -> str:
    return str(event.get("session_id") or "")


def _reset_for_tests() -> None:
    """Reset all profile-scoped Relay hosts for isolated tests."""
    with SESSION_COORDINATOR._active_turns_lock:
        SESSION_COORDINATOR._active_turns.clear()
    HOST_REGISTRY.shutdown_all()
    _PLUGIN_CONFIGURATION.reset_for_tests()
    _PROFILE_KEY_CACHE.clear()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from enum import auto  # noqa: F401,E402

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

def get_host(
    *,
    create: bool = True,
    profile_key: str | None = None,
) -> RelayHost | None:
    """Return the explicit real or reduced-capability host for a profile."""
    return HOST_REGISTRY.for_profile(profile_key, create=create)

def get_session_handle(session_id: str) -> Any:
    """Return the shared Relay handle for direct core instrumentation."""
    runtime = get_runtime(create=False)
    return None if runtime is None else runtime.get_session_handle(session_id)

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
# ---- END PLUGIN-COMPAT ----

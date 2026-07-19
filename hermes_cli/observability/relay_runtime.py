"""Profile-scoped NeMo Relay runtimes owned by Hermes core."""

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
RUNTIME_SCHEMA_KEY = "hermes.relay.schema_version"
RUNTIME_SCHEMA_VERSION = "hermes.relay.runtime.v1"
RUNTIME_INSTANCE_KEY = "hermes.relay.runtime_instance"

SESSION_START_HOOKS = frozenset({"on_session_start"})
SESSION_CLOSE_HOOKS = frozenset({"on_session_finalize", "on_session_reset"})
SUBAGENT_START_HOOKS = frozenset({"subagent_start"})
SUBAGENT_STOP_HOOKS = frozenset({"subagent_stop"})
HANDLED_HOOKS = (
    SESSION_START_HOOKS
    | SESSION_CLOSE_HOOKS
    | SUBAGENT_START_HOOKS
    | SUBAGENT_STOP_HOOKS
)

_RUNTIME_FAILED = object()
_RUNTIMES: dict[str, RelayRuntime | object] = {}
_RUNTIME_LOCK = threading.RLock()


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
        self._shutdown_registered = True
        atexit.register(self.shutdown)

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

    def register_subagent(self, event: dict[str, Any]) -> None:
        """Record the parent used when a delegated Hermes session starts."""
        parent_session_id = str(event.get("parent_session_id") or "")
        child_session_id = str(event.get("child_session_id") or "")
        if (
            not parent_session_id
            or not child_session_id
            or parent_session_id == child_session_id
        ):
            return
        self.ensure_session({"session_id": parent_session_id})
        with self._sessions_lock:
            self._subagent_parents[child_session_id] = parent_session_id

    def unregister_subagent(self, event: dict[str, Any]) -> None:
        """Close a delegated session and forget its parent relationship."""
        child_session_id = str(event.get("child_session_id") or "")
        if not child_session_id:
            return
        self.close_session({"session_id": child_session_id})
        with self._sessions_lock:
            self._subagent_parents.pop(child_session_id, None)

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

            def invoke() -> Any:
                self.relay.get_scope_stack()
                return callback(*args, **kwargs)

            # A copy permits a helper called by an existing Relay callback to
            # re-enter the same logical session without re-entering Context.
            return session.context.copy().run(invoke)

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
            context = session.context.copy()

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
                    )
                except Exception as exc:
                    failures.append(f"session scope close failed: {exc}")
        try:
            self.relay.subscribers.flush()
        except Exception as exc:
            failures.append(f"subscriber flush failed: {exc}")
        with self._sessions_lock:
            if self._sessions.get(session_id) is session:
                self._sessions.pop(session_id, None)
            self._subagent_parents.pop(session_id, None)
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


def handles_hook(hook_name: str) -> bool:
    """Return whether the core Relay host consumes this lifecycle hook."""
    return hook_name in HANDLED_HOOKS


def observe_lifecycle(hook_name: str, **kwargs: Any) -> None:
    """Apply session lifecycle events to the core Relay host."""
    if not handles_hook(hook_name):
        return
    # Session hooks do not activate Relay by themselves. A direct core
    # producer or an enabled built-in consumer creates the host lazily, after
    # which these hooks keep its session lifetime correct.
    runtime = get_runtime(create=False)
    if runtime is None:
        return
    try:
        if hook_name in SESSION_START_HOOKS:
            runtime.ensure_session(kwargs)
        elif hook_name in SESSION_CLOSE_HOOKS:
            runtime.close_session(kwargs)
        elif hook_name in SUBAGENT_START_HOOKS:
            runtime.register_subagent(kwargs)
        else:
            runtime.unregister_subagent(kwargs)
    except Exception:
        logger.warning("Hermes Relay lifecycle failed: %s", hook_name, exc_info=True)


def emit_mark(
    name: str,
    *,
    session_id: str,
    data: Any = None,
    metadata: Any = None,
) -> bool:
    """Emit a fail-open Relay mark under a Hermes session."""
    runtime = get_runtime()
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
    runtime = get_runtime()
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


def get_runtime(
    *,
    create: bool = True,
    profile_key: str | None = None,
) -> RelayRuntime | None:
    """Return the Relay host for the active Hermes profile."""
    key = profile_key or current_profile_key()
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(key)
        if isinstance(runtime, RelayRuntime):
            return runtime
        if runtime is _RUNTIME_FAILED or not create:
            return None
        try:
            runtime = RelayRuntime(profile_key=key)
        except Exception:
            logger.warning("Hermes Relay runtime initialization failed", exc_info=True)
            _RUNTIMES[key] = _RUNTIME_FAILED
            return None
        _RUNTIMES[key] = runtime
        return runtime


def current_profile_key() -> str:
    """Return the canonical profile identity used for runtime isolation."""
    return str(get_hermes_home().expanduser().resolve())


def _load_nemo_relay() -> Any:
    """Load the binding only when a producer or consumer needs Relay."""
    return importlib.import_module("nemo_relay")


def _session_id(event: dict[str, Any]) -> str:
    return str(event.get("session_id") or "")


def _reset_for_tests() -> None:
    """Reset all profile-scoped Relay hosts for isolated tests."""
    with _RUNTIME_LOCK:
        runtimes = list(_RUNTIMES.values())
        _RUNTIMES.clear()
    for runtime in runtimes:
        if isinstance(runtime, RelayRuntime):
            runtime.shutdown()

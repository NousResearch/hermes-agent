"""Physical gateway chat-execution identity for sequential tool observation."""
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from dataclasses import dataclass
from threading import Lock

@dataclass(frozen=True)
class ConcurrentToolResource:
    home: str
    session_key: str
    generation: int
    session_id: str
    resource_id: str
    kind: str = "concurrent_tool_thread"

_CHAT_EXECUTION: ContextVar[tuple[str, str, int] | None] = ContextVar("chat_execution_identity", default=None)
_CONCURRENT_RESOURCE: ContextVar[ConcurrentToolResource | None] = ContextVar("concurrent_tool_resource", default=None)
# The started detached resource (background delegate unit) whose subtree this context runs in. A copied
# context also inherits the dispatcher's execution identity, which alone refuses once that worker settles.
_DETACHED_RESOURCE: ContextVar[ConcurrentToolResource | None] = ContextVar("detached_tool_resource", default=None)


# Detached reservations this process issued and whose owner has not started them yet. A detached row
# may be started only once, by the very handle its dispatcher was given: never by a handle rebuilt from
# journal rows, after its submission failed or was cancelled, or after a restart (units are process-local).
_ISSUED_DETACHED: dict[tuple[str, str, int, str], ConcurrentToolResource] = {}
_ISSUED_DETACHED_LOCK = Lock()


def _resource_key(handle: ConcurrentToolResource) -> tuple[str, str, int, str]:
    return handle.home, handle.session_key, handle.generation, handle.resource_id


def _claim_detached_owner(handle: ConcurrentToolResource) -> bool:
    with _ISSUED_DETACHED_LOCK:
        if _ISSUED_DETACHED.get(_resource_key(handle)) is not handle:
            return False
        del _ISSUED_DETACHED[_resource_key(handle)]
        return True


def _detached_under(home: str, session_key: str, generation: int):
    """``(resource_id, session_id)`` of the bound detached resource owned by exactly this execution."""
    root = _DETACHED_RESOURCE.get()
    if root is None or (root.home, root.session_key, root.generation) != (home, session_key, generation):
        return None
    return root.resource_id, root.session_id


@contextmanager
def bind_chat_execution(home: str | Path, session_key: str, generation: int) -> Iterator[None]:
    token = _CHAT_EXECUTION.set((str(Path(home).resolve()), session_key, generation))
    try:
        yield
    finally:
        _CHAT_EXECUTION.reset(token)


def concurrent_tool_resource_for_submit(resource_id: str, kind="concurrent_tool_thread"):
    identity = _CHAT_EXECUTION.get()
    if identity is None:
        return None
    home, session_key, generation = identity
    from gateway.chat_execution_recovery import ChatExecutionStore
    store = ChatExecutionStore(home)
    with store._db() as db:
        parent = db.execute("SELECT session_id FROM executions WHERE session_key=? AND generation=?", (session_key, generation)).fetchone()
    under = _detached_under(home, session_key, generation)
    if _DETACHED_RESOURCE.get() is not None and under is None:
        raise RuntimeError("Durable concurrent tool resource reservation refused")
    if parent is None or not parent["session_id"] or not store.reserve_concurrent_tool_resource(session_key, generation, resource_id, kind=kind, under=under):
        raise RuntimeError("Durable concurrent tool resource reservation refused")
    handle = ConcurrentToolResource(home, session_key, generation, parent["session_id"], resource_id, kind)
    from gateway.chat_execution_recovery import DETACHED_RESOURCE_KINDS
    if kind in DETACHED_RESOURCE_KINDS:
        with _ISSUED_DETACHED_LOCK:
            _ISSUED_DETACHED[_resource_key(handle)] = handle
    return handle


def start_reserved_tool_resource(handle: ConcurrentToolResource | None) -> bool:
    """Start with the caller's authority, without claiming thread or process settlement."""
    if handle is None:
        return True
    from gateway.chat_execution_recovery import DETACHED_RESOURCE_KINDS, ChatExecutionStore
    under = _detached_under(handle.home, handle.session_key, handle.generation)
    if _DETACHED_RESOURCE.get() is not None and under is None:
        return False
    if handle.kind in DETACHED_RESOURCE_KINDS and not _claim_detached_owner(handle):
        return False
    return ChatExecutionStore(handle.home).start_concurrent_tool_resource(
        handle.session_key, handle.generation, handle.resource_id, handle.session_id, handle.kind, under)


@contextmanager
def bind_concurrent_tool_resource(handle: ConcurrentToolResource | None) -> Iterator[bool]:
    if handle is None:
        yield True
        return
    from gateway.chat_execution_recovery import DETACHED_RESOURCE_KINDS, ChatExecutionStore
    store = ChatExecutionStore(handle.home)
    if not start_reserved_tool_resource(handle):
        yield False
        return
    token = _CONCURRENT_RESOURCE.set(handle)
    detached = _DETACHED_RESOURCE.set(handle) if handle.kind in DETACHED_RESOURCE_KINDS else None
    try:
        yield True
    finally:
        if detached is not None:
            _DETACHED_RESOURCE.reset(detached)
        _CONCURRENT_RESOURCE.reset(token)
        try:
            if not store.finish_concurrent_tool_resource(handle.session_key, handle.generation, handle.resource_id, handle.session_id, handle.kind):
                raise RuntimeError("durable resource finish refused")
        except Exception:
            import logging
            logging.getLogger(__name__).exception("failed to persist concurrent tool resource completion")


def discard_detached_resource(handle: ConcurrentToolResource | None) -> None:
    """Drop the dispatcher's claim on a detached reservation whose work will never run (failed or
    cancelled submission). The row stays ``reserved`` (unresolved) and can never be started."""
    if handle is not None:
        with _ISSUED_DETACHED_LOCK:
            if _ISSUED_DETACHED.get(_resource_key(handle)) is handle:
                del _ISSUED_DETACHED[_resource_key(handle)]


def bind_tool_continuation(dispatch):
    """Reserve before middleware handoff; settle only the actual callback target."""
    from threading import Lock
    from uuid import uuid4
    from tools.thread_context import propagate_context_to_thread

    handle = concurrent_tool_resource_for_submit(str(uuid4()), kind="middleware_tool_continuation")
    if handle is None:
        return dispatch
    lock = Lock()
    claimed = False

    def target(*args, **kwargs):
        with bind_concurrent_tool_resource(handle) as admitted:
            if not admitted:
                raise RuntimeError("Durable middleware tool continuation admission refused")
            return dispatch(*args, **kwargs)

    invoke = propagate_context_to_thread(target, restore_callbacks=True)

    def once(*args, **kwargs):
        nonlocal claimed
        with lock:
            if claimed:
                raise RuntimeError("Hermes tool execution callback invoked more than once")
            claimed = True
        return invoke(*args, **kwargs)

    return once


def observe_tool_before_execution():
    """Fail closed for bound gateway work; legacy/unbound calls remain unchanged."""
    identity = _CHAT_EXECUTION.get()
    if identity is None:
        return False
    from gateway.chat_execution_recovery import ChatExecutionStore

    home, session_key, generation = identity
    store = ChatExecutionStore(home)
    under = _detached_under(home, session_key, generation)
    if (_DETACHED_RESOURCE.get() is not None and under is None
            or not store.observe_started_worker_tool(session_key, generation, under=under)):
        raise RuntimeError("Durable tool observation refused")
    return True

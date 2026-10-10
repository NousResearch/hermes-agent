"""Physical gateway chat-execution identity for sequential tool observation."""
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from dataclasses import dataclass

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
    if parent is None or not parent["session_id"] or not store.reserve_concurrent_tool_resource(session_key, generation, resource_id, kind=kind):
        raise RuntimeError("Durable concurrent tool resource reservation refused")
    return ConcurrentToolResource(home, session_key, generation, parent["session_id"], resource_id, kind)


@contextmanager
def bind_concurrent_tool_resource(handle: ConcurrentToolResource | None) -> Iterator[bool]:
    if handle is None:
        yield True
        return
    from gateway.chat_execution_recovery import ChatExecutionStore
    store = ChatExecutionStore(handle.home)
    if not store.start_concurrent_tool_resource(handle.session_key, handle.generation, handle.resource_id, handle.session_id, handle.kind):
        yield False
        return
    token = _CONCURRENT_RESOURCE.set(handle)
    try:
        yield True
    finally:
        _CONCURRENT_RESOURCE.reset(token)
        try:
            if not store.finish_concurrent_tool_resource(handle.session_key, handle.generation, handle.resource_id, handle.session_id, handle.kind):
                raise RuntimeError("durable resource finish refused")
        except Exception:
            import logging
            logging.getLogger(__name__).exception("failed to persist concurrent tool resource completion")


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
    if not ChatExecutionStore(home).observe_started_worker_tool(session_key, generation):
        raise RuntimeError("Durable tool observation refused")
    return True

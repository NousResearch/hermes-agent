"""Attempt-scoped durable writes for a native compressor runtime restore."""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import get_ident
from typing import Callable


_PENDING_STATE_WRITES: ContextVar[tuple[object, int, list[Callable[[], bool]]] | None] = (
    ContextVar("hermes_pending_compressor_state_writes", default=None)
)


def pending_compressor_state_writes(compressor):
    pending = _PENDING_STATE_WRITES.get()
    if pending is not None and pending[0] is compressor and pending[1] == get_ident():
        return pending[2]
    return None


@contextmanager
def defer_compressor_state_writes(compressor):
    """Discard native state resets on failure; retain each setter's best-effort policy.

    This is not a database transaction: successful restores flush the existing
    setters separately. Other instances and detached workers keep their writes.
    """
    if pending_compressor_state_writes(compressor) is not None:
        yield
        return
    writes = []
    token = _PENDING_STATE_WRITES.set((compressor, get_ident(), writes))
    try:
        yield
    finally:
        _PENDING_STATE_WRITES.reset(token)
    for write in writes:
        write()

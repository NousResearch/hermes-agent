"""Attempt-scoped durable writes for a native compressor runtime restore."""

from contextlib import contextmanager
from contextvars import ContextVar
import logging
import sqlite3
from threading import get_ident
from typing import Callable


logger = logging.getLogger(__name__)

_PENDING_STATE_WRITES: ContextVar[tuple[object, int, list[Callable[[], object]]] | None] = (
    ContextVar("hermes_pending_compressor_state_writes", default=None)
)


def pending_compressor_state_writes(compressor):
    pending = _PENDING_STATE_WRITES.get()
    if pending is not None and pending[0] is compressor and pending[1] == get_ident():
        return pending[2]
    return None


@contextmanager
def defer_compressor_state_writes(compressor):
    """Discard pending resets on failure and expose any failure during their flush.

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


class CompressorDurableStateMixin:
    def _durable_write(self, method: str, label: str, *args) -> bool:
        """Write a bound native state value or queue it for the current restore."""
        setter = getattr(getattr(self, '_session_db', None), method, None)
        if not getattr(self, '_session_id', '') or not callable(setter):
            return False
        session_id = self._session_id

        def write():
            setter(session_id, *args)

        pending = pending_compressor_state_writes(self)
        if pending is not None:
            pending.append(write)
            return True
        try:
            write()
        except sqlite3.Error:
            logger.exception('%s persist failed', label)
            return False
        return True

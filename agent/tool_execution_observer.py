"""Turn-scoped observations at registry handler invocation, after dispatch guards.

The factory resolves the current owner once at invocation. Completion retains that callback;
corrections cannot attribute a late result to a replacement request. Tool worker threads already
inherit ContextVars. No prompt or tool-schema state is involved.
"""
import logging
from contextlib import contextmanager
from contextvars import ContextVar

logger = logging.getLogger(__name__)
_observer = ContextVar('tool_execution_observer', default=None)


@contextmanager
def observe_tool_execution(callback=None, *, factory=None):
    token = _observer.set(factory or (lambda: callback))
    try:
        yield
    finally:
        _observer.reset(token)


def execution_observer():
    factory = _observer.get()
    try:
        return factory() if factory is not None else None
    except Exception:
        logger.debug('Tool observer resolution failed', exc_info=True)
        return None


def emit_tool_execution(*, callback=None, **event):
    try:
        observer = callback or execution_observer()
        if observer is not None:
            observer(**event)
    except Exception:
        logger.debug('Tool execution observer failed', exc_info=True)

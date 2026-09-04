"""Opt-in refusal of text-only document fallback for verified file delivery."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_required = ContextVar("native_document_required", default=False)


class NativeDocumentFallback(RuntimeError):
    """A native upload could not be confirmed; a text notice is not delivery."""


def check_document_fallback():
    """Refuse a text fallback only for the current native-only operation."""
    if _required.get():
        raise NativeDocumentFallback("native document delivery was not confirmed")


def guard_document_fallback(function):
    @wraps(function)
    async def guarded(*args, **kwargs):
        check_document_fallback()
        return await function(*args, **kwargs)

    guarded.strict_native_document_guard = True
    return guarded


@contextmanager
def require_native_document():
    token = _required.set(True)
    try:
        yield
    finally:
        _required.reset(token)

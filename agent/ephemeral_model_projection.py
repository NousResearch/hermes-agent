"""Scope model-projection writes without changing live compressor DB bindings."""

import contextlib
import contextvars

_PROJECTION_TARGET: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "hermes_ephemeral_model_projection", default=None
)


@contextlib.contextmanager
def suppress_ephemeral_projection_writes(compressor: object):
    """Suppress only this compressor's projection resets in this execution context."""
    token = _PROJECTION_TARGET.set(compressor)
    try:
        yield
    finally:
        _PROJECTION_TARGET.reset(token)


def projection_writes_suppressed(compressor: object) -> bool:
    """A sibling thread retains permission to persist real events."""
    return _PROJECTION_TARGET.get() is compressor

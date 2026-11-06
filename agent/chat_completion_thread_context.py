"""Context propagation for API helper worker threads."""

from __future__ import annotations

import contextvars


def context_thread_target(callback):
    """Bind a no-argument thread target to the caller's ContextVars."""
    context = contextvars.copy_context()
    return lambda: context.run(callback)

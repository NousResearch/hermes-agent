"""The per-thread tool whitelist for background-review forks and ``/btw`` side questions.

Every tool call — sequential included — is dispatched on a ``DaemonThreadPoolExecutor``
worker wrapped in ``propagate_context_to_thread``, which copies ContextVars and the prompt
callbacks but not thread-locals. The whitelist therefore lives in a ``threading.local``
(for same-thread reads) mirrored in a ContextVar, so the propagated context carries it to
the worker; the thread-local wins when both are set, and a bare thread (no propagation)
keeps seeing no whitelist.

Split out of ``hermes_cli/plugins.py``, which re-exports the setters so plugins and tests
keep one import path.
"""

from __future__ import annotations

import contextvars
import threading
from typing import Optional

_thread_tool_whitelist = threading.local()

# What a tool-dispatch worker reads: propagate_context_to_thread carries ContextVars (not
# thread-locals) across, so the worker falls back to this snapshot of the parent's whitelist.
_tool_whitelist_var: contextvars.ContextVar[Optional[tuple[frozenset, str]]] = (
    contextvars.ContextVar("tool_whitelist", default=None)
)


def set_thread_tool_whitelist(
    allowed: Optional[set[str]],
    deny_msg_fmt: str = "Tool '{tool_name}' denied: not in this thread's tool whitelist",
) -> None:
    _thread_tool_whitelist.allowed = allowed
    _thread_tool_whitelist.fmt = deny_msg_fmt
    _tool_whitelist_var.set(
        None if allowed is None else (frozenset(allowed), deny_msg_fmt)
    )


def clear_thread_tool_whitelist() -> None:
    _thread_tool_whitelist.allowed = None
    _tool_whitelist_var.set(None)


def thread_tool_whitelist_block_message(tool_name: str) -> Optional[str]:
    """The whitelist's block message for *tool_name* (``None`` when the tool is allowed or no
    whitelist is in effect on this thread)."""
    allowed = getattr(_thread_tool_whitelist, "allowed", None)
    fmt = getattr(_thread_tool_whitelist, "fmt", None)
    if allowed is None:
        # On a propagate_context_to_thread worker the thread-local is unset; the ContextVar
        # mirror carries the parent's whitelist across.
        ctx_whitelist = _tool_whitelist_var.get()
        if ctx_whitelist is not None:
            allowed, fmt = ctx_whitelist
    if allowed is not None and tool_name not in allowed:
        return (fmt or "Tool '{tool_name}' denied").format(tool_name=tool_name)
    return None

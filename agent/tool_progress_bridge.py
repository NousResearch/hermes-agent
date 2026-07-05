"""Tool-progress bridge — lets a long-running tool handler stream sub-steps to the host UI.

Hermes fires ``tool.started`` / ``tool.completed`` AROUND a tool, but a black-box handler
(e.g. a subprocess that runs for minutes, like ``torya_build``) has no way to report progress
in between. This ContextVar carries the active agent's ``tool_progress_callback`` into the
handler's scope, so the handler can emit sub-steps that render on the SAME path (Discord/TUI)
as normal progress.

Install: copy this file to ``agent/tool_progress_bridge.py`` and set the ContextVar around
tool execution (see README.md — ~3 lines in ``agent/tool_executor.py``). The Torya plugin
imports ``emit_subtool_progress`` softly, so this is optional: without it the plugin still
returns its final verified result, just without live streaming.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Callable, Optional

# The active agent's tool_progress_callback, made available to tool handlers (task-local).
_CURRENT: "ContextVar[Optional[Callable]]" = ContextVar("HERMES_TOOL_PROGRESS", default=None)


def set_tool_progress(cb: Optional[Callable]) -> Token:
    """Bind the current agent's progress callback for the duration of a tool call."""
    return _CURRENT.set(cb)


def reset_tool_progress(token: Token) -> None:
    """Unbind (call in a finally: after the tool returns)."""
    _CURRENT.reset(token)


def emit_subtool_progress(label: str, detail: str = "") -> None:
    """A tool handler calls this to stream a sub-step to the host UI.

    Emitted as a ``tool.started`` event: that is the ONLY event type Hermes' gateway relays to
    chat surfaces (see gateway/run.py — the progress path explicitly ignores tool.completed,
    reasoning.available, etc.). ``label`` becomes the tool_name and ``detail`` the preview, so a
    sub-step renders as a normal progress bubble on Discord/CLI. No tool.completed is paired, so
    it never touches tool-count accounting.
    """
    cb = _CURRENT.get()
    if cb is None:
        return
    try:
        cb("tool.started", label, detail, None)
    except Exception:
        # Streaming is best-effort — a rendering hiccup must never fail the tool.
        return

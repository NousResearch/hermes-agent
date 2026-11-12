"""Live progress from inside a running tool call (MCP ``notifications/progress``) onto the agent's
``tool_progress_callback`` as ``tool.progress`` events.

The executor binds a relay for each dispatched call (``bound_tool_progress``); a tool reads it on its
own thread with ``current_tool_progress()`` and may invoke the result from any thread (the MCP client
fires it on the MCP event loop). Emission is throttled per call so a chatty server cannot flood the
UI; the first update and the one reaching ``total`` always pass.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import threading
import time
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)

TOOL_PROGRESS_EVENT = "tool.progress"
_MIN_INTERVAL_S = 0.5

ProgressReporter = Callable[[float, Optional[float], Optional[str]], None]
_current: contextvars.ContextVar[Optional[ProgressReporter]] = contextvars.ContextVar("tool_progress_reporter", default=None)


def _reporter(callback, tool_name: str, tool_call_id: str) -> ProgressReporter:
    lock = threading.Lock()
    last = [0.0]

    def report(progress: float, total: Optional[float] = None, message: Optional[str] = None) -> None:
        done = total is not None and total > 0 and progress >= total
        now = time.monotonic()
        with lock:
            if last[0] and not done and now - last[0] < _MIN_INTERVAL_S:
                return
            last[0] = now
        percent = round(min(max(progress / total, 0.0), 1.0) * 100) if total else None
        try:
            callback(TOOL_PROGRESS_EVENT, tool_name, message or None, None, tool_call_id=tool_call_id,
                     progress=float(progress), total=None if total is None else float(total), percent=percent)
        except Exception:  # health: allow BLE001 -- a UI sink failing must never fail the tool call
            logger.debug("tool.progress callback failed for %s", tool_name, exc_info=True)
    return report


@contextlib.contextmanager
def bound_tool_progress(callback, tool_name: str, tool_call_id: str) -> Iterator[None]:
    """Bind this call's relay for the block (no-op when the agent has no progress callback)."""
    token = _current.set(_reporter(callback, tool_name, tool_call_id or "") if callback else None)
    try:
        yield
    finally:
        _current.reset(token)


def current_tool_progress() -> Optional[ProgressReporter]:
    """The running call's reporter, or None outside a dispatched tool call."""
    return _current.get()


def tool_progress_payload(name: Optional[str], message: Optional[str], kwargs: dict) -> dict:
    """The ``tool.progress`` wire payload every viewer surface publishes (gateway event stream and
    the legacy TUI gateway): ``percent`` is 0-100 when the server sent a ``total``, else None."""
    return {"tool_id": str(kwargs.get("tool_call_id") or ""), "name": str(name or ""),
            "progress": float(kwargs.get("progress") or 0.0), "total": kwargs.get("total"),
            "percent": kwargs.get("percent"), "message": str(message) if message else None}

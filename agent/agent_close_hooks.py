"""On-agent-close callback registry for plugin cleanup.

Plugins holding per-agent resources (watchers, temp state, background handles)
Plugins register via register_on_agent_close(cb); cb receives the closing task_id.

Contract (narrow by design):

- run-once: run_on_agent_close drains the registry, so a repeated close
  never re-fires a callback;
- pre-cleanup: ClientLifecycleMixin._close_task_resources runs this
  before its teardown steps;
- late-register runs immediately with the most recent task_id;
- errors swallowed (debug-logged): one bad plugin never breaks agent close.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger("run_agent")

_lock = threading.Lock()
_callbacks: list = []
_fired = False
_last_task_id = ""


def register_on_agent_close(cb: Callable) -> None:
    """Register *cb(task_id)*; runs immediately if the agent already closed."""
    if not callable(cb):
        return
    with _lock:
        global _fired
        if _fired:
            pending = [cb]
            task_id = _last_task_id
        elif any(cb is existing for existing in _callbacks):
            return
        else:
            _callbacks.append(cb)
            return
    for fn in pending:
        try:
            fn(task_id)
        except Exception as exc:
            logger.debug("on-agent-close callback failed: %s", exc)


def run_on_agent_close(task_id: str) -> None:
    """Fire registered callbacks once, pre-cleanup. Never raises."""
    with _lock:
        global _fired, _last_task_id
        _last_task_id = task_id or ""
        if _fired:
            return
        _fired = True
        pending = _callbacks[:]
        _callbacks.clear()
    for fn in pending:
        try:
            fn(task_id)
        except Exception as exc:
            logger.debug("on-agent-close callback failed: %s", exc)


def reset_for_tests() -> None:
    """Clear the registry (tests only)."""
    with _lock:
        global _fired, _last_task_id
        _callbacks.clear()
        _fired = False
        _last_task_id = ""

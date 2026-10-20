"""On-agent-close callback registry for plugin cleanup.

Plugins holding per-agent resources (watchers, temp state, background handles)
Plugins register via register_on_agent_close(cb); cb receives the closing task_id.

Contract (narrow by design):

- run-once per agent: run_on_agent_close drains the registry, so a repeated
  close never re-fires a callback;
- pre-cleanup: ClientLifecycleMixin._close_task_resources runs this
  before its teardown steps;
- late-register runs immediately with the most recent task_id;
- per-agent generations: a plugin serving a new agent passes its task_id at
  registration; registering for a different task after a latched close starts
  a fresh generation instead of firing immediately with the stale task_id;
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


def register_on_agent_close(cb: Callable, task_id: str | None = None) -> None:
    """Register *cb(task_id)*; runs immediately if its agent already closed.

    Pass the owning agent's *task_id* when registering for a new agent after
    a previous agent closed; otherwise the callback fires immediately with the
    previous agent's task_id (legacy late-register behavior).
    """
    if not callable(cb):
        return
    with _lock:
        global _fired, _last_task_id
        target = task_id or _last_task_id
        if _fired and target != _last_task_id:
            # New agent generation after a latched close: start fresh instead
            # of firing immediately with the previous agent's task_id.
            _callbacks.clear()
            _fired = False
            _last_task_id = target
        if _fired:
            pending = [cb]
            fire_with = _last_task_id
        elif any(cb is existing for existing in _callbacks):
            return
        else:
            _callbacks.append(cb)
            return
    for fn in pending:
        try:
            fn(fire_with)
        except Exception as exc:
            logger.debug("on-agent-close callback failed: %s", exc)


def run_on_agent_close(task_id: str) -> None:
    """Fire registered callbacks once, pre-cleanup. Never raises."""
    with _lock:
        global _fired, _last_task_id
        task_id = task_id or ""
        if _fired and task_id != _last_task_id:
            # A different agent is closing after a latched close: its
            # generation starts here.
            _callbacks.clear()
            _fired = False
        if not _fired and _callbacks and _last_task_id and task_id != _last_task_id:
            # Stale duplicate close for an older agent while a newer
            # generation owns the registry: do not steal its callbacks.
            return
        _last_task_id = task_id
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

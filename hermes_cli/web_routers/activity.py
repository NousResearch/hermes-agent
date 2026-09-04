"""``GET /api/activity`` — the backend's busy signal for supervisors that reap idle
``hermes serve`` processes (the desktop app runs one per profile).

Wall-clock idleness of the RPC transport is not idleness of the process: a running turn,
a detached ``delegate_task`` subagent (a thread in this process) or a background terminal
process is live work with no request traffic. Reaping on transport idleness alone killed a
backend mid-delegation; this endpoint is what the reaper must consult first.

Answers in milliseconds from in-memory counters only: no DB, no threadpool, no imports of
the TUI gateway (a process that never opened a chat has no running turns by definition).
"""

from __future__ import annotations

import sys
from typing import Callable

from fastapi import APIRouter, Request
from hermes_cli.web_deps import late

router = APIRouter()

_require_token = late("_require_token")


def activity_snapshot(running_turns: int | Callable[[], int], active_subagents: int | Callable[[], int],
                      background_processes: int | Callable[[], int]) -> dict:
    """Aggregate the three live-work counters into the endpoint payload. ``busy`` is true iff any
    counter is positive; a counter whose source fails counts as 0 (the reaper falls back to its
    own idle policy rather than being told a healthy backend is stuck busy)."""
    counts = {}
    for name, source in (("running_turns", running_turns), ("active_subagents", active_subagents),
                         ("background_processes", background_processes)):
        try:
            value = int(source() if callable(source) else source)
        except Exception:
            value = 0
        counts[name] = max(0, value)
    return {"ok": True, "busy": any(counts.values()), **counts}


def _count_running_turns() -> int:
    """Live TUI/desktop sessions with a turn in flight. Read through ``sys.modules`` so the
    probe never imports the gateway (and its threads) into a process that never needed it."""
    server = sys.modules.get("tui_gateway.server")
    if server is None:
        return 0
    with server._sessions_lock:
        return sum(1 for s in server._sessions.values() if s.get("running") and not s.get("_finalized"))


def _count_active_subagents() -> int:
    from tools.async_delegation import active_count
    return active_count()


def _count_background_processes() -> int:
    from tools.process_registry import process_registry
    return process_registry.count_running()


@router.get("/api/activity")
async def get_activity(request: Request):
    _require_token(request)
    return activity_snapshot(_count_running_turns, _count_active_subagents, _count_background_processes)

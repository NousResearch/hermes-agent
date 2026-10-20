"""``active-work`` control verb: what a stop/restart of this gateway would interrupt, answered live.

``gateway_state.json`` names each unit only while draining and its ``active_agents`` total leaves out
async delegations, so a CLI asking "may I stop this gateway?" could not see a background subagent
that ``stop()`` interrupts (``tools.async_delegation.interrupt_all``). The verb answers from the
running process instead: the same units the drain report publishes plus every live delegation.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable


def _delegation_units(now: float) -> list[dict[str, Any]]:
    from tools.async_delegation import _LIVE_STATES, list_async_delegations

    units = []
    for record in list_async_delegations():
        if record.get("status") not in _LIVE_STATES:
            continue
        unit: dict[str, Any] = {"kind": "delegation", "delegation_id": record.get("delegation_id"),
                                "goal": str(record.get("goal") or "")[:80], "pid": os.getpid()}
        if record.get("dispatched_at"):
            unit["elapsed_s"] = round(now - float(record["dispatched_at"]), 1)
        units.append(unit)
    return units


def active_work_verb(runner, loop) -> Callable[[], dict[str, Any]]:
    """Handler for the control socket: runs on its executor thread, reads the runner on the loop."""

    def _handler() -> dict[str, Any]:
        box: list[list] = []
        done = threading.Event()

        def _collect() -> None:
            try:
                box.append(runner._describe_active_work())
            finally:
                done.set()

        loop.call_soon_threadsafe(_collect)
        done.wait(timeout=5.0)
        units = list(box[0]) if box else []
        units += _delegation_units(time.time())
        return {"active_agents": runner._active_work_count(), "active_work": units, "pid": os.getpid()}

    return _handler

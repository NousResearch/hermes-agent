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


def _profile_units(runner, profile: str, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What an ``unserve-profile`` of *profile* stops (``stop_authority_work``): its chat turns (session
    keys in its namespace) plus its authority's managed turns and cron runs. Delegations survive it."""
    from gateway.session import _session_key_namespace

    prefix = _session_key_namespace(profile) + ":"
    mine = [u for u in units if u.get("kind") == "chat" and str(u.get("session") or "").startswith(prefix)]
    home = (getattr(runner, "_served_profile_homes", None) or {}).get(profile)
    registry = getattr(runner, "session_authorities", None)
    authority = registry.for_home(home) if registry is not None and home is not None else None
    managed = sum(not w.closed.is_set() for w in getattr(authority, "_managed_workers", {}).values())
    crons = len(getattr(authority, "_cron_cancellations", None) or {})
    return mine + [{"kind": "managed", "pid": os.getpid()}] * managed + [{"kind": "cron", "job_id": "?", "pid": os.getpid()}] * crons


def active_work_verb(runner, loop) -> Callable[[], dict[str, Any]]:
    """Handler for the control socket: runs on its executor thread, reads the runner on the loop."""

    def _handler(params: dict[str, Any]) -> dict[str, Any]:
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
        profile = params.get("profile")
        if isinstance(profile, str) and profile:
            units = _profile_units(runner, profile, units)
            return {"active_agents": len(units), "active_work": units, "pid": os.getpid(), "profile": profile}
        return {"active_agents": runner._active_work_count(), "active_work": units, "pid": os.getpid()}

    return _handler

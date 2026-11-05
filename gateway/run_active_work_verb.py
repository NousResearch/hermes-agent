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
    from gateway.session_runtime_workers import side_agents

    managed = sum(not w.closed.is_set() for w in getattr(authority, "_managed_workers", {}).values())
    crons = len(getattr(authority, "_cron_cancellations", None) or {})
    sides = len(side_agents(authority)) if authority is not None else 0
    pid = os.getpid()
    return (mine + [{"kind": "managed", "pid": pid}] * managed + [{"kind": "cron", "job_id": "?", "pid": pid}] * crons
            + [{"kind": "side_agent", "pid": pid}] * sides)


def active_work_verb(runner, loop) -> Callable[[], dict[str, Any]]:
    """Handler for the control socket: runs on its executor thread, reads the runner on the loop."""

    def _handler(params: dict[str, Any]) -> dict[str, Any]:
        profile = params.get("profile")
        scoped = isinstance(profile, str) and bool(profile)
        box: list[dict[str, Any]] = []
        done = threading.Event()

        def _collect() -> None:
            # Everything that reads live authority state runs here, on the loop: a read on the socket
            # thread can race a worker being added and raise, and an error reads as "no work".
            try:
                units = list(runner._describe_active_work()) + _delegation_units(time.time())
                if scoped:
                    units = _profile_units(runner, profile, units)
                    box.append({"active_agents": len(units), "active_work": units, "profile": profile})
                else:
                    box.append({"active_agents": runner._active_work_count(), "active_work": units})
            finally:
                done.set()

        loop.call_soon_threadsafe(_collect)
        if not done.wait(timeout=5.0) or not box:
            raise RuntimeError("active-work: the gateway loop did not answer")
        return {**box[0], "pid": os.getpid()}

    return _handler

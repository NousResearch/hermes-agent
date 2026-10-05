"""Post-update escalation for gateways the fleet matrix proved stale.

``_verify_fleet_after_update`` compares every live gateway's stamped ``code_sha`` against the
fresh checkout. Until now a ``stale`` row only failed the update (exit 1) and left the process
running pre-update modules: its cron ticker then yields every tick to the "fresh gateway" it
assumes exists, and nothing ever restarts it (#117275). A proven-stale survivor is now handed to
the same drain-first ``request_restart`` path (SIGUSR1) the restart phase uses — a supervised
gateway respawns on the new code, a bare ``gateway run`` stops and is listed for a manual restart.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def stale_fleet_survivor_pids(fleet: list, already_signalled: set) -> list[int]:
    """Live PIDs the fleet matrix stamped ``stale`` that the restart phase never touched."""
    pids: list[int] = []
    for row in fleet or []:
        pid = row.get("pid") if isinstance(row, dict) else None
        if row.get("state") == "stale" and isinstance(pid, int) and pid > 0 and pid not in already_signalled:
            pids.append(pid)
    return pids


def signal_stale_fleet_survivors(fleet: list, restart, drain_budget: float) -> list[int]:
    """Drain-first restart every proven-stale gateway; returns the PIDs signalled.

    Bookkeeping lands in ``restart.killed_pids`` so the receipt and the survivor sweep see them.
    Never raises: verification must still finalize the receipt and exit 1.
    """
    from hermes_cli.update_cmd_fleet import _drain_or_signal_gateway_for_update

    pids = stale_fleet_survivor_pids(fleet, set(restart.killed_pids))
    if not pids:
        return []
    try:
        from hermes_cli.gateway import _get_service_pids
        service_pids = set(_get_service_pids(all_profiles=True))
    except Exception:
        service_pids = set()
    labels = {row.get("pid"): str(row.get("profile") or "gateway") for row in fleet if isinstance(row, dict)}
    print()
    print(f"  ⚠ {len(pids)} gateway process(es) still run the pre-update code — requesting a restart")
    signalled: list[int] = []
    manual: list[int] = []
    for pid in pids:
        label = f"{labels.get(pid, 'gateway')} (PID {pid})"
        try:
            if _drain_or_signal_gateway_for_update(pid, drain_budget, label):
                signalled.append(pid)
                restart.killed_pids.add(pid)
                if pid not in service_pids:
                    manual.append(pid)
        except Exception as exc:
            logger.warning("Could not signal stale gateway PID %s: %s", pid, exc)
            print(f"  ⚠ {label}: could not be signalled ({exc}) — restart it by hand")
    if manual:
        print(f"  → Stopped {len(manual)} manual gateway process(es) that had no supervisor to respawn them")
        print("    Restart manually: hermes gateway run")
        if len(manual) > 1:
            print("    (or: hermes -p <profile> gateway run  for each profile)")
    return signalled

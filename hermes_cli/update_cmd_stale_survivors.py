"""Post-update escalation for gateways the fleet matrix proved stale.

``_verify_fleet_after_update`` compares every live gateway's stamped ``code_sha`` against the
fresh checkout. A plain ``stale`` verdict only fails the update (exit 1) and leaves the process
running pre-update modules: its cron ticker then yields every tick to the "fresh gateway" it
assumes exists and nothing ever restarts it (#117275). A proven-stale survivor is therefore
handed to the same drain-first ``request_restart`` path (SIGUSR1) the restart phase uses.

Supervision is decided by OUTCOME, not by pid sets. On macOS the launchd job's registered pid is
the ``osascript`` launcher while the gateway that actually serves sits one or two levels below it
(``osascript`` → shim python → gateway python; the serving pid is what the fleet matrix, the state
file, and SIGUSR1 all use). Comparing the serving pid against ``gateway._get_service_pids()``
therefore misread every supervised restart as a bare manual process: the drain signal worked, the
supervisor queued the successor, and the update still printed "Restart manually" and failed. So:
signal every stale survivor, give the supervisor a bounded settle window, re-verify the fleet with
the same snapshot the happy path uses, and only report a manual restart for survivors that truly
never come back.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# Bounded successor-settle window after the drain-first signals. The observed launchd KeepAlive
# respawn latency for the default multiplexer is ~20s once the old PID exits; 90s covers a slow
# boot without stalling a genuinely failed restart much beyond the waits already spent draining.
SURVIVOR_SETTLE_BUDGET_S = 90.0
SURVIVOR_SETTLE_POLL_S = 5.0


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
    Whether a supervisor actually brings a successor back is decided afterwards by
    :func:`settle_survivor_restarts` (outcome-based — see the module docstring). Never raises:
    verification must still finalize the receipt and exit 1.
    """
    from hermes_cli.update_cmd_fleet import _drain_or_signal_gateway_for_update

    pids = stale_fleet_survivor_pids(fleet, set(restart.killed_pids))
    if not pids:
        return []
    labels = {row.get("pid"): str(row.get("profile") or "gateway") for row in fleet if isinstance(row, dict)}
    print()
    print(f"  ⚠ {len(pids)} gateway process(es) still run the pre-update code — requesting a restart")
    signalled: list[int] = []
    for pid in pids:
        label = f"{labels.get(pid, 'gateway')} (PID {pid})"
        try:
            if _drain_or_signal_gateway_for_update(pid, drain_budget, label):
                signalled.append(pid)
                restart.killed_pids.add(pid)
        except Exception as exc:
            logger.warning("Could not signal stale gateway PID %s: %s", pid, exc)
            print(f"  ⚠ {label}: could not be signalled ({exc}) — restart it by hand")
    return signalled


def settle_survivor_restarts(snapshot: list, collect, budget_s: float = SURVIVOR_SETTLE_BUDGET_S) -> tuple[bool, list]:
    """Wait up to ``budget_s`` for supervisors to replace every signalled stale gateway.

    ``collect`` is the caller's fleet-snapshot function (already bound to its probe arguments).
    Returns ``(converged, fresh_snapshot)``: converged when every profile that was ``stale`` or
    ``down`` in ``snapshot`` now reports a ``current`` row — the fresh successor verified on the
    updated checkout, the same evidence the happy-path matrix demands. Never raises; a failed or
    empty re-probe keeps the update incomplete (fail-closed).
    """
    need = {
        row.get("profile")
        for row in snapshot or []
        if isinstance(row, dict) and row.get("state") in ("stale", "down") and row.get("profile")
    }
    if not need:
        return False, list(snapshot or [])
    deadline = time.monotonic() + max(float(budget_s), 0.0)
    fresh: list = []
    while True:
        try:
            fresh = collect() or []
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Survivor settle re-probe failed: %s", exc)
            fresh = []
        rows = {row.get("profile"): row for row in fresh if isinstance(row, dict)}
        if all(rows.get(p, {}).get("state") == "current" for p in need):
            return True, fresh
        if time.monotonic() >= deadline:
            return False, fresh
        try:
            time.sleep(SURVIVOR_SETTLE_POLL_S)
        except Exception:  # pragma: no cover - defensive
            return False, fresh


def report_unsettled_survivors(snapshot: list, fresh: list) -> None:
    """Manual-restart fallback for survivors no supervisor brought back in the window."""
    fresh_by_profile = {row.get("profile"): row for row in fresh or [] if isinstance(row, dict)}
    still_out = [
        row for row in snapshot or []
        if isinstance(row, dict)
        and row.get("state") in ("stale", "down")
        and fresh_by_profile.get(row.get("profile"), {}).get("state") != "current"
    ]
    if not still_out:
        return
    print(f"  → Stopped {len(still_out)} manual gateway process(es) that had no supervisor to respawn them")
    print("    Restart manually: hermes gateway run")
    if len(still_out) > 1:
        print("    (or: hermes -p <profile> gateway run  for each profile)")

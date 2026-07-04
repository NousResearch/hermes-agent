from __future__ import annotations

from pathlib import Path
from typing import Any

from hermes_cli import control_db as cp


def supervise_once(*, root: Path | None, actor_instance_id: str, dry_run: bool = True, stale_ms: int = 600_000) -> dict[str, Any]:
    """Record one watchdog pass over the control DB.

    In dry_run mode this is observational only. Non-dry mode currently only
    reaps expired dispatch leases through the existing DB state transition; it
    does not kill processes or restart services.
    """
    conn = cp.connect(root=root)
    try:
        run_id = cp.start_supervision_run(conn, actor_instance_id=actor_instance_id, scope={"dry_run": dry_run, "stale_ms": stale_ms})
        now = cp.now_ms()
        stale = [
            dict(r)
            for r in conn.execute(
                "SELECT dispatch_id, receiver_profile, status, updated_at_ms FROM cp_dispatches WHERE status IN ('pending','running','blocked') AND updated_at_ms < ? ORDER BY updated_at_ms",
                (now - stale_ms,),
            ).fetchall()
        ]
        open_blockers = cp.list_blockers(conn, status="open", limit=200)
        actions: list[dict[str, Any]] = []
        if not dry_run:
            reaped = cp.reap_expired_dispatches(conn, now_ms=now)
            actions.append({"action": "reap_expired_dispatches", "count": reaped})
            expired_workers = cp.mark_expired_worker_instances_offline(conn, now_ms_value=now)
            actions.append({"action": "mark_expired_worker_instances_offline", "count": len(expired_workers), "instance_ids": expired_workers})
        findings = [
            {"code": "stale_dispatches", "count": len(stale), "dispatches": [r["dispatch_id"] for r in stale]},
            {"code": "open_blockers", "count": len(open_blockers), "blockers": [r["blocker_id"] for r in open_blockers]},
        ]
        cp.finish_supervision_run(conn, run_id, status="completed", findings=findings, actions=actions)
        return {"run_id": run_id, "findings": findings, "actions": actions, "dry_run": dry_run}
    finally:
        conn.close()

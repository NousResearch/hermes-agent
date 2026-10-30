#!/usr/bin/env python3
"""Recover a mission after an isolated gateway-dispatched worker has exited."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hermes_cli import mission_ownership as missions
from hermes_cli.kanban_db_connect import connect


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    conn = connect(db_path=Path(os.environ["HERMES_KANBAN_DB"]))
    try:
        mission_row = conn.execute(
            "SELECT id,owner_generation,owner_expires FROM missions "
            "WHERE owner_kind='task' AND owner_id=? AND status='active'", (args.task_id,),
        ).fetchone()
        if mission_row is None:
            raise SystemExit("no active task-owned mission for isolated worker")
        before = conn.execute(
            "SELECT status,worker_pid,current_run_id FROM tasks WHERE id=?", (args.task_id,),
        ).fetchone()
        if before is None or before["status"] != "running" or not before["worker_pid"]:
            raise SystemExit("controlled worker did not leave a running task record")
        pid = int(before["worker_pid"])
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit("refusing recovery while isolated worker PID is still alive")
        generation = missions.recover(
            conn, mission_id=mission_row["id"], expected_generation=mission_row["owner_generation"],
            new_owner_kind="session", new_owner_id="phase1-recovery-session",
            recovery_actor="phase1-isolated-recovery-probe", now=mission_row["owner_expires"] + 1,
        )
        task = conn.execute("SELECT status,worker_pid,current_run_id FROM tasks WHERE id=?", (args.task_id,)).fetchone()
        run = conn.execute("SELECT status,outcome FROM task_runs WHERE id=?", (before["current_run_id"],)).fetchone()
        mission = missions.get(conn, mission_row["id"])
        if task["status"] != "blocked" or task["worker_pid"] is not None or run["status"] != "crashed":
            raise SystemExit("dead worker was not fenced and parked during recovery")
        print(json.dumps({
            "worker_pid": pid,
            "worker_alive_before_recovery": False,
            "task_after_recovery": task["status"],
            "run_after_recovery": run["status"],
            "mission_owner_after_recovery": [mission["owner_kind"], mission["owner_id"]],
            "generation": generation,
            "automatic_reclaim": False,
        }, sort_keys=True))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

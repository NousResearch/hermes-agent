#!/usr/bin/env python3
"""Demonstrate whether detached subprocesses remain fenced by mission execution leases."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from hermes_cli import kanban_db, mission_ownership as missions
from hermes_cli.kanban_db_connect import connect


def main() -> int:
    db_path = Path(os.environ["HERMES_KANBAN_DB"])
    marker = Path(os.environ["TMPDIR"]) / "detached-worker-finished"
    if marker.exists():
        raise SystemExit("refusing to reuse detached-worker marker")
    conn = connect(db_path=db_path)
    try:
        mission_id, _ = missions.admit(
            conn,
            ingress_key="phase1-detached-subprocess-probe",
            request="Controlled isolated detached subprocess probe",
            contract={"kind": "test", "outcome": "measure lease boundary", "acceptance": ["child state recorded"]},
            owner_id="phase1-detached-session",
        )
        child_code = (
            "import pathlib,time; time.sleep(2); "
            f"pathlib.Path({str(marker)!r}).write_text('finished')"
        )
        with missions.execution_scope(session_id="phase1-detached-session", tool_name="detached-probe"):
            child = subprocess.Popen([sys.executable, "-c", child_code], start_new_session=True)
        lease_count = conn.execute(
            "SELECT count(*) FROM mission_execution_leases WHERE mission_id=?", (mission_id,)
        ).fetchone()[0]
        task_id = kanban_db.create_task(
            conn, title="Isolated handoff probe", body="Controlled probe", assignee="fred", triage=True,
        )
        generation = missions.handoff(
            conn, mission_id=mission_id, expected_owner_kind="session",
            expected_owner_id="phase1-detached-session", expected_generation=1,
            new_owner_kind="task", new_owner_id=task_id, progress={"probe": "detached child"},
        )
        child_alive_at_handoff = child.poll() is None
        child.wait(timeout=8)
        if child.returncode != 0 or not marker.is_file():
            raise SystemExit("controlled detached child did not finish its isolated marker write")
        result = {
            "mission_generation_after_handoff": generation,
            "lease_count_after_handler_return": lease_count,
            "child_alive_when_handoff_committed": child_alive_at_handoff,
            "child_completed_after_handoff": marker.read_text(encoding="utf-8") == "finished",
            "result": "unsafe detached-operation gap reproduced",
        }
        print(json.dumps(result, sort_keys=True))
        if lease_count != 0 or not child_alive_at_handoff:
            raise SystemExit("probe did not reproduce the intended detached-operation condition")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

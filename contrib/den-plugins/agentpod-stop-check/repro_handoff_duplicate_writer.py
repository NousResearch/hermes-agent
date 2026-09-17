"""Reproduction: review handoff leaves the previous worker alive in the same
workspace, and the very next claim spawns a second writer into that directory.

This is NOT part of the plugin's test suite — it is the evidence attached to
the carve-out card for the core lifecycle defect. Run it directly:

    python contrib/den-plugins/agentpod-stop-check/repro_handoff_duplicate_writer.py

It uses an isolated temporary board and a process this script owns.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="handoff-repro-"))
    os.environ["HERMES_HOME"] = str(tmp)
    from hermes_cli import kanban_db as kb

    conn = kb.connect(tmp / "board.db")
    ws = tmp / "shared-workspace"
    ws.mkdir()
    tid = kb.create_task(
        conn, title="repro", assignee="software-engineer",
        workspace_kind="dir", workspace_path=str(ws),
    )

    # A process standing in for the reviewer worker that is still running its
    # turn when it calls kanban_request_changes.
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        kb.request_review(conn, tid, summary="impl done", reviewer="reviewer")
        claimed = kb.claim_review_task(conn, tid)
        assert claimed is not None, "review claim failed"
        run_id = kb.get_task(conn, tid).current_run_id
        # The dispatcher records the spawned worker pid on the task row.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid = ? WHERE id = ?",
                (worker.pid, tid),
            )
        ok, implementer = kb.request_changes(
            conn, tid, reason="needs rework", expected_run_id=run_id,
        )
        assert ok, implementer

        alive = worker.poll() is None
        t = kb.get_task(conn, tid)
        # The next dispatcher tick claims immediately.
        second = kb.claim_task(conn, tid)

        print(f"previous worker pid {worker.pid} alive after handoff: {alive}")
        print(f"task row worker_pid after handoff: {t.worker_pid}")
        print(f"task status after handoff:          {t.status}")
        print(f"next claim succeeded immediately:   {second is not None}")
        print(f"same workspace for both writers:    {t.workspace_path == str(ws)}")
        duplicate = alive and second is not None
        print()
        print("RESULT:", "DUPLICATE WRITER REPRODUCED" if duplicate else "no duplicate")
        return 0 if duplicate else 1
    finally:
        worker.kill()
        worker.wait(timeout=5)
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

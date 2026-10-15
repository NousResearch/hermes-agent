"""An operator stop confirms worker exit; a failed stop retains retry evidence."""
import argparse
import json
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect
from hermes_cli.kanban_db_dispatch import _set_worker_pid
from hermes_cli.kanban_stop import stop_task


@pytest.mark.platforms("posix")
def test_stop_reaps_worker_without_handoff_grace_and_preserves_workspace(tmp_path, monkeypatch, capsys):
    from hermes_cli.kanban import kanban_command
    from hermes_cli.kanban_parser import build_parser

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers(dest="command"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    retained = workspace / "retained.txt"
    retained.write_text("research evidence")
    with connect(tmp_path / "board.db") as conn:
        worker = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"])
        try:
            task = kb.create_task(conn, title="stop proof", workspace_kind="dir", workspace_path=str(workspace))
            assert kb.claim_task(conn, task) is not None
            _set_worker_pid(conn, task, worker.pid)
            args = parser.parse_args(["kanban", "stop", task, "--reason", "operator stop", "--json"])
            assert kanban_command(args) == 0
            result = json.loads(capsys.readouterr().out)
            assert result == {"task_id": task, "status": "blocked", "stopped": True, "pending_runs": []}
            assert worker.wait(timeout=10) != 0
            assert retained.read_text() == "research evidence"
            assert stop_task(conn, task, "retry").stopped
            assert kb.claim_task(conn, task) is None
        finally:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=10)


@pytest.mark.platforms("posix")
def test_stop_retains_unverified_worker_and_retry_terminates_after_identity_is_known(tmp_path):
    from hermes_cli.kanban_db_dispatch import UNVERIFIED_WORKER_FINGERPRINT

    with connect(tmp_path / "board.db") as conn:
        worker = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"])
        try:
            task = kb.create_task(conn, title="unverified worker")
            assert kb.claim_task(conn, task) is not None
            _set_worker_pid(conn, task, worker.pid)
            run = kb._current_run_id(conn, task)
            fingerprint = conn.execute("SELECT worker_started_at FROM task_runs WHERE id=?", (run,)).fetchone()[0]
            conn.execute("UPDATE task_runs SET worker_started_at=? WHERE id=?", (UNVERIFIED_WORKER_FINGERPRINT, run))
            result = stop_task(conn, task, "operator stop")
            assert not result.stopped and result.pending_runs == (run,)
            assert worker.poll() is None
            assert conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (run,)).fetchone()[0] == worker.pid
            conn.execute("UPDATE task_runs SET worker_started_at=? WHERE id=?", (fingerprint, run))
            assert stop_task(conn, task, "retry").stopped
            assert worker.wait(timeout=10) != 0
        finally:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=10)

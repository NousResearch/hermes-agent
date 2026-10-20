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


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # Cleanup must reap the exact child if its worker orphaned it.
@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("resist", [False, True])
def test_stop_reaps_real_hermes_background_process(tmp_path, monkeypatch, persist, resist):
    import os
    from pathlib import Path

    from hermes_cli.kanban_db_dispatch import _pid_alive
    from tests.waiting import wait_until

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    ready = tmp_path / "ready.json"
    script = tmp_path / "worker.py"
    script.write_text('''
import json, os, signal, sys
from pathlib import Path
from types import SimpleNamespace
from run_agent import AIAgent
from hermes_cli.cli_single_query import _install_single_query_signal_handlers
from tools.process_registry import process_registry
agent = AIAgent(model="test/stop-proof", provider="openrouter", api_key="dummy-proof-key",
    base_url="http://127.0.0.1:1/v1", enabled_toolsets=[], skip_memory=True,
    skip_background_review=True, quiet_mode=True, skip_context_files=True)
_install_single_query_signal_handlers(SimpleNamespace(agent=agent))
session = process_registry.spawn_local("sleep 300", task_id="proof", owner_task_id="proof",
    persist_on_release=sys.argv[2] == "True")
if sys.argv[3] == "True":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(json.dumps({"pid": session.pid,
    "handler": str(signal.getsignal(signal.SIGTERM))}))
signal.pause()
''')
    with connect(tmp_path / "board.db") as conn, (tmp_path / "worker.log").open("w+") as log:
        task = kb.create_task(conn, title="background stop", workspace_kind="dir", workspace_path=str(tmp_path))
        assert kb.claim_task(conn, task) is not None
        env = {**os.environ, "HERMES_KANBAN_TASK": task,
               "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
        worker = subprocess.Popen([sys.executable, str(script), str(ready), str(persist), str(resist)],
                                  env=env, stdout=log, stderr=log)
        child = None
        try:
            wait_until(lambda: ready.exists() or worker.poll() is not None, timeout=60,
                       what="Hermes worker background process")
            log.seek(0)
            assert ready.exists(), log.read()
            receipt = json.loads(ready.read_text())
            if not resist:
                assert "_signal_handler_q" in receipt["handler"], receipt
            import psutil
            child = psutil.Process(receipt["pid"])
            _set_worker_pid(conn, task, worker.pid)
            assert stop_task(conn, task, "operator stop").stopped
            worker.wait(timeout=10)
            assert not _pid_alive(child.pid), "worker exited but its background work survived"
        finally:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=10)
            if child is not None and child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                child.kill()

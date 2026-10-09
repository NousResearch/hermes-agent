"""The redaction wrapper, not its child, owns dispatcher recovery and cleanup."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


_PAYLOAD = """
import json, os, signal, subprocess, sys, time
from pathlib import Path

root = Path(sys.argv[1])
def receipt(name, data):
    temporary = root / (name + '.tmp')
    temporary.write_text(json.dumps(data))
    temporary.replace(root / name)

signal.signal(signal.SIGTERM, signal.SIG_IGN)
if len(sys.argv) > 2:
    receipt('descendant.json', {'pid': os.getpid()})
else:
    from tools.kanban_tools import register_current_worker_from_env
    registered = register_current_worker_from_env()
    subprocess.Popen([sys.executable, __file__, str(root), 'descendant'])
    print('token=sk-' + 'C' * 80, flush=True)
    receipt('child.json', {
        'pid': os.getpid(), 'ppid': os.getppid(), 'pgid': os.getpgid(0),
        'cwd': os.getcwd(), 'registered': registered,
        'task': os.environ['HERMES_KANBAN_TASK'],
        'run': os.environ['HERMES_KANBAN_RUN_ID'],
    })
while True:
    time.sleep(1)
"""


def _wait_for(predicate, *, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    assert predicate(), "worker lifecycle condition did not arrive before deadline"


def _alive(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / ".hermes" / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    with kbc.connect_closing() as conn:
        yield conn


# Real group signals are restricted to the detached private subtree spawned below.
@pytest.mark.live_system_guard_bypass
@pytest.mark.platforms("linux")
@pytest.mark.parametrize("dispatcher_persists_pid", [True, False])
def test_reclaim_cleans_the_wrapped_worker_tree(board, tmp_path, monkeypatch, dispatcher_persists_pid):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = tmp_path / "payload.py"
    payload.write_text(_PAYLOAD)
    tid = kb.create_task(board, title="wrapper cleanup", assignee="default")
    task = kb.claim_task(board, tid)
    assert task is not None
    monkeypatch.setattr(kbd, "_worker_argv", lambda *_: [sys.executable, str(payload), str(tmp_path)])
    wrapper = kbd._default_spawn(task, str(workspace))
    assert wrapper is not None
    try:
        if dispatcher_persists_pid:
            kbd._set_worker_pid(board, tid, wrapper)
        _wait_for(lambda: (tmp_path / "child.json").exists() and (tmp_path / "descendant.json").exists())
        child = json.loads((tmp_path / "child.json").read_text())
        descendant = json.loads((tmp_path / "descendant.json").read_text())["pid"]
        row = board.execute("SELECT worker_pid, worker_started_at, claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert child["registered"] is True
        assert child["ppid"] == child["pgid"] == wrapper
        assert child["cwd"] == str(workspace)
        assert child["task"] == tid and child["run"] == str(task.current_run_id)
        assert row["worker_pid"] == wrapper
        assert row["worker_started_at"] == kbd._process_fingerprint(wrapper)
        log = kb.worker_logs_dir() / f"{tid}.log"
        assert log.stat().st_mode & 0o777 == 0o600

        signals = []

        def signal_worker(pid, signum):
            signals.append((pid, signum))
            os.kill(pid, signum)

        kbd._terminate_reclaimed_worker(
            row["worker_pid"], row["claim_lock"], started_at=row["worker_started_at"], signal_fn=signal_worker,
        )
        _wait_for(lambda: not any(_alive(pid) for pid in (wrapper, child["pid"], descendant)))
        assert signals == [(-wrapper, signal.SIGTERM), (-wrapper, signal.SIGKILL)]
        assert b"C" * 80 not in log.read_bytes()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(wrapper, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(wrapper, 0)


@pytest.mark.platforms("linux")
def test_reclaimed_wrapper_never_launches_its_child(board, tmp_path, monkeypatch):
    tid = kb.create_task(board, title="late wrapper", assignee="default")
    task = kb.claim_task(board, tid)
    assert task is not None
    assert kb.reclaim_task(board, tid, reason="operator retry")
    new_task = kb.claim_task(board, tid)
    assert new_task is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    marker = tmp_path / "child-started"
    command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).touch()", str(marker)]
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.kanban_worker_log", str(tmp_path / "worker.log"), "--", *command],
        cwd=tmp_path, capture_output=True, timeout=20, start_new_session=True,
    )
    assert result.returncode != 0
    assert not marker.exists()
    current = kb.get_task(board, tid)
    assert current is not None
    assert current.current_run_id == new_task.current_run_id and current.worker_pid is None

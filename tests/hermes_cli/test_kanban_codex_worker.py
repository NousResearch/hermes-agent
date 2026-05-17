"""Native Codex CLI worker routing for Hermes Kanban dispatch."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def captured_popen(monkeypatch):
    captured: dict[str, object] = {}

    class FakeProc:
        pid = 98765

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    return captured


def _ready_task(conn, *, assignee: str) -> kb.Task:
    task_id = kb.create_task(conn, title=f"{assignee} task", assignee=assignee)
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "ready"
    return task


class FakeCodexProc:
    def __init__(self, lines: list[str], returncode: int):
        import io
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("".join(lines))
        self._returncode = returncode

    def wait(self, timeout=None):
        return self._returncode

    def terminate(self):
        self._returncode = 124


def test_codex_assignee_routes_to_native_codex_worker(kanban_home, captured_popen):
    assignee = "codex"
    with kb.connect() as conn:
        task = _ready_task(conn, assignee=assignee)
        workspace = kb.resolve_workspace(task)

    pid = kb._default_spawn(task, str(workspace), board="default")

    assert pid == 98765
    cmd = captured_popen["cmd"]
    assert cmd[:3] == [sys.executable, "-m", "hermes_cli.codex_worker"]
    assert cmd[cmd.index("--task-id") + 1] == task.id
    assert cmd[cmd.index("--workspace") + 1] == str(workspace)
    assert cmd[cmd.index("--board") + 1] == "default"
    env = captured_popen["kwargs"]["env"]
    assert env["HERMES_PROFILE"] == "codex-worker"
    assert env["HERMES_KANBAN_TASK"] == task.id
    assert env["HERMES_KANBAN_WORKSPACE"] == str(workspace)
    assert env["HERMES_KANBAN_DB"] == str(kb.kanban_db_path(board="default"))



def test_codex_assignee_is_spawnable_without_profile(kanban_home, monkeypatch):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="codex task", assignee="codex")
        assert kb.has_spawnable_ready(conn) is True
        res = kb.dispatch_once(conn, dry_run=True)
    assert task_id not in res.skipped_nonspawnable
    assert task_id in {item[0] for item in res.spawned}

def test_non_codex_assignee_still_routes_to_hermes_profile(kanban_home, captured_popen, monkeypatch):
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "resolve_profile_env", lambda name: str(Path.home() / ".hermes" / "profiles" / name))
    with kb.connect() as conn:
        task = _ready_task(conn, assignee="engineer")
        workspace = kb.resolve_workspace(task)

    pid = kb._default_spawn(task, str(workspace), board="default")

    assert pid == 98765
    cmd = captured_popen["cmd"]
    assert "-p" in cmd
    assert cmd[cmd.index("-p") + 1] == "engineer"
    assert cmd[-2:] == ["-q", f"work kanban task {task.id}"]


def test_codex_worker_success_blocks_for_review_with_log_and_metadata(kanban_home, tmp_path, monkeypatch):
    from hermes_cli import codex_worker

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="implement", assignee="codex")
        task = kb.claim_task(conn, task_id)
        assert task is not None

    popen_calls = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        assert cmd[0] == "/usr/bin/codex"
        assert cmd[-2:] == ["exec", "-"]
        assert "OPENAI_API_KEY" not in kwargs["env"]
        return FakeCodexProc(["working\n", "done\n"], 0)

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "status"]:
            return CompletedProcess(cmd, 0, stdout=" M app.py\n", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return CompletedProcess(cmd, 0, stdout="app.py | 1 +\n", stderr="")
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(codex_worker.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    monkeypatch.setattr(codex_worker.subprocess, "run", fake_run)
    monkeypatch.setattr(codex_worker.subprocess, "Popen", fake_popen)

    rc = codex_worker.main(["--task-id", task_id, "--workspace", str(repo), "--board", "default"])

    assert rc == 0
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "blocked"
        run = kb.list_runs(conn, task_id)[-1]
        assert run.summary.startswith("review-required:")
        assert run.metadata["codex"]["exit_code"] == 0
        assert "app.py" in run.metadata["git"]["status"]
    assert "done" in kb.read_worker_log(task_id, board="default")


def test_codex_worker_failure_blocks_with_error_metadata(kanban_home, tmp_path, monkeypatch):
    from hermes_cli import codex_worker

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="implement", assignee="codex")
        task = kb.claim_task(conn, task_id)
        assert task is not None

    def fake_popen(cmd, **kwargs):
        return FakeCodexProc(["boom"], 2)

    def fake_run(cmd, **kwargs):
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(codex_worker.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setattr(codex_worker.subprocess, "run", fake_run)
    monkeypatch.setattr(codex_worker.subprocess, "Popen", fake_popen)

    rc = codex_worker.main(["--task-id", task_id, "--workspace", str(repo), "--board", "default"])

    assert rc == 2
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "blocked"
        run = kb.list_runs(conn, task_id)[-1]
        assert run.summary.startswith("codex-failed:")
        assert run.error == "boom"
        assert run.metadata["codex"]["exit_code"] == 2

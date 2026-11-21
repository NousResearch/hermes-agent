"""Controlled Symphony E2E against a real temporary git worktree."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import projects_db, symphony


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(cwd) if cwd else None, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(symphony, "get_hermes_home", lambda: home)
    return home


def test_development_task_gets_real_isolated_worktree_policy_artifact_and_cancel(tmp_path, monkeypatch):
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    origin = tmp_path / "origin.git"
    repo = tmp_path / "repo"
    _git("init", "--bare", str(origin))
    _git("clone", str(origin), str(repo))
    _git("config", "user.email", "test@example.invalid", cwd=repo)
    _git("config", "user.name", "Symphony Test", cwd=repo)
    (repo / "README.md").write_text("seed\n", encoding="utf-8")
    _git("add", "README.md", cwd=repo)
    _git("commit", "-m", "seed", cwd=repo)
    _git("push", "origin", "HEAD", cwd=repo)

    with projects_db.connect_closing() as conn:
        project_id = projects_db.create_project(conn, name="E2E", primary_path=str(repo))

    task = symphony.create_task(
        title="E2E task", source_text="Create isolated worker result", project_id=project_id,
        telegram_chat_id="-100", telegram_thread_id="17", requester_id="42",
    )
    symphony.mark_ready(task.task_id)
    queued = symphony.enqueue(task.task_id, assignee="default")
    assert queued.status == "queued"
    assert queued.workspace_path

    spawned = []
    worker_processes = []

    def fake_worker(_kanban_task, workspace: str, _board: str):
        root = Path(workspace)
        spawned.append(root)
        assert root.is_dir()
        assert (root / "WORKFLOW.md").is_file()
        worker = subprocess.Popen(
            [
                sys.executable, "-c",
                "from pathlib import Path; import time; "
                "Path('worker-started.txt').write_text('process worker\\n'); "
                "time.sleep(600)",
            ],
            cwd=root,
        )
        worker_processes.append(worker)
        return worker.pid

    def controlled_dispatch(conn, *, board, spawn_fn, max_spawn, **_kwargs):
        kanban_task = symphony.kb.get_task(conn, queued.kanban_task_id)
        workspace = symphony.kb.resolve_workspace(kanban_task, board=board)
        spawn_fn(kanban_task, str(workspace), board=board)
        return type("Result", (), {"spawned": [queued.kanban_task_id]})()

    monkeypatch.setattr(symphony.kb, "dispatch_once", controlled_dispatch)
    dispatched = symphony.SymphonyRunner(max_concurrency=1).run_once(spawn_fn=fake_worker)
    assert dispatched == [task.task_id]
    assert len(spawned) == 1
    assert worker_processes and worker_processes[0].poll() is None
    for _ in range(50):
        if (spawned[0] / "worker-started.txt").is_file():
            break
        time.sleep(0.1)
    assert (spawned[0] / "worker-started.txt").read_text(encoding="utf-8") == "process worker\n"
    assert spawned[0] != repo
    assert spawned[0].parent.name == ".worktrees"
    assert Path(_git("rev-parse", "--show-toplevel", cwd=spawned[0]).strip()).resolve() == spawned[0].resolve()

    artifact = symphony.run_worker_checks(
        task.task_id,
        commands=[[sys.executable, "-c", "from pathlib import Path; Path('worker.txt').write_text('controlled worker\\n')"]],
    )["artifact"]
    artifact = Path(artifact)
    assert artifact.is_file()
    assert (spawned[0] / "worker.txt").read_text(encoding="utf-8") == "controlled worker\n"
    assert symphony.validate_result_artifact(task.task_id, spawned[0])["task_id"] == task.task_id
    assert symphony.latest_agent_run(task.task_id).status == "completed"
    kinds = [item["kind"] for item in reversed(symphony.list_events(task.task_id, limit=50))]
    assert "workspace.created" in kinds
    assert kinds.index("workspace.created") < kinds.index("agent.started")
    assert kinds.index("validation.started") < kinds.index("validation.finished") < kinds.index("agent.completed")
    assert "agent.finished" in kinds

    adapter = object.__new__(__import__("plugins.platforms.telegram.adapter", fromlist=["TelegramAdapter"]).TelegramAdapter)
    from unittest.mock import AsyncMock
    adapter._bot = AsyncMock()
    import asyncio
    asyncio.run(adapter._flush_symphony_delivery_events())
    assert adapter._bot.send_message.await_count >= 1
    assert int(adapter._bot.send_message.await_args.kwargs["message_thread_id"]) == 17
    cancelled = symphony.cancel(task.task_id)
    assert cancelled.status == "cancelled"
    assert worker_processes[0].wait(timeout=10) is not None

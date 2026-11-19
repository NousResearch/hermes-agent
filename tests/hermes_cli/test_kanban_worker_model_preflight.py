"""A card whose profile names no model is refused before dispatch spawns a worker; a profile the
CLI and gateway serve through the provider's catalog default still spawns (#144027).

Fixture and refusal assertions adapted from hehuang139's #144088."""

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def dispatch_profile(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch

    root = tmp_path / ".hermes"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text("model:\n  default: launch-model\n", encoding="utf-8")
    (profile / ".env").write_text("", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(root))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "normal")
    monkeypatch.setattr(dispatch, "_restart_safe_worker_argv", lambda task, command: command)
    spawned = []
    monkeypatch.setattr("subprocess.Popen", lambda command, **kw: spawned.append(command) or SimpleNamespace(pid=424242))
    kb.create_board("work")
    return profile, spawned


def _dispatch(profile, config_yaml, **task):
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch
    from hermes_cli.kanban_db_connect import connect_closing

    (profile / "config.yaml").write_text(config_yaml, encoding="utf-8")
    with connect_closing(board="work") as conn:
        task_id = kb.create_task(conn, title="needs a model", assignee="worker", board="work", **task)
        result = dispatch.dispatch_once(conn, board="work")
        return task_id, result, kb.get_task(conn, task_id), kb.list_events(conn, task_id), kb.list_runs(conn, task_id)


def test_unresolved_model_refused_before_spawn_with_task_log(dispatch_profile):
    from hermes_cli import kanban as cli

    profile, spawned = dispatch_profile
    task_id, result, task, events, runs = _dispatch(
        profile, "model:\n  provider: custom\n  base_url: https://llm.example.test/v1\n")
    assert not spawned and not result.spawned
    assert task.status == "ready" and task.worker_pid is None
    assert "No model configured for profile worker" in task.last_failure_error
    assert any(e.kind == "spawn_failed" and "hermes -p worker model" in e.payload["error"] for e in events)
    assert len(runs) == 1 and runs[0].outcome == "spawn_failed"
    assert "hermes -p worker model" in cli.run_slash(f"--board work log {task_id}")


@pytest.mark.parametrize("config_yaml, task", [
    ("model:\n  provider: openrouter\n", {}),
    ("model:\n  provider: custom\n  base_url: https://llm.example.test/v1\n", {"model_override": "task-model"}),
], ids=["catalog-default", "task-override"])
def test_resolvable_model_still_spawns(dispatch_profile, config_yaml, task):
    profile, spawned = dispatch_profile
    task_id, result, row, _, _ = _dispatch(profile, config_yaml, **task)
    assert len(spawned) == 1 and result.spawned[0][0] == task_id
    assert row.status == "running" and row.worker_pid == 424242

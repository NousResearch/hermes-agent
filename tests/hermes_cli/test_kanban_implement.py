from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _task(*, assignee="rozmilo-codex", title="Implement this"):
    with kbc.connect() as conn:
        return kb.create_task(conn, title=title, assignee=assignee)


def test_implement_gate_is_disabled_by_default(kanban_home):
    from hermes_cli.kanban_implement import implement_command_enabled

    assert implement_command_enabled() is False


def test_route_defaults_to_codex_with_independent_claude_review():
    from hermes_cli.kanban_implement import select_implementation_route

    route = select_implementation_route("rozmilo-codex")
    assert route.selected_profile == "rozmilo-codex"
    assert route.reviewer_profile == "rozmilo-claude"
    assert route.independence_valid is True


def test_route_explicit_claude_uses_codex_review():
    from hermes_cli.kanban_implement import select_implementation_route

    route = select_implementation_route("rozmilo-claude")
    assert route.selected_profile == "rozmilo-claude"
    assert route.reviewer_profile == "rozmilo-codex"
    assert route.independence_valid is True


def test_null_profile_is_not_authoritative():
    from hermes_cli.kanban_implement import select_implementation_route

    with pytest.raises(ValueError, match="profile"):
        select_implementation_route(None)


def test_disabled_command_does_not_mutate_or_dispatch(kanban_home, monkeypatch):
    from hermes_cli.kanban_implement import run_implement_slash

    task_id = _task()
    monkeypatch.setattr("hermes_cli.kanban_implement.implement_command_enabled", lambda: False)
    result = run_implement_slash(f"/implement {task_id}")
    assert result["dispatch_status"] == "disabled"
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.current_run_id is None


def test_renderer_is_compact():
    from hermes_cli.kanban_implement import render_implement_result

    text = render_implement_result({
        "command": "implement", "task_id": "t_123", "run_id": 9,
        "selected_profile": "rozmilo-codex", "reviewer_profile": "rozmilo-claude",
        "dispatch_status": "started",
    })
    assert "Implement started" in text
    assert "t_123" in text
    assert "rozmilo-claude" in text


def test_exact_id_dispatch_persists_route_before_worker(kanban_home, monkeypatch):
    from hermes_cli.kanban_implement import run_implement_slash
    from hermes_cli import kanban_db_dispatch as kbd

    task_id = _task()
    monkeypatch.setattr("hermes_cli.kanban_implement.implement_command_enabled", lambda: True)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    observed = {}

    def spawn(task, workspace, **_kwargs):
        with kbc.connect() as conn:
            observed["decision"] = kb.get_run(conn, task.current_run_id).metadata.get("routing_decision")
        return 123

    monkeypatch.setattr(kbd, "_default_spawn", spawn)
    result = run_implement_slash(f"/implement {task_id}")
    assert result["dispatch_status"] == "started"
    assert result["decision_id"]
    assert observed["decision"]["selected_profile"] == "rozmilo-codex"
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "running"
        assert len([e for e in kb.list_events(conn, task_id) if e.kind == "routing_selected"]) == 1


def test_repeated_request_does_not_create_second_run_or_decision(kanban_home, monkeypatch):
    from hermes_cli.kanban_implement import run_implement_slash
    from hermes_cli import kanban_db_dispatch as kbd

    task_id = _task()
    monkeypatch.setattr("hermes_cli.kanban_implement.implement_command_enabled", lambda: True)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    monkeypatch.setattr(kbd, "_default_spawn", lambda task, workspace, **_kwargs: 123)
    first = run_implement_slash(task_id)
    second = run_implement_slash(task_id)
    assert first["run_id"] == second["run_id"]
    assert second["dispatch_status"] == "already_running"
    with kbc.connect() as conn:
        assert len(kb.list_runs(conn, task_id)) == 1
        assert len([e for e in kb.list_events(conn, task_id) if e.kind == "routing_selected"]) == 1


def test_routing_persistence_failure_does_not_spawn(kanban_home, monkeypatch):
    from hermes_cli.kanban_implement import run_implement_slash
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_db_routing as kbr

    task_id = _task()
    monkeypatch.setattr("hermes_cli.kanban_implement.implement_command_enabled", lambda: True)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    spawned = []
    monkeypatch.setattr(kbd, "_default_spawn", lambda *args, **kwargs: spawned.append(True))
    monkeypatch.setattr(kbr, "persist_run_routing_decision", lambda *args, **kwargs: False)
    result = run_implement_slash(task_id)
    assert result["dispatch_status"] == "routing_failed"
    assert spawned == []
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.current_run_id is None


def test_explicit_claude_dispatch_uses_claude_run_profile(kanban_home, monkeypatch):
    from hermes_cli.kanban_implement import run_implement_slash
    from hermes_cli import kanban_db_dispatch as kbd

    task_id = _task(assignee="some-other-lane")
    monkeypatch.setattr("hermes_cli.kanban_implement.implement_command_enabled", lambda: True)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    monkeypatch.setattr(kbd, "_default_spawn", lambda task, workspace, **_kwargs: 123)
    result = run_implement_slash(f"/implement {task_id} --profile rozmilo-claude")
    assert result["dispatch_status"] == "started"
    assert result["selected_profile"] == "rozmilo-claude"
    assert result["reviewer_profile"] == "rozmilo-codex"
    with kbc.connect() as conn:
        assert kb.latest_run(conn, task_id).profile == "rozmilo-claude"


@pytest.mark.parametrize("status", ["blocked", "done", "archived"])
def test_ineligible_status_does_not_claim_or_spawn(kanban_home, monkeypatch, status):
    from hermes_cli.kanban_implement import run_implement_slash
    from hermes_cli import kanban_db_dispatch as kbd

    task_id = _task()
    with kbc.connect() as conn:
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
        conn.commit()
    monkeypatch.setattr("hermes_cli.kanban_implement.implement_command_enabled", lambda: True)
    monkeypatch.setattr(kbd, "_default_spawn", lambda *_args, **_kwargs: pytest.fail("spawned"))
    result = run_implement_slash(task_id)
    assert result["dispatch_status"] == "not_eligible"
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.current_run_id is None


def test_human_gate_event_is_authoritative(kanban_home, monkeypatch):
    from hermes_cli import kanban_implement as adapter
    from hermes_cli.kanban_implement import run_implement_slash

    task_id = _task()
    monkeypatch.setattr(adapter, "implement_command_enabled", lambda: True)
    with kbc.connect() as conn:
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, "human_gate_required", {"required": True})
    result = run_implement_slash(task_id)
    assert result["dispatch_status"] == "not_eligible"
    assert result["human_gate_required"] is True
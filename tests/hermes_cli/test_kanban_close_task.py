from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hermes_cli import commands
from hermes_cli import kanban_close_task as close_task
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
    monkeypatch.setattr(close_task, "close_task_command_enabled", lambda: True)
    return home


def test_close_task_gate_default_off():
    assert close_task._GATE_KEY == "close_task_command"
    assert close_task.close_task_command_enabled() is False


def test_close_task_disabled_fails_closed(kanban_home, monkeypatch):
    monkeypatch.setattr(close_task, "close_task_command_enabled", lambda: False)
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="t", assignee="a")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()
    result = close_task.run_close_task_slash(task_id)
    assert result["dispatch_status"] == "disabled"
    assert result["mutation_performed"] is False
    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).status == "ready"


def test_close_task_missing_task_fails_closed(kanban_home):
    result = close_task.run_close_task_slash("t_doesnotexist")
    assert result["dispatch_status"] == "not_eligible"
    assert result["mutation_performed"] is False


def test_close_task_ambiguous_target_fails_closed(kanban_home):
    with kbc.connect() as conn:
        kb.create_task(conn, title="duplicate name", assignee="a")
        kb.create_task(conn, title="duplicate name", assignee="a")
    result = close_task.run_close_task_slash("duplicate name")
    assert result["dispatch_status"] == "not_eligible"
    assert result["mutation_performed"] is False


def test_close_task_invalid_lifecycle_state_rejected(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="triage", assignee="a", triage=True)
        assert kb.get_task(conn, task_id).status == "triage"
    result = close_task.run_close_task_slash(task_id)
    assert result["dispatch_status"] == "not_eligible"
    assert result["closure_state"] == "not-eligible"
    assert result["mutation_performed"] is False
    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).status == "triage"


def test_close_task_successful_eligible_close(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="eligible", assignee="a")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()
    result = close_task.run_close_task_slash(f"{task_id} --result done --summary finished")
    assert result["dispatch_status"] == "closed"
    assert result["closure_state"] == "closed"
    assert result["mutation_performed"] is True
    assert result["task_status"] == "done"
    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).status == "done"


def test_close_task_already_closed_is_idempotent(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="closed", assignee="a")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()
    first = close_task.run_close_task_slash(task_id)
    assert first["dispatch_status"] == "closed"
    second = close_task.run_close_task_slash(task_id)
    assert second["dispatch_status"] == "already_closed"
    assert second["closure_state"] == "already-closed"
    assert second["mutation_performed"] is False


@pytest.mark.parametrize("status", ["archived"])
def test_close_task_archived_is_idempotent_no_mutation(kanban_home, status):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="archived", assignee="a")
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
        conn.commit()
    result = close_task.run_close_task_slash(task_id)
    assert result["dispatch_status"] == "already_closed"
    assert result["mutation_performed"] is False


def test_close_task_ownership_cas_rejection_via_unsatisfied_parent(kanban_home):
    with kbc.connect() as conn:
        parent_id = kb.create_task(conn, title="parent", assignee="a")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent_id,))
        child_id = kb.create_task(conn, title="child", assignee="a", parents=[parent_id])
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (child_id,))
        conn.commit()
    # Force child into an eligible status while its parent is unsatisfied; complete_task
    # must reject via the existing dependency/CAS check, not a new one.
    result = close_task.run_close_task_slash(child_id)
    assert result["dispatch_status"] == "not_eligible"
    assert result["closure_state"] == "rejected"
    assert result["mutation_performed"] is False
    with kbc.connect() as conn:
        assert kb.get_task(conn, child_id).status == "ready"


def test_close_task_no_merge_deploy_release_side_effect(kanban_home, monkeypatch):
    """Closing a task must never invoke git/deploy/release machinery."""
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append((a, k)) or pytest.fail("subprocess invoked"))
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="no-side-effects", assignee="a")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()
    result = close_task.run_close_task_slash(task_id)
    assert result["dispatch_status"] == "closed"
    assert calls == []


def test_close_task_deterministic_structured_output(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="deterministic", assignee="a")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()
    result = close_task.run_close_task_slash(task_id)
    expected_keys = {
        "command", "task_id", "board", "task_status", "closure_state", "action",
        "mutation_performed", "run_id", "dispatch_status", "message",
    }
    assert set(result.keys()) == expected_keys
    assert result["command"] == "close-task"


def test_close_task_concurrent_calls_close_exactly_once(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="concurrent", assignee="a")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: close_task.run_close_task_slash(task_id), range(4)))
    assert sum(item["mutation_performed"] for item in results) == 1
    assert all(item["dispatch_status"] in {"closed", "already_closed"} for item in results)
    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).status == "done"


def test_close_task_command_registry_and_gate_default():
    command = commands.resolve_command("close-task")
    assert command is not None
    assert command.name == "close-task"
    assert command.gateway_config_gate == "kanban.close_task_command"
    assert command.busy_policy == "dispatch"
    assert close_task._GATE_KEY == "close_task_command"


def test_close_task_default_config_gate_is_off():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["kanban"]["close_task_command"] is False

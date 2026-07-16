import json

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(kb, "_INITIALIZED_PATHS", set())
    return home


def _argv(extra=""):
    return (
        "program create --title 'Ship it' --assignee Planner "
        "--allowed-assignee planner --allowed-assignee worker "
        "--orchestrator planner --max-depth 2 --max-tasks 8 "
        "--max-concurrency 2 --max-runtime-seconds 60 "
        "--max-wall-clock-seconds 300 --goal-max-turns 5 --json " + extra
    )


def test_program_create_cli_emits_deterministic_safe_json_and_canonical_root(kanban_home):
    payload = json.loads(kc.run_slash(_argv()))
    assert set(payload) == {"id", "status", "orchestration_root_id", "policy"}
    assert payload["id"] == payload["orchestration_root_id"]
    assert payload["status"] == "ready"
    assert payload["policy"]["allowed_assignees"] == ["planner", "worker"]
    assert payload["policy"]["goal_max_turns"] == 5
    with kb.connect() as conn:
        tasks = kb.list_tasks(conn)
    assert len(tasks) == 1
    assert tasks[0].orchestration_depth == 0
    assert tasks[0].goal_max_turns == 5


@pytest.mark.parametrize("extra", [
    "--max-concurrency 0",
    "--orchestrator outsider",
    "--allowed-assignee planner",
    "--assignee worker",
    "--project missing-project",
])
def test_program_create_cli_invalid_input_leaves_no_root(kanban_home, extra):
    output = kc.run_slash(_argv(extra))
    assert "error" in output.lower() or "must" in output.lower() or "match" in output.lower()
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0

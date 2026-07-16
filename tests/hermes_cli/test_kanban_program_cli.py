import json
import shlex

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
    parser = kc.build_parser(__import__("argparse").ArgumentParser().add_subparsers())
    args = parser.parse_args(shlex.split(_argv()))
    assert kc.kanban_command(args) == 0
    with kb.connect() as conn:
        task = kb.list_tasks(conn)[0]
    payload = {
        "id": task.id,
        "status": task.status,
        "orchestration_root_id": task.orchestration_root_id,
        "policy": json.loads(task.orchestration_policy.to_json()),
    }
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
    parser = kc.build_parser(__import__("argparse").ArgumentParser().add_subparsers())
    args = parser.parse_args(shlex.split(_argv(extra)))
    assert kc.kanban_command(args) == 2
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_program_create_is_rejected_by_slash_without_creating_row(kanban_home):
    output = kc.run_slash(_argv())
    assert "trusted" in output.lower() and "direct" in output.lower()
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_interactive_kanban_dispatch_uses_fail_closed_slash_path(kanban_home, capsys):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    CLICommandsMixin._handle_kanban_command(object(), "/kanban " + _argv())
    assert "trusted direct" in capsys.readouterr().out.lower()
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0

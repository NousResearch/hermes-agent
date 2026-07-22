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


VALID_CGROUP = b"0::/system.slice/vigo-mc-command-0123456789abcdef01.service\n"


@pytest.fixture
def mission_control_cgroup(monkeypatch):
    monkeypatch.setattr(kb, "_read_self_cgroup", lambda: VALID_CGROUP)


def test_program_create_broker_emits_deterministic_safe_json_and_canonical_root(
    kanban_home, mission_control_cgroup,
):
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
def test_program_create_cli_invalid_input_leaves_no_root(
    kanban_home, mission_control_cgroup, extra,
):
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


def test_program_create_is_rejected_by_direct_argv_without_creating_row(
    kanban_home, capsys, monkeypatch,
):
    """A terminal-capable model must not turn direct argv into root authority."""
    parser = kc.build_parser(__import__("argparse").ArgumentParser().add_subparsers())
    args = parser.parse_args(shlex.split(_argv()))
    monkeypatch.setattr(kb, "_read_self_cgroup", lambda: b"0::/user.slice/test.service\n")
    monkeypatch.setattr(kb, "init_db", lambda *a, **k: pytest.fail("DB initialized"))
    assert kc.kanban_command(args) == 2
    assert "mission control" in capsys.readouterr().err.lower()
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_program_create_direct_argv_rejects_forged_namespace_marker(
    kanban_home, capsys, monkeypatch,
):
    parser = kc.build_parser(__import__("argparse").ArgumentParser().add_subparsers())
    args = parser.parse_args(shlex.split(_argv()))
    args.trusted = True
    args.broker = "mission-control"
    args._trusted_program_create = True
    monkeypatch.setattr(kb, "_read_self_cgroup", lambda: b"0::/user.slice/test.service\n")
    assert kc.kanban_command(args) == 2
    assert "mission control" in capsys.readouterr().err.lower()
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


@pytest.mark.parametrize("cgroup", [
    b"0::/system.slice/vigo-mc-command-0123456789abcdef0.service\n",
    b"0::/system.slice/hermes-mc-command-0123456789abcdef01.service\n",
    b"0::/system.slice/vigo-mc-command-0123456789ABCDEf01.service\n",
    b"0::/system.slice/vigo-mc-command-0123456789abcdef01.service.extra\n",
    b"0::/system.slice/../system.slice/vigo-mc-command-0123456789abcdef01.service\n",
    VALID_CGROUP + b"1:name=systemd:/system.slice/other.service\n",
    b"not:a:cgroup:line\n",
    b"",
    b"x" * (kb._CGROUP_READ_LIMIT + 1),
])
def test_program_create_rejects_non_exact_cgroup(kanban_home, monkeypatch, cgroup):
    parser = kc.build_parser(__import__("argparse").ArgumentParser().add_subparsers())
    args = parser.parse_args(shlex.split(_argv()))
    monkeypatch.setattr(kb, "_read_self_cgroup", lambda: cgroup)
    assert kc.kanban_command(args) == 2
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


@pytest.mark.parametrize("error", [PermissionError("denied"), FileNotFoundError()])
def test_program_create_rejects_unreadable_or_missing_cgroup(
    kanban_home, monkeypatch, error,
):
    parser = kc.build_parser(__import__("argparse").ArgumentParser().add_subparsers())
    args = parser.parse_args(shlex.split(_argv()))

    def unreadable():
        raise error

    monkeypatch.setattr(kb, "_read_self_cgroup", unreadable)
    assert kc.kanban_command(args) == 2
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_program_create_accepts_exact_v1_systemd_cgroup(
    kanban_home, monkeypatch,
):
    parser = kc.build_parser(__import__("argparse").ArgumentParser().add_subparsers())
    args = parser.parse_args(shlex.split(_argv()))
    monkeypatch.setattr(
        kb, "_read_self_cgroup",
        lambda: b"7:cpu,cpuacct:/ordinary\n1:name=systemd:/system.slice/"
        b"vigo-mc-command-fedcba9876543210ab.service\n",
    )
    assert kc.kanban_command(args) == 0


def test_interactive_kanban_dispatch_uses_fail_closed_slash_path(kanban_home, capsys):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    CLICommandsMixin._handle_kanban_command(object(), "/kanban " + _argv())
    assert "trusted direct" in capsys.readouterr().out.lower()
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0

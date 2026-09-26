"""Tests for the profile-scoped cross-process agent fleet command."""

from argparse import ArgumentParser
import json

from hermes_cli.subcommands.agents import _table, build_agents_parser, cmd_agents


def _args(*argv):
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_agents_parser(subparsers)
    return parser.parse_args(["agents", *argv])


def test_agents_parser_exposes_snapshot_modes():
    args = _args("--json")
    assert args.json is True
    assert args.once is False
    assert args.interval == 1.5
    assert args.func is cmd_agents


def test_top_level_parser_registers_agents_command():
    from hermes_cli.main import _build_cli_parser

    parser, _subparsers = _build_cli_parser()
    args = parser.parse_args(["agents", "--once"])
    assert args.command == "agents"
    assert args.once is True
    assert args.func is cmd_agents


def test_agents_json_emits_stable_profile_snapshot(monkeypatch, capsys, tmp_path):
    child = {
        "subagent_id": "sa-0-abcd",
        "parent_id": None,
        "depth": 0,
        "delegation_id": "deleg_abcd1234",
        "task_index": 0,
        "owner_session_id": "session-parent",
        "owner_pid": 123,
        "owner_started_at": 456,
        "goal": "Inspect parser",
        "model": "openai/gpt-5",
        "provider": "openai",
        "status": "running",
        "started_at": 1000.0,
        "updated_at": 1001.0,
        "tool_count": 2,
        "last_tool": "read_file",
        "transcript": str(tmp_path / "task-0.log"),
    }
    monkeypatch.setattr(
        "tools.delegation_live_log.scan_live_delegations", lambda: [child]
    )
    monkeypatch.setattr(
        "hermes_cli.subcommands.agents.get_hermes_home", lambda: tmp_path
    )
    monkeypatch.setattr("hermes_cli.subcommands.agents.time.time", lambda: 2000.0)

    assert cmd_agents(_args("--json")) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload == {
        "schema_version": 1,
        "observed_at": 2000.0,
        "profile_home": str(tmp_path),
        "children": [child],
    }


def test_agents_once_reports_empty_fleet(monkeypatch, capsys):
    monkeypatch.setattr(
        "tools.delegation_live_log.scan_live_delegations", lambda: []
    )

    assert cmd_agents(_args("--once")) == 0
    assert "No live delegated agents" in capsys.readouterr().out


def test_agents_rejects_non_finite_interval():
    import pytest

    with pytest.raises(SystemExit):
        _args("--interval", "nan")


def test_agents_table_renders_manifest_text_literally():
    from io import StringIO
    from rich.console import Console

    snapshot = {
        "observed_at": 2.0,
        "children": [{
            "owner_session_id": "[session]", "owner_pid": 1,
            "delegation_id": "deleg", "task_index": 0,
            "status": "running", "updated_at": "-inf",
            "model": "[model]", "last_tool": "[tool]",
            "goal": "[notatag] hello",
        }],
    }
    output = StringIO()
    Console(file=output, width=160, color_system=None).print(_table(snapshot))
    rendered = output.getvalue()
    assert "[notatag] hello" in rendered
    assert "[session]" in rendered

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import mission


def test_build_preview_has_ack_watchdog_and_no_side_effects():
    spec = {
        "mission": {
            "name": "overnight mission wrapper dry-run",
            "origin": "discord:#hermes-main",
            "return_to": "discord:#hermes-main",
        },
        "graph": [
            {"id": "impl", "title": "Implement dry-run", "assignee": "ccsupervisor"},
            {"id": "review", "title": "Review dry-run", "assignee": "ccreviewer", "depends_on": ["impl"]},
        ],
        "ack": {"final_task": "review"},
        "watchdog": {"schedule": "every 15m"},
    }

    preview = mission.build_preview(spec)

    assert preview["status"] == "mission_dry_run_preview"
    assert preview["sent"] is False
    assert preview["created"] is False
    assert preview["live_dispatch"] is False
    assert preview["safety"] == {
        "dry_run_only": True,
        "would_send": False,
        "would_write_kanban": False,
        "would_create_cron": False,
        "would_trigger_agent": False,
    }
    assert [task["id"] for task in preview["would_create_tasks"]] == ["impl", "review"]
    assert preview["would_subscribe_final_ack"] == {
        "enabled": True,
        "origin": "discord:#hermes-main",
        "return_to": "discord:#hermes-main",
        "final_task": "review",
        "verdict_schema": ["GO", "BLOCK", "NEED_MORE"],
    }
    assert preview["would_create_watchdog"] == {
        "enabled": True,
        "schedule": "every 15m",
        "material_change_only": True,
        "deliver": "discord:#hermes-main",
    }


def test_create_requires_dry_run_before_reading_file(capsys):
    args = argparse.Namespace(dry_run=False, file="/path/that/should/not/be/read.yml", json=True)

    rc = mission.mission_create(args)

    assert rc == 2
    assert "pass --dry-run" in capsys.readouterr().err


def test_create_dry_run_json_from_yaml_file(tmp_path, capsys):
    spec = tmp_path / "mission.yml"
    spec.write_text(
        """
mission:
  name: Agent OS MISSION-M0
  objective: overnight mission wrapper dry-run
  origin: discord:#hermes-main
  return_to: discord:#hermes-mission-control
graph:
  - id: create
    title: hermes mission create --dry-run
    assignee: ccsupervisor
  - id: review
    title: review preview contract
    assignee: ccreviewer
    depends_on: create
watchdog:
  schedule: every 15m
""".strip(),
        encoding="utf-8",
    )
    args = argparse.Namespace(dry_run=True, file=str(spec), json=True)

    rc = mission.mission_create(args)

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mission"]["name"] == "Agent OS MISSION-M0"
    assert out["mission"]["return_to"] == "discord:#hermes-mission-control"
    assert out["would_create_tasks"][1]["depends_on"] == ["create"]
    assert out["would_create_watchdog"]["enabled"] is True
    assert out["safety"]["would_create_cron"] is False


def test_malformed_yaml_error_is_sanitized(tmp_path, capsys):
    spec = tmp_path / "secret-mission.yml"
    spec.write_text(
        "graph:\n  - title: ok\nsecret_token: sk-live-should-not-echo: [",
        encoding="utf-8",
    )
    args = argparse.Namespace(dry_run=True, file=str(spec), json=True)

    rc = mission.mission_create(args)

    err = capsys.readouterr().err
    assert rc == 1
    assert "mission spec is not valid JSON/YAML" in err
    assert str(spec) not in err
    assert "sk-live-should-not-echo" not in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "content,expected",
    [
        ('["not", "a", "mapping"]', "mission spec must be a JSON/YAML object"),
        ('{"graph": {"title": "not a list"}}', "graph/tasks must be a list"),
        ('{"graph": []}', "mission spec must include at least one graph/task item"),
    ],
)
def test_invalid_schema_reports_validation_error(tmp_path, capsys, content, expected):
    spec = tmp_path / "mission.json"
    spec.write_text(content, encoding="utf-8")
    args = argparse.Namespace(dry_run=True, file=str(spec), json=True)

    rc = mission.mission_create(args)

    err = capsys.readouterr().err
    assert rc == 1
    assert expected in err
    assert "Traceback" not in err


def test_missing_file_error_is_sanitized(tmp_path, capsys):
    missing = tmp_path / "private" / "mission.yml"
    args = argparse.Namespace(dry_run=True, file=str(missing), json=True)

    rc = mission.mission_create(args)

    err = capsys.readouterr().err
    assert rc == 1
    assert "mission spec file could not be read" in err
    assert str(missing) not in err
    assert "Traceback" not in err


def test_build_parser_wires_create_command():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    mission.build_parser(subparsers)

    args = parser.parse_args(["mission", "create", "--dry-run", "--json"])

    assert args.command == "mission"
    assert args.mission_command == "create"
    assert args.dry_run is True
    assert args.json is True
    assert args.func is mission.mission_command


def test_valid_preview_json_is_deterministic(tmp_path, capsys):
    spec = tmp_path / "mission.json"
    spec.write_text(
        json.dumps(
            {
                "graph": [{"title": "review preview", "assignee": "ccreviewer"}],
                "return_to": "discord:#ops",
                "name": "M0",
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(dry_run=True, file=str(spec), json=True)

    first_rc = mission.mission_create(args)
    first = capsys.readouterr().out
    second_rc = mission.mission_create(args)
    second = capsys.readouterr().out

    assert first_rc == second_rc == 0
    assert first == second
    assert json.loads(first)["safety"]["would_trigger_agent"] is False


def test_depends_on_entries_must_be_stringish():
    spec = {
        "graph": [
            {"id": "one", "title": "one"},
            {"id": "two", "title": "two", "depends_on": [{"nested": "not allowed"}]},
        ]
    }

    with pytest.raises(mission.MissionSpecError, match="depends_on entries"):
        mission.build_preview(spec)


def test_top_level_cli_missing_dry_run_exits_nonzero(tmp_path):
    spec = tmp_path / "mission.json"
    spec.write_text('{"graph": ["preview only"]}', encoding="utf-8")
    env = {**os.environ, "HERMES_HOME": str(tmp_path / "hermes-home")}

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "mission",
            "create",
            "--file",
            str(spec),
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "pass --dry-run" in result.stderr


def test_top_level_cli_valid_preview_has_no_side_effect_files(tmp_path):
    spec = tmp_path / "mission.json"
    spec.write_text('{"graph": ["preview only"], "return_to": "discord:#ops"}', encoding="utf-8")
    home = tmp_path / "hermes-home"
    env = {**os.environ, "HERMES_HOME": str(home)}

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "mission",
            "create",
            "--dry-run",
            "--file",
            str(spec),
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    out = json.loads(result.stdout)
    assert out["created"] is False
    assert out["sent"] is False
    assert not (home / "kanban.db").exists()
    assert not (home / "cron").exists()
    assert not (home / "webhooks").exists()


def test_mission_is_not_shared_slash_command():
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

    assert resolve_command("mission") is None
    assert "mission" not in GATEWAY_KNOWN_COMMANDS


def test_mission_top_level_subcommand_skips_plugin_discovery_fast_path():
    from unittest.mock import patch

    from hermes_cli.main import _BUILTIN_SUBCOMMANDS, _plugin_cli_discovery_needed

    assert "mission" in _BUILTIN_SUBCOMMANDS
    with patch.object(sys, "argv", ["hermes", "mission", "create", "--dry-run"]):
        assert _plugin_cli_discovery_needed() is False

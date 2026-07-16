from __future__ import annotations

import json
import argparse
from pathlib import Path

import pytest

from hermes_cli.subcommands.conductor import build_conductor_parser, cmd_conductor


def _definition(tmp_path: Path) -> Path:
    path = tmp_path / "campaign.json"
    path.write_text(
        json.dumps({
            "campaign_id": "cli-campaign",
            "cwd": str(tmp_path),
            "mutable_manifest": [],
            "steps": [
                {"step_id": "observe", "kind": "observation", "command": ["true"]}
            ],
        })
    )
    return path


def test_conductor_parser_registers_narrow_definition_surface():
    parser = argparse.ArgumentParser()
    build_conductor_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["conductor", "campaign.json"])
    assert args.command == "conductor"
    assert args.definition == "campaign.json"
    assert args.func is cmd_conductor


def test_conductor_command_requires_opt_in(tmp_path, monkeypatch, capsys):
    definition = _definition(tmp_path)
    monkeypatch.setattr(
        "hermes_cli.subcommands.conductor.load_config",
        lambda: {"conductor": {"enabled": False}},
    )
    with pytest.raises(SystemExit, match="conductor.enabled"):
        cmd_conductor(type("Args", (), {"definition": str(definition)})())


def test_conductor_command_runs_exactly_one_tick_and_prints_json(
    tmp_path, monkeypatch, capsys
):
    definition = _definition(tmp_path)
    config = {
        "conductor": {
            "enabled": True,
            "state_path": "conductor/test.sqlite",
            "tick_lease_seconds": 15,
            "writer": {"command": []},
            "reviewer": {"command": []},
        }
    }
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("hermes_cli.subcommands.conductor.load_config", lambda: config)
    assert cmd_conductor(type("Args", (), {"definition": str(definition)})()) == 0
    status = json.loads(capsys.readouterr().out)
    assert status == {
        "campaign_id": "cli-campaign",
        "conductor_turns": 1,
        "result": "OBSERVED_SILENT",
        "state": "READY",
        "step_index": 1,
    }


def test_conductor_definition_rejects_unbounded_or_unknown_surface(
    tmp_path, monkeypatch
):
    definition = _definition(tmp_path)
    value = json.loads(definition.read_text())
    value["budgets"] = {"max_runs_per_day": 999999}
    definition.write_text(json.dumps(value))
    monkeypatch.setattr(
        "hermes_cli.subcommands.conductor.load_config",
        lambda: {"conductor": {"enabled": True}},
    )
    with pytest.raises(SystemExit, match="unsupported field"):
        cmd_conductor(type("Args", (), {"definition": str(definition)})())

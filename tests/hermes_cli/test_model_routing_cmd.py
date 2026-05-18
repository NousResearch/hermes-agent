"""Tests for the ``hermes routing preview`` dry-run command."""

from __future__ import annotations

import argparse
import json

import pytest


def _parse(argv: list[str]) -> argparse.Namespace:
    from hermes_cli.model_routing_cmd import register_routing_subparser

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    register_routing_subparser(subparsers)
    return parser.parse_args(argv)


def test_registers_routing_preview_subcommand():
    args = _parse(
        [
            "routing",
            "preview",
            "--task-type",
            "daily_ops",
            "--risk-level",
            "low",
        ]
    )

    assert args.command == "routing"
    assert args.routing_command == "preview"
    assert args.task_type == "daily_ops"
    assert callable(args.func)


def test_preview_json_emits_dry_run_decision_without_config_mutation(capsys):
    from hermes_cli.model_routing_cmd import cmd_routing

    args = _parse(
        [
            "routing",
            "preview",
            "--task-type",
            "client_facing_draft",
            "--agent-role",
            "cmo",
            "--risk-level",
            "high",
            "--client-facing",
            "--final-authority",
            "--json",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        cmd_routing(args)

    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["provider"] == "openrouter"
    assert payload["tier"] == "S"
    assert payload["approval_required"] is True
    assert "would_change_runtime" not in payload


def test_preview_human_output_calls_out_dry_run_and_approval(capsys):
    from hermes_cli.model_routing_cmd import cmd_routing

    args = _parse(
        [
            "routing",
            "preview",
            "--task-type",
            "security_privacy_review",
            "--risk-level",
            "high",
            "--sensitive-data",
            "--final-authority",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        cmd_routing(args)

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Routing preview" in out
    assert "DRY RUN" in out
    assert "Approval required: yes" in out
    assert "Runtime config changed: no" in out
    assert "openrouter" in out


def test_preview_defaults_are_safe_for_unknown_tasks(capsys):
    from hermes_cli.model_routing_cmd import cmd_routing

    args = _parse(
        [
            "routing",
            "preview",
            "--task-type",
            "brand_new_workflow",
            "--json",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        cmd_routing(args)

    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tier"] == "B"
    assert payload["model"] == "deepseek/deepseek-v4-flash"
    assert payload["policy_warnings"] == ["Unknown task type: brand_new_workflow"]


def test_routing_command_without_subcommand_is_usage_error(capsys):
    from hermes_cli.model_routing_cmd import cmd_routing

    args = _parse(["routing"])

    with pytest.raises(SystemExit) as exc:
        cmd_routing(args)

    assert exc.value.code == 2
    assert "hermes routing: subcommand required" in capsys.readouterr().err

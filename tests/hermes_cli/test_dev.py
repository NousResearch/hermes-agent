from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import dev


def _parse_dev_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    dev.build_parser(subparsers)
    return parser.parse_args(["dev", *argv])


def test_dump_parse_reports_counts_and_tokens(tmp_path: Path) -> None:
    dump = tmp_path / "context.message.txt"
    dump.write_text(
        "\n".join(
            [
                "# Hermes Raw Context Dump",
                "",
                "## Raw API Messages",
                "",
                json.dumps(
                    [
                        {"role": "system", "content": "SOUL"},
                        {"role": "user", "content": "hi"},
                    ]
                ),
                "",
                "## Tool Schemas",
                "",
                json.dumps(
                    [{"type": "function", "function": {"name": "chat_startup_context"}}]
                ),
                "",
                "## Debug Metadata",
                "",
                json.dumps(
                    {"schema": "hermes.context_dump.v2", "session_key": "discord:1"}
                ),
            ]
        ),
        encoding="utf-8",
    )

    result = dev.parse_context_dump(dump)

    assert result.raw_message_count == 2
    assert result.tool_schema_count == 1
    assert result.rough_total_tokens > 0
    assert result.missing_required_sections == ()


def test_dump_parse_flags_missing_required_sections(tmp_path: Path) -> None:
    dump = tmp_path / "broken.message.txt"
    dump.write_text(
        "\n".join(
            [
                "# Hermes Raw Context Dump",
                "",
                "## Raw API Messages",
                "",
                "[]",
                "",
                "## Tool Schemas",
                "",
                "[]",
                "",
                "## Debug Metadata",
                "",
                "{}",
            ]
        ),
        encoding="utf-8",
    )

    result = dev.parse_context_dump(dump)

    assert "raw_api_messages" in result.missing_required_sections
    assert "metadata.schema" in result.missing_required_sections
    assert "metadata.session_key" in result.missing_required_sections


@pytest.mark.parametrize("command", ["sync", "restart", "logs", "smoke", "diff-odin"])
def test_odin_commands_require_explicit_odin_flag(command: str) -> None:
    args = _parse_dev_args(
        [command, "--to", "odin"] if command == "sync" else [command]
    )

    with pytest.raises(SystemExit, match="requires --odin"):
        dev.cmd_dev(args)


def test_sync_plan_uses_rsync_and_required_excludes() -> None:
    args = _parse_dev_args(["sync", "--to", "odin", "--odin", "--dry-run"])

    plans = dev.build_plans(args)

    assert len(plans) == 1
    rendered = plans[0].render()
    assert "rsync" in rendered
    assert "--dry-run" in plans[0].argv
    for pattern in ("venv", ".git", "*.bak.*", ".env", "node_modules"):
        assert pattern in plans[0].argv


def test_verify_dry_run_plan_orders_local_checks_before_odin() -> None:
    args = _parse_dev_args(["verify", "--odin", "--dry-run"])

    labels = [plan.label for plan in dev.build_plans(args)]

    assert labels[:6] == [
        "ruff check",
        "ruff gateway run undefined names",
        "compile gateway runtime",
        "ruff format check",
        "mypy",
        "vulture advisory",
    ]
    assert "focused pytest" in labels
    assert labels.index("rsync to odin") > labels.index("focused pytest")
    assert labels.index("restart gateway") > labels.index("rsync to odin")
    assert labels.index("discord smoke") > labels.index("restart gateway")
    assert labels.index("gateway logs") > labels.index("discord smoke")


def test_smoke_plan_uses_odin_venv_python() -> None:
    args = _parse_dev_args(["smoke", "--odin", "--dry-run"])

    plan = dev.build_plans(args)[0]

    assert dev.ODIN_PYTHON in plan.render()
    assert "--mode auto" in plan.render()
    assert "&& python -m gateway.validation.discord_context_smoke" not in plan.render()


def test_smoke_plan_can_request_live_mode() -> None:
    args = _parse_dev_args(["smoke", "--odin", "--dry-run", "--mode", "live"])

    plan = dev.build_plans(args)[0]

    assert "--mode live" in plan.render()


def test_odin_plan_runs_locally_when_already_on_odin(monkeypatch) -> None:
    monkeypatch.setattr(dev, "_running_on_odin", lambda: True)
    args = _parse_dev_args(["smoke", "--odin", "--dry-run"])

    plan = dev.build_plans(args)[0]

    assert plan.argv[0] == "bash"
    assert "ssh odin" not in plan.render()


def test_dev_parser_registers_expected_subcommands() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    dev_parser = dev.build_parser(subparsers)
    action = next(
        action
        for action in dev_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert {
        "status",
        "sync",
        "diff-odin",
        "lint",
        "typecheck",
        "dead-code",
        "test",
        "restart",
        "logs",
        "smoke",
        "dump-parse",
        "verify",
    }.issubset(action.choices)

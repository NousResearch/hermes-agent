"""Unit tests for the extracted ``hermes cron`` parser builder.

Confirms ``build_cron_parser`` wires up the same subactions, aliases, options,
and ``func=cmd_cron`` dispatch that lived inline in ``main()`` before the
god-file Phase 2 extraction.
"""

from __future__ import annotations

import argparse

import pytest

from hermes_cli.subcommands.cron import build_cron_parser


def _sentinel_handler(args):  # pragma: no cover - only identity is asserted
    return "cron-handler"


def _build():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_cron_parser(subparsers, cmd_cron=_sentinel_handler)
    return parser


def test_cron_subactions_present():
    parser = _build()
    for action in ("list", "create", "edit", "pause", "resume", "run", "remove", "status", "runs", "doctor", "tick"):
        ns = parser.parse_args(["cron", action] if action in ("list", "status", "runs", "doctor", "tick")
                               else ["cron", action, "jobid"] if action in ("pause", "resume", "run", "remove", "edit")
                               else ["cron", "create", "30m"])
        assert ns.command == "cron"
        assert ns.cron_command == action


def test_cron_edit_no_agent_tristate():
    parser = _build()
    # --no-agent -> True, --agent -> False, neither -> None
    assert parser.parse_args(["cron", "edit", "j", "--no-agent"]).no_agent is True
    assert parser.parse_args(["cron", "edit", "j", "--agent"]).no_agent is False
    assert parser.parse_args(["cron", "edit", "j"]).no_agent is None


def test_cron_accept_hooks_flag_on_run_and_tick():
    parser = _build()
    # --accept-hooks is suppressed-default; present only when passed.
    ns = parser.parse_args(["cron", "run", "jid", "--accept-hooks"])
    assert ns.accept_hooks is True
    ns2 = parser.parse_args(["cron", "tick", "--accept-hooks"])
    assert ns2.accept_hooks is True


@pytest.mark.parametrize(
    "argv, expected",
    [
        pytest.param(
            ["cron", "create", "0 3 * * *", "--name", "test-prompt-job",
             "--deliver", "local", "Test prompt to see if positionals work"],
            {"command": "cron", "cron_command": "create", "schedule": "0 3 * * *",
             "name": "test-prompt-job", "deliver": "local",
             "prompt": "Test prompt to see if positionals work"},
            id="issue-multiword-prompt",
        ),
        pytest.param(
            ["cron", "create", "0 3 * * *", "--name", "test-prompt-job",
             "Single-word-prompt"],
            {"prompt": "Single-word-prompt", "name": "test-prompt-job"},
            id="issue-single-word-prompt",
        ),
        pytest.param(
            ["cron", "add", "30m", "--name", "job", "alias prompt"],
            {"cron_command": "add", "prompt": "alias prompt", "name": "job"},
            id="create-alias",
        ),
        pytest.param(
            ["cron", "create", "30m", "contiguous prompt", "--deliver", "local"],
            {"prompt": "contiguous prompt", "deliver": "local"},
            id="contiguous-positionals",
        ),
        pytest.param(
            ["cron", "create", "30m", "--", "--literal-prompt"],
            {"prompt": "--literal-prompt"}, id="end-of-options",
        ),
        pytest.param(
            ["cron", "create", "--name", "job", "30m", "--skill", "first",
             "prompt", "--skill", "second"],
            {"schedule": "30m", "prompt": "prompt", "name": "job",
             "skills": ["first", "second"]},
            id="options-around-positionals",
        ),
        pytest.param(
            ["cron", "create", "0 3 * * *", "--name", "test-prompt-job",
             "--deliver", "local", "--no-agent", "--script", "x.sh"],
            {"prompt": None, "no_agent": True, "script": "x.sh", "deliver": "local"},
            id="script-without-prompt",
        ),
        pytest.param(
            ["cron", "list", "--all"],
            {"command": "cron", "cron_command": "list", "all": True},
            id="flags-only-cron-action",
        ),
        pytest.param(
            ["-c", "multi word session"],
            {"command": None, "continue_last": "multi word session"},
            id="quoted-session-name",
        ),
        pytest.param(
            ["-c", "multi", "word", "session"],
            {"command": None, "continue_last": "multi word session"},
            id="coalesced-session-name",
        ),
        pytest.param(
            ["-c", "model"],
            {"command": None, "continue_last": "model"},
            id="session-name-matching-command",
        ),
        pytest.param(
            ["model"], {"command": "model"}, id="other-subcommand",
        ),
        pytest.param(
            ["--version"], {"command": None, "version": True}, id="top-level-flag",
        ),
    ],
)
def test_cron_and_session_args_through_top_level_routing(argv, expected):
    from hermes_cli.main import _build_cli_parser, _parse_cli_args

    parser, subparsers = _build_cli_parser()
    args = _parse_cli_args(parser, subparsers, argv)

    for name, value in expected.items():
        assert getattr(args, name) == value


@pytest.mark.parametrize(
    "argv, exit_code",
    [
        (["30m", "prompt", "extra"], 2),
        (["30m", "--unknown", "prompt"], 2),
        (["30m", "--name"], 2),
        (["--name", "job"], 2),
        (["--help"], 0),
    ],
)
def test_cron_parser_remains_usable_after_exit(argv, exit_code):
    from hermes_cli.main import _build_cli_parser, _parse_cli_args

    parser, subparsers = _build_cli_parser()
    with pytest.raises(SystemExit) as exc:
        _parse_cli_args(parser, subparsers, ["cron", "create", *argv])
    assert exc.value.code == exit_code

    args = _parse_cli_args(
        parser, subparsers, ["cron", "create", "30m", "--name", "job", "prompt"])
    assert args.prompt == "prompt"
    assert args.name == "job"

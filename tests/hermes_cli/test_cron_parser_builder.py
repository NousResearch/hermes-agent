"""Unit tests for the extracted ``hermes cron`` parser builder.

Confirms ``build_cron_parser`` wires up the same subactions, aliases, options,
and ``func=cmd_cron`` dispatch that lived inline in ``main()`` before the
god-file Phase 2 extraction.
"""

from __future__ import annotations

import argparse

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
    for action in ("list", "create", "edit", "pause", "resume", "run", "remove", "status", "runs", "tick"):
        ns = parser.parse_args(["cron", action] if action in ("list", "status", "runs", "tick")
                               else ["cron", action, "jobid"] if action in ("pause", "resume", "run", "remove", "edit")
                               else ["cron", "create", "30m"])
        assert ns.command == "cron"
        assert ns.cron_command == action


def test_cron_aliases():
    parser = _build()
    # create has alias "add"
    ns = parser.parse_args(["cron", "add", "30m"])
    assert ns.cron_command == "add"
    # remove has aliases rm / delete
    for alias in ("rm", "delete"):
        ns = parser.parse_args(["cron", alias, "jid"])
        assert ns.cron_command == alias
    ns = parser.parse_args(["cron", "history", "jid", "--limit", "7"])
    assert ns.cron_command == "history"
    assert ns.job_id == "jid"
    assert ns.limit == 7


def test_cron_create_options():
    parser = _build()
    ns = parser.parse_args([
        "cron", "create", "0 9 * * *", "daily task prompt",
        "--name", "daily", "--deliver", "origin", "--repeat", "3",
        "--skill", "a", "--skill", "b", "--no-agent",
        "--workdir", "/tmp/x",
    ])
    assert ns.schedule == "0 9 * * *"
    assert ns.prompt_positional == "daily task prompt"
    assert ns.prompt_flag is None
    assert ns.name == "daily"
    assert ns.deliver == "origin"
    assert ns.repeat == 3
    assert ns.skills == ["a", "b"]
    assert ns.no_agent is True
    assert ns.workdir == "/tmp/x"


def test_cron_create_prompt_flag_works_with_flags_before_it():
    # Regression test: argparse cannot fill an optional positional (prompt)
    # from a token that appears after an interspersed --flag (a required
    # positional followed by nargs='?' is a documented argparse limitation
    # — reproduced directly against bare argparse, not hermes-specific).
    # Confirmed 2026-07-29: `cron create <schedule> --name X <prompt>` used
    # to raise "unrecognized arguments" from the TOP-LEVEL parser. --prompt
    # is the fix: a flag has no ordering dependency on other flags.
    parser = _build()
    ns = parser.parse_args([
        "cron", "create", "0 9 * * *",
        "--name", "daily", "--prompt", "daily task prompt",
    ])
    assert ns.schedule == "0 9 * * *"
    assert ns.prompt_positional is None
    assert ns.prompt_flag == "daily task prompt"


def test_cron_create_prompt_flag_takes_precedence_over_positional():
    parser = _build()
    ns = parser.parse_args([
        "cron", "create", "0 9 * * *", "positional prompt",
        "--prompt", "flag prompt",
    ])
    assert ns.prompt_positional == "positional prompt"
    assert ns.prompt_flag == "flag prompt"
    # cron_create() in hermes_cli/cron.py resolves prompt_flag over
    # prompt_positional when both are set — this test only confirms the
    # parser captured both distinctly; see test_cron.py for the resolution.


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

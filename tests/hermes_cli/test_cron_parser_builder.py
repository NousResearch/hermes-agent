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
    for action in ("list", "create", "edit", "pause", "resume", "run", "remove", "status", "tick"):
        ns = parser.parse_args(["cron", action] if action in ("list", "status", "tick")
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


def test_cron_create_options():
    parser = _build()
    ns = parser.parse_args([
        "cron", "create", "0 9 * * *", "daily task prompt",
        "--name", "daily", "--deliver", "origin", "--repeat", "3",
        "--skill", "a", "--skill", "b", "--no-agent",
        "--workdir", "/tmp/x",
    ])
    assert ns.schedule == "0 9 * * *"
    assert ns.prompt == "daily task prompt"
    assert ns.name == "daily"
    assert ns.deliver == "origin"
    assert ns.repeat == 3
    assert ns.skills == ["a", "b"]
    assert ns.no_agent is True
    assert ns.workdir == "/tmp/x"


def test_cron_create_accepts_profile_flag():
    """``cron create`` must accept ``--profile``."""
    parser = _build()
    ns = parser.parse_args(["cron", "create", "every 1h", "task", "--profile", "trading"])
    assert ns.profile == "trading"

    # Empty string clears profile
    ns = parser.parse_args(["cron", "create", "every 1h", "task", "--profile", ""])
    assert ns.profile == ""


def test_cron_edit_options():
    """``cron edit`` must accept all standard flags plus ``--profile``."""
    parser = _build()
    ns = parser.parse_args([
        "cron", "edit", "abc123",
        "--schedule", "every 2h",
        "--prompt", "new prompt",
        "--name", "renamed",
        "--deliver", "origin",
        "--repeat", "5",
        "--skill", "a", "--skill", "b",
        "--add-skill", "c",
        "--remove-skill", "d",
        "--clear-skills",
        "--script", "watchdog.sh",
        "--workdir", "/tmp/proj",
    ])
    assert ns.job_id == "abc123"
    assert ns.schedule == "every 2h"
    assert ns.prompt == "new prompt"
    assert ns.name == "renamed"
    assert ns.deliver == "origin"
    assert ns.repeat == 5
    assert ns.skills == ["a", "b"]
    assert ns.add_skills == ["c"]
    assert ns.remove_skills == ["d"]
    assert ns.clear_skills is True
    assert ns.script == "watchdog.sh"
    assert ns.workdir == "/tmp/proj"


def test_cron_edit_accepts_profile_flag():
    """``cron edit`` must accept ``--profile <name>`` as a job field, not a context switch."""
    parser = _build()
    ns = parser.parse_args(["cron", "edit", "abc123", "--profile", "trading"])
    assert ns.profile == "trading"

    # Empty string clears profile
    ns = parser.parse_args(["cron", "edit", "abc123", "--profile", ""])
    assert ns.profile == ""


def test_cron_edit_no_agent_tristate():
    parser = _build()
    # --no-agent -> True, --agent -> False, neither -> None
    assert parser.parse_args(["cron", "edit", "j", "--no-agent"]).no_agent is True
    assert parser.parse_args(["cron", "edit", "j", "--agent"]).no_agent is False
    assert parser.parse_args(["cron", "edit", "j"]).no_agent is None


def test_cron_dispatch_func_is_injected_handler():
    parser = _build()
    ns = parser.parse_args(["cron", "list"])
    assert ns.func is _sentinel_handler


def test_cron_accept_hooks_flag_on_run_and_tick():
    parser = _build()

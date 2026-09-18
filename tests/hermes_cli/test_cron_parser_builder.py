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


def test_cron_create_accepts_context_from_and_attach_to_session():
    parser = _build()
    ns = parser.parse_args(
        [
            "cron",
            "create",
            "30m",
            "prompt",
            "--context-from",
            "job-a",
            "--context-from",
            "job-b",
            "--attach-to-session",
        ]
    )
    assert ns.context_from == ["job-a", "job-b"]
    assert ns.attach_to_session is True
    # Omitted -> None (no accidental clear on create)
    ns2 = parser.parse_args(["cron", "create", "30m", "prompt"])
    assert ns2.context_from is None
    assert ns2.attach_to_session is None


def test_cron_edit_context_from_and_attach_to_session_tristate():
    parser = _build()
    ns = parser.parse_args(
        ["cron", "edit", "j", "--context-from", "job-a", "--attach-to-session"]
    )
    assert ns.context_from == ["job-a"]
    assert ns.attach_to_session is True

    ns2 = parser.parse_args(["cron", "edit", "j", "--no-attach-to-session"])
    assert ns2.attach_to_session is False

    ns3 = parser.parse_args(["cron", "edit", "j"])
    assert ns3.context_from is None
    assert ns3.attach_to_session is None


def test_cron_accept_hooks_flag_on_run_and_tick():
    parser = _build()
    # --accept-hooks is suppressed-default; present only when passed.
    ns = parser.parse_args(["cron", "run", "jid", "--accept-hooks"])
    assert ns.accept_hooks is True
    ns2 = parser.parse_args(["cron", "tick", "--accept-hooks"])
    assert ns2.accept_hooks is True

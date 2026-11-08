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


def test_cron_timezone_and_grace_flags():
    parser = _build()
    created = parser.parse_args(
        ["cron", "create", "0 9 * * *", "hi", "--timezone", "Asia/Hong_Kong",
         "--grace-minutes", "0"])
    assert created.timezone == "Asia/Hong_Kong"
    assert created.grace_minutes == "0", "an int-ish string must survive to the normalizer"

    edited = parser.parse_args(["cron", "edit", "j", "--timezone", "", "--grace-minutes", ""])
    assert edited.timezone == "", "an empty value is the documented clear"
    assert edited.grace_minutes == ""

    assert parser.parse_args(["cron", "edit", "j"]).timezone is None


def test_cron_repeat_accepts_user_facing_strings():
    parser = _build()
    # argparse used to hardcode type=int, so these user-facing strings (and the
    # tool/API path that already accepts them) hit "invalid int value".
    assert parser.parse_args(["cron", "create", "in 1h", "x", "--repeat", "forever"]).repeat is None
    assert parser.parse_args(["cron", "create", "in 1h", "x", "--repeat", "once"]).repeat == 1
    assert parser.parse_args(["cron", "create", "in 1h", "x", "--repeat", "3"]).repeat == 3
    assert parser.parse_args(["cron", "edit", "j", "--repeat", "once"]).repeat == 1

    with pytest.raises(SystemExit):
        parser.parse_args(["cron", "create", "in 1h", "x", "--repeat", "bogus"])

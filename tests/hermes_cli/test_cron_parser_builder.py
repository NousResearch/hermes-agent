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


def test_cron_script_help_explains_the_active_profile_home():
    """Every cron script option must explain the profile home, even in cached help."""
    parser = _build()
    cron = parser._subparsers._group_actions[0].choices["cron"]
    commands = cron._subparsers._group_actions[0].choices
    for command, flag in (("create", "--script"), ("create", "--monitor-script"),
                          ("edit", "--script")):
        action = commands[command]._option_string_actions[flag]
        help_text = action.help
        assert "$HERMES_HOME/scripts/" in help_text
        assert "active profile" in help_text
        assert "-p <name>" in help_text


def test_webhook_script_help_distinguishes_route_and_gateway_profiles():
    """The subscription owner and script owner can differ on multiplexed gateways."""
    from hermes_cli.subcommands.webhook import build_webhook_parser

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_webhook_parser(subparsers, cmd_webhook=_sentinel_handler)
    webhook = subparsers.choices["webhook"]
    commands = next(action for action in webhook._actions
                    if isinstance(action, argparse._SubParsersAction))
    subscribe = commands.choices["subscribe"]
    action = subscribe._option_string_actions["--script"]
    assert "$HERMES_HOME/scripts/" in action.help
    assert "--route-profile" in action.help
    assert "bare URLs" in action.help and "gateway's own profile" in action.help
    assert "-p <name>" not in action.help
    rendered = " ".join(subscribe.format_help().split())
    assert " ".join(action.help.split()) in rendered

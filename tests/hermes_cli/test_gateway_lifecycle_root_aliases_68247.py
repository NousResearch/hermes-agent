"""Regression (#68247): top-level ``hermes start|stop|restart`` aliases.

Users (and agent-generated scripts) often expect supervisor-style verbs at the
root CLI. Historically only ``hermes gateway start|stop`` worked, and
``hermes stop`` printed ``invalid choice: 'stop'``. These aliases parse to the
same ``cmd_gateway`` handler with ``gateway_command`` set.
"""

from __future__ import annotations

import argparse

from hermes_cli.subcommands.gateway import build_gateway_parser


def _parser():
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    calls = []

    def _cmd_gateway(args):
        calls.append(
            (
                args.gateway_command,
                bool(getattr(args, "system", False)),
                bool(getattr(args, "all", False)),
            )
        )
        return 0

    build_gateway_parser(
        sub,
        cmd_gateway=_cmd_gateway,
        cmd_proxy=lambda _a: None,
        cmd_gateway_enroll=lambda _a: None,
    )
    return parser, calls


def test_root_stop_dispatches_like_gateway_stop():
    parser, calls = _parser()
    ns = parser.parse_args(["stop"])
    assert ns.command == "stop"
    assert ns.gateway_command == "stop"
    ns.func(ns)
    assert calls == [("stop", False, False)]


def test_root_start_system_all_flags():
    parser, calls = _parser()
    ns = parser.parse_args(["start", "--system", "--all"])
    ns.func(ns)
    assert calls == [("start", True, True)]


def test_root_restart_parity_with_nested():
    parser, calls = _parser()
    nested = parser.parse_args(["gateway", "restart", "--all"])
    root = parser.parse_args(["restart", "--all"])
    nested.func(nested)
    root.func(root)
    assert calls[0] == calls[1] == ("restart", False, True)


def test_nested_gateway_stop_still_works():
    parser, calls = _parser()
    ns = parser.parse_args(["gateway", "stop"])
    assert ns.command == "gateway"
    assert ns.gateway_command == "stop"
    ns.func(ns)
    assert calls == [("stop", False, False)]

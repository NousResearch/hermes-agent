import argparse
from hermes_cli.subcommands.tunnel import build_tunnel_parser


def _sentinel(args):  # pragma: no cover
    return "tunnel-handler"


def _build():
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_tunnel_parser(sub, cmd_tunnel=_sentinel)
    return parser


def test_up_subaction_parses_origins():
    p = _build()
    ns = p.parse_args(["tunnel", "up", "--origin", "alice=127.0.0.1:3000",
                       "--origin", "alice-api=127.0.0.1:8080", "--hold-request",
                       "--reason", "demo", "--until", "4h"])
    assert ns.command == "tunnel"
    assert ns.tunnel_command == "up"
    assert ns.origins == ["alice=127.0.0.1:3000", "alice-api=127.0.0.1:8080"]
    assert ns.hold_request is True
    assert ns.reason == "demo"
    assert ns.func is _sentinel


def test_down_kill_origins():
    p = _build()
    ns = p.parse_args(["tunnel", "down", "--kill-origins"])
    assert ns.tunnel_command == "down"
    assert ns.kill_origins is True


def test_approve_positional_id_and_until():
    p = _build()
    ns = p.parse_args(["tunnel", "approve", "abc123", "--until", "6h"])
    assert ns.tunnel_command == "approve"
    assert ns.id == "abc123"
    assert ns.until == "6h"


def test_deny_positional_id():
    p = _build()
    ns = p.parse_args(["tunnel", "deny", "abc123", "--reason", "too long"])
    assert ns.tunnel_command == "deny"
    assert ns.id == "abc123"


def test_all_subactions_present():
    p = _build()
    for action in ("up", "down", "status", "doctor", "hold", "requests", "approve", "deny"):
        ns = p.parse_args(["tunnel", action])
        assert ns.tunnel_command == action

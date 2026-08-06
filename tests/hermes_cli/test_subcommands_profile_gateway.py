"""Unit tests for extracted subcommand parser builders (profile, gateway).

Confirms the builders attach the same subactions and ``func=`` dispatch that
lived inline in ``main()`` before the god-file Phase 2 extraction.
"""

from __future__ import annotations

import argparse

from hermes_cli.subcommands.gateway import build_gateway_parser
from hermes_cli.subcommands.profile import build_profile_parser


def _h_gateway(args):  # pragma: no cover - identity only
    return "gateway"


def _h_proxy(args):  # pragma: no cover - identity only
    return "proxy"


def _h_gateway_enroll(args):  # pragma: no cover - identity only
    return "gateway_enroll"


def _h_profile(args):  # pragma: no cover - identity only
    return "profile"


def _profile_parser():
    p = argparse.ArgumentParser(prog="hermes")
    sub = p.add_subparsers(dest="command")
    build_profile_parser(sub, cmd_profile=_h_profile)
    return p


def _gateway_parser():
    p = argparse.ArgumentParser(prog="hermes")
    sub = p.add_subparsers(dest="command")
    build_gateway_parser(
        sub,
        cmd_gateway=_h_gateway,
        cmd_proxy=_h_proxy,
        cmd_gateway_enroll=_h_gateway_enroll,
    )
    return p






def test_gateway_and_proxy_dispatch():
    p = _gateway_parser()
    gw = p.parse_args(["gateway", "run"])
    assert gw.command == "gateway"
    assert gw.func is _h_gateway
    px = p.parse_args(["proxy"])
    assert px.command == "proxy"
    assert px.func is _h_proxy




def test_gateway_enroll_dispatch():
    p = _gateway_parser()
    ns = p.parse_args(
        [
            "gateway",
            "enroll",
            "--token",
            "tok",
            "--connector-url",
            "wss://connector.example.com/relay",
            "--gateway-id",
            "gw-1",
        ]
    )
    assert ns.command == "gateway"
    assert ns.gateway_command == "enroll"
    assert ns.func is _h_gateway_enroll
    assert ns.token == "tok"
    assert ns.connector_url == "wss://connector.example.com/relay"
    assert ns.gateway_id == "gw-1"

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


def test_profile_subactions_and_dispatch():
    p = _profile_parser()
    ns = p.parse_args(["profile", "list"])
    assert ns.command == "profile"
    assert ns.profile_action == "list"
    assert ns.func is _h_profile
    # a representative arg-taking subaction
    ns2 = p.parse_args(["profile", "show", "work"])
    assert ns2.profile_action == "show"


def test_profile_has_expected_actions():
    p = _profile_parser()
    # Map each subaction to a minimal valid argv suffix.
    cases = {
        "list": [],
        "use": ["work"],
        "create": ["work"],
        "delete": ["work"],
        "show": ["work"],
        "rename": ["old", "new"],
        "export": ["work"],
        "import": ["/tmp/x.zip"],
    }
    for action, extra in cases.items():
        ns = p.parse_args(["profile", action, *extra])
        assert ns.profile_action == action


def test_gateway_and_proxy_dispatch():
    p = _gateway_parser()
    gw = p.parse_args(["gateway", "run"])
    assert gw.command == "gateway"
    assert gw.func is _h_gateway
    px = p.parse_args(["proxy"])
    assert px.command == "proxy"
    assert px.func is _h_proxy


def test_gateway_accept_hooks_flag():
    p = _gateway_parser()
    ns = p.parse_args(["gateway", "run", "--accept-hooks"])
    assert ns.accept_hooks is True


def test_gateway_run_accepts_external_supervisor_flag():
    p = _gateway_parser()
    ns = p.parse_args(["gateway", "run", "--external-supervisor"])
    assert ns.external_supervisor is True


def test_gateway_lifecycle_accepts_legacy_platform_flag():
    p = _gateway_parser()
    for action in ("start", "restart", "status"):
        ns = p.parse_args(["gateway", action, "--platform", "photon"])
        assert ns.gateway_command == action
        assert ns.platform == "photon"
        assert ns.func is _h_gateway


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

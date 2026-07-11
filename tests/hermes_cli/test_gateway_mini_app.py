from __future__ import annotations

import argparse

from hermes_cli import gateway
from hermes_cli.subcommands.gateway import build_gateway_parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_gateway_parser(
        subparsers,
        cmd_gateway=lambda args: None,
        cmd_proxy=lambda args: None,
        cmd_gateway_enroll=lambda args: None,
    )
    return parser


def test_gateway_mini_app_setup_parser_contract() -> None:
    args = _parser().parse_args([
        "gateway",
        "mini-app",
        "setup",
        "--public-url",
        "https://mini.example.com",
        "--owner",
        "111",
        "--owner",
        "222",
    ])
    assert args.gateway_command == "mini-app"
    assert args.mini_app_command == "setup"
    assert args.public_url == "https://mini.example.com"
    assert args.owner == ["111", "222"]
    assert args.listen_port is None


def test_gateway_mini_app_all_lifecycle_verbs_parse() -> None:
    for verb in ("status", "start", "stop", "restart", "uninstall", "serve"):
        args = _parser().parse_args(["gateway", "mini-app", verb])
        assert args.gateway_command == "mini-app"
        assert args.mini_app_command == verb


def test_gateway_dispatches_mini_app(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(
        "plugins.platforms.telegram.mini_app.cli.command",
        lambda args: seen.append(args.mini_app_command),
    )
    gateway._gateway_command_inner(
        argparse.Namespace(gateway_command="mini-app", mini_app_command="status")
    )
    assert seen == ["status"]

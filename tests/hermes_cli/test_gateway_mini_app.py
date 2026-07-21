from __future__ import annotations

import argparse

from gateway.platform_registry import PlatformRegistry
from hermes_cli import plugins
from plugins.platforms import telegram
from plugins.platforms.telegram.mini_app import cli


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    mini_app = subparsers.add_parser("telegram-mini-app")
    cli.register_cli(mini_app)
    mini_app.set_defaults(func=cli.command)
    return parser


def test_plugin_mini_app_setup_parser_contract() -> None:
    args = _parser().parse_args([
        "telegram-mini-app",
        "setup",
        "--public-url",
        "https://mini.example.com",
        "--owner",
        "111",
        "--owner",
        "222",
    ])
    assert args.command == "telegram-mini-app"
    assert args.mini_app_command == "setup"
    assert args.public_url == "https://mini.example.com"
    assert args.owner == ["111", "222"]
    assert args.listen_port is None


def test_plugin_mini_app_all_lifecycle_verbs_parse() -> None:
    for verb in ("status", "start", "stop", "restart", "uninstall", "serve"):
        args = _parser().parse_args(["telegram-mini-app", verb])
        assert args.command == "telegram-mini-app"
        assert args.mini_app_command == verb


def test_telegram_plugin_owns_mini_app_cli_registration() -> None:
    class Context:
        def __init__(self):
            self.platform = None
            self.cli_command = None

        def register_platform(self, **kwargs):
            self.platform = kwargs

        def register_cli_command(self, **kwargs):
            self.cli_command = kwargs

    context = Context()
    telegram.register(context)

    assert context.platform["name"] == "telegram"
    assert context.cli_command["name"] == "telegram-mini-app"
    assert context.cli_command["setup_fn"] is cli.register_cli
    assert context.cli_command["handler_fn"] is cli.command


def test_platform_cli_loader_resolves_hyphenated_command_prefix(monkeypatch) -> None:
    registry = PlatformRegistry()
    loaded = []
    registry.register_deferred("telegram", lambda: loaded.append("telegram"))
    monkeypatch.setattr("gateway.platform_registry.platform_registry", registry)

    assert plugins.load_platform_cli_commands_for("telegram-mini-app") is True
    assert loaded == ["telegram"]
    assert plugins.load_platform_cli_commands_for("unknown-command") is False

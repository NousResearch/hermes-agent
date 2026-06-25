"""CLI for the freellmapi Hermes plugin."""

from __future__ import annotations

import argparse

from . import core


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="freellmapi_command")

    subs.add_parser("status", help="Show FreeLLMAPI provider and config state")
    subs.add_parser("doctor", help="Run setup health checks (models probe when keyed)")

    setup = subs.add_parser(
        "setup",
        help="Enable plugin, prepend fallback chain, optionally set primary model",
    )
    setup.add_argument(
        "--apply-model",
        action="store_true",
        help="Set model.provider=freellmapi and model.default=auto",
    )
    setup.add_argument(
        "--no-enable",
        action="store_true",
        help="Skip adding freellmapi to plugins.enabled",
    )

    subparser.set_defaults(func=freellmapi_command)


def freellmapi_command(args: argparse.Namespace) -> int:
    command = getattr(args, "freellmapi_command", None)
    if not command:
        print("usage: hermes freellmapi {setup,doctor,status}")
        return 2

    if command == "status":
        payload = core.status()
    elif command == "doctor":
        payload = core.doctor()
    elif command == "setup":
        payload = core.setup(
            apply_model=getattr(args, "apply_model", False),
            enable_plugin=not getattr(args, "no_enable", False),
        )
    else:
        print(f"unknown freellmapi subcommand: {command}")
        return 2

    print(core.to_json(payload))
    return 0 if payload.get("ok", True) else 1

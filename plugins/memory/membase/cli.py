"""CLI shim for the Membase memory provider."""

from __future__ import annotations

import argparse
from typing import Any

try:
    from membase_hermes.plugin.cli import membase_command, register_cli
except ModuleNotFoundError as exc:
    if exc.name != "membase_hermes":
        raise

    def register_cli(subparser: argparse.ArgumentParser) -> None:
        subparser.description = (
            "Membase CLI is available after installing hermes-membase. "
            "Run `hermes memory setup` first."
        )
        subparser.set_defaults(func=membase_command)

    def membase_command(args: argparse.Namespace) -> None:
        raise SystemExit(
            "Membase requires hermes-membase>=0.1.5. "
            "Run `hermes memory setup` to install dependencies, then `hermes membase login`."
        )


def register(ctx: Any) -> None:
    """Compat shim for plugin hosts that call cli.register(ctx) directly."""
    return None

"""``hermes evolution`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_evolution_parser(subparsers, *, cmd_evolution: Callable) -> None:
    """Attach the ``evolution`` subcommand to ``subparsers``."""
    from hermes_cli.evolution_cmd import register_parser
    register_parser(subparsers)

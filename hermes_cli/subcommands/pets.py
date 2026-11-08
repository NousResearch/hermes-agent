"""``hermes pets`` subcommand parser."""

from __future__ import annotations

import logging


def build_pets_parser(subparsers) -> None:
    """Attach the ``pets`` subcommand to ``subparsers``."""
    pets_parser = subparsers.add_parser(
        "pets", help="Browse, install, and select petdex animated pets",
        description="Petdex (https://github.com/crafter-station/petdex) is a public "
            "gallery of animated sprite pets for coding agents. Install one "
            "and Hermes shows it reacting to agent activity across the CLI, "
            "TUI, and desktop app.")
    try:
        from hermes_cli.pets import register_cli as _register_pets_cli

        _register_pets_cli(pets_parser)
    except Exception as _exc:
        logging.getLogger("hermes_cli.main").debug("pets CLI wiring failed: %s", _exc)

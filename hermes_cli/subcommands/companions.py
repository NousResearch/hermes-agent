"""``hermes companions`` subcommand parser."""

from __future__ import annotations

import logging


def build_companions_parser(subparsers) -> None:
    """Attach the ``companions`` subcommand to ``subparsers``."""
    companions_parser = subparsers.add_parser(
        "companions", help="Browse, install, and select companiondex animated companions",
        description="Companiondex (https://github.com/crafter-station/companiondex) is a public "
            "gallery of animated sprite companions for coding agents. Install one "
            "and Hermes shows it reacting to agent activity across the CLI, "
            "TUI, and desktop app.")
    try:
        from hermes_cli.companions import register_cli as _register_companions_cli

        _register_companions_cli(companions_parser)
    except Exception as _exc:
        logging.getLogger("hermes_cli.main").debug("companions CLI wiring failed: %s", _exc)

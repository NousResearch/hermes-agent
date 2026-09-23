"""``hermes journey`` subcommand parser."""

from __future__ import annotations

import logging


def build_journey_parser(subparsers) -> None:
    """Attach the ``journey`` subcommand to ``subparsers``."""
    journey_parser = subparsers.add_parser(
        "journey", aliases=["learning", "memory-graph"],
        help="Timeline of learned skills + memories over time",
        description="A terminal rendition of the desktop Star Map / Memory Graph: a "
            "timeline bar chart of learned skills and memories over time "
            "(oldest at top, newest at bottom) plus a playable constellation "
            "scrubber. Mirrors the TUI `/journey` overlay and the desktop panel.")
    try:
        from hermes_cli.journey import register_cli as _register_journey_cli

        _register_journey_cli(journey_parser)
    except Exception as _exc:
        logging.getLogger("hermes_cli.main").debug("journey CLI wiring failed: %s", _exc)

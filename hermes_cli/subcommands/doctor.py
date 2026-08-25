"""``hermes doctor`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_doctor_parser(subparsers, *, cmd_doctor: Callable) -> None:
    """Attach the ``doctor`` subcommand to ``subparsers``."""
    doctor_parser = subparsers.add_parser(
        "doctor", help="Check configuration and dependencies",
        description="Diagnose issues with Hermes Agent setup")
    doctor_parser.add_argument(
        "--fix", action="store_true", help="Attempt to fix issues automatically")
    doctor_parser.add_argument(
        "--live", action="store_true",
        help="Opt-in: run one bounded, read-only real-call health probe per "
            "configured tool backend (Firecrawl/FAL/browser/MCP/TTS/STT) "
            "after the static checks. Makes real network calls."
    )
    output_mode = doctor_parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--ack",
        metavar="ADVISORY_ID",
        default=None,
        help=(
            "Acknowledge a security advisory by ID and exit. After ack, the "
            "advisory will no longer trigger startup banners. Run `hermes "
            "doctor` first to see active advisories and their IDs."
        ),
    )
    output_mode.add_argument(
        "--json",
        action="store_true",
        help="Emit the complete diagnostic report as JSON only",
    )
    output_mode.add_argument(
        "--verbose",
        action="store_true",
        help="Show resolved route, fallback, and auxiliary-task diagnostics",
    )
    doctor_parser.set_defaults(func=cmd_doctor)

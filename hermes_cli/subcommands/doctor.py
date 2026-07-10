"""``hermes doctor`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_doctor_parser(subparsers, *, cmd_doctor: Callable) -> None:
    """Attach the ``doctor`` subcommand to ``subparsers``."""
    # =========================================================================
    # doctor command
    # =========================================================================
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check configuration and dependencies",
        description="Diagnose issues with Hermes Agent setup",
    )
    doctor_parser.add_argument(
        "--fix", action="store_true", help="Attempt to fix issues automatically"
    )
    doctor_parser.add_argument(
        "--ack",
        metavar="ADVISORY_ID",
        default=None,
        help=(
            "Acknowledge a security advisory by ID and exit. After ack, the "
            "advisory will no longer trigger startup banners. Run `hermes "
            "doctor` first to see active advisories and their IDs."
        ),
    )
    # ------------------------------------------------------------------
    # --upstream : READONLY diagnostic for UH1..UH5, UH9 + UH10.
    # Pure inspection. --json / --compact are only effective with
    # --upstream and are mutually exclusive with --fix / --ack.
    # ------------------------------------------------------------------
    doctor_parser.add_argument(
        "--upstream",
        action="store_true",
        help=(
            "READONLY diagnostic of upstream reference, tracking, "
            "ahead/behind, mutual paths, and update safety (UH10). "
            "Combines with --json / --compact. Does not modify the "
            "working tree or invoke any mutating git command."
        ),
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit a pure JSON object describing upstream health. "
            "Only effective with --upstream."
        ),
    )
    doctor_parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Emit a single stable line summarizing upstream health. "
            "Only effective with --upstream."
        ),
    )
    doctor_parser.set_defaults(func=cmd_doctor)

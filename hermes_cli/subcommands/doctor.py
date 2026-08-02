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
    doctor_targets = doctor_parser.add_subparsers(dest="doctor_target")
    for target_name in ("skill", "cron"):
        target_parser = doctor_targets.add_parser(
            target_name,
            help=f"Run targeted {target_name} diagnostics",
        )
        target_parser.add_argument(
            "target", nargs="?", help=f"{target_name} name or ID"
        )
        target_parser.add_argument("--all", action="store_true", default=False)
        target_parser.add_argument("--json", action="store_true", default=False)
    doctor_parser.set_defaults(func=cmd_doctor)

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
            "after the static checks. Makes real network calls.")
    diagnostic_mode = doctor_parser.add_mutually_exclusive_group()
    diagnostic_mode.add_argument(
        "--runtime", action="store_true",
        help="Opt-in: diagnose the selected profile's resolved agent startup and "
            "first-chunk path with at most one minimal inference request.")
    diagnostic_mode.add_argument(
        "--isolate", action="store_true",
        help="Opt-in: compare a sterile temporary profile with cumulative source-state "
            "slices without mutating the source profile. Makes up to seven minimal "
            "inference requests and stops after the first regression.")
    doctor_parser.add_argument(
        "--json", action="store_true",
        help="Emit the --runtime or --isolate report as machine-readable JSON.")
    doctor_parser.add_argument(
        "--ack", metavar="ADVISORY_ID", default=None,
        help="Acknowledge a security advisory by ID and exit. After ack, the "
            "advisory will no longer trigger startup banners. Run `hermes "
            "doctor` first to see active advisories and their IDs.")
    doctor_parser.set_defaults(func=cmd_doctor)

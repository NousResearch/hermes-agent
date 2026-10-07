"""``rabbit debug`` subcommand parser."""

from __future__ import annotations

import argparse
from typing import Callable


def build_debug_parser(subparsers, *, cmd_debug: Callable) -> None:
    """Attach the ``debug`` subcommand to ``subparsers``."""
    debug_parser = subparsers.add_parser(
        "debug", help="Debug tools — upload logs and system info for support",
        description="Debug utilities for Rabbit Agent. Use 'rabbit debug share' to "
        "upload a debug report (system info + recent logs) to a paste "
        "service and get a shareable URL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    rabbit debug share              Upload debug report (asks for confirmation)
    rabbit debug share --yes        Skip confirmation (for scripts/CI)
    rabbit debug share --lines 500  Include more log lines
    rabbit debug share --expire 30  Keep dpaste.com fallback pastes for 30 days
    rabbit debug share --local      Print report locally (no upload)
    rabbit debug share --no-redact  Disable upload-time secret redaction
    rabbit debug delete <url>       Delete a previously uploaded paste
""")
    debug_sub = debug_parser.add_subparsers(dest="debug_command")
    share_parser = debug_sub.add_parser(
        "share", help="Upload debug report to a paste service and print a shareable URL")
    share_parser.add_argument(
        "--lines", type=int, default=200,
        help="Number of log lines to include per log file (default: 200)")
    share_parser.add_argument(
        "--expire", type=int, default=1,
        help="dpaste.com fallback retention in days (minimum 1; default: 1). "
            "paste.rs pastes are always deleted after 6 hours, but if the "
            "upload falls back to dpaste.com the pastes live for this many "
            "days and cannot be deleted.")
    share_parser.add_argument(
        "--local", action="store_true", help="Print the report locally instead of uploading")
    share_parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip the confirmation prompt and upload immediately. Required "
            "in non-interactive contexts (scripts/CI); without it, and with "
            "no TTY on stdin, the command refuses rather than upload silently.")
    share_parser.add_argument(
        "--no-redact", action="store_true",
        help="Disable upload-time secret redaction (default: redact). Logs "
            "are normally run through agent.redact.redact_sensitive_text "
            "with force=True before upload so credentials are not leaked "
            "into the public paste service.")
    delete_parser = debug_sub.add_parser(
        "delete", help="Delete a paste uploaded by 'rabbit debug share'")
    delete_parser.add_argument(
        "urls", nargs="*", default=[],
        help="One or more paste URLs to delete (e.g. https://paste.rs/abc123)")
    debug_parser.set_defaults(func=cmd_debug)

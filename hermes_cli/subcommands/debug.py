"""``hermes debug`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

import argparse
from typing import Callable


def build_debug_parser(subparsers, *, cmd_debug: Callable) -> None:
    """Attach the ``debug`` subcommand to ``subparsers``."""
    # =========================================================================
    # debug command
    # =========================================================================
    debug_parser = subparsers.add_parser(
        "debug",
        help="Debug tools — upload logs and system info for support",
        description="Debug utilities for Hermes Agent. Use 'hermes debug share' to "
        "upload a debug report (system info + recent logs) to a paste "
        "service and get a shareable URL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    hermes debug share              Upload debug report and print URL
    hermes debug share --lines 500  Include more log lines
    hermes debug share --expire 30  Keep paste for 30 days
    hermes debug share --local      Print report locally (no upload)
    hermes debug share --no-redact  Disable upload-time secret redaction
    hermes debug share --nous       Upload to Nous-internal storage (private)
    hermes debug delete <url>       Delete a previously uploaded paste
""",
    )
    debug_sub = debug_parser.add_subparsers(dest="debug_command")
    share_parser = debug_sub.add_parser(
        "share",
        help="Upload debug report to a paste service and print a shareable URL",
    )
    share_parser.add_argument(
        "--lines",
        type=int,
        default=200,
        help="Number of log lines to include per log file (default: 200)",
    )
    share_parser.add_argument(
        "--expire",
        type=int,
        default=7,
        help="Paste expiry in days (default: 7)",
    )
    share_parser.add_argument(
        "--local",
        action="store_true",
        help="Print the report locally instead of uploading",
    )
    share_parser.add_argument(
        "--no-redact",
        action="store_true",
        help=(
            "Disable upload-time secret redaction (default: redact). Logs "
            "are normally run through agent.redact.redact_sensitive_text "
            "with force=True before upload so credentials are not leaked "
            "into the public paste service."
        ),
    )
    share_parser.add_argument(
        "--nous",
        action="store_true",
        help=(
            "Upload the debug bundle to Nous-internal storage (AWS S3) instead "
            "of a public paste service. The bundle is private — viewable only "
            "by Nous staff (and allowlisted Discord mods) via a Google-login-"
            "gated viewer — and auto-deletes after 14 days. Still force-redacts "
            "secrets unless --no-redact is also passed."
        ),
    )
    delete_parser = debug_sub.add_parser(
        "delete",
        help="Delete a paste uploaded by 'hermes debug share'",
    )
    delete_parser.add_argument(
        "urls",
        nargs="*",
        default=[],
        help="One or more paste URLs to delete (e.g. https://paste.rs/abc123)",
    )
    debug_parser.set_defaults(func=cmd_debug)

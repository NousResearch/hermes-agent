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
        "--live",
        action="store_true",
        help=(
            "Opt-in: run one bounded, read-only real-call health probe per "
            "configured tool backend (Firecrawl/FAL/browser/MCP/TTS/STT) "
            "after the static checks. Makes real network calls."
        ),
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
    doctor_parser.add_argument(
        "--routing", action="store_true",
        help="Explain one profile-routing decision (read-only; no gateway or network access)",
    )
    doctor_parser.add_argument("--routing-profile", default="default", metavar="NAME", help="Fallback/default profile for routing diagnostics")
    doctor_parser.add_argument("--platform", dest="routing_platform", default="", help="Inbound platform")
    doctor_parser.add_argument("--guild-id", dest="routing_guild_id", help="Inbound guild/server ID")
    doctor_parser.add_argument("--chat-id", dest="routing_chat_id", help="Inbound chat/channel ID")
    doctor_parser.add_argument("--thread-id", dest="routing_thread_id", help="Inbound thread ID")
    doctor_parser.add_argument("--user-id", dest="routing_user_id", help="Inbound user ID (shown only as a redacted dimension)")
    doctor_parser.add_argument("--json", action="store_true", help="Emit deterministic JSON (routing mode only)")
    doctor_parser.set_defaults(func=cmd_doctor)

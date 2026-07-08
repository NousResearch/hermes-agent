"""``hermes acp`` subcommand parser.

Extracted from ``hermes_cli/main.py:main()`` (god-file Phase 2 follow-up).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable

from hermes_cli.subcommands._shared import add_accept_hooks_flag


def build_acp_parser(subparsers, *, cmd_acp: Callable) -> None:
    """Attach the ``acp`` subcommand to ``subparsers``."""
    acp_parser = subparsers.add_parser(
        "acp",
        help="Run HT AI Agent as an ACP (Agent Client Protocol) server",
        description="Start HT AI Agent in ACP mode for editor integration (VS Code, Zed, JetBrains)",
    )
    add_accept_hooks_flag(acp_parser)
    acp_parser.add_argument(
        "--version",
        action="store_true",
        dest="acp_version",
        help="Print HT AI Agent ACP version and exit",
    )
    acp_parser.add_argument(
        "--check",
        action="store_true",
        help="Verify ACP dependencies and adapter imports, then exit",
    )
    acp_parser.add_argument(
        "--setup",
        action="store_true",
        help="Run interactive HT AI Agent provider/model setup for ACP terminal auth",
    )
    acp_parser.add_argument(
        "--setup-browser",
        action="store_true",
        help="Install agent-browser + Playwright Chromium into ~/.hermes/node/ "
             "for browser tool support (idempotent).",
    )
    acp_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        dest="assume_yes",
        help="Accept all prompts (used by --setup-browser to skip the "
             "~400 MB Chromium download confirmation).",
    )
    acp_parser.set_defaults(func=cmd_acp)

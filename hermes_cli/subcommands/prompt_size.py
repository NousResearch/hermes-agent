"""``hermes prompt-size`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_prompt_size_parser(subparsers, *, cmd_prompt_size: Callable) -> None:
    """Attach the ``prompt-size`` subcommand to ``subparsers``."""
    # =========================================================================
    # prompt-size command
    # =========================================================================
    prompt_size_parser = subparsers.add_parser(
        "prompt-size",
        help="Show a byte breakdown of the system prompt + tool schemas",
        description=(
            "Report the fixed prompt budget for a fresh session: system "
            "prompt total, skills index, memory, user profile, and tool-schema "
            "JSON. Runs offline (no API call)."
        ),
    )
    prompt_size_parser.add_argument(
        "--platform",
        default="cli",
        help="Platform to simulate (cli, telegram, discord, ...). Default: cli",
    )
    prompt_size_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the breakdown as JSON",
    )
    prompt_size_parser.set_defaults(func=cmd_prompt_size)

"""``hermes usage`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_usage_parser(subparsers, *, cmd_usage: Callable) -> None:
    parser = subparsers.add_parser(
        "usage",
        help="Show official OAuth quota windows (Codex, Claude, OpenRouter, Nous)",
        description=(
            "Fetch first-party subscription quota windows. "
            "Does not scrape undocumented endpoints (Grok is unsupported)."
        ),
    )
    parser.add_argument(
        "provider",
        nargs="?",
        help="Provider id or alias (openai-codex, anthropic, openrouter, nous). Omit for all official sources.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON report (script-friendly)",
    )
    parser.set_defaults(func=cmd_usage)

"""`hermes preview` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_preview_parser(subparsers, cmd_preview: Callable | None = None) -> None:
    """Register the preview subcommand on the given subparsers container."""
    parser = subparsers.add_parser(
        "preview",
        help="Preview Hermes runtime configuration without executing anything",
        description=(
            "Inspect Hermes configuration (model, provider, auth, skills, tools, MCP) "
            "and report readiness. Does NOT call the model, execute tools, or spawn subagents."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format: text (default) or json (for CI/scripts)",
    )
    parser.set_defaults(func=cmd_preview)

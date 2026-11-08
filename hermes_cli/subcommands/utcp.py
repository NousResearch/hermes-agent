"""``hermes utcp`` subcommand parser.

Mirrors ``hermes_cli/subcommands/mcp.py``: the handler is injected so this
module never imports ``hermes_cli.main``.
"""

from __future__ import annotations

from typing import Callable


def build_utcp_parser(subparsers, *, cmd_utcp: Callable) -> None:
    """Attach the ``utcp`` subcommand to *subparsers*."""
    parser = subparsers.add_parser(
        "utcp",
        help="Manage native UTCP tool providers",
        description=(
            "Manage Universal Tool Calling Protocol (UTCP) providers.\n\n"
            "UTCP providers publish a manual (tool definitions + transport) that\n"
            "Hermes consumes natively — no MCP server required. Tools appear as\n"
            "'utcp__<provider>__<tool>' once discovery runs."
        ),
    )
    sub = parser.add_subparsers(dest="utcp_action")

    sub.add_parser("list", help="List configured UTCP providers")
    sub.add_parser("status", help="Show live provider status (runs discovery)")
    sub.add_parser("refresh", help="Re-discover one provider or all")
    sub.choices["refresh"].add_argument(
        "provider", nargs="?", help="Provider name (default: all)"
    )

    tools_p = sub.add_parser("tools", help="List discovered UTCP tools")
    tools_p.add_argument("provider", nargs="?", help="Filter by provider name")

    add_p = sub.add_parser("add", help="Add or update a UTCP provider")
    add_p.add_argument("name", help="Provider name (letters, digits, underscores)")
    add_p.add_argument("--url", required=True, help="UTCP manual URL (https, or http on loopback)")
    add_p.add_argument(
        "--transport",
        default="http",
        choices=["http", "sse", "streamable_http"],
        help="UTCP transport (default: http)",
    )
    add_p.add_argument(
        "--header",
        action="append",
        metavar="KEY=VALUE",
        help="Static header sent to the provider (repeatable)",
    )
    add_p.add_argument(
        "--variable",
        action="append",
        metavar="NAME=env:ENV_VAR",
        help="Expose an environment variable to the provider's templates (repeatable)",
    )
    add_p.add_argument("--timeout", type=float, help="Per-tool-call timeout in seconds")

    remove_p = sub.add_parser("remove", help="Remove a UTCP provider")
    remove_p.add_argument("name", help="Provider name")

    parser.set_defaults(func=cmd_utcp)

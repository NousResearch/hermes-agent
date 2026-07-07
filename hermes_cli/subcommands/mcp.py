"""``hermes mcp`` subcommand parser.

Extracted from ``hermes_cli/main.py:main()`` (god-file Phase 2 follow-up).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

import argparse
from typing import Callable

from hermes_cli.subcommands._shared import add_accept_hooks_flag


def build_mcp_parser(subparsers, *, cmd_mcp: Callable) -> None:
    """Attach the ``mcp`` subcommand to ``subparsers``."""
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Manage MCP servers and run Hermes as an MCP server",
        description=(
            "Manage MCP server connections and run Hermes as an MCP server.\n\n"
            "MCP servers provide additional tools via the Model Context Protocol.\n"
            "Use 'hermes mcp add' to connect to a new server, or\n"
            "'hermes mcp serve' to expose Hermes conversations over MCP."
        ),
    )
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_action")

    mcp_serve_p = mcp_sub.add_parser(
        "serve",
        help="Run Hermes as an MCP server (expose conversations to other agents)",
    )
    mcp_serve_p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging on stderr",
    )
    mcp_serve_p.add_argument(
        "--profile-router",
        action="store_true",
        help=(
            "Expose only the no-model profile-router MCP tools instead of the "
            "legacy conversation/messaging bridge"
        ),
    )
    mcp_serve_p.add_argument(
        "--http",
        action="store_true",
        help=(
            "Run --profile-router over Streamable HTTP instead of stdio. "
            "HTTP is bearer-token protected and binds to localhost by default."
        ),
    )
    mcp_serve_p.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport for --profile-router (default: stdio; use streamable-http for local HTTP)",
    )
    mcp_serve_p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for --profile-router --http (default: 127.0.0.1)",
    )
    mcp_serve_p.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for --profile-router --http (default: 8765)",
    )
    mcp_serve_p.add_argument(
        "--streamable-http-path",
        default="/mcp",
        help="Streamable HTTP path for --profile-router --http (default: /mcp)",
    )
    mcp_serve_p.add_argument(
        "--public-url",
        help=(
            "Externally reachable origin for --profile-router --http behind a "
            "TLS reverse proxy, e.g. https://mcp.example.com. Do not include /mcp."
        ),
    )
    add_accept_hooks_flag(mcp_serve_p)

    token_p = mcp_sub.add_parser(
        "profile-router-token",
        aliases=["profile-router-tokens"],
        help="Manage local bearer tokens for the self-hosted profile-router HTTP server",
    )
    token_sub = token_p.add_subparsers(dest="token_action")
    token_create = token_sub.add_parser("create", help="Create a new profile-router bearer token")
    token_create.add_argument("--name", default="", help="Optional token label for local list/revoke UX")
    token_create.add_argument(
        "--scope",
        action="append",
        dest="scopes",
        help="Scope to grant (repeatable; default: context:read, workspace:read, diff:read)",
    )
    token_create.add_argument("--expires-at", help="Optional ISO timestamp expiry")
    token_sub.add_parser("list", aliases=["ls"], help="List profile-router token metadata")
    token_revoke = token_sub.add_parser("revoke", help="Revoke a profile-router token by token_id")
    token_revoke.add_argument("token_id")
    token_rotate = token_sub.add_parser("rotate", help="Revoke and replace a profile-router token by token_id")
    token_rotate.add_argument("token_id")

    mcp_add_p = mcp_sub.add_parser(
        "add", help="Add an MCP server (discovery-first install)"
    )
    mcp_add_p.add_argument("name", help="Server name (used as config key)")
    mcp_add_p.add_argument("--url", help="HTTP/SSE endpoint URL")
    # dest="mcp_command" so this flag does not clobber the top-level
    # subparser's args.command attribute, which the dispatcher reads to
    # route to cmd_mcp.  Without an explicit dest, argparse derives
    # dest="command" from the flag name and sets it to None when the
    # flag is omitted, causing `hermes mcp add ...` to fall through to
    # interactive chat.
    mcp_add_p.add_argument(
        "--command", dest="mcp_command", help="Stdio command (e.g. npx)"
    )
    mcp_add_p.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments for stdio command; must be the last option",
    )
    mcp_add_p.add_argument("--auth", choices=["oauth", "header"], help="Auth method")
    mcp_add_p.add_argument("--preset", help="Known MCP preset name")
    mcp_add_p.add_argument(
        "--connect-timeout",
        type=float,
        help="Timeout in seconds for initial connection and tool discovery",
    )
    mcp_add_p.add_argument(
        "--env",
        nargs="*",
        default=[],
        help="Environment variables for stdio servers (KEY=VALUE)",
    )

    mcp_rm_p = mcp_sub.add_parser("remove", aliases=["rm"], help="Remove an MCP server")
    mcp_rm_p.add_argument("name", help="Server name to remove")

    mcp_sub.add_parser("list", aliases=["ls"], help="List configured MCP servers")

    mcp_test_p = mcp_sub.add_parser("test", help="Test MCP server connection")
    mcp_test_p.add_argument("name", help="Server name to test")

    mcp_cfg_p = mcp_sub.add_parser(
        "configure", aliases=["config"], help="Toggle tool selection"
    )
    mcp_cfg_p.add_argument("name", help="Server name to configure")

    mcp_login_p = mcp_sub.add_parser(
        "login",
        help="Force re-authentication for an OAuth-based MCP server",
    )
    mcp_login_p.add_argument("name", help="Server name to re-authenticate")

    mcp_reauth_p = mcp_sub.add_parser(
        "reauth",
        help="Re-authenticate one OAuth MCP server, or all of them (--all)",
    )
    mcp_reauth_p.add_argument(
        "name", nargs="?", help="Server name to re-authenticate (omit with --all)"
    )
    mcp_reauth_p.add_argument(
        "--all",
        action="store_true",
        help="Re-authenticate every OAuth server in config, one at a time",
    )

    # ── Catalog (Nous-approved MCPs shipped with the repo) ─────────────────
    mcp_sub.add_parser(
        "picker",
        help="Interactive catalog picker (also the default for `hermes mcp`)",
    )
    mcp_sub.add_parser(
        "catalog",
        help="List Nous-approved MCPs available for one-click install",
    )
    mcp_install_p = mcp_sub.add_parser(
        "install",
        help="Install a catalog MCP by name (e.g. `hermes mcp install n8n`)",
    )
    mcp_install_p.add_argument(
        "identifier",
        help="Catalog entry name (or `official/<name>`)",
    )

    add_accept_hooks_flag(mcp_parser)
    mcp_parser.set_defaults(func=cmd_mcp)

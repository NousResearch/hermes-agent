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
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport to serve (default: stdio)",
    )
    mcp_serve_p.add_argument("--host", default="127.0.0.1", help="HTTP host to bind")
    mcp_serve_p.add_argument("--port", type=int, default=8666, help="HTTP port to bind")
    mcp_serve_p.add_argument("--path", default="/mcp", help="HTTP MCP endpoint path")
    mcp_serve_p.add_argument(
        "--public-base-url",
        help="Externally visible base URL for OAuth metadata (for reverse proxies/Tailscale)",
    )
    mcp_serve_p.add_argument(
        "--auth-token-env",
        help="Environment variable containing the shared MCP HTTP auth token/PSK",
    )
    mcp_serve_p.add_argument(
        "--auth-header",
        default="X-Hermes-MCP-PSK",
        help="Additional PSK header name accepted by HTTP auth",
    )
    mcp_serve_p.add_argument(
        "--allow-query-token",
        action="store_true",
        help="Accept ?access_token= or ?psk= auth for clients that cannot send headers; disables access logs",
    )
    mcp_serve_p.add_argument(
        "--allow-insecure-http",
        action="store_true",
        help="Explicitly allow cleartext non-loopback serving (still requires auth)",
    )
    mcp_serve_p.add_argument(
        "--oauth-compatible",
        action="store_true",
        help="Expose OAuth-compatible metadata plus /mcp/authorize and /mcp/token endpoints",
    )
    mcp_serve_p.add_argument(
        "--oauth-client-id-env",
        help="Environment variable containing the OAuth client id (default: hermes-mcp)",
    )
    mcp_serve_p.add_argument(
        "--oauth-client-secret-env",
        help="OAuth client-secret environment variable (required with a custom client id; otherwise uses the PSK)",
    )
    mcp_serve_p.add_argument(
        "--oauth-token-ttl-seconds",
        type=int,
        default=3600,
        help="Issued OAuth bearer token lifetime in seconds (default: 3600)",
    )
    mcp_serve_p.add_argument(
        "--oauth-code-ttl-seconds",
        type=int,
        default=300,
        help="Authorization code lifetime in seconds",
    )
    mcp_serve_p.add_argument(
        "--oauth-redirect-uri",
        action="append",
        default=[],
        help="Registered HTTP-loopback or HTTPS OAuth redirect URI; repeat or comma-separate",
    )
    mcp_serve_p.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="Allowed HTTP Host value for Streamable HTTP transport security; repeat or comma-separate",
    )
    mcp_serve_p.add_argument(
        "--allowed-origin",
        action="append",
        default=[],
        help="Allowed Origin value for Streamable HTTP transport security; repeat or comma-separate",
    )
    mcp_serve_p.add_argument(
        "--health-path",
        default="/health",
        help="Health check endpoint path for HTTP transport",
    )
    mcp_serve_p.add_argument(
        "--expose-toolset",
        action="append",
        default=[],
        help="Expose registered Hermes tools from a toolset over this MCP server; repeat or comma-separate",
    )
    mcp_serve_p.add_argument(
        "--expose-tool",
        action="append",
        default=[],
        help="Expose an individual registered Hermes tool over this MCP server; repeat or comma-separate",
    )
    mcp_serve_p.add_argument(
        "--expose-plugin-tools",
        action="store_true",
        help="Expose all tools registered by enabled Hermes plugins over this MCP server",
    )
    mcp_serve_p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging on stderr",
    )
    add_accept_hooks_flag(mcp_serve_p)

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

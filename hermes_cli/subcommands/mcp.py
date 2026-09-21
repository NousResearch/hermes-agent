"""``hermes mcp`` subcommand parser."""

from __future__ import annotations

import argparse
from typing import Callable

from hermes_cli.subcommands._shared import add_accept_hooks_flag


def build_mcp_parser(subparsers, *, cmd_mcp: Callable) -> None:
    """Attach the ``mcp`` subcommand to ``subparsers``."""
    mcp_parser = subparsers.add_parser(
        "mcp", help="Manage MCP servers and run Hermes as an MCP server",
        description="Manage MCP server connections and run Hermes as an MCP server.\n\n"
            "MCP servers provide additional tools via the Model Context Protocol.\n"
            "Use 'hermes mcp add' to connect to a new server, or\n"
            "'hermes mcp serve' to expose Hermes conversations over MCP.")
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_action")

    mcp_serve_p = mcp_sub.add_parser(
        "serve", help="Run Hermes as an MCP server (expose conversations to other agents)")
    mcp_serve_p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging on stderr")
    mcp_serve_p.add_argument(
        "--transport", choices=("stdio", "http"), default="stdio",
        help="MCP transport (default: stdio)")
    mcp_serve_p.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    mcp_serve_p.add_argument("--port", type=int, default=8000, help="HTTP bind port")
    mcp_serve_p.add_argument("--path", default="/mcp", help="Streamable HTTP endpoint path")
    mcp_serve_p.add_argument(
        "--token-env", default="HERMES_MCP_SERVER_TOKEN",
        help="Environment variable containing the HTTP bearer token")
    mcp_serve_p.add_argument(
        "--allowed-host", action="append", dest="allowed_hosts",
        help="Allowed HTTP Host value (repeatable; required for non-loopback binds)")
    mcp_serve_p.add_argument(
        "--public-url", help="Public MCP URL advertised in protected-resource metadata")
    add_accept_hooks_flag(mcp_serve_p)

    mcp_add_p = mcp_sub.add_parser("add", help="Add an MCP server (discovery-first install)")
    mcp_add_p.add_argument("name", help="Server name (used as config key)")
    mcp_add_p.add_argument("--url", help="HTTP/SSE endpoint URL")
    # dest="mcp_command": the default dest="command" would clobber the top-level
    # subcommand name the dispatcher routes on (None when omitted → falls into chat).
    mcp_add_p.add_argument("--command", dest="mcp_command", help="Stdio command (e.g. npx)")
    mcp_add_p.add_argument(
        "--args", nargs=argparse.REMAINDER, default=[],
        help="Arguments for stdio command; must be the last option")
    mcp_add_p.add_argument("--auth", choices=["oauth", "header"], help="Auth method")
    mcp_add_p.add_argument("--preset", help="Known MCP preset name")
    mcp_add_p.add_argument(
        "--connect-timeout", type=float,
        help="Timeout in seconds for initial connection and tool discovery")
    mcp_add_p.add_argument(
        "--env", nargs="*", default=[], help="Environment variables for stdio servers (KEY=VALUE)")

    mcp_rm_p = mcp_sub.add_parser("remove", aliases=["rm"], help="Remove an MCP server")
    mcp_rm_p.add_argument("name", help="Server name to remove")

    mcp_sub.add_parser("list", aliases=["ls"], help="List configured MCP servers")

    mcp_test_p = mcp_sub.add_parser("test", help="Test MCP server connection")
    mcp_test_p.add_argument("name", help="Server name to test")

    mcp_cfg_p = mcp_sub.add_parser("configure", aliases=["config"], help="Toggle tool selection")
    mcp_cfg_p.add_argument("name", help="Server name to configure")

    mcp_login_p = mcp_sub.add_parser(
        "login", help="Force re-authentication for an OAuth-based MCP server")
    mcp_login_p.add_argument("name", help="Server name to re-authenticate")
    mcp_login_p.add_argument(
        "--flow", choices=["browser", "device"], default=None,
        help="OAuth flow (overrides oauth.flow): browser PKCE or RFC 8628 device code")

    mcp_reauth_p = mcp_sub.add_parser(
        "reauth", help="Re-authenticate one OAuth MCP server, or all of them (--all)")
    mcp_reauth_p.add_argument(
        "name", nargs="?", help="Server name to re-authenticate (omit with --all)")
    mcp_reauth_p.add_argument(
        "--all", action="store_true",
        help="Re-authenticate every OAuth server in config, one at a time")

    # Catalog (Nous-approved MCPs shipped with the repo)
    mcp_sub.add_parser(
        "picker", help="Interactive catalog picker (also the default for `hermes mcp`)")
    mcp_sub.add_parser("catalog", help="List Nous-approved MCPs available for one-click install")
    mcp_install_p = mcp_sub.add_parser(
        "install", help="Install a catalog MCP by name (e.g. `hermes mcp install deepwiki`)")
    mcp_install_p.add_argument("identifier", help="Catalog entry name (or `official/<name>`)")

    add_accept_hooks_flag(mcp_parser)
    mcp_parser.set_defaults(func=cmd_mcp)

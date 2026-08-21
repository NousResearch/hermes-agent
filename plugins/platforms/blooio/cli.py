"""
``hermes blooio ...`` CLI subcommands — registered by the plugin via
``ctx.register_cli_command()``.

Subcommands:

    login     one-click OAuth: opens the browser to the Blooio consent screen,
              captures the code on a loopback redirect, and stores the tokens
    logout    revoke + clear the stored OAuth tokens
    status    show the current auth state (mode, org, token expiry)
    setup     alias for `login` plus the inbound-webhook / public-URL guidance

Blooio auth is OAuth by default (public app + PKCE). ``BLOOIO_API_KEY`` remains
a headless/CI fallback; when it is set, ``login`` is unnecessary.
"""

from __future__ import annotations

import argparse
import asyncio

from hermes_cli.colors import Colors, color

from . import auth as blooio_auth


def register_cli(parser: argparse.ArgumentParser) -> None:
    """Wire up `hermes blooio ...` subcommands."""
    subs = parser.add_subparsers(dest="blooio_command", required=False)

    p_login = subs.add_parser("login", help="Connect Blooio via OAuth (opens a browser)")
    p_login.add_argument("--no-browser", action="store_true",
                         help="Print the authorize URL instead of opening a browser")
    p_login.add_argument("--org", default="",
                         help="Organization id to scope the token to (org_…); "
                              "needed only if the app is installed on multiple orgs")

    subs.add_parser("logout", help="Revoke and clear stored Blooio credentials")
    subs.add_parser("status", help="Show Blooio auth state")

    p_setup = subs.add_parser("setup", help="Connect Blooio + inbound-webhook guidance")
    p_setup.add_argument("--no-browser", action="store_true")
    p_setup.add_argument("--org", default="")


def dispatch(args: argparse.Namespace) -> int:
    command = getattr(args, "blooio_command", None)
    if command in (None, "status"):
        return _cmd_status(args)
    if command in ("login", "setup"):
        return _cmd_login(args)
    if command == "logout":
        return _cmd_logout(args)
    return 2


def _cmd_login(args: argparse.Namespace) -> int:
    if blooio_auth.os.getenv("BLOOIO_API_KEY"):
        print(color("BLOOIO_API_KEY is set — OAuth login isn't needed.", Colors.YELLOW))
        return 0
    try:
        record = asyncio.run(
            blooio_auth.login(
                open_browser=not getattr(args, "no_browser", False),
                organization_id=getattr(args, "org", "") or "",
            )
        )
    except blooio_auth.BlooioAuthError as exc:
        print(color(f"Login failed: {exc}", Colors.RED))
        return 1
    org = record.get("organization_id") or "(resolve at first request)"
    print(color("\n✓ Connected Blooio to Hermes.", Colors.GREEN))
    print(f"  organization: {org}")
    print(f"  scopes:       {record.get('scope', '')}")
    if getattr(args, "blooio_command", None) == "setup":
        print(
            "\nInbound messages arrive via webhook, so Hermes needs a public "
            "HTTPS URL. Expose it (Cloudflare Tunnel / ngrok), set "
            "BLOOIO_PUBLIC_URL, and set BLOOIO_AUTO_REGISTER_WEBHOOK=true to "
            "auto-create the webhook and capture its signing secret on connect."
        )
    return 0


def _cmd_logout(_args: argparse.Namespace) -> int:
    asyncio.run(blooio_auth.logout())
    print(color("Signed out of Blooio.", Colors.GREEN))
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    st = blooio_auth.status()
    mode = st.get("mode")
    if mode == "none":
        print(color("Blooio: not connected. Run `hermes blooio login`.", Colors.YELLOW))
        return 0
    print(color(f"Blooio auth: {mode}", Colors.GREEN))
    if st.get("organization_id"):
        print(f"  organization: {st['organization_id']}")
    if mode == "oauth":
        print(f"  scopes:            {st.get('scope', '')}")
        print(f"  access expires in: {st.get('access_expires_in', 0)}s")
        print(f"  refresh token:     {'present' if st.get('has_refresh_token') else 'missing'}")
    return 0

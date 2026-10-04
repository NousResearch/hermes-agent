"""``hermes feishu ...`` CLI subcommands (registered via ``ctx.register_cli_command()``):
login, status, logout for the **user** grant (``user_access_token``).

The bot's own credentials are still owned by ``hermes setup`` — this command only manages the
optional, opt-in authorization that lets Hermes read the user's own Feishu/Lark message history
(cross-chat search and other bots' messages in shared chats). The flow itself lives in
``tools/feishu_user_auth.py`` so the tools and this CLI share one implementation.
"""
from __future__ import annotations

import argparse
import sys

from tools import feishu_user_auth


def register_cli(parser: argparse.ArgumentParser) -> None:
    """Wire up `hermes feishu ...` subcommands."""
    subs = parser.add_subparsers(dest="feishu_command", required=False)
    p_login = subs.add_parser(
        "login", help="Authorize Hermes to read your own Feishu/Lark messages (user_access_token)")
    p_login.add_argument(
        "--redirect-uri", default=None,
        help=f"Loopback redirect URI to listen on (default {feishu_user_auth.DEFAULT_REDIRECT_URI}); "
             "must be allow-listed in the app console")
    p_login.add_argument(
        "--scope", default=None,
        help="Space-separated scopes to request (default: offline_access plus the message read/search set)")
    p_login.add_argument(
        "--no-browser", action="store_true", help="Print the authorize URL instead of opening a browser")
    p_login.add_argument(
        "--timeout", type=float, default=180.0, help="Seconds to wait for the callback (default 180)")
    subs.add_parser("status", help="Show whether a Feishu/Lark user grant is stored, and its scopes")
    subs.add_parser("logout", help="Forget the stored Feishu/Lark user grant")
    parser.set_defaults(func=dispatch)


def dispatch(args: argparse.Namespace) -> int:
    sub = getattr(args, "feishu_command", None)
    handler = _cmd_status if sub is None else _COMMANDS.get(sub)  # no subcommand — status by default
    if handler is None:
        print(f"unknown subcommand: {sub}", file=sys.stderr)
        return 2
    return handler(args)


def _cmd_login(args: argparse.Namespace) -> int:
    try:
        state = feishu_user_auth.login(
            redirect_uri=getattr(args, "redirect_uri", None), scope=getattr(args, "scope", None),
            open_browser=not getattr(args, "no_browser", False),
            timeout_seconds=float(getattr(args, "timeout", None) or 180.0))
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"login failed: {exc}", file=sys.stderr)
        return 1
    # Never print token material (shoulder-surfing / screen recordings): scopes and expiry only.
    print("\nFeishu / Lark user access authorized.")
    print(f"  Granted scopes: {state.get('granted_scope') or state.get('scope')}")
    print(f"  Expires at:     {state.get('expires_at')}")
    print("  feishu_message_search / feishu_message_list are available from the next session.")
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    status = feishu_user_auth.auth_status()
    if not status.get("logged_in"):
        reason = status.get("error")
        print("feishu user access: not authorized" + (f" ({reason})" if reason else ""))
        print("  Authorize with: hermes feishu login")
        return 0
    print("feishu user access: authorized")
    for key in ("domain", "client_id", "scope", "expires_at", "redirect_uri", "has_refresh_token"):
        value = status.get(key)
        if value not in (None, ""):
            print(f"  {key}: {value}")
    if status.get("error"):
        print(f"  last error: {status['error']}")
    return 0


def _cmd_logout(_args: argparse.Namespace) -> int:
    if feishu_user_auth.clear_state():
        print("Forgot the stored Feishu / Lark user grant.")
    else:
        print("No Feishu / Lark user grant was stored.")
    return 0


_COMMANDS = {"login": _cmd_login, "status": _cmd_status, "logout": _cmd_logout}

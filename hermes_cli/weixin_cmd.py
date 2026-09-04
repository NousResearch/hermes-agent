"""``hermes weixin`` subcommand group.

Currently provides ``hermes weixin login`` — a standalone QR-login flow that
persists credentials for an ADDITIONAL Weixin account under
``~/.hermes/weixin/accounts/<account_id>.json`` (#47129).

The PRIMARY account is still configured through ``hermes gateway setup``
(``_setup_weixin``), which writes the legacy ``WEIXIN_*`` env vars. Extra
accounts discovered from the accounts/ directory are registered by
``gateway.platforms.weixin_multi.register_persisted_weixin_accounts`` at
gateway startup as ``weixin:<account>`` platforms.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from hermes_cli.cli_output import (
    color,
    print_error,
    print_info,
    print_success,
    print_warning,
)
from hermes_constants import get_hermes_home

_ACCOUNT_SUBDIR = Path("weixin") / "accounts"


def _persist_account(account_id: str, credentials: dict) -> Path:
    """Write one account credential file and return its path."""
    target = Path(get_hermes_home()) / _ACCOUNT_SUBDIR / f"{account_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "account_id": account_id,
        "token": credentials.get("token", ""),
        "base_url": credentials.get("base_url", ""),
        "user_id": credentials.get("user_id", ""),
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def cmd_weixin(args) -> int:
    """Dispatch ``hermes weixin <action>``."""
    action = getattr(args, "weixin_action", None)
    if action == "login":
        return _cmd_login(getattr(args, "account_id", "") or "")
    print_error("Usage: hermes weixin login [--account-id <id>]")
    return 2


def _cmd_login(account_id: str) -> int:
    try:
        from gateway.platforms.weixin import check_weixin_requirements, qr_login
    except Exception as exc:
        print_error(f"Weixin adapter import failed: {exc}")
        print_info("Install gateway dependencies first, then retry.")
        return 1

    if not check_weixin_requirements():
        print_error("Missing dependencies: Weixin needs aiohttp and cryptography.")
        print_info("Install them, then rerun this command.")
        return 1

    if not account_id:
        print_info(
            "No --account-id given; the returned iLink identity will be used "
            "as the account id."
        )

    print()
    print_info("Starting iLink QR login. Scan with the WeChat account to connect.")
    try:
        credentials = asyncio.run(qr_login(str(get_hermes_home())))
    except KeyboardInterrupt:
        print_warning("\nQR login cancelled.")
        return 1
    except Exception as exc:
        print_error(f"QR login failed: {exc}")
        return 1

    if not credentials:
        print_warning("QR login did not complete.")
        return 1

    resolved_account = str(
        account_id or credentials.get("account_id", "")
    ).strip()
    if not resolved_account:
        print_error(
            "Could not determine an account id. Re-run with "
            "--account-id <name>."
        )
        return 1

    path = _persist_account(resolved_account, credentials)
    print_success(f"Saved Weixin account '{resolved_account}' → {path}")
    print_info(
        "Restart the gateway to register it as platform "
        f"'weixin:{resolved_account}'."
    )
    print_info(
        f"Address it via send_message deliver='weixin:{resolved_account}' "
        "or as a cron delivery target."
    )
    return 0


def build_weixin_parser(subparsers, cmd_weixin=cmd_weixin) -> None:
    """Register the ``hermes weixin`` subcommand tree on ``subparsers``."""
    weixin_parser = subparsers.add_parser(
        "weixin",
        help="Weixin personal-account utilities (multi-account login)",
    )
    weixin_sub = weixin_parser.add_subparsers(dest="weixin_action")

    login_parser = weixin_sub.add_parser(
        "login",
        help="Run iLink QR login and persist an extra Weixin account",
    )
    login_parser.add_argument(
        "--account-id",
        default="",
        help="Stable identifier for this account (defaults to the iLink identity)",
    )
    weixin_parser.set_defaults(func=cmd_weixin)
    # Bare `hermes weixin` shows help.
    weixin_sub.required = False

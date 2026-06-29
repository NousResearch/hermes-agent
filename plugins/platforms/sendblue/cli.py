"""``hermes sendblue ...`` CLI subcommands for the Sendblue platform plugin."""

from __future__ import annotations

import argparse
import os
import sys

from .adapter import (
    DEFAULT_API_BASE_URL,
    DEFAULT_WEBHOOK_HOST,
    DEFAULT_WEBHOOK_PATH,
    DEFAULT_WEBHOOK_PORT,
    _default_sticky_state_path,
    _normalize_webhook_path,
    _split_csv,
    interactive_setup,
)


def register_cli(parser: argparse.ArgumentParser) -> None:
    """Wire up ``hermes sendblue ...`` subcommands."""
    subs = parser.add_subparsers(dest="sendblue_command", required=False)
    subs.add_parser("status", help="Show Sendblue configuration state")
    subs.add_parser("setup", help="Prompt for Sendblue environment variables")
    parser.set_defaults(func=dispatch)


def dispatch(args: argparse.Namespace) -> int:
    sub = getattr(args, "sendblue_command", None)
    if sub is None or sub == "status":
        return _cmd_status()
    if sub == "setup":
        interactive_setup()
        return 0
    print(f"unknown subcommand: {sub}", file=sys.stderr)
    return 2


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        from hermes_cli.config import get_env_var
    except ImportError:
        return default
    try:
        stored = get_env_var(name)
    except Exception:
        stored = None
    return str(stored or default).strip()


def _configured(name: str) -> str:
    return "configured" if _env(name) else "missing"


def _redact_phone(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value.startswith("group:"):
        return f"group:{value[6:12]}..."
    if len(value) <= 6:
        return "***"
    return f"{value[:3]}...{value[-4:]}"


def _redact_list(values: list[str]) -> str:
    return ", ".join(_redact_phone(v) for v in values) if values else "missing"


def _cmd_status() -> int:
    from_number = _env("SENDBLUE_FROM_NUMBER")
    from_numbers = _split_csv(_env("SENDBLUE_FROM_NUMBERS"))
    if from_number and from_number not in from_numbers:
        from_numbers.insert(0, from_number)

    webhook_host = _env("SENDBLUE_WEBHOOK_HOST", DEFAULT_WEBHOOK_HOST)
    webhook_port = _env("SENDBLUE_WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT))
    webhook_path = _normalize_webhook_path(
        _env("SENDBLUE_WEBHOOK_PATH", DEFAULT_WEBHOOK_PATH)
    )
    sticky_path = _env("SENDBLUE_STICKY_STATE_PATH", str(_default_sticky_state_path()))
    allow_all = _env("SENDBLUE_ALLOW_ALL_USERS").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    allowed_users = _split_csv(_env("SENDBLUE_ALLOWED_USERS"))
    access = (
        "allow-all"
        if allow_all
        else f"allowlist ({len(allowed_users)})"
        if allowed_users
        else "pairing required"
    )

    print("Sendblue")
    print(f"  API key id:      {_configured('SENDBLUE_API_KEY_ID')}")
    print(f"  API secret key:  {_configured('SENDBLUE_API_SECRET_KEY')}")
    print(f"  API base URL:    {_env('SENDBLUE_API_BASE_URL', DEFAULT_API_BASE_URL)}")
    print(f"  From numbers:    {_redact_list(from_numbers)}")
    print(f"  Webhook bind:    {webhook_host}:{webhook_port}{webhook_path}")
    print(f"  Webhook secret:  {_configured('SENDBLUE_WEBHOOK_SECRET')}")
    print(f"  Access:          {access}")
    print(
        f"  Home channel:    {_redact_phone(_env('SENDBLUE_HOME_CHANNEL')) or 'unset'}"
    )
    print(f"  Sticky state:    {sticky_path}")
    return 0

"""Targeted messaging setup, including noninteractive Telegram managed-bot pairing."""

from __future__ import annotations

import sys

from hermes_cli.config import get_env_value, require_env_writable, save_env_value, write_platform_config_field
from hermes_cli.gateway_setup_wizard import _gw
from hermes_cli.telegram_managed_bot import (
    DEFAULT_POLL_TIMEOUT, _api_url, create_pairing, is_valid_telegram_bot_token, poll_for_setup_result,
    print_qr_code,
)


def run_platform_setup(args) -> int:
    """Configure one platform, returning the command's exit status."""
    platforms = {p["key"]: p for p in _gw()._all_platforms()}
    if args.platform not in platforms:
        print(f"Unknown platform. Valid keys: {', '.join(platforms)}")
        return 2
    if args.quick:
        if args.platform != "telegram":
            print("--quick exists for Telegram only.")
            return 2
        return _quick_telegram(args)
    if not sys.stdin.isatty():
        print("`hermes gateway setup <platform>` needs a terminal. For Telegram use `--quick`. "
              "Never paste a bot token into a chat.")
        return 2
    _gw()._configure_platform(platforms[args.platform])
    _gw()._wizard_post_setup()
    return 0


def _quick_telegram(args) -> int:
    from gateway.relay.link import relay_channel_view

    if "telegram" in relay_channel_view()[0]:
        print("Telegram here comes through Team Gateway; link it in the desktop app instead.")
        return 1
    if get_env_value("TELEGRAM_BOT_TOKEN") and not args.replace:
        print("A Telegram bot is already configured. Use --replace to replace it.")
        return 1
    allowed = [uid.strip() for uid in args.allow.split(",")] if args.allow else []
    if any(not uid.isascii() or not uid.isdecimal() for uid in allowed):
        print("--allow must contain comma-separated numeric Telegram user IDs.")
        return 2
    try:  # a managed scope can pin these keys; save_env_value would refuse silently after the bot exists
        for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USERS"):
            require_env_writable(key, "set")
    except ValueError as exc:
        print(exc)
        return 1
    return _pair_telegram(DEFAULT_POLL_TIMEOUT if args.timeout is None else args.timeout, allowed)


def _pair_telegram(timeout: float, allowed: list[str]) -> int:
    pairing = create_pairing(_api_url(None))
    if not pairing:
        print("  ✗ Could not reach the Hermes Telegram onboarding service.\n"
              "    Try the manual setup instead, or check your network.")
        return 1
    print(f"Link: {pairing.deep_link}", flush=True)
    if sys.stdout.isatty():
        print_qr_code(pairing.qr_payload, include_link=False)
    result = poll_for_setup_result(None, pairing, timeout, on_tick=None)
    if result is None:
        print("Timed out waiting for Telegram bot creation. Re-run setup to try again.")
        return 1
    if not is_valid_telegram_bot_token(result.token):
        print("Automatic setup returned an invalid Telegram token.")
        return 1
    _save_telegram(result, allowed)
    connected = f"@{result.bot_username} connected." if result.bot_username else "Telegram bot connected."
    next_step = "restart" if _gw()._is_service_running() else "ensure"
    print(f"{connected} Next: hermes gateway {next_step}")
    return 0


def _save_telegram(result, allowed: list[str]) -> None:
    save_env_value("TELEGRAM_BOT_TOKEN", result.token)
    owner = str(result.owner_user_id) if result.owner_user_id else ""
    ids = list(dict.fromkeys(([owner] if owner else []) + allowed))
    if ids:
        save_env_value("TELEGRAM_ALLOWED_USERS", ",".join(ids))
    else:
        kept = "the existing allowlist stays in force" if get_env_value("TELEGRAM_ALLOWED_USERS") else (
            "with no other access rule the gateway denies everyone")
        from hermes_cli.profiles import get_active_profile_name
        explicit = getattr(sys.modules.get("hermes_cli.main"), "_explicit_cli_profile", None)  # -p NAME given
        profile = explicit or get_active_profile_name()
        flag = f"-p {profile} " if explicit or profile not in ("default", "custom") else ""
        print(f"No owner was detected, so no user was added; {kept}. "
              f"Set allowed user IDs with: hermes {flag}config set TELEGRAM_ALLOWED_USERS ID1,ID2")
    write_platform_config_field("telegram", "enabled", True)
    if owner and not get_env_value("TELEGRAM_HOME_CHANNEL"):
        save_env_value("TELEGRAM_HOME_CHANNEL", owner)

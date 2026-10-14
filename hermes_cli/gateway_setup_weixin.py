"""Profile-local Weixin login selection and editable gateway settings."""

from __future__ import annotations

from hermes_cli.gateway_setup_wizard import _gw


_WEIXIN_GROUP_NOTE = (
    "  QR login connects an iLink bot identity, not a scriptable personal WeChat account.",
    "  Group settings only apply if iLink delivers group events for your account type.",
)
_LEGACY_SETTINGS = (
    "WEIXIN_ACCOUNT_ID", "WEIXIN_BASE_URL", "WEIXIN_CDN_BASE_URL", "WEIXIN_DM_POLICY",
    "WEIXIN_ALLOW_ALL_USERS", "WEIXIN_ALLOWED_USERS", "WEIXIN_GROUP_POLICY", "WEIXIN_GROUP_ALLOWED_USERS",
    "WEIXIN_HOME_CHANNEL", "WEIXIN_HOME_CHANNEL_NAME", "WEIXIN_SPLIT_MULTILINE_MESSAGES",
)


def _weixin_config():
    from gateway.config import Platform, PlatformConfig, load_gateway_config
    return load_gateway_config().platforms.get(Platform.WEIXIN, PlatformConfig())


def _weixin_accounts(current):
    from gateway.platforms.weixin import ILINK_BASE_URL, list_weixin_accounts

    accounts = list_weixin_accounts(str(_gw().get_hermes_home()))
    account_id = str(current.extra.get("account_id") or "").strip()
    saved = next((account for account in accounts if account["account_id"] == account_id), {})
    token = str(current.token or current.extra.get("token") or saved.get("token") or "").strip()
    if account_id and token:
        # The configured account stays first even if another login was saved more recently.
        configured = {**saved, "account_id": account_id, "token": token,
                      "base_url": current.extra.get("base_url") or saved.get("base_url") or ILINK_BASE_URL}
        accounts = [configured, *(account for account in accounts if account["account_id"] != account_id)]
    return accounts


def _weixin_status() -> str:
    from gateway.platforms.weixin_group import has_weixin_credentials
    current = _weixin_config()
    accounts = _weixin_accounts(current)
    if has_weixin_credentials(current):
        return "configured"
    if accounts:
        return "login saved"
    return "partially configured" if current.token or current.extra.get("account_id") else "not configured"


def _select_weixin_credentials(current):
    from gateway.platforms.weixin import aiohttp, check_weixin_requirements, qr_login

    accounts = _weixin_accounts(current)
    if accounts:
        choices = [f"Use saved account: {account['account_id']}" for account in accounts]
        choices.extend(["Scan a new QR code to connect an account", "Cancel"])
        selected = _gw().prompt_choice("  Choose a Weixin login to configure", choices, 0)
        if selected < len(accounts):
            return accounts[selected]
        if selected == len(accounts) + 1:
            return None
    elif not _gw().prompt_yes_no("  Start QR login now?", True):
        return None
    if not check_weixin_requirements():
        _gw().print_error("  Weixin QR login needs aiohttp and cryptography. Install messaging dependencies, then retry.")
        return None
    try:
        return _gw().asyncio.run(qr_login(
            str(_gw().get_hermes_home()), bot_agent=current.extra.get("bot_agent"), route_tag=current.extra.get("route_tag")))
    except KeyboardInterrupt:
        _gw().print_warning("  Weixin setup cancelled.")
    except (aiohttp.ClientError, OSError, RuntimeError, ValueError) as exc:
        _gw().print_error(f"  QR login failed: {exc}")
    return None


def _prompt_weixin_allowlist(question: str, existing) -> list[str]:
    default = ",".join(str(value) for value in existing) if isinstance(existing, list) else str(existing or "")
    value = _gw().prompt(f"{question} (comma-separated; '-' to clear)", default, password=False)
    return [] if value == "-" else [item.strip() for item in value.split(",") if item.strip()]


def _prompt_weixin_access(current, user_id: str) -> dict:
    dm_policies = ("pairing", "open", "allowlist", "disabled")
    dm_choices = ["Use DM pairing approval (recommended)", "Allow all direct messages",
                  "Only allow listed user IDs", "Disable direct messages"]
    dm = str(current.extra.get("dm_policy") or "pairing").lower()
    default = dm_policies.index(dm) if dm in dm_policies else 0
    dm = dm_policies[_gw().prompt_choice("  How should direct messages be authorized?", dm_choices, default)]
    allow_from = _prompt_weixin_allowlist("  Allowed Weixin user IDs", current.extra.get("allow_from") or user_id) if dm == "allowlist" else []
    print()
    _gw()._print_info_lines(*_WEIXIN_GROUP_NOTE)
    group_policies = ("disabled", "open", "allowlist")
    group_choices = ["Disable group chats (recommended)", "Allow all group chats", "Only allow listed group chat IDs"]
    group = str(current.extra.get("group_policy") or "disabled").lower()
    default = group_policies.index(group) if group in group_policies else 0
    group = group_policies[_gw().prompt_choice("  How should group chats be handled?", group_choices, default)]
    group_allow = _prompt_weixin_allowlist("  Allowed group chat IDs", current.extra.get("group_allow_from")) if group == "allowlist" else []
    return {"dm_policy": dm, "allow_all_users": dm == "open", "allow_from": allow_from,
            "group_policy": group, "group_allow_from": group_allow}


def _prompt_weixin_home(current, user_id: str):
    existing = current.home_channel
    default = existing.chat_id if existing else user_id
    chat_id = _gw().prompt("  Home channel ID for notifications ('-' to clear)", default, password=False)
    if not chat_id or chat_id == "-":
        return None
    if existing and chat_id == existing.chat_id:
        return existing.to_dict()
    return {"platform": "weixin", "chat_id": chat_id, "name": "Home"}


def _save_weixin_settings(credentials: dict, settings: dict, home_channel, *, enabled=True, set_default=True, retired=()) -> None:
    from hermes_cli.config import remove_env_value, require_env_writable, save_config
    from gateway.platforms.weixin import save_weixin_account

    # Clear only the migrated settings, after YAML is durable; stale env overrides would undo edits.
    if set_default:
        require_env_writable("WEIXIN_TOKEN", "set")
    for name in _LEGACY_SETTINGS:
        if _gw().get_env_value(name) is not None:
            require_env_writable(name, "remove")
    raw = _gw().read_raw_config()
    platform = raw.setdefault("platforms", {}).setdefault("weixin", {})
    extra = platform.setdefault("extra", {})
    accounts = extra.setdefault("accounts", {})
    old_id = extra.get("account_id")
    for account_id in retired:
        accounts.pop(account_id, None)
    if old_id in retired:
        extra.pop("account_id", None)
        old_id = None
    if extra.get("default_account") in retired:
        extra.pop("default_account", None)
    if old_id and old_id not in accounts:
        accounts[old_id] = {key: value for key, value in extra.items()
                            if key not in {"accounts", "account_id", "default_account", "token"}}
        accounts[old_id]["home_channel"] = platform.get("home_channel")
    if old_id and old_id != credentials["account_id"]:
        old_credentials = next((row for row in _weixin_accounts(_weixin_config()) if row["account_id"] == old_id), None)
        if old_credentials:
            save_weixin_account(str(_gw().get_hermes_home()), account_id=old_id, token=old_credentials["token"],
                                base_url=old_credentials["base_url"], user_id=old_credentials.get("user_id", ""))
    account_id = credentials["account_id"]
    accounts[account_id] = {**accounts.get(account_id, {}), **settings, "enabled": enabled, "home_channel": home_channel}
    save_weixin_account(str(_gw().get_hermes_home()), account_id=account_id, token=credentials["token"],
                        base_url=settings.get("base_url") or credentials["base_url"], user_id=credentials.get("user_id", ""))
    if set_default:
        _gw().save_env_value("WEIXIN_TOKEN", credentials["token"])
        extra.update(settings)
        extra["account_id"] = extra["default_account"] = account_id
        platform["home_channel"] = home_channel
    platform["enabled"] = True
    save_config(raw, strip_defaults=False)
    for name in _LEGACY_SETTINGS:
        remove_env_value(name)


def _setup_weixin():
    from gateway.platforms.weixin import ILINK_BASE_URL, WEIXIN_CDN_BASE_URL, _coerce_bool

    _gw()._print_setup_header("💬 Weixin / WeChat")
    _gw()._print_info_lines(
        "  Reuse a saved login or scan a new QR code with WeChat.",
        "  Enter a verification code only if WeChat asks for one.",
        "  Multiple accounts can run together in this profile; tokens stay in its saved login files.",
    )
    current = _weixin_config()
    previous_accounts = _weixin_accounts(current)
    credentials = _select_weixin_credentials(current)
    if not credentials:
        _gw().print_info("  Weixin setup did not complete.")
        return
    retired = _retired_weixin_accounts(previous_accounts, credentials)
    enabled = _gw().prompt_yes_no("  Enable this account? (No disables its connection)", True)
    if not enabled:
        _save_weixin_settings(credentials, {}, None, enabled=False, set_default=False, retired=retired)
        _gw().print_success("Weixin account disabled; a running gateway will apply the change automatically.")
        return
    default_id = current.extra.get("default_account") or current.extra.get("account_id")
    set_default = _gw().prompt_yes_no(
        "  Use this account for notifications and sends without an account ID?",
        not default_id or default_id in retired or default_id == credentials["account_id"],
    )
    from gateway.platforms.weixin_group import account_configs
    from pathlib import Path
    selected = account_configs(current, Path(_gw().get_hermes_home())).get(credentials["account_id"])
    current = selected or current
    settings = _prompt_weixin_access(current, credentials.get("user_id", ""))
    settings["base_url"] = _gw().prompt("  iLink API base URL", credentials.get("base_url") or ILINK_BASE_URL, password=False).rstrip("/")
    settings["cdn_base_url"] = _gw().prompt("  Media CDN base URL", current.extra.get("cdn_base_url") or WEIXIN_CDN_BASE_URL, password=False).rstrip("/")
    settings["split_multiline_messages"] = _gw().prompt_yes_no(
        "  Split multi-line replies into separate messages?",
        _coerce_bool(current.extra.get("split_multiline_messages"), default=False),
    )
    settings["use_platform_transcription"] = _gw().prompt_yes_no(
        "  Use WeChat's voice transcription when available (no separate STT needed)?",
        _coerce_bool(current.extra.get("use_platform_transcription"), default=True),
    )
    settings["reply_progress_messages"] = _gw().prompt_yes_no(
        "  Show native tool-call progress in WeChat?",
        _coerce_bool(current.extra.get("reply_progress_messages"), default=True),
    )
    quote_cache = current.extra.get("quote_cache") or {}
    settings["quote_cache"] = {**quote_cache, "enabled": _gw().prompt_yes_no(
        "  Cache messages and attachments to restore quoted replies?",
        _coerce_bool(quote_cache.get("enabled"), default=True),
    )}
    home_channel = _prompt_weixin_home(current, credentials.get("user_id", ""))
    _save_weixin_settings(credentials, settings, home_channel, set_default=set_default, retired=retired)
    _gw().print_success("Weixin configured!")
    _gw().print_info(f"  Account ID: {credentials['account_id']}")
    _gw().print_info("  Run `hermes gateway setup` again to add, edit or disable accounts. A running gateway applies changes automatically after active replies finish.")


def _retired_weixin_accounts(previous_accounts, credentials):
    from gateway.platforms.weixin import list_weixin_accounts
    if not credentials.get("user_id"):
        return []
    current_ids = {row["account_id"] for row in list_weixin_accounts(str(_gw().get_hermes_home()))}
    return [row["account_id"] for row in previous_accounts
            if row.get("user_id") == credentials["user_id"] and row["account_id"] != credentials["account_id"]
            and row["account_id"] not in current_ids]

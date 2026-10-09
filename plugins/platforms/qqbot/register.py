"""QQ Bot platform plugin: env/YAML seeding, interactive setup, and registry entry.

The adapter stays in ``adapter.py``. This module is the only thing ``__init__``
exports, so importing the package registers the platform and does not open a portal
connection.
"""

from __future__ import annotations

from typing import Any, Optional

from gateway.platforms._shared import apply_yaml_bridge, get_scoped_secret

from .adapter import QQAdapter, check_qq_requirements
from .constants import MAX_MESSAGE_LENGTH

_PLATFORM_HINT = (
    "You are on QQ, a popular Chinese messaging platform. QQ supports markdown formatting "
    "and emoji. You can send media files natively: include MEDIA:/absolute/path/to/file in "
    "your response. Images are sent as native photos, and other files arrive as downloadable documents."
)

_YAML_BRIDGE = (
    ("app_id", "QQ_APP_ID", "str"),
    ("client_secret", "QQ_CLIENT_SECRET", "str"),
    ("allow_from", "QQ_ALLOWED_USERS", "csv"),
    ("group_allow_from", "QQ_GROUP_ALLOWED_USERS", "csv"),
    ("markdown_support", "QQ_MARKDOWN_SUPPORT", "lower"),
)
_YAML_EXTRA_KEYS = ("dm_policy", "group_policy", "group_allowed_chats", "stt", "sandbox")


def _scoped(name: str) -> str:
    return str(get_scoped_secret(name, "") or "").strip()


def _truthy(raw: str) -> bool:
    return raw.lower() in {"1", "true", "yes", "on"}


def _credentials_configured(extra: Optional[dict] = None) -> bool:
    extra = extra or {}
    app_id = str(extra.get("app_id") or "").strip() or _scoped("QQ_APP_ID")
    secret = str(extra.get("client_secret") or "").strip() or _scoped("QQ_CLIENT_SECRET")
    return bool(app_id and secret)


def env_enablement() -> Optional[dict]:
    """Seed ``platforms.qqbot.extra`` from the profile's QQ env. ``None`` when the bot
    is not configured, so importing the plugin never enables a connect loop."""
    if not _credentials_configured():
        return None
    seed: dict[str, Any] = {"app_id": _scoped("QQ_APP_ID"), "client_secret": _scoped("QQ_CLIENT_SECRET")}
    if allow := _scoped("QQ_ALLOWED_USERS"):
        seed["allow_from"] = allow
    if groups := _scoped("QQ_GROUP_ALLOWED_USERS"):
        seed["group_allow_from"] = groups
    if markdown := _scoped("QQ_MARKDOWN_SUPPORT"):
        seed["markdown_support"] = _truthy(markdown)
    chat_id = _scoped("QQBOT_HOME_CHANNEL") or _scoped("QQ_HOME_CHANNEL")
    name = _scoped("QQBOT_HOME_CHANNEL_NAME") or _scoped("QQ_HOME_CHANNEL_NAME")
    thread_id = _scoped("QQBOT_HOME_CHANNEL_THREAD_ID") or _scoped("QQ_HOME_CHANNEL_THREAD_ID")
    if chat_id:
        home = {"chat_id": chat_id, "name": name or "Home"}
        if thread_id:
            home["thread_id"] = thread_id
        seed["home_channel"] = home
    return seed


def apply_yaml_config(_yaml_cfg: dict, platform_cfg: dict) -> Optional[dict]:
    if not isinstance(platform_cfg, dict):
        return None
    seeded = apply_yaml_bridge(platform_cfg, _YAML_BRIDGE) or {}
    for key in _YAML_EXTRA_KEYS:
        if key in platform_cfg:
            seeded[key] = platform_cfg[key]
    return seeded or None


def is_connected(pconfig) -> bool:
    extra = getattr(pconfig, "extra", None) or {}
    return _credentials_configured(extra)


async def standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None,
                          force_document=False, caption=None):
    """Late-import the REST sender so cron delivery works without a live gateway process."""
    from .send import send_qqbot
    return await send_qqbot(
        pconfig, chat_id, message, media_files=media_files, caption=caption,
        thread_id=thread_id, force_document=force_document,
    )


def setup_qqbot() -> None:
    """Interactive setup: scan-to-configure or manual App ID / App Secret."""
    from hermes_cli.gateway_setup_wizard import (
        _confirm_reconfigure, _gw, _offer_home_channel, _print_setup_header, _prompt_csv, _save_env_values,
    )

    _print_setup_header("🐧 QQ Bot")
    if not _confirm_reconfigure("QQ Bot", "QQ_APP_ID", "QQ_CLIENT_SECRET"):
        return

    print()
    method_choices = [
        "Scan QR code to add bot automatically (recommended)",
        "Enter existing App ID and App Secret manually",
    ]
    credentials = None
    if _gw().prompt_choice("  How would you like to set up QQ Bot?", method_choices, 0) == 0:
        try:
            from .onboard import qr_register
            credentials = qr_register()
        except KeyboardInterrupt:
            print()
            _gw().print_warning("  QQ Bot setup cancelled.")
            return
        if not credentials:
            _gw().print_info("  QR setup did not complete. Continuing with manual input.")

    if not credentials:
        print()
        _gw()._print_info_lines(
            "  Go to https://q.qq.com to register a QQ Bot application.",
            "  Note your App ID and App Secret from the application page.",
        )
        print()
        app_id = _gw().prompt("  App ID", password=False)
        if not app_id:
            _gw().print_warning("  Skipped — QQ Bot won't work without an App ID.")
            return
        app_secret = _gw().prompt("  App Secret", password=True)
        if not app_secret:
            _gw().print_warning("  Skipped — QQ Bot won't work without an App Secret.")
            return
        credentials = {"app_id": app_id.strip(), "client_secret": app_secret.strip(), "user_openid": ""}

    _gw().save_env_value("QQ_APP_ID", credentials["app_id"])
    _gw().save_env_value("QQ_CLIENT_SECRET", credentials["client_secret"])
    user_openid = credentials.get("user_openid", "")

    print()
    access_choices = [
        "Use DM pairing approval (recommended)",
        "Allow all direct messages",
        "Only allow listed user OpenIDs",
    ]
    access_idx = _gw().prompt_choice("  How should direct messages be authorized?", access_choices, 0)
    if access_idx == 0:
        _gw().save_env_value("QQ_ALLOW_ALL_USERS", "false")
        allowed = ""
        if user_openid:
            print()
            if _gw().prompt_yes_no(f"  Add yourself ({user_openid}) to the allow list?", True):
                allowed = user_openid
                _gw().print_success(f"  Allow list set to {user_openid}")
        _gw().save_env_value("QQ_ALLOWED_USERS", allowed)
        _gw().print_success("  DM pairing enabled.")
        _gw().print_info("  Unknown users can request access; approve with `hermes pairing approve`.")
    elif access_idx == 1:
        _save_env_values(QQ_ALLOW_ALL_USERS="true", QQ_ALLOWED_USERS="")
        _gw().print_warning("  Open DM access enabled for QQ Bot.")
    else:
        allowlist = _prompt_csv("  Allowed user OpenIDs (comma-separated)", user_openid or "")
        _save_env_values(QQ_ALLOW_ALL_USERS="false", QQ_ALLOWED_USERS=allowlist)
        _gw().print_success("  Allowlist saved.")

    print()
    if user_openid:
        _offer_home_channel("QQBOT_HOME_CHANNEL", user_openid, "your QQ user ID")
    else:
        home_channel = _gw().prompt("  Home channel OpenID (for cron/notifications, or empty)", password=False)
        if home_channel:
            _gw().save_env_value("QQBOT_HOME_CHANNEL", home_channel.strip())
            _gw().print_success(f"  Home channel set to {home_channel.strip()}")

    print()
    _gw().print_success("🐧 QQ Bot configured!")
    _gw().print_info(f"  App ID: {credentials['app_id']}")


def register(ctx) -> None:
    ctx.register_platform(
        name="qqbot",
        label="QQ Bot",
        adapter_factory=QQAdapter,
        check_fn=check_qq_requirements,
        validate_config=is_connected,
        is_connected=is_connected,
        required_env=["QQ_APP_ID", "QQ_CLIENT_SECRET"],
        install_hint="aiohttp and httpx are required. Run: hermes pm repair",
        setup_fn=setup_qqbot,
        allowed_users_env="QQ_ALLOWED_USERS",
        allow_all_env="QQ_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🐧",
        platform_hint=_PLATFORM_HINT,
        env_enablement_fn=env_enablement,
        apply_yaml_config_fn=apply_yaml_config,
        cron_deliver_env_var="QQBOT_HOME_CHANNEL",
        standalone_sender_fn=standalone_send,
    )

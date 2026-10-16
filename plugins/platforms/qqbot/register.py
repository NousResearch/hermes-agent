"""QQ Bot platform plugin: interactive setup and registry entry.

The adapter stays in ``adapter.py``. This module is the only thing ``__init__``
exports, so importing the package registers the platform and does not open a portal
connection.
"""

from __future__ import annotations

from gateway.platforms._shared import env_is_connected

from .adapter import QQAdapter, check_qq_requirements


_env_configured = env_is_connected("QQ_APP_ID", "QQ_CLIENT_SECRET")


def is_connected(pconfig) -> bool:
    """Both credentials from ``extra`` (config.yaml) or the profile env / ``.env``."""
    extra = getattr(pconfig, "extra", None) or {}
    if str(extra.get("app_id") or "").strip() and str(extra.get("client_secret") or "").strip():
        return True
    return _env_configured(pconfig)


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
        is_connected=is_connected,
        required_env=["QQ_APP_ID", "QQ_CLIENT_SECRET"],
        install_hint="QQBot: aiohttp/httpx missing or QQ_APP_ID/QQ_CLIENT_SECRET not configured",
        setup_fn=setup_qqbot,
        allowed_users_env="QQ_ALLOWED_USERS",
        allow_all_env="QQ_ALLOW_ALL_USERS",
        emoji="🐧",
        cron_deliver_env_var="QQBOT_HOME_CHANNEL",
        standalone_sender_fn=standalone_send,
    )

"""QQ Bot platform plugin: interactive setup and registry entry.

The adapter stays in ``adapter.py``. This module is the only thing ``__init__``
exports, so importing the package registers the platform and does not open a portal
connection.
"""

from __future__ import annotations

from gateway.platforms._shared import env_is_connected
from gateway.platforms._shared import config_connected_extra
from gateway.platforms.send_targets import StandaloneSendPolicy

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
    from hermes_cli.cli_output import print_info, print_success, print_warning
    from hermes_cli.config import save_env_value
    from hermes_cli.gateway_setup_wizard import (
        confirm_reconfigure, offer_home_channel, print_setup_header, prompt_csv, save_env_values,
    )
    from hermes_cli.setup import prompt, prompt_choice, prompt_yes_no

    print_setup_header("🐧 QQ Bot")
    if not confirm_reconfigure("QQ Bot", "QQ_APP_ID", "QQ_CLIENT_SECRET"):
        return

    print()
    method_choices = [
        "Scan QR code to add bot automatically (recommended)",
        "Enter existing App ID and App Secret manually",
    ]
    credentials = None
    if prompt_choice("  How would you like to set up QQ Bot?", method_choices, 0) == 0:
        try:
            from .onboard import qr_register
            credentials = qr_register()
        except KeyboardInterrupt:
            print()
            print_warning("  QQ Bot setup cancelled.")
            return
        if not credentials:
            print_info("  QR setup did not complete. Continuing with manual input.")

    if not credentials:
        print()
        print_info("  Go to https://q.qq.com to register a QQ Bot application.")
        print_info("  Note your App ID and App Secret from the application page.")
        print()
        app_id = prompt("  App ID", password=False)
        if not app_id:
            print_warning("  Skipped — QQ Bot won't work without an App ID.")
            return
        app_secret = prompt("  App Secret", password=True)
        if not app_secret:
            print_warning("  Skipped — QQ Bot won't work without an App Secret.")
            return
        credentials = {"app_id": app_id.strip(), "client_secret": app_secret.strip(), "user_openid": ""}

    save_env_value("QQ_APP_ID", credentials["app_id"])
    save_env_value("QQ_CLIENT_SECRET", credentials["client_secret"])
    user_openid = credentials.get("user_openid", "")

    print()
    access_choices = [
        "Use DM pairing approval (recommended)",
        "Allow all direct messages",
        "Only allow listed user OpenIDs",
    ]
    access_idx = prompt_choice("  How should direct messages be authorized?", access_choices, 0)
    if access_idx == 0:
        save_env_value("QQ_ALLOW_ALL_USERS", "false")
        allowed = ""
        if user_openid:
            print()
            if prompt_yes_no(f"  Add yourself ({user_openid}) to the allow list?", True):
                allowed = user_openid
                print_success(f"  Allow list set to {user_openid}")
        save_env_value("QQ_ALLOWED_USERS", allowed)
        print_success("  DM pairing enabled.")
        print_info("  Unknown users can request access; approve with `hermes pairing approve`.")
    elif access_idx == 1:
        save_env_values(QQ_ALLOW_ALL_USERS="true", QQ_ALLOWED_USERS="")
        print_warning("  Open DM access enabled for QQ Bot.")
    else:
        allowlist = prompt_csv("  Allowed user OpenIDs (comma-separated)", user_openid or "")
        save_env_values(QQ_ALLOW_ALL_USERS="false", QQ_ALLOWED_USERS=allowlist)
        print_success("  Allowlist saved.")

    print()
    if user_openid:
        offer_home_channel("QQBOT_HOME_CHANNEL", user_openid, "your QQ user ID")
    else:
        home_channel = prompt("  Home channel OpenID (for cron/notifications, or empty)", password=False)
        if home_channel:
            save_env_value("QQBOT_HOME_CHANNEL", home_channel.strip())
            print_success(f"  Home channel set to {home_channel.strip()}")

    print()
    print_success("🐧 QQ Bot configured!")
    print_info(f"  App ID: {credentials['app_id']}")


# System-prompt platform hint (PlatformEntry.platform_hint); cached-prompt bytes, keep stable.
_PLATFORM_HINT = (
    "You are on QQ, a popular Chinese messaging platform. QQ supports markdown formatting "
    "and emoji. You can send media files natively: include MEDIA:/absolute/path/to/file in "
    "your response. Images are sent as native photos, and other files arrive as downloadable documents."
)


def register(ctx) -> None:
    from . import gateway_env  # env -> GatewayConfig bridge (gateway.config_env)
    ctx.register_platform(
        standalone_send=StandaloneSendPolicy(label="QQ Bot", media=True, captionable=True, text_direct=True),
        name="qqbot",
        dashboard_description="Connect Hermes to a QQ Bot from the QQ Open Platform.",
        docs_url="https://q.qq.com",
        dashboard_env_vars=("QQ_APP_ID", "QQ_CLIENT_SECRET", "QQ_ALLOWED_USERS"),
        label="QQ Bot",
        adapter_factory=QQAdapter,
        check_fn=check_qq_requirements,
        is_connected=is_connected,
        config_connected_fn=config_connected_extra("app_id", "client_secret"),
        required_env=["QQ_APP_ID", "QQ_CLIENT_SECRET"],
        install_hint="QQBot: aiohttp/httpx missing or QQ_APP_ID/QQ_CLIENT_SECRET not configured",
        setup_fn=setup_qqbot,
        emoji="🐧",
        cron_deliver_env_var="QQBOT_HOME_CHANNEL",
        standalone_sender_fn=standalone_send,
        # QQ's dm/group policy is YAML-only (no env var); QQ_ALLOW_ALL_USERS is still the opt-in.
        platform_hint=_PLATFORM_HINT,
        env_overrides=gateway_env.ENV_OVERRIDES,
        env_enable_credentials=gateway_env.ENV_ENABLE_CREDENTIALS, **gateway_env.AUTHZ,
    )

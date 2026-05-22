"""
Adapter factory — create the appropriate platform adapter for a given platform.

Extracted from gateway/run.py to decouple adapter creation from GatewayRunner.
"""

import logging
import os
from typing import Any, Optional

from gateway.platforms.base import BasePlatformAdapter
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)


def create_adapter(
    platform: Any,
    config: Any,
    *,
    group_sessions_per_user: bool = True,
    thread_sessions_per_user: bool = False,
    gateway_runner: Optional[Any] = None,
) -> Optional[BasePlatformAdapter]:
    """Create the appropriate adapter for a platform.

    Checks the platform_registry first (plugin adapters), then falls
    through to the built-in if/elif chain for core platforms.

    Args:
        platform: The Platform enum value.
        config: Platform configuration object.
        group_sessions_per_user: Global group sessions-per-user setting.
        thread_sessions_per_user: Global thread sessions-per-user setting.
        gateway_runner: Optional GatewayRunner instance — needed by some
            adapters (Discord, Webhook) for cross-platform callbacks.
    """
    if hasattr(config, "extra") and isinstance(config.extra, dict):
        config.extra.setdefault("group_sessions_per_user", group_sessions_per_user)
        config.extra.setdefault("thread_sessions_per_user", thread_sessions_per_user)

    # ── Plugin-registered platforms (checked first) ───────────────────
    try:
        from gateway.platform_registry import platform_registry

        if platform_registry.is_registered(platform.value):
            adapter = platform_registry.create_adapter(platform.value, config)
            if adapter is not None:
                return adapter
            # Registered but failed to instantiate — don't silently fall
            # through to built-ins (there are none for plugin platforms).
            logger.error(
                "Platform '%s' is registered but adapter creation failed "
                "(check dependencies and config)",
                platform.value,
            )
            return None
    except Exception as e:
        logger.debug(
            "Platform registry lookup for '%s' failed: %s", platform.value, e
        )
    # Fall through to built-in adapters below

    if platform.value == "telegram":
        from gateway.platforms.telegram import TelegramAdapter, check_telegram_requirements

        if not check_telegram_requirements():
            logger.warning("Telegram: python-telegram-bot not installed")
            return None
        adapter = TelegramAdapter(config)
        # Apply Telegram notification mode from config.  Controls whether
        # intermediate messages (tool progress, streaming, status) trigger
        # push notifications.  Supports ENV override for quick testing.
        _notify_mode = os.getenv("HERMES_TELEGRAM_NOTIFICATIONS", "")
        if not _notify_mode:
            try:
                from gateway.run import _load_gateway_config

                _gw_cfg = _load_gateway_config()
                _raw = cfg_get(
                    _gw_cfg, "display", "platforms", "telegram", "notifications"
                )
                if _raw not in {None, ""}:
                    _notify_mode = str(_raw).strip().lower()
            except Exception:
                pass
        _notify_mode = _notify_mode or "important"
        if _notify_mode not in {"all", "important"}:
            logger.warning(
                "Unknown telegram notifications mode '%s', "
                "defaulting to 'important' (valid: all, important)",
                _notify_mode,
            )
            _notify_mode = "important"
        adapter._notifications_mode = _notify_mode
        return adapter

    elif platform.value == "discord":
        from gateway.platforms.discord import DiscordAdapter, check_discord_requirements

        if not check_discord_requirements():
            logger.warning("Discord: discord.py not installed")
            return None
        adapter = DiscordAdapter(config)
        if gateway_runner is not None:
            adapter.gateway_runner = gateway_runner
        return adapter

    elif platform.value == "whatsapp":
        from gateway.platforms.whatsapp import WhatsAppAdapter, check_whatsapp_requirements

        if not check_whatsapp_requirements():
            logger.warning("WhatsApp: Node.js not installed or bridge not configured")
            return None
        return WhatsAppAdapter(config)

    elif platform.value == "slack":
        from gateway.platforms.slack import SlackAdapter, check_slack_requirements

        if not check_slack_requirements():
            logger.warning(
                "Slack: slack-bolt not installed. Run: pip install 'hermes-agent[slack]'"
            )
            return None
        return SlackAdapter(config)

    elif platform.value == "signal":
        from gateway.platforms.signal import SignalAdapter, check_signal_requirements

        if not check_signal_requirements():
            logger.warning(
                "Signal: SIGNAL_HTTP_URL or SIGNAL_ACCOUNT not configured"
            )
            return None
        return SignalAdapter(config)

    elif platform.value == "homeassistant":
        from gateway.platforms.homeassistant import (
            HomeAssistantAdapter,
            check_ha_requirements,
        )

        if not check_ha_requirements():
            logger.warning(
                "HomeAssistant: aiohttp not installed or HASS_TOKEN not set"
            )
            return None
        return HomeAssistantAdapter(config)

    elif platform.value == "email":
        from gateway.platforms.email import EmailAdapter, check_email_requirements

        if not check_email_requirements():
            logger.warning(
                "Email: EMAIL_ADDRESS, EMAIL_PASSWORD, "
                "EMAIL_IMAP_HOST, or EMAIL_SMTP_HOST not set"
            )
            return None
        return EmailAdapter(config)

    elif platform.value == "sms":
        from gateway.platforms.sms import SmsAdapter, check_sms_requirements

        if not check_sms_requirements():
            logger.warning(
                "SMS: aiohttp not installed or "
                "TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN not set"
            )
            return None
        return SmsAdapter(config)

    elif platform.value == "dingtalk":
        from gateway.platforms.dingtalk import (
            DingTalkAdapter,
            check_dingtalk_requirements,
        )

        if not check_dingtalk_requirements():
            logger.warning(
                "DingTalk: dingtalk-stream not installed or "
                "DINGTALK_CLIENT_ID/SECRET not set"
            )
            return None
        return DingTalkAdapter(config)

    elif platform.value == "feishu":
        from gateway.platforms.feishu import FeishuAdapter, check_feishu_requirements

        if not check_feishu_requirements():
            logger.warning(
                "Feishu: lark-oapi not installed or "
                "FEISHU_APP_ID/SECRET not set"
            )
            return None
        return FeishuAdapter(config)

    elif platform.value == "wecom_callback":
        from gateway.platforms.wecom_callback import (
            WecomCallbackAdapter,
            check_wecom_callback_requirements,
        )

        if not check_wecom_callback_requirements():
            logger.warning("WeComCallback: aiohttp/httpx not installed")
            return None
        return WecomCallbackAdapter(config)

    elif platform.value == "wecom":
        from gateway.platforms.wecom import WeComAdapter, check_wecom_requirements

        if not check_wecom_requirements():
            logger.warning(
                "WeCom: aiohttp not installed or "
                "WECOM_BOT_ID/SECRET not set"
            )
            return None
        return WeComAdapter(config)

    elif platform.value == "weixin":
        from gateway.platforms.weixin import WeixinAdapter, check_weixin_requirements

        if not check_weixin_requirements():
            logger.warning("Weixin: aiohttp/cryptography not installed")
            return None
        return WeixinAdapter(config)

    elif platform.value == "mattermost":
        from gateway.platforms.mattermost import (
            MattermostAdapter,
            check_mattermost_requirements,
        )

        if not check_mattermost_requirements():
            logger.warning(
                "Mattermost: MATTERMOST_TOKEN or MATTERMOST_URL "
                "not set, or aiohttp missing"
            )
            return None
        return MattermostAdapter(config)

    elif platform.value == "matrix":
        from gateway.platforms.matrix import MatrixAdapter, check_matrix_requirements

        if not check_matrix_requirements():
            logger.warning(
                "Matrix: mautrix not installed or credentials not set. "
                "Run: pip install 'mautrix[encryption]'"
            )
            return None
        return MatrixAdapter(config)

    elif platform.value == "api_server":
        from gateway.platforms.api_server import (
            APIServerAdapter,
            check_api_server_requirements,
        )

        if not check_api_server_requirements():
            logger.warning("API Server: aiohttp not installed")
            return None
        return APIServerAdapter(config)

    elif platform.value == "webhook":
        from gateway.platforms.webhook import WebhookAdapter, check_webhook_requirements

        if not check_webhook_requirements():
            logger.warning("Webhook: aiohttp not installed")
            return None
        adapter = WebhookAdapter(config)
        if gateway_runner is not None:
            adapter.gateway_runner = gateway_runner
        return adapter

    elif platform.value == "msgraph_webhook":
        from gateway.platforms.msgraph_webhook import (
            MSGraphWebhookAdapter,
            check_msgraph_webhook_requirements,
        )

        if not check_msgraph_webhook_requirements():
            logger.warning("MSGraph webhook: aiohttp not installed")
            return None
        return MSGraphWebhookAdapter(config)

    elif platform.value == "bluebubbles":
        from gateway.platforms.bluebubbles import (
            BlueBubblesAdapter,
            check_bluebubbles_requirements,
        )

        if not check_bluebubbles_requirements():
            logger.warning(
                "BlueBubbles: aiohttp/httpx missing or "
                "BLUEBUBBLES_SERVER_URL/BLUEBUBBLES_PASSWORD not configured"
            )
            return None
        return BlueBubblesAdapter(config)

    elif platform.value == "qqbot":
        from gateway.platforms.qqbot import QQAdapter, check_qq_requirements

        if not check_qq_requirements():
            logger.warning(
                "QQBot: aiohttp/httpx missing or "
                "QQ_APP_ID/QQ_CLIENT_SECRET not configured"
            )
            return None
        return QQAdapter(config)

    elif platform.value == "yuanbao":
        from gateway.platforms.yuanbao import YuanbaoAdapter, WEBSOCKETS_AVAILABLE

        if not WEBSOCKETS_AVAILABLE:
            logger.warning("Yuanbao: websockets not installed. Run: pip install websockets")
            return None
        return YuanbaoAdapter(config)

    return None

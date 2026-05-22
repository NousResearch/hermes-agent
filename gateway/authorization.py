"""Gateway user authorization — extracted from gateway/run.py.

Validates user access against platform allowlists, DM pairing, and
global allow-all flags.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def is_user_authorized(
    source: Any,
    config: Dict[str, Any],
    pairing: Any = None,
) -> bool:
    """Check if a user is authorized to use the bot.

    Checks in order:
    1. Per-platform allow-all flag
    2. Environment variable allowlists
    3. DM pairing approved list
    4. Global allow-all
    5. Default: deny
    """
    Platform = _get_platform_enum()

    # Home Assistant and Webhook events are always authorized
    if getattr(source, "platform", None) in {Platform.HOMEASSISTANT, Platform.WEBHOOK}:
        return True

    user_id = getattr(source, "user_id", None)
    chat_type = getattr(source, "chat_type", "")
    chat_id = getattr(source, "chat_id", None)
    platform = getattr(source, "platform", None)

    # Group/forum/channel allowlists
    if chat_type in {"group", "forum", "channel"} and chat_id:
        chat_allowlist_env = {
            Platform.TELEGRAM: "TELEGRAM_GROUP_ALLOWED_CHATS",
            Platform.QQBOT: "QQ_GROUP_ALLOWED_USERS",
        }.get(platform, "")
        if chat_allowlist_env:
            raw_chat_allowlist = os.getenv(chat_allowlist_env, "").strip()
            if raw_chat_allowlist:
                allowed_ids = {cid.strip() for cid in raw_chat_allowlist.split(",") if cid.strip()}
                if "*" in allowed_ids or chat_id in allowed_ids:
                    return True

    if not user_id:
        return False

    # Per-platform user allowlists
    platform_env_map = {
        getattr(Platform, "TELEGRAM", None): "TELEGRAM_ALLOWED_USERS",
        getattr(Platform, "DISCORD", None): "DISCORD_ALLOWED_USERS",
        getattr(Platform, "WHATSAPP", None): "WHATSAPP_ALLOWED_USERS",
        getattr(Platform, "SLACK", None): "SLACK_ALLOWED_USERS",
        getattr(Platform, "SIGNAL", None): "SIGNAL_ALLOWED_USERS",
        getattr(Platform, "EMAIL", None): "EMAIL_ALLOWED_USERS",
        getattr(Platform, "SMS", None): "SMS_ALLOWED_USERS",
        getattr(Platform, "MATTERMOST", None): "MATTERMOST_ALLOWED_USERS",
        getattr(Platform, "MATRIX", None): "MATRIX_ALLOWED_USERS",
        getattr(Platform, "DINGTALK", None): "DINGTALK_ALLOWED_USERS",
        getattr(Platform, "FEISHU", None): "FEISHU_ALLOWED_USERS",
        getattr(Platform, "WECOM", None): "WECOM_ALLOWED_USERS",
        getattr(Platform, "WECOM_CALLBACK", None): "WECOM_CALLBACK_ALLOWED_USERS",
        getattr(Platform, "WEIXIN", None): "WEIXIN_ALLOWED_USERS",
        getattr(Platform, "BLUEBUBBLES", None): "BLUEBUBBLES_ALLOWED_USERS",
        getattr(Platform, "QQBOT", None): "QQ_ALLOWED_USERS",
        getattr(Platform, "YUANBAO", None): "YUANBAO_ALLOWED_USERS",
    }
    platform_env = platform_env_map.get(platform)
    if platform_env:
        raw_allowlist = os.getenv(platform_env, "").strip()
        if raw_allowlist:
            allowed = {uid.strip() for uid in raw_allowlist.split(",") if uid.strip()}
            if "*" in allowed or user_id in allowed:
                return True
        return False  # Env var set but user not in it = denied

    # Group user allowlists
    platform_group_user_map = {
        Platform.DISCORD: "DISCORD_GROUP_ALLOWED_USERS",
        Platform.TELEGRAM: "TELEGRAM_GROUP_ALLOWED_USERS",
        Platform.QQBOT: "QQ_GROUP_ALLOWED_USERS",
    }
    group_env = platform_group_user_map.get(platform)
    if group_env and chat_type in {"group", "forum", "channel"}:
        raw_group_allowlist = os.getenv(group_env, "").strip()
        if raw_group_allowlist:
            allowed = {uid.strip() for uid in raw_group_allowlist.split(",") if uid.strip()}
            if "*" in allowed or user_id in allowed:
                return True
            return False

    # DM pairing check
    if pairing is not None:
        try:
            user_id_str = str(user_id or "")
            platform_str = str(getattr(platform, "value", platform)) if platform else ""
            if hasattr(pairing, "is_approved") and pairing.is_approved(platform_str, user_id_str):
                return True
        except Exception:
            pass

    # Global allow-all
    if os.getenv("GATEWAY_ALLOW_ALL_USERS", "").strip().lower() in ("1", "true", "yes"):
        return True

    return False


def get_unauthorized_dm_behavior(platform: Any) -> str:
    """Get the configured behavior for unauthorized DM access.

    Returns one of: "ignore" (default), "reply", "pair_request".
    """
    # Platform-specific override
    platform_key = str(getattr(platform, "value", platform)) if platform else ""
    env_key = f"{platform_key.upper()}_UNAUTHORIZED_DM_BEHAVIOR"
    raw = os.getenv(env_key, "").strip().lower()
    if raw in ("reply", "ignore", "pair_request", "pair"):
        return "reply" if raw == "pair" else raw

    # Global default
    raw = os.getenv("GATEWAY_UNAUTHORIZED_DM_BEHAVIOR", "ignore").strip().lower()
    if raw in ("reply", "ignore", "pair_request", "pair"):
        return "reply" if raw == "pair" else raw
    return "ignore"


def _get_platform_enum():
    """Lazy import to avoid circular dependency."""
    try:
        from gateway.platform_registry import Platform
        return Platform
    except ImportError:
        # Fallback: create a simple namespace
        class _P:
            TELEGRAM = "telegram"
            DISCORD = "discord"
            WHATSAPP = "whatsapp"
            SLACK = "slack"
            SIGNAL = "signal"
            EMAIL = "email"
            SMS = "sms"
            MATTERMOST = "mattermost"
            MATRIX = "matrix"
            DINGTALK = "dingtalk"
            FEISHU = "feishu"
            WECOM = "wecom"
            WECOM_CALLBACK = "wecom_callback"
            WEIXIN = "weixin"
            BLUEBUBBLES = "bluebubbles"
            QQBOT = "qqbot"
            YUANBAO = "yuanbao"
            HOMEASSISTANT = "homeassistant"
            WEBHOOK = "webhook"
            LOCAL = "local"
        return _P()

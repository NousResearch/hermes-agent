"""Env cleanup mappings: feature name → list of env vars.

Used to remove stale env vars from ~/.hermes/.env when a user disables a
platform or tool. This prevents the "zombie resurrection" bug where env vars
left behind silently re-enable a feature on the next gateway restart.
"""

from typing import Dict, List

# Mapping of platform names to env vars that configure them.
# Derived from gateway/config.py::_apply_env_overrides() and platform adapters.
PLATFORM_ENV_VARS: Dict[str, List[str]] = {
    "slack": [
        "SLACK_BOT_TOKEN",
        "SLACK_APP_TOKEN",
        "SLACK_HOME_CHANNEL",
        "SLACK_HOME_CHANNEL_NAME",
        "SLACK_HOME_CHANNEL_THREAD_ID",
    ],
    "telegram": [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_REPLY_TO_MODE",
        "TELEGRAM_FALLBACK_IPS",
        "TELEGRAM_HOME_CHANNEL",
        "TELEGRAM_HOME_CHANNEL_NAME",
        "TELEGRAM_HOME_CHANNEL_THREAD_ID",
    ],
    "discord": [
        "DISCORD_BOT_TOKEN",
        "DISCORD_HOME_CHANNEL",
        "DISCORD_HOME_CHANNEL_NAME",
        "DISCORD_HOME_CHANNEL_THREAD_ID",
        "DISCORD_REPLY_TO_MODE",
    ],
    "whatsapp": [
        "WHATSAPP_ENABLED",
        "WHATSAPP_HOME_CHANNEL",
        "WHATSAPP_HOME_CHANNEL_NAME",
        "WHATSAPP_HOME_CHANNEL_THREAD_ID",
    ],
    "signal": [
        "SIGNAL_HTTP_URL",
        "SIGNAL_ACCOUNT",
        "SIGNAL_IGNORE_STORIES",
        "SIGNAL_HOME_CHANNEL",
        "SIGNAL_HOME_CHANNEL_NAME",
        "SIGNAL_HOME_CHANNEL_THREAD_ID",
    ],
    "mattermost": [
        "MATTERMOST_TOKEN",
        "MATTERMOST_URL",
        "MATTERMOST_HOME_CHANNEL",
        "MATTERMOST_HOME_CHANNEL_NAME",
        "MATTERMOST_HOME_CHANNEL_THREAD_ID",
    ],
    "matrix": [
        "MATRIX_ACCESS_TOKEN",
        "MATRIX_HOMESERVER",
        "MATRIX_PASSWORD",
        "MATRIX_USER_ID",
        "MATRIX_ENCRYPTION",
        "MATRIX_DEVICE_ID",
        "MATRIX_HOME_ROOM",
        "MATRIX_HOME_ROOM_NAME",
        "MATRIX_HOME_ROOM_THREAD_ID",
    ],
    "homeassistant": [
        "HASS_TOKEN",
        "HASS_URL",
    ],
    "email": [
        "EMAIL_ADDRESS",
        "EMAIL_PASSWORD",
        "EMAIL_IMAP_HOST",
        "EMAIL_SMTP_HOST",
        "EMAIL_HOME_ADDRESS",
        "EMAIL_HOME_ADDRESS_NAME",
        "EMAIL_HOME_ADDRESS_THREAD_ID",
    ],
    "sms": [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "SMS_HOME_CHANNEL",
        "SMS_HOME_CHANNEL_NAME",
        "SMS_HOME_CHANNEL_THREAD_ID",
    ],
    "dingtalk": [
        "DINGTALK_CLIENT_ID",
        "DINGTALK_CLIENT_SECRET",
        "DINGTALK_HOME_CHANNEL",
        "DINGTALK_HOME_CHANNEL_NAME",
        "DINGTALK_HOME_CHANNEL_THREAD_ID",
    ],
    "api_server": [
        "API_SERVER_ENABLED",
        "API_SERVER_KEY",
        "API_SERVER_CORS_ORIGINS",
        "API_SERVER_PORT",
        "API_SERVER_HOST",
        "API_SERVER_MODEL_NAME",
    ],
    "webhook": [
        "WEBHOOK_ENABLED",
        "WEBHOOK_PORT",
        "WEBHOOK_SECRET",
    ],
    "msgraph_webhook": [
        "MSGRAPH_WEBHOOK_ENABLED",
        "MSGRAPH_WEBHOOK_PORT",
        "MSGRAPH_WEBHOOK_CLIENT_STATE",
        "MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES",
        "MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS",
    ],
    "feishu": [
        "FEISHU_APP_ID",
        "FEISHU_APP_SECRET",
        "FEISHU_DOMAIN",
        "FEISHU_CONNECTION_MODE",
        "FEISHU_ENCRYPT_KEY",
        "FEISHU_VERIFICATION_TOKEN",
        "FEISHU_HOME_CHANNEL",
        "FEISHU_HOME_CHANNEL_NAME",
        "FEISHU_HOME_CHANNEL_THREAD_ID",
    ],
    "wecom": [
        "WECOM_BOT_ID",
        "WECOM_SECRET",
        "WECOM_WEBSOCKET_URL",
        "WECOM_HOME_CHANNEL",
        "WECOM_HOME_CHANNEL_NAME",
        "WECOM_HOME_CHANNEL_THREAD_ID",
    ],
    "wecom_callback": [
        "WECOM_CALLBACK_CORP_ID",
        "WECOM_CALLBACK_CORP_SECRET",
        "WECOM_CALLBACK_AGENT_ID",
        "WECOM_CALLBACK_TOKEN",
        "WECOM_CALLBACK_ENCODING_AES_KEY",
        "WECOM_CALLBACK_HOST",
        "WECOM_CALLBACK_PORT",
    ],
    "weixin": [
        "WEIXIN_TOKEN",
        "WEIXIN_ACCOUNT_ID",
        "WEIXIN_BASE_URL",
        "WEIXIN_CDN_BASE_URL",
        "WEIXIN_DM_POLICY",
        "WEIXIN_GROUP_POLICY",
        "WEIXIN_ALLOWED_USERS",
        "WEIXIN_GROUP_ALLOWED_USERS",
        "WEIXIN_SPLIT_MULTILINE_MESSAGES",
        "WEIXIN_HOME_CHANNEL",
        "WEIXIN_HOME_CHANNEL_NAME",
        "WEIXIN_HOME_CHANNEL_THREAD_ID",
    ],
    "bluebubbles": [
        "BLUEBUBBLES_SERVER_URL",
        "BLUEBUBBLES_PASSWORD",
        "BLUEBUBBLES_WEBHOOK_HOST",
        "BLUEBUBBLES_WEBHOOK_PORT",
        "BLUEBUBBLES_WEBHOOK_PATH",
        "BLUEBUBBLES_SEND_READ_RECEIPTS",
        "BLUEBUBBLES_REQUIRE_MENTION",
        "BLUEBUBBLES_MENTION_PATTERNS",
        "BLUEBUBBLES_HOME_CHANNEL",
        "BLUEBUBBLES_HOME_CHANNEL_NAME",
        "BLUEBUBBLES_HOME_CHANNEL_THREAD_ID",
    ],
    "qqbot": [
        "QQ_APP_ID",
        "QQ_CLIENT_SECRET",
        "QQ_ALLOWED_USERS",
        "QQ_GROUP_ALLOWED_USERS",
        "QQBOT_HOME_CHANNEL",
        "QQBOT_HOME_CHANNEL_NAME",
        "QQBOT_HOME_CHANNEL_THREAD_ID",
        "QQ_HOME_CHANNEL",
        "QQ_HOME_CHANNEL_NAME",
        "QQ_HOME_CHANNEL_THREAD_ID",
    ],
    "yuanbao": [
        "YUANBAO_APP_ID",
        "YUANBAO_APP_KEY",
        "YUANBAO_APP_SECRET",
    ],
}

# Mapping of tool names to env vars that configure them.
# Derived from TOOL_CATEGORIES and TOOLSET_ENV_REQUIREMENTS in tools_config.py.
TOOL_ENV_VARS: Dict[str, List[str]] = {
    "vision": ["OPENROUTER_API_KEY"],
    "moa": ["OPENROUTER_API_KEY"],
    "tts": [
        "VOICE_TOOLS_OPENAI_KEY",
        "ELEVENLABS_API_KEY",
        "MISTRAL_API_KEY",
        "GEMINI_API_KEY",
    ],
    "web": [
        "FIRECRAWL_API_URL",
        "FIRECRAWL_API_KEY",
    ],
    "image_gen": ["FAL_KEY"],
    "video_gen": ["FAL_KEY"],
    "x_search": ["XAI_API_KEY"],
    "browser": [
        "BROWSER_USE_API_KEY",
        "CAMOFOX_URL",
    ],
    "homeassistant": [
        "HASS_TOKEN",
        "HASS_URL",
    ],
    "langfuse": [
        "HERMES_LANGFUSE_PUBLIC_KEY",
        "HERMES_LANGFUSE_SECRET_KEY",
        "HERMES_LANGFUSE_BASE_URL",
    ],
    "spotify": [],  # Spotify uses OAuth post-setup, no env vars
    "computer_use": [],  # cua-driver is local, no env vars required
}


def remove_env_vars_for_feature(category: str, name: str) -> List[str]:
    """Remove env vars for a disabled feature. Returns list of removed keys.

    Args:
        category: "platform" or "tool".
        name: Platform or tool name (e.g. "slack", "homeassistant").

    Raises:
        ValueError: If category is not "platform" or "tool".
    """
    from hermes_cli.config import remove_env_value

    if category not in ("platform", "tool"):
        raise ValueError(f"Invalid category {category!r}: must be 'platform' or 'tool'")

    removed: List[str] = []
    mapping = PLATFORM_ENV_VARS if category == "platform" else TOOL_ENV_VARS
    for key in mapping.get(name, []):
        if remove_env_value(key):
            removed.append(key)
    return removed

"""WeCom (bot + callback sibling) env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

from gateway.config import Platform
from gateway.platform_env import EnvCred, int_or

ENV_ENABLE_CREDENTIALS = ("WECOM_BOT_ID", "WECOM_SECRET")
ENV_OVERRIDES = (
    EnvCred(
        Platform.WECOM, ("WECOM_BOT_ID", "WECOM_SECRET"),
        fixed=(("bot_id", "WECOM_BOT_ID"), ("secret", "WECOM_SECRET")),
        optional=(("websocket_url", "WECOM_WEBSOCKET_URL"),),
        home="WECOM_HOME_CHANNEL",
    ),
)

CALLBACK_ENV_ENABLE_CREDENTIALS = ("WECOM_CALLBACK_CORP_ID", "WECOM_CALLBACK_CORP_SECRET")
CALLBACK_ENV_OVERRIDES = (
    EnvCred(
        Platform.WECOM_CALLBACK, ("WECOM_CALLBACK_CORP_ID", "WECOM_CALLBACK_CORP_SECRET"),
        fixed=(
            ("corp_id", "WECOM_CALLBACK_CORP_ID"), ("corp_secret", "WECOM_CALLBACK_CORP_SECRET"),
            ("agent_id", "WECOM_CALLBACK_AGENT_ID"), ("token", "WECOM_CALLBACK_TOKEN"),
            ("encoding_aes_key", "WECOM_CALLBACK_ENCODING_AES_KEY"),
            # No host default: falsy extra.host lets the adapter's dual-stack DEFAULT_HOST=None bind v4+v6.
            ("host", "WECOM_CALLBACK_HOST"), ("port", "WECOM_CALLBACK_PORT", "", int_or(8645)),
        ),
    ),
)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "WECOM_ALLOWED_USERS",
    "allow_all_env": "WECOM_ALLOW_ALL_USERS",
    "own_policy_env": ("WECOM_DM_POLICY", "WECOM_GROUP_POLICY"),
    "allowlist_suppresses_pairing": True,
}
CALLBACK_AUTHZ = {
    "allowed_users_env": "WECOM_CALLBACK_ALLOWED_USERS",
    "allow_all_env": "WECOM_CALLBACK_ALLOW_ALL_USERS",
    "allowlist_suppresses_pairing": True,
}

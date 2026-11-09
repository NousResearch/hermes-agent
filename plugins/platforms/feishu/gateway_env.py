"""Feishu/Lark env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

from gateway.config import Platform
from gateway.platform_env import EnvCred

ENV_ENABLE_CREDENTIALS = ("FEISHU_APP_ID", "FEISHU_APP_SECRET")
ENV_OVERRIDES = (
    EnvCred(
        Platform.FEISHU, ("FEISHU_APP_ID", "FEISHU_APP_SECRET"),
        fixed=(
            ("app_id", "FEISHU_APP_ID"), ("app_secret", "FEISHU_APP_SECRET"),
            ("domain", "FEISHU_DOMAIN", "feishu"), ("connection_mode", "FEISHU_CONNECTION_MODE", "websocket"),
        ),
        optional=(("encrypt_key", "FEISHU_ENCRYPT_KEY"), ("verification_token", "FEISHU_VERIFICATION_TOKEN")),
        home="FEISHU_HOME_CHANNEL",
    ),
)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "FEISHU_ALLOWED_USERS",
    "allow_all_env": "FEISHU_ALLOW_ALL_USERS",
    "allow_bots_env": "FEISHU_ALLOW_BOTS",
    "allowlist_suppresses_pairing": True,
}

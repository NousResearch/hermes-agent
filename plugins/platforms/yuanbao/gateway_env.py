"""Yuanbao env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

from gateway.config import Platform
from gateway.platform_env import EnvCred, strip_lower

ENV_ENABLE_CREDENTIALS = ("YUANBAO_APP_ID", "YUANBAO_APP_SECRET")
# YUANBAO_APP_ID preferred over the legacy YUANBAO_APP_KEY.
ENV_OVERRIDES = (
    EnvCred(
        Platform.YUANBAO, (("YUANBAO_APP_ID", "YUANBAO_APP_KEY"), "YUANBAO_APP_SECRET"),
        fixed=(("app_id", ("YUANBAO_APP_ID", "YUANBAO_APP_KEY")), ("app_secret", "YUANBAO_APP_SECRET")),
        optional=(
            ("bot_id", "YUANBAO_BOT_ID"), ("ws_url", "YUANBAO_WS_URL"),
            ("api_domain", "YUANBAO_API_DOMAIN"), ("route_env", "YUANBAO_ROUTE_ENV"),
            ("dm_policy", "YUANBAO_DM_POLICY", strip_lower), ("dm_allow_from", "YUANBAO_DM_ALLOW_FROM"),
            ("group_policy", "YUANBAO_GROUP_POLICY", strip_lower), ("group_allow_from", "YUANBAO_GROUP_ALLOW_FROM"),
            ("text_mention_fallback", "YUANBAO_TEXT_MENTION_FALLBACK"),
        ),
        home="YUANBAO_HOME_CHANNEL",
    ),
)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "YUANBAO_ALLOWED_USERS",
    "allow_all_env": "YUANBAO_ALLOW_ALL_USERS",
    "own_policy_env": ("YUANBAO_DM_POLICY", "YUANBAO_GROUP_POLICY"),
}

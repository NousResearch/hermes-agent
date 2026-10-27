"""QQ (Official Bot API v2) env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

import logging

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platform_env import EnvCred, getenv

# Logger name parity with the origin module (gateway.config_env): records stay under "gateway.config".
logger = logging.getLogger("gateway.config")


def _home(config: GatewayConfig, qq_config: PlatformConfig) -> None:
    qq_home = getenv("QQBOT_HOME_CHANNEL").strip()
    name_env = "QQBOT_HOME_CHANNEL_NAME"
    if not qq_home and (qq_home := getenv("QQ_HOME_CHANNEL").strip()):
        # Back-compat: accept the pre-rename name and warn.
        name_env = "QQ_HOME_CHANNEL_NAME"
        logger.warning(
            "QQ_HOME_CHANNEL is deprecated; rename to QQBOT_HOME_CHANNEL "
            "in your .env for consistency with the platform key."
        )
    if qq_home:
        qq_config.home_channel = HomeChannel(
            platform=Platform.QQBOT, chat_id=qq_home,
            name=getenv("QQBOT_HOME_CHANNEL_NAME") or getenv(name_env, "Home"),
            thread_id=getenv("QQBOT_HOME_CHANNEL_THREAD_ID") or getenv("QQ_HOME_CHANNEL_THREAD_ID") or None,
        )


ENV_ENABLE_CREDENTIALS = ("QQ_APP_ID", "QQ_CLIENT_SECRET")
ENV_OVERRIDES = (
    EnvCred(
        Platform.QQBOT, (("QQ_APP_ID", "QQ_CLIENT_SECRET"),),
        optional=(("app_id", "QQ_APP_ID"), ("client_secret", "QQ_CLIENT_SECRET")),
        optional_stripped=(("allow_from", "QQ_ALLOWED_USERS"), ("group_allow_from", "QQ_GROUP_ALLOWED_USERS")),
        then=_home,
    ),
)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "QQ_ALLOWED_USERS",
    "allow_all_env": "QQ_ALLOW_ALL_USERS",
    "own_policy_env": ("", ""),
    "group_allowed_chats_env": "QQ_GROUP_ALLOWED_USERS",
    "allowlist_suppresses_pairing": True,
}

"""Slack env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platform_env import EnvCred, getenv


def _home(config: GatewayConfig) -> None:
    """SLACK_HOME_CHANNEL creates a disabled Slack entry if needed; user_id/scope_id provenance survives an unchanged chat_id."""
    if not (slack_home := getenv("SLACK_HOME_CHANNEL")):
        return
    slack_config = config.platforms.setdefault(Platform.SLACK, PlatformConfig(enabled=False))
    existing_home = slack_config.home_channel
    same_home = existing_home is not None and existing_home.chat_id == slack_home
    slack_config.home_channel = HomeChannel(
        platform=Platform.SLACK, chat_id=slack_home,
        name=getenv("SLACK_HOME_CHANNEL_NAME"), thread_id=getenv("SLACK_HOME_CHANNEL_THREAD_ID") or None,
        user_id=existing_home.user_id if same_home else None,
        scope_id=existing_home.scope_id if same_home else None,
    )


ENV_ENABLE_CREDENTIALS = ("SLACK_BOT_TOKEN",)
ENV_OVERRIDES = (EnvCred(Platform.SLACK, ("SLACK_BOT_TOKEN",), token="SLACK_BOT_TOKEN"), _home)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "SLACK_ALLOWED_USERS",
    "allow_all_env": "SLACK_ALLOW_ALL_USERS",
    "allow_bots_env": "SLACK_ALLOW_BOTS",
    "allowlist_suppresses_pairing": True,
}

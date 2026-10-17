"""Discord env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

from gateway.config import Platform
from gateway.platform_env import EnvCred, home_step, reply_mode_step

ENV_ENABLE_CREDENTIALS = ("DISCORD_BOT_TOKEN",)
# Order matters: the home channel is read BEFORE the reply mode may create the entry.
ENV_OVERRIDES = (
    EnvCred(Platform.DISCORD, ("DISCORD_BOT_TOKEN",), token="DISCORD_BOT_TOKEN"),
    home_step(Platform.DISCORD, "DISCORD_HOME_CHANNEL"),
    reply_mode_step(Platform.DISCORD, "DISCORD_REPLY_TO_MODE"),
)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "DISCORD_ALLOWED_USERS",
    "allow_all_env": "DISCORD_ALLOW_ALL_USERS",
    "allow_bots_env": "DISCORD_ALLOW_BOTS",
}

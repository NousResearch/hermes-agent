"""Telegram env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_env import EnvCred, csv_list, getenv, home_step, reply_mode_step


def _fallback_ips(config: GatewayConfig) -> None:
    if ips := getenv("TELEGRAM_FALLBACK_IPS"):
        config.platforms.setdefault(Platform.TELEGRAM, PlatformConfig()).extra["fallback_ips"] = csv_list(ips)


ENV_ENABLE_CREDENTIALS = ("TELEGRAM_BOT_TOKEN",)
# Order matters: the reply mode / fallback IPs may create the entry the home channel then attaches to.
ENV_OVERRIDES = (
    EnvCred(Platform.TELEGRAM, ("TELEGRAM_BOT_TOKEN",), token="TELEGRAM_BOT_TOKEN"),
    reply_mode_step(Platform.TELEGRAM, "TELEGRAM_REPLY_TO_MODE"),
    _fallback_ips,
    home_step(Platform.TELEGRAM, "TELEGRAM_HOME_CHANNEL"),
)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "TELEGRAM_ALLOWED_USERS",
    "allow_all_env": "TELEGRAM_ALLOW_ALL_USERS",
    "allow_bots_env": "TELEGRAM_ALLOW_BOTS",
}

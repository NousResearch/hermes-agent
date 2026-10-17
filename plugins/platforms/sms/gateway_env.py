"""SMS (Twilio) env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_env import EnvCred, getenv, home_step


def _api_key(config: GatewayConfig, sms_config: PlatformConfig) -> None:
    sms_config.api_key = getenv("TWILIO_AUTH_TOKEN")


ENV_ENABLE_CREDENTIALS = ("TWILIO_ACCOUNT_SID",)
ENV_OVERRIDES = (
    EnvCred(Platform.SMS, ("TWILIO_ACCOUNT_SID",), then=_api_key),
    home_step(Platform.SMS, "SMS_HOME_CHANNEL"),
)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "SMS_ALLOWED_USERS",
    "allow_all_env": "SMS_ALLOW_ALL_USERS",
}

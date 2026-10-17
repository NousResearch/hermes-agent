"""WhatsApp Cloud API env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

from gateway.config import Platform
from gateway.platform_env import INT, EnvCred, home_step

ENV_ENABLE_CREDENTIALS = ("WHATSAPP_CLOUD_PHONE_NUMBER_ID", "WHATSAPP_CLOUD_ACCESS_TOKEN")
# Distinct from the Baileys bridge; both may run against different numbers.
ENV_OVERRIDES = (
    EnvCred(
        Platform.WHATSAPP_CLOUD, ("WHATSAPP_CLOUD_PHONE_NUMBER_ID", "WHATSAPP_CLOUD_ACCESS_TOKEN"),
        fixed=(("phone_number_id", "WHATSAPP_CLOUD_PHONE_NUMBER_ID"), ("access_token", "WHATSAPP_CLOUD_ACCESS_TOKEN")),
        optional=(
            ("app_id", "WHATSAPP_CLOUD_APP_ID"), ("app_secret", "WHATSAPP_CLOUD_APP_SECRET"),
            ("waba_id", "WHATSAPP_CLOUD_WABA_ID"), ("verify_token", "WHATSAPP_CLOUD_VERIFY_TOKEN"),
            ("webhook_host", "WHATSAPP_CLOUD_WEBHOOK_HOST"), ("webhook_port", "WHATSAPP_CLOUD_WEBHOOK_PORT", INT),
            ("webhook_path", "WHATSAPP_CLOUD_WEBHOOK_PATH"), ("api_version", "WHATSAPP_CLOUD_API_VERSION"),
        ),
    ),
    home_step(Platform.WHATSAPP_CLOUD, "WHATSAPP_CLOUD_HOME_CHANNEL"),
)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "WHATSAPP_CLOUD_ALLOWED_USERS",
    "allow_all_env": "WHATSAPP_CLOUD_ALLOW_ALL_USERS",
}

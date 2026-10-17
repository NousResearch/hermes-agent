"""BlueBubbles (iMessage) env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

from gateway.config import Platform
from gateway.platform_env import EnvCred, home_step, int_or, mention_patterns, strip_slash, truthy_token
from utils import is_truthy_value

ENV_ENABLE_CREDENTIALS = ("BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_PASSWORD")
# ``require_mention`` is always written: an unset env reads as "" -> False.
ENV_OVERRIDES = (
    EnvCred(
        Platform.BLUEBUBBLES, ("BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_PASSWORD"),
        fixed=(
            ("server_url", "BLUEBUBBLES_SERVER_URL", "", strip_slash), ("password", "BLUEBUBBLES_PASSWORD"),
            ("webhook_host", "BLUEBUBBLES_WEBHOOK_HOST", "127.0.0.1"),
            ("webhook_port", "BLUEBUBBLES_WEBHOOK_PORT", "", int_or(8645)),
            ("webhook_path", "BLUEBUBBLES_WEBHOOK_PATH", "/bluebubbles-webhook"),
            ("send_read_receipts", "BLUEBUBBLES_SEND_READ_RECEIPTS", "true", is_truthy_value),
            ("require_mention", "BLUEBUBBLES_REQUIRE_MENTION", "", truthy_token),
        ),
        optional=(("mention_patterns", "BLUEBUBBLES_MENTION_PATTERNS", mention_patterns),),
    ),
    home_step(Platform.BLUEBUBBLES, "BLUEBUBBLES_HOME_CHANNEL"),
)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "BLUEBUBBLES_ALLOWED_USERS",
    "allow_all_env": "BLUEBUBBLES_ALLOW_ALL_USERS",
}

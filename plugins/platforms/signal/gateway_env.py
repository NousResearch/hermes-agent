"""Signal env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

from gateway.config import Platform
from gateway.platform_env import EnvCred, home_step
from utils import is_truthy_value

ENV_ENABLE_CREDENTIALS = ("SIGNAL_HTTP_URL",)
ENV_OVERRIDES = (
    EnvCred(
        Platform.SIGNAL, ("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"),
        fixed=(
            ("http_url", "SIGNAL_HTTP_URL"), ("account", "SIGNAL_ACCOUNT"),
            ("ignore_stories", "SIGNAL_IGNORE_STORIES", "true", is_truthy_value),
        ),
    ),
    home_step(Platform.SIGNAL, "SIGNAL_HOME_CHANNEL"),
)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "SIGNAL_ALLOWED_USERS",
    "allow_all_env": "SIGNAL_ALLOW_ALL_USERS",
    "allowlist_suppresses_pairing": True,
}

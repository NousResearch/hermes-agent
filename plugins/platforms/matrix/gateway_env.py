"""Matrix env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_env import EnvCred, env_extras, getenv, home_step
from utils import is_truthy_value


def _e2ee(config: GatewayConfig, matrix_config: PlatformConfig) -> None:
    mode = getenv("MATRIX_E2EE_MODE").strip().lower()
    matrix_config.extra["encryption"] = mode in ("required", "require", "optional", "prefer", "preferred") or is_truthy_value(getenv("MATRIX_ENCRYPTION"))
    if mode:
        matrix_config.extra["e2ee_mode"] = mode
    env_extras(matrix_config.extra, (("device_id", "MATRIX_DEVICE_ID"),))


ENV_ENABLE_CREDENTIALS = ("MATRIX_ACCESS_TOKEN", "MATRIX_PASSWORD")
ENV_OVERRIDES = (
    EnvCred(
        Platform.MATRIX, (("MATRIX_ACCESS_TOKEN", "MATRIX_PASSWORD"),), token="MATRIX_ACCESS_TOKEN",
        warn_missing=("MATRIX_HOMESERVER", "MATRIX_ACCESS_TOKEN/MATRIX_PASSWORD set but MATRIX_HOMESERVER is missing"),
        fixed=(("homeserver", "MATRIX_HOMESERVER"),),
        optional=(("user_id", "MATRIX_USER_ID"), ("password", "MATRIX_PASSWORD")),
        then=_e2ee,
    ),
    home_step(Platform.MATRIX, "MATRIX_HOME_ROOM"),
)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "MATRIX_ALLOWED_USERS",
    "allow_all_env": "MATRIX_ALLOW_ALL_USERS",
    "allowlist_suppresses_pairing": True,
}

"""DingTalk env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

from gateway.config import Platform
from gateway.platform_env import EnvCred

ENV_ENABLE_CREDENTIALS = ("DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET")
ENV_OVERRIDES = (
    EnvCred(
        Platform.DINGTALK, ("DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET"),
        fixed=(("client_id", "DINGTALK_CLIENT_ID"), ("client_secret", "DINGTALK_CLIENT_SECRET")),
        home="DINGTALK_HOME_CHANNEL",
    ),
)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "DINGTALK_ALLOWED_USERS",
    "allow_all_env": "DINGTALK_ALLOW_ALL_USERS",
    "allowlist_suppresses_pairing": True,
}

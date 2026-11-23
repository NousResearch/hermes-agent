"""Weixin (personal WeChat via iLink Bot API) env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

from gateway.config import Platform
from gateway.platform_env import EnvCred, strip_slash

ENV_ENABLE_CREDENTIALS = ("WEIXIN_TOKEN", "WEIXIN_ACCOUNT_ID")
ENV_OVERRIDES = (
    EnvCred(
        Platform.WEIXIN, (("WEIXIN_TOKEN", "WEIXIN_ACCOUNT_ID"),), token="WEIXIN_TOKEN",
        optional=(("account_id", "WEIXIN_ACCOUNT_ID"),),
        optional_stripped=(
            ("base_url", "WEIXIN_BASE_URL", strip_slash), ("cdn_base_url", "WEIXIN_CDN_BASE_URL", strip_slash),
            ("dm_policy", "WEIXIN_DM_POLICY", str.lower), ("group_policy", "WEIXIN_GROUP_POLICY", str.lower),
            ("allow_from", "WEIXIN_ALLOWED_USERS"), ("group_allow_from", "WEIXIN_GROUP_ALLOWED_USERS"),
            ("split_multiline_messages", "WEIXIN_SPLIT_MULTILINE_MESSAGES"),
        ),
        home="WEIXIN_HOME_CHANNEL", home_strip=True,
    ),
)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "WEIXIN_ALLOWED_USERS",
    "allow_all_env": "WEIXIN_ALLOW_ALL_USERS",
    "own_policy_env": ("WEIXIN_DM_POLICY", "WEIXIN_GROUP_POLICY"),
    "allowlist_suppresses_pairing": True,
}

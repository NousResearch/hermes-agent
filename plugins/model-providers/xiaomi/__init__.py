"""Xiaomi MiMo provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

# Default base_url is Pay As You Go.  Token Plan users get a regional
# endpoint selected during ``hermes model`` setup and stored in
# ``model.base_url`` in config.yaml (env var: XIAOMI_BASE_URL).
#
# Token Plan endpoints:
#   CN:  https://token-plan-cn.xiaomimimo.com/v1
#   SGP: https://token-plan-sgp.xiaomimimo.com/v1
#   EU:  https://token-plan-ams.xiaomimimo.com/v1
xiaomi = ProviderProfile(
    name="xiaomi",
    aliases=("mimo", "xiaomi-mimo"),
    env_vars=("XIAOMI_API_KEY",),
    base_url="https://api.xiaomimimo.com/v1",
    supports_health_check=False,  # /v1/models returns 401 even with valid key
    supports_vision=True,  # mimo-v2-omni is vision-capable
    supports_vision_tool_messages=False,  # rejects list-type tool content (400 "text is not set")
)

register_provider(xiaomi)

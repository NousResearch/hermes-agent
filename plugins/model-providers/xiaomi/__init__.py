"""Xiaomi MiMo provider profile."""

from providers import register_provider
from providers.model_normalizers import LowercaseMatchingPrefixProviderProfile

xiaomi = LowercaseMatchingPrefixProviderProfile(
    name="xiaomi", aliases=("mimo", "xiaomi-mimo"), display_name="Xiaomi MiMo",
    description="Xiaomi MiMo (MiMo-V2.5 and V2 models: pro, omni, flash)",
    signup_url="https://platform.xiaomimimo.com", env_vars=("XIAOMI_API_KEY",), base_url="https://api.xiaomimimo.com/v1",
    base_url_env_var="XIAOMI_BASE_URL",
    supports_health_check=False,  # /v1/models returns 401 even with valid key
    supports_vision=True,  # mimo-v2-omni is vision-capable
    supports_vision_tool_messages=False,  # rejects list-type tool content (400 "text is not set")
    default_vision_model_id="mimo-v2.5",
)

register_provider(xiaomi)

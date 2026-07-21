"""Z.ai Indirect provider using Z.ai's Claude Code-compatible endpoint."""
from providers import ProviderProfile, register_provider


zai_indirect = ProviderProfile(
    name="zai-indirect",
    aliases=("z.ai-indirect", "zai_indirect"),
    display_name="Z.ai Indirect",
    description="GLM-5.2 through Z.ai's Claude Code-compatible Anthropic endpoint",
    signup_url="https://z.ai/subscribe",
    env_vars=("ZAI_INDIRECT_API_KEY", "ZAI_API_KEY", "GLM_API_KEY", "Z_AI_API_KEY"),
    base_url="https://api.z.ai/api/anthropic",
    api_mode="anthropic_messages",
    auth_type="api_key",
    supports_health_check=False,
    supports_vision=True,
    fallback_models=("glm-5.2",),
    default_aux_model="glm-5.2",
    default_max_tokens=32000,
)

register_provider(zai_indirect)

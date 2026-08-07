"""Z.AI GLM Coding Plan provider profile.

Separate from the standard `zai` profile because the coding plan
uses the Anthropic-compatible endpoint (api.z.ai/api/anthropic)
instead of the OpenAI-compatible /api/paas/v4 endpoint.

The coding plan quota is ONLY accessible via the Anthropic endpoint.
"""

from providers import register_provider
from providers.base import ProviderProfile


zai_coding = ProviderProfile(
    name="zai-coding",
    aliases=("zai-coding-plan", "glm-coding", "z-ai-coding"),
    api_mode="anthropic_messages",
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
    display_name="Z.AI (GLM Coding Plan)",
    description="Z.AI GLM Coding Plan — Anthropic-compatible endpoint",
    signup_url="https://z.ai/",
    base_url="https://api.z.ai/api/anthropic",
    auth_type="api_key",
    fallback_models=(
        "glm-5.1",
        "glm-5-turbo",
        "glm-4.7",
        "glm-4.5-air",
    ),
    default_aux_model="glm-4.5-air",
)

register_provider(zai_coding)

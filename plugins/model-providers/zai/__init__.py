"""ZAI / GLM provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

zai = ProviderProfile(
    name="zai",
    aliases=("glm", "z-ai", "z.ai", "zhipu"),
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
    display_name="Z.AI / GLM Direct API",
    description="Z.AI / GLM direct API endpoint",
    signup_url="https://z.ai/",
    fallback_models=(
        "glm-5.2",
        "glm-5",
        "glm-4-9b",
    ),
    base_url="https://api.z.ai/api/paas/v4",
    default_aux_model="glm-4.5-flash",
)

zai_coding = ProviderProfile(
    name="zai-coding",
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
    display_name="Z.AI / GLM Coding Plan API",
    description="Z.AI / GLM Coding Plan API endpoint",
    signup_url="https://z.ai/",
    fallback_models=(
        "glm-5.2",
        "glm-5-turbo",
        "glm-5.1",
        "glm-4.7",
        "glm-4.5-air",
    ),
    base_url="https://api.z.ai/api/coding/paas/v4",
    default_aux_model="glm-5-turbo",
)

register_provider(zai)
register_provider(zai_coding)

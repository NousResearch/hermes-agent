"""AIgateway provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


aigateway = ProviderProfile(
    name="aigateway", aliases=("aigw", "aigateway-sh"), display_name="AIgateway",
    description="AIgateway — 1,000+ models from 85+ labs behind one OpenAI-compatible endpoint",
    signup_url="https://aigateway.sh/dashboard/keys", env_vars=("AIGATEWAY_API_KEY", "AIGATEWAY_BASE_URL"),
    base_url="https://api.aigateway.sh/v1", auth_type="api_key",
    default_aux_model="zai-org/glm-5.3-flash",
    fallback_models=(
        "zai-org/glm-5.3-flash", "anthropic/claude-sonnet-4.6", "openai/gpt-5.4",
        "moonshot/kimi-k2.7-code", "google/gemini-3.7-flash", "deepseek/deepseek-v4-flash-0731",
    ),
)

register_provider(aigateway)

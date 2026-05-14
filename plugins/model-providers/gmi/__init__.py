"""GMI Cloud provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

gmi = ProviderProfile(
    name="gmi",
    aliases=("gmi-cloud", "gmicloud"),
    display_name="GMI Cloud",
    description="GMI Cloud — multi-model direct API (slash-form model IDs)",
    signup_url="https://www.gmicloud.ai/",
    env_vars=("GMI_API_KEY", "GMI_BASE_URL"),
    base_url="https://api.gmi-serving.com/v1",
    auth_type="api_key",
    default_aux_model="google/gemini-3.1-flash-lite-preview",
    fallback_models=(
        "zai-org/GLM-5.1-FP8",
        "deepseek-ai/DeepSeek-V3.2",
        "moonshotai/Kimi-K2.5",
        "google/gemini-3.1-flash-lite-preview",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
    ),
)

register_provider(gmi)

"""Atlas Cloud provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

atlascloud = ProviderProfile(
    name="atlascloud",
    aliases=("atlas", "atlas-cloud"),
    display_name="Atlas Cloud",
    description="Atlas Cloud - OpenAI-compatible unified model API",
    signup_url="https://www.atlascloud.ai/console/api-keys",
    env_vars=("ATLASCLOUD_API_KEY", "ATLASCLOUD_BASE_URL"),
    base_url="https://api.atlascloud.ai/v1",
    auth_type="api_key",
    default_aux_model="qwen/qwen3.5-flash",
    fallback_models=(
        "qwen/qwen3.5-flash",
        "deepseek-ai/deepseek-v4-pro",
    ),
)

register_provider(atlascloud)

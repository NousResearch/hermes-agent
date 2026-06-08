"""Cloud Temple LLMaaS provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


cloud_temple = ProviderProfile(
    name="cloud-temple",
    aliases=("cloud_temple", "cloudtemple"),
    display_name="Cloud Temple",
    description="Cloud Temple LLMaaS — sovereign OpenAI-compatible inference",
    signup_url="https://docs.cloud-temple.com/llmaas",
    env_vars=("CLOUD_TEMPLE_API_KEY", "CLOUD_TEMPLE_BASE_URL"),
    base_url="https://api.ai.cloud-temple.com/v1",
    auth_type="api_key",
    default_aux_model="qwen3.6:35b",
    fallback_models=("qwen3.6:35b", "gemma4:31b"),
)

register_provider(cloud_temple)

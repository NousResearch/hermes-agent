"""Together AI provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

together = ProviderProfile(
    name="together",
    aliases=("together-ai", "togetherai"),
    display_name="Together AI",
    description="Together AI — cloud platform for open models",
    signup_url="https://api.together.xyz/settings/api-keys",
    env_vars=("TOGETHER_API_KEY",),
    base_url="https://api.together.xyz/v1",
    auth_type="api_key",
    default_aux_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    fallback_models=(
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ),
)

register_provider(together)
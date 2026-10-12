"""Prime Inference provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

prime = ProviderProfile(
    name="prime",
    aliases=("prime-inference", "primeintellect", "pinference"),
    display_name="Prime Inference",
    description="Prime Intellect — OpenAI-compatible inference API",
    signup_url="https://app.primeintellect.ai/",
    env_vars=("PRIME_API_KEY",),
    base_url="https://api.pinference.ai/api/v1",
    auth_type="api_key",
    default_aux_model="deepseek/deepseek-v4.1-flash",
    fallback_models=(
        "deepseek/deepseek-v4.1-flash",
        "z-ai/glm-5.3",
        "moonshotai/kimi-k3",
        "openai/gpt-5.4",
        "anthropic/claude-opus-5.5",
    ),
)

register_provider(prime)

"""PleumRouter provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

pleumrouter = ProviderProfile(
    name="pleumrouter",
    aliases=("pleum",),
    display_name="PleumRouter",
    description="Korea-region OpenAI-compatible multi-provider LLM gateway",
    signup_url="https://router.pleum.ai",
    env_vars=("PLEUMROUTER_API_KEY",),
    base_url="https://router.pleum.ai/v1",
    # models_url falls back to {base_url}/models — PleumRouter exposes a public
    # GET /v1/models (no key required), so model discovery works out of the box.
    fallback_models=(
        "claude-sonnet-4-6",
        "gpt-5.5",
        "gemini-2.5-pro",
    ),
)

register_provider(pleumrouter)

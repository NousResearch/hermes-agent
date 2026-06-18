"""TokenRouter provider profile.

TokenRouter (https://tokenrouter.com) is an OpenAI-compatible multi-model
router: one API key and base URL front models from Anthropic, OpenAI, DeepSeek,
Qwen, Moonshot, Z.AI, xAI, MiniMax and others. It speaks standard
chat-completions with no provider-specific request quirks, so the base
``ProviderProfile`` handles everything — no subclass needed.

The live model catalog is fetched from ``/v1/models`` via the default
``ProviderProfile.fetch_models`` (Bearer auth). ``fallback_models`` below is a
small curated set of tool-calling text models shown in the ``/model`` picker
only when that live fetch fails; image / video / audio models are omitted.

``TOKENROUTER_BASE_URL`` is listed in ``env_vars`` so the auth registry derives
a base-URL override env var (see ``hermes_cli/auth.py`` auto-extend), letting a
user point at a self-hosted / regional gateway without code changes.
"""

from providers import register_provider
from providers.base import ProviderProfile

tokenrouter = ProviderProfile(
    name="tokenrouter",
    aliases=("token-router",),
    env_vars=("TOKENROUTER_API_KEY", "TOKENROUTER_BASE_URL"),
    display_name="TokenRouter",
    description="TokenRouter — OpenAI-compatible multi-model router",
    signup_url="https://tokenrouter.com",
    base_url="https://api.tokenrouter.com/v1",
    fallback_models=(
        "anthropic/claude-opus-4.8",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.5",
        "deepseek/deepseek-v4-pro",
        "moonshotai/kimi-k2.7-code",
        "qwen/qwen3.7-max",
        "z-ai/glm-5",
        "MiniMax-M3",
    ),
)

register_provider(tokenrouter)

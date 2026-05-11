"""Perplexity provider profile.

Perplexity offers an OpenAI-compatible Agent API at https://api.perplexity.ai
that routes to many frontier models behind a single key. Configure with
``PERPLEXITY_API_KEY`` (``PPLX_API_KEY`` accepted as a fallback).

Get an API key at https://www.perplexity.ai/account/api/keys
"""

from providers import register_provider
from providers.base import ProviderProfile

perplexity = ProviderProfile(
    name="perplexity",
    aliases=("pplx",),
    env_vars=("PERPLEXITY_API_KEY", "PPLX_API_KEY"),
    display_name="Perplexity",
    description="Perplexity Agent API — OpenAI-compatible multi-model gateway",
    signup_url="https://www.perplexity.ai/account/api/keys",
    # Perplexity Agent API is a model-agnostic gateway — these are
    # representative frontier models exposed through it. The live picker
    # fetches the full catalog via /models.
    fallback_models=(
        "openai/gpt-5.4",
        "anthropic/claude-sonnet-4-6",
        "google/gemini-3-1-pro",
    ),
    base_url="https://api.perplexity.ai",
)

register_provider(perplexity)

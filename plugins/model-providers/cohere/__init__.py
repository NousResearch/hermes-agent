"""Cohere provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


cohere = ProviderProfile(
    name="cohere",
    aliases=('cohere-ai',),
    display_name="Cohere",
    description="Cohere — Command A family via OpenAI-compatible Compatibility API",
    signup_url="https://dashboard.cohere.com/api-keys",
    env_vars=('COHERE_API_KEY', 'COHERE_BASE_URL'),
    base_url="https://api.cohere.ai/compatibility/v1",
    auth_type="api_key",
    supports_vision=True,
    default_aux_model="command-a-03-2025",
    fallback_models=('command-a-plus-05-2026', 'command-a-03-2025', 'command-a-reasoning-08-2025', 'command-r-plus', 'command-r'),
)

register_provider(cohere)

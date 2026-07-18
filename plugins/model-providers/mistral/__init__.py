"""Mistral AI provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


mistral = ProviderProfile(
    name="mistral",
    aliases=('mistral-ai', 'mistralai'),
    display_name="Mistral AI",
    description="Mistral AI — Large, Medium, Codestral, Pixtral, Devstral",
    signup_url="https://console.mistral.ai/",
    env_vars=('MISTRAL_API_KEY', 'MISTRAL_BASE_URL'),
    base_url="https://api.mistral.ai/v1",
    auth_type="api_key",
    supports_vision=True,
    default_aux_model="mistral-small-latest",
    fallback_models=('mistral-large-latest', 'mistral-medium-latest', 'mistral-small-latest', 'codestral-latest', 'pixtral-large-latest', 'devstral-medium-latest'),
)

register_provider(mistral)

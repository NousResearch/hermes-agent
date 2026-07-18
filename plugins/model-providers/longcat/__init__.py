"""LongCat provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


longcat = ProviderProfile(
    name="longcat",
    aliases=('longcat-ai',),
    display_name="LongCat",
    description="LongCat — LongCat-2.0 agentic / coding model API",
    signup_url="https://longcat.chat/platform/api_keys",
    env_vars=('LONGCAT_API_KEY', 'LONGCAT_BASE_URL'),
    base_url="https://api.longcat.chat/openai",
    auth_type="api_key",
    supports_vision=False,
    default_aux_model="LongCat-2.0",
    fallback_models=('LongCat-2.0',),
)

register_provider(longcat)

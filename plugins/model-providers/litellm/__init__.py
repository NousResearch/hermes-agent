"""LiteLLM Proxy provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


litellm = ProviderProfile(
    name="litellm",
    aliases=('litellm-proxy', 'lite-llm'),
    display_name="LiteLLM Proxy",
    description="LiteLLM — unified multi-provider proxy (default localhost:4000)",
    signup_url="https://docs.litellm.ai/",
    env_vars=('LITELLM_API_KEY', 'LITELLM_BASE_URL'),
    base_url="http://127.0.0.1:4000/v1",
    auth_type="api_key",
    supports_vision=True,
    default_aux_model="",
    fallback_models=(),
)

register_provider(litellm)

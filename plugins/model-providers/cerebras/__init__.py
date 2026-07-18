"""Cerebras provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


cerebras = ProviderProfile(
    name="cerebras",
    aliases=('cerebras-ai',),
    display_name="Cerebras",
    description="Cerebras — high-speed OpenAI-compatible inference",
    signup_url="https://cloud.cerebras.ai",
    env_vars=('CEREBRAS_API_KEY', 'CEREBRAS_BASE_URL'),
    base_url="https://api.cerebras.ai/v1",
    auth_type="api_key",
    supports_vision=False,
    default_aux_model="llama-3.3-70b",
    fallback_models=('llama-3.3-70b', 'llama3.1-8b', 'qwen-3-32b', 'gpt-oss-120b'),
)

register_provider(cerebras)

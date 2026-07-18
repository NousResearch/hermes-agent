"""Featherless AI provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


featherless = ProviderProfile(
    name="featherless",
    aliases=('featherless-ai',),
    display_name="Featherless AI",
    description="Featherless AI — open models via OpenAI-compatible API",
    signup_url="https://featherless.ai",
    env_vars=('FEATHERLESS_API_KEY', 'FEATHERLESS_BASE_URL'),
    base_url="https://api.featherless.ai/v1",
    auth_type="api_key",
    supports_vision=False,
    default_aux_model="Qwen/Qwen3-32B",
    fallback_models=('Qwen/Qwen3-32B', 'meta-llama/Meta-Llama-3.1-70B-Instruct', 'mistralai/Mistral-Small-24B-Instruct-2501'),
)

register_provider(featherless)

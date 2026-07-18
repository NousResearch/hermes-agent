"""Chutes provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


chutes = ProviderProfile(
    name="chutes",
    aliases=('chutes-ai',),
    display_name="Chutes",
    description="Chutes — open-source model catalog (OpenAI-compatible; API key)",
    signup_url="https://chutes.ai/settings/api-keys",
    env_vars=('CHUTES_API_KEY', 'CHUTES_BASE_URL'),
    base_url="https://llm.chutes.ai/v1",
    auth_type="api_key",
    supports_vision=False,
    default_aux_model="zai-org/GLM-4.5-Air",
    fallback_models=('zai-org/GLM-5-TEE', 'zai-org/GLM-4.5-Air', 'deepseek-ai/DeepSeek-V3.1', 'Qwen/Qwen3-235B-A22B'),
)

register_provider(chutes)

"""Baseten provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


baseten = ProviderProfile(
    name="baseten",
    aliases=('baseten-ai',),
    display_name="Baseten",
    description="Baseten — Model APIs and Inkling (OpenAI-compatible)",
    signup_url="https://app.baseten.co",
    env_vars=('BASETEN_API_KEY', 'BASETEN_BASE_URL'),
    base_url="https://inference.baseten.co/v1",
    auth_type="api_key",
    supports_vision=False,
    default_aux_model="deepseek-ai/DeepSeek-V3.2",
    fallback_models=('deepseek-ai/DeepSeek-V3.2', 'moonshotai/Kimi-K2.5', 'zai-org/GLM-5'),
)

register_provider(baseten)

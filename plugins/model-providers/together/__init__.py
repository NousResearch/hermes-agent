"""Together AI provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


together = ProviderProfile(
    name="together",
    aliases=('together-ai', 'togetherai'),
    display_name="Together AI",
    description="Together AI — open models (Llama, DeepSeek, Kimi, Qwen, GLM, …)",
    signup_url="https://api.together.ai/settings/api-keys",
    env_vars=('TOGETHER_API_KEY', 'TOGETHER_BASE_URL'),
    base_url="https://api.together.xyz/v1",
    auth_type="api_key",
    supports_vision=True,
    default_aux_model="Qwen/Qwen2.5-7B-Instruct-Turbo",
    fallback_models=('meta-llama/Llama-3.3-70B-Instruct-Turbo', 'moonshotai/Kimi-K2.5', 'deepseek-ai/DeepSeek-V3', 'Qwen/Qwen2.5-7B-Instruct-Turbo', 'zai-org/GLM-4.5'),
)

register_provider(together)

"""Groq provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


groq = ProviderProfile(
    name="groq",
    aliases=('groq-cloud',),
    display_name="Groq",
    description="Groq — ultra-fast LPU inference (Llama, Gemma, Qwen, …)",
    signup_url="https://console.groq.com/keys",
    env_vars=('GROQ_API_KEY', 'GROQ_BASE_URL'),
    base_url="https://api.groq.com/openai/v1",
    auth_type="api_key",
    supports_vision=False,
    default_aux_model="llama-3.1-8b-instant",
    fallback_models=('llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'openai/gpt-oss-120b', 'qwen/qwen3-32b', 'moonshotai/kimi-k2-instruct'),
)

register_provider(groq)

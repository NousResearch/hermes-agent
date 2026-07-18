"""Venice AI provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


venice = ProviderProfile(
    name="venice",
    aliases=('venice-ai',),
    display_name="Venice AI",
    description="Venice AI — privacy-focused inference (private + anonymized proxy models)",
    signup_url="https://venice.ai",
    env_vars=('VENICE_API_KEY', 'VENICE_BASE_URL'),
    base_url="https://api.venice.ai/api/v1",
    auth_type="api_key",
    supports_vision=True,
    default_aux_model="llama-3.3-70b",
    fallback_models=('llama-3.3-70b', 'qwen3-235b', 'deepseek-v3.2', 'kimi-k2-5', 'minimax-m2.5'),
)

register_provider(venice)

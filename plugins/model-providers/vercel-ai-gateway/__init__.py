"""Vercel AI Gateway provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


vercel_ai_gateway = ProviderProfile(
    name="vercel-ai-gateway",
    aliases=('vercel', 'ai-gateway', 'vercel-gateway'),
    display_name="Vercel AI Gateway",
    description="Vercel AI Gateway — multi-provider routing via one API key",
    signup_url="https://vercel.com/docs/ai-gateway",
    env_vars=('AI_GATEWAY_API_KEY', 'AI_GATEWAY_BASE_URL'),
    base_url="https://ai-gateway.vercel.sh/v1",
    auth_type="api_key",
    supports_vision=True,
    default_aux_model="google/gemini-2.5-flash",
    fallback_models=('anthropic/claude-sonnet-4.5', 'openai/gpt-5', 'google/gemini-2.5-flash', 'xai/grok-4'),
)

register_provider(vercel_ai_gateway)

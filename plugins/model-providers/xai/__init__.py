"""xAI (Grok) provider profile (API key version).

For the recommended browser-based OAuth experience that reuses your existing
Grok CLI / Grok Build login (~/.grok/auth.json), use the "xai-oauth" provider instead.
"""

from providers import register_provider
from providers.base import ProviderProfile

xai = ProviderProfile(
    name="xai",
    aliases=("grok", "x-ai", "x.ai"),
    api_mode="codex_responses",
    env_vars=("XAI_API_KEY",),
    display_name="xAI (API Key)",
    description="xAI Grok — API key (direct, per-token billing)",
    signup_url="https://console.x.ai/",
    fallback_models=(
        "grok-4.3",
        "grok-4.3-latest",
    ),
    base_url="https://api.x.ai/v1",
    auth_type="api_key",
)

register_provider(xai)

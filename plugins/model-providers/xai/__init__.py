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
    base_url="https://api.x.ai/v1",
    auth_type="api_key",
)

register_provider(xai)

"""xAI (Grok) OAuth Provider — unified under xai-coding-plan.

This provider delegates to the xai-coding-plan profile, which supports:
  - Automatic import from the official Grok CLI / Grok Build login
    (~/.grok/auth.json). This provides the best experience for users who
    already use the official Grok tools.
  - Full browser-based PKCE OAuth login flow as fallback.

Renamed from the original xai-oauth to xai-coding-plan for consistency.
Credits am423 for the original OAuth flow pattern.
"""

from providers import register_provider
from providers.base import ProviderProfile


xai_coding_plan = ProviderProfile(
    name="xai-coding-plan",
    aliases=(
        "xai-oauth",
        "xai-oauth",
        "grok-oauth",
        "xai-portal",
        "grok-login",
        "grok-oauth-login",
        "xai-browser",
        "xai-oauth-plan",
        "grok-plan",
        "grok-code",
        "xai-grok-build",
    ),
    api_mode="codex_responses",
    env_vars=("XAI_API_KEY",),
    display_name="xAI Coding Plan",
    description="xAI Grok Coding Plan (OAuth — requires grok CLI)",
    signup_url="https://x.ai/",
    base_url="https://api.x.ai/v1",
    auth_type="oauth_external",
    default_aux_model="grok-3-mini",
    default_max_tokens=32768,
    fallback_models=(
        "grok-build",
        "grok-4-1-fast",
        "grok-code-fast-1",
        "grok-4.3",
        "grok-4.3-latest",
    ),
)

register_provider(xai_coding_plan)

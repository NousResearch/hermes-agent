"""xAI (Grok) provider profiles.

Two authentication paths:
  1. xai         — API key via XAI_API_KEY env var
  2. xai-coding-plan — OAuth delegated to grok CLI binary
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

xai_coding_plan = ProviderProfile(
    name="xai-coding-plan",
    aliases=("xai-oauth", "grok-plan", "grok-code", "xai-grok-build"),
    api_mode="codex_responses",
    env_vars=(),  # OAuth via grok CLI — no API key
    display_name="xAI Coding Plan",
    description="xAI Grok Coding Plan (OAuth — requires grok CLI)",
    signup_url="https://x.ai/",
    fallback_models=(
        "grok-build",
        "grok-4-1-fast",
        "grok-code-fast-1",
        "grok-4.3",
        "grok-4.3-latest",
    ),
    base_url="https://api.x.ai/v1",
    auth_type="oauth_external",
)

register_provider(xai)
register_provider(xai_coding_plan)

"""xAI (Grok) OAuth Provider for Hermes Agent.

This provider adds first-class support for xAI's Grok models using OAuth
authentication against auth.x.ai.

Primary authentication method:
    Automatic import from the official Grok CLI / Grok Build login
    (~/.grok/auth.json). This provides the best experience for users who
    already use the official Grok tools.

Fallback:
    Full browser-based PKCE OAuth login flow.

Technical details:
    - Uses `codex_responses` API mode for native reasoning support.
    - Benefits from xAI's built-in prompt caching via the `x-grok-conv-id` header
      (automatically handled by the transport layer).
"""

from providers import register_provider
from providers.base import ProviderProfile


class XaiOAuthProfile(ProviderProfile):
    """xAI Grok via OAuth.

    Prefers credentials imported from the official Grok CLI.
    Falls back to browser OAuth login when needed.
    """

    # Note: xAI prompt caching (x-grok-conv-id) is handled automatically
    # in the transport layer when the base URL contains "x.ai".
    # No custom prepare_messages or build_extra_body hooks are required
    # at this time.


xai_oauth = XaiOAuthProfile(
    name="xai-oauth",
    aliases=(
        "xai-oauth",
        "grok-oauth",
        "xai-portal",
        "grok-login",
        "grok-oauth-login",
        "xai-browser",
    ),
    api_mode="codex_responses",
    env_vars=("XAI_API_KEY",),
    base_url="https://api.x.ai/v1",
    auth_type="oauth_external",
    default_aux_model="grok-3-mini",
    default_max_tokens=32768,
    description="xAI Grok — OAuth (auto-imports from official Grok CLI login)",
    signup_url="https://grok.com",
)

register_provider(xai_oauth)

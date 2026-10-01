"""xAI Grok OAuth provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

xai_oauth = ProviderProfile(
    name="xai-oauth",
    aliases=("grok-oauth", "x-ai-oauth", "xai-grok-oauth"),
    display_name="xAI Grok OAuth (SuperGrok / Premium+)",
    description="xAI Grok OAuth (SuperGrok / Premium+ subscription)",
    api_mode="codex_responses",
    auth_type="oauth_external",
    base_url="https://api.x.ai/v1",
    base_url_env_var="XAI_BASE_URL",
    supports_model_listing=False,
)

register_provider(xai_oauth)

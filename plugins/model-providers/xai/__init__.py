"""xAI (Grok) provider profile."""

from hermes_cli.version_info import hermes_user_agent
from hermes_agent.providers import register_provider
from hermes_agent.providers.base import LazyHeaders, ProviderProfile

xai = ProviderProfile(
    name="xai", aliases=("grok", "x-ai", "x.ai"), api_mode="codex_responses", env_vars=("XAI_API_KEY",),
    base_url="https://api.x.ai/v1", auth_type="api_key",
    default_headers=LazyHeaders({"User-Agent": lambda: hermes_user_agent("Hermes-Agent")}),
)

register_provider(xai)

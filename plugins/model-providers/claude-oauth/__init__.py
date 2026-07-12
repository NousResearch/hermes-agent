"""Claude subscription OAuth through the official Claude Code runtime."""

from providers import register_provider
from providers.base import ProviderProfile

claude_oauth = ProviderProfile(
    name="claude-oauth",
    aliases=("claude-code",),
    display_name="Claude OAuth",
    description="Claude subscription via official Claude Code runtime (fail-closed)",
    signup_url="https://claude.ai/",
    api_mode="claude_agent_sdk",
    env_vars=(),
    base_url="",
    auth_type="external_process",
    supports_health_check=False,
    fallback_models=("claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"),
)

register_provider(claude_oauth)

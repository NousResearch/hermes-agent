"""Claude Code CLI provider profile.

Uses Claude Code CLI (subscription-based) as the LLM backend instead of
API tokens. No API key needed — authentication is handled by the Claude
Code CLI's own login/subscription.
"""

from providers import register_provider
from providers.base import ProviderProfile


claude_cli = ProviderProfile(
    name="claude-cli",
    api_mode="claude_cli",
    aliases=("cc", "claude-code", "claude-code-cli"),
    auth_type="none",  # No API key — uses CC subscription
    env_vars=(),        # No env vars needed
    base_url="",        # No API endpoint — subprocess-based
    display_name="Claude Code CLI",
    description="Claude Code CLI (subscription) — subprocess-based, no API tokens needed",
    supports_health_check=False,  # No /models endpoint to probe
    fallback_models=(
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ),
)

register_provider(claude_cli)

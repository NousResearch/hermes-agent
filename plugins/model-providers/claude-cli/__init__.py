"""Claude Code CLI provider profile.

claude-cli uses the local Claude Code CLI in non-interactive print mode. It is
not a REST endpoint; run_agent.py routes the marker base URL to ClaudeCLIClient.
"""

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeCLIProfile(ProviderProfile):
    """Claude Code CLI external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by Hermes' static CLI catalog."""
        return None


claude_cli = ClaudeCLIProfile(
    name="claude-cli",
    aliases=("anthropic-cli", "claude-code-cli"),
    api_mode="chat_completions",
    env_vars=(),
    base_url="claude-cli://local",
    auth_type="external_process",
)

register_provider(claude_cli)

"""Claude Code ACP provider profile.

claude-code-acp drives the local Claude Code CLI through the vendor-maintained
``@zed-industries/claude-code-acp`` ACP bridge (an external subprocess) — NOT
the standard HTTP transport. This keeps inference billed against the user's
Claude subscription (Pro/Max) instead of failing as third-party API usage.
``api_mode="chat_completions"`` is used for routing; the actual ACP handling is
dispatched in agent_runtime_helpers.py based on provider / base_url. The profile
captures auth + endpoint metadata for registry migration.
"""

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeCodeACPProfile(ProviderProfile):
    """Claude Code ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return None


claude_code_acp = ClaudeCodeACPProfile(
    name="claude-code-acp",
    aliases=("claude-acp", "claude-code-acp-agent"),
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    env_vars=(),  # Managed by ACP subprocess (Claude Code keychain / ~/.claude)
    base_url="acp://claude",  # ACP internal scheme
    auth_type="external_process",
)

register_provider(claude_code_acp)

"""Claude Code ACP provider profile.

claude-acp drives Claude Code through its official ACP adapter as an
external subprocess — NOT the standard transport. Routing happens in
agent/acp_client.py via the acp://claude base URL; the launch command
comes from agent/acp_agent_registry.py.
"""

from providers import register_provider
from providers.base import ProviderProfile


class ACPSubprocessProfile(ProviderProfile):
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


claude_acp = ACPSubprocessProfile(
    name="claude-acp",
    aliases=("claude-acp-agent",),
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    env_vars=(),  # Credentials are managed by the agent's own CLI login
    base_url="acp://claude",  # ACP internal scheme
    auth_type="external_process",
)

register_provider(claude_acp)

"""Codex CLI ACP provider profile.

codex-acp drives Codex CLI through its official ACP adapter as an
external subprocess — NOT the standard transport. Routing happens in
agent/acp_client.py via the acp://codex base URL; the launch command
comes from agent/acp_agent_registry.py.
"""

from providers import register_provider
from providers.base import ProviderProfile


class ACPSubprocessProfile(ProviderProfile):
    """Codex CLI ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return None


codex_acp = ACPSubprocessProfile(
    name="codex-acp",
    aliases=("codex-acp-agent",),
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    env_vars=(),  # Credentials are managed by the agent's own CLI login
    base_url="acp://codex",  # ACP internal scheme
    auth_type="external_process",
)

register_provider(codex_acp)

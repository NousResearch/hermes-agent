"""Claude Code subscription through the local ACP subprocess.

Reuses the generic ``CopilotACPClient`` stdio shim (see
``plugins/model-providers/copilot-acp/`` for the pattern this mirrors): the client
picks its default command/args from ``base_url`` (``acp://claude-code`` here), so no
new client class is needed — just a profile pointing it at the ``claude-agent-acp``
CLI instead of ``copilot``.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeCodeACPProfile(ProviderProfile):
    """Claude Code ACP — external process, no REST models endpoint."""

    def create_client(self, **client_kwargs: Any) -> Any:
        """Build the ACP stdio shim rather than an HTTP client."""
        from agent.copilot_acp_client import CopilotACPClient

        return CopilotACPClient(**client_kwargs)

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return None


claude_code_acp = ClaudeCodeACPProfile(
    name="claude-code-acp",
    aliases=("claude-acp",),
    api_mode="chat_completions",
    env_vars=(),  # Claude Code owns its saved authentication.
    base_url="acp://claude-code",
    auth_type="external_process",
    supports_health_check=False,
    process_command="claude-agent-acp",
    process_args=(),
    process_command_env_vars=("HERMES_CLAUDE_CODE_ACP_COMMAND",),
    process_args_env_var="HERMES_CLAUDE_CODE_ACP_ARGS",
)

register_provider(claude_code_acp)

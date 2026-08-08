"""Claude Code ACP provider profile.

claude-code-acp uses an external ACP subprocess (the `cc-acp` binary,
installed by the `claude-code-acp` npm package) -- NOT the standard
transport. Wiring this profile into api_mode/run_agent.py-level provider
resolution (the equivalent of copilot-acp's `if provider == "copilot-acp":`
branch in agent/auxiliary_client.py) is deliberately deferred -- a real, not-yet-ticketed follow-up, not an
oversight -- see agent/claude_code_acp_client.py's module docstring for the
full explanation (this profile was built for Linear LIA-529, which is the
ticket that intentionally scoped the wiring OUT, not a tracker for doing it).
The profile captures auth + endpoint metadata for registry migration, same
as plugins/model-providers/copilot-acp/.

Note: the provider `name`/alias `"cc-acp"` below is the same string as the
literal binary name `_resolve_command()` falls back to in
agent/claude_code_acp_client.py, but they are different namespaces
(ProviderProfile.aliases is config-selection surface; _resolve_command()'s
fallback is the literal argv[0] spawned) -- not a bug, just worth a reader
not conflating `provider: cc-acp` with an actual CLI flag.
"""

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeCodeACPProfile(ProviderProfile):
    """Claude Code ACP -- external process, no REST models endpoint."""

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
    aliases=("claude-acp", "cc-acp", "claude-code-acp-agent"),
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    env_vars=(),  # Managed by ACP subprocess
    base_url="acp://claude-code",  # ACP internal scheme
    auth_type="external_process",
)

register_provider(claude_code_acp)

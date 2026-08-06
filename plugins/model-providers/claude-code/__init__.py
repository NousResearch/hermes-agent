"""Claude subscription provider profile — routed through the official Claude Agent SDK.

Inference runs inside ``claude-agent-sdk`` (which wraps the Claude Code CLI),
so ``api_mode="claude_agent_sdk"`` is handled separately from the HTTP
transports; the profile only captures auth + identity metadata.

Boundary: Hermes never reads, writes, refreshes, or deletes Claude
credentials.  ``env_vars`` is empty on purpose — there is no credential for
Hermes to hold.  The user runs ``claude auth login`` and the SDK resolves auth
itself.  ``base_url`` is an internal scheme, not a reachable endpoint, so no
code path can mistake this provider for a REST backend.
"""

from hermes_cli.claude_code import (
    CLAUDE_CODE_API_MODE,
    CLAUDE_CODE_BASE_URL,
    CLAUDE_CODE_DESCRIPTION,
    CLAUDE_CODE_DISPLAY_NAME,
    CLAUDE_CODE_PROVIDER_ID,
    CLAUDE_DOCS_URL,
)
from providers import register_provider
from providers.base import ProviderProfile


class ClaudeCodeProfile(ProviderProfile):
    """Claude subscription via the Agent SDK — no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is owned by the Agent SDK, not a REST catalog."""
        return None


claude_code = ClaudeCodeProfile(
    name=CLAUDE_CODE_PROVIDER_ID,
    # `claude-oauth` used to be an anthropic alias meaning "Claude subscription".
    # Once the gate is open the slug belongs here; while it is closed
    # hermes_cli.claude_code.legacy_alias_target() keeps pointing it at anthropic.
    aliases=("claude-subscription", "claude-agent-sdk", "claude-oauth"),
    api_mode=CLAUDE_CODE_API_MODE,
    display_name=CLAUDE_CODE_DISPLAY_NAME,
    description=CLAUDE_CODE_DESCRIPTION,
    signup_url=CLAUDE_DOCS_URL,
    env_vars=(),  # No credential env var — the SDK resolves auth itself.
    base_url=CLAUDE_CODE_BASE_URL,
    auth_type="external_process",
    supports_health_check=False,  # no HTTP endpoint to probe
)

register_provider(claude_code)

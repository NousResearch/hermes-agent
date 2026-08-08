"""Claude via the official claude-agent-sdk — subscription OAuth (#25267).

Unlike the ``anthropic`` provider (raw Messages API: API-key pay-per-token,
or OAuth that bills overage credits), this provider hands the whole turn to
Anthropic's ``claude-agent-sdk``, which authenticates with the **Claude
subscription** (``CLAUDE_CODE_OAUTH_TOKEN`` / the ``~/.claude`` credential
store) — the officially supported third-party path. Hermes resolves NO
credentials for it: the SDK subprocess self-authenticates, which is why
``auth_type="oauth_external"`` and ``env_vars`` is only advisory.

Runtime: ``api_mode="claude_agent_sdk"`` — an agent-loop runtime dispatched
by an early return in run_conversation(), exactly like ``codex_app_server``.
"""

from providers import register_provider
from providers.base import ProviderProfile

claude_agent_sdk = ProviderProfile(
    name="claude-agent-sdk",
    # NB: "claude"/"claude-code"/"claude-oauth" are already claimed by the
    # `anthropic` profile — keep alias namespaces disjoint.
    aliases=("claude-sdk", "claude-code-sdk", "claude_agent_sdk"),
    display_name="Claude (Agent SDK / subscription)",
    description=(
        "Claude Code's agent loop via the official Agent SDK, billed to the "
        "Claude subscription (never a metered API key)."
    ),
    api_mode="claude_agent_sdk",
    env_vars=("CLAUDE_CODE_OAUTH_TOKEN",),
    base_url="",
    auth_type="oauth_external",
    default_aux_model="claude-haiku-4-5-20251001",
)
register_provider(claude_agent_sdk)

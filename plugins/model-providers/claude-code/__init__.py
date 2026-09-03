"""Claude subscription via the Claude Code CLI (``claude_code`` api_mode).

Turns are handed to a long-lived, Hermes-owned ``claude -p`` subprocess that
authenticates with a ``claude setup-token`` credential from
``$CLAUDE_CODE_OAUTH_TOKEN`` — no API key, no keychain access, and no token
is ever read out of the CLI. See ``agent/claude_code_runtime.py`` (#25267).
"""

from providers import register_provider
from providers.base import ProviderProfile

claude_code = ProviderProfile(
    name="claude-code-cli",
    aliases=("claude-subscription", "claude_code_cli"),
    api_mode="claude_code",
    auth_type="oauth_external",
    env_vars=(),  # credential is CLAUDE_CODE_OAUTH_TOKEN, consumed by the CLI only
    base_url="",  # no HTTP endpoint — the CLI owns the network
    display_name="Claude (subscription via Claude Code)",
    description=(
        "Claude Pro/Max subscription through the official `claude` CLI, with "
        "Claude Code's native tools plus Hermes tools over MCP"
    ),
    signup_url="https://claude.ai/",
    supports_health_check=False,
    fallback_models=("opus", "sonnet", "haiku"),
)

register_provider(claude_code)

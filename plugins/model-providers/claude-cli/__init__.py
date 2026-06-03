"""Claude Code CLI provider — uses claude CLI credentials (no API key required).

Lets users run Hermes with Claude as the AI backend using the same
authentication as their local Claude Code CLI installation.  No separate
ANTHROPIC_API_KEY is needed; credentials are read from
~/.claude/.credentials.json (or the macOS Keychain on macOS) exactly as
the existing 'anthropic' provider does when oauth credentials are present.
"""

import logging

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

CLAUDE_CLI_MODELS = (
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
)


class ClaudeCliProfile(ProviderProfile):
    """Uses Claude Code CLI OAuth credentials — no Anthropic API key needed."""

    def fetch_models(self, *, api_key=None, timeout=8.0):
        # Claude Code OAuth tokens work against the Anthropic models endpoint,
        # but we can't be sure of the token at profile-load time.  Return a
        # static curated list for the model picker so 'hermes model' is fast
        # and doesn't require the token to be available at startup.
        return list(CLAUDE_CLI_MODELS)


claude_cli = ClaudeCliProfile(
    name="claude-cli",
    aliases=("claude-code-cli", "claudecli"),
    api_mode="anthropic_messages",
    env_vars=("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_TOKEN"),
    base_url="https://api.anthropic.com",
    auth_type="api_key",
    display_name="Claude Code",
    description="Claude via Claude Code CLI authentication (no API key required)",
    signup_url="https://claude.ai/",
    supports_health_check=False,
    default_aux_model="claude-haiku-4-5-20251001",
    fallback_models=CLAUDE_CLI_MODELS,
)

register_provider(claude_cli)

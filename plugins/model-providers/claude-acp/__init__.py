"""Claude Code ACP provider (argus-acp fork).

Routes to the streaming ``ClaudeACPClient`` via the ``create_openai_client``
dispatch (matched on provider name / ``acp://claude`` base_url). Claude Code
runs as a real ACP agent — executes its own MCP tools, streams thinking + tool
traces to the reasoning channel and the answer to content. Uses the local
``claude`` Max auth (no API key, Anthropic-permitted).
"""
from providers import register_provider
from providers.base import ProviderProfile


class ClaudeACPProfile(ProviderProfile):
    """External ACP subprocess — no REST models endpoint."""

    def fetch_models(self, *, api_key=None, base_url=None, timeout: float = 8.0):
        return None


register_provider(ClaudeACPProfile(
    name="claude-acp",
    aliases=("claude-code-acp", "claude-acp-agent"),
    api_mode="chat_completions",   # dispatched to ClaudeACPClient in create_openai_client
    env_vars=(),                   # auth handled by the local claude CLI (Max)
    base_url="acp://claude",
    auth_type="external_process",
))

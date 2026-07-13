"""Claude Code CLI provider profile.

claude-cli: routes Hermes requests through the local `claude -p` (Claude Code
headless) binary as a pure model endpoint. The genuine first-party Claude Code
client makes the network call using the user's existing OAuth/subscription
session, so traffic draws normal included usage rather than pay-per-token API
billing. Hermes keeps its own agent loop and executes ALL tools — the subprocess
only turns messages into an assistant reply (text + <tool_call> blocks).

Reports api_mode="chat_completions" but is dispatched at client-construction
time to ClaudeCLIClient (see agent/agent_runtime_helpers.create_openai_client),
exactly like the copilot-acp provider. The profile carries the effort-threading
hook so a live `/effort` change reaches the subprocess on the next turn.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Model aliases + full ids accepted by `claude -p --model <x>`. Kept static so
# the model picker populates without a network call (there is no REST catalog
# for a subprocess provider). `/model` switches between these live.
_CLAUDE_CLI_MODELS = (
    "sonnet",
    "opus",
    "haiku",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-haiku-4-5-20251001",
    "claude-fable-5",
)


class ClaudeCLIProfile(ProviderProfile):
    """Claude Code CLI subprocess — thread reasoning effort to the client.

    `claude -p` takes reasoning effort as a `--effort` CLI flag, not an API
    field. build_extra_body runs per request with the live reasoning_config,
    so stashing the effort here is how a mid-session `/effort` change reaches
    the subprocess on the very next turn. ClaudeCLIClient pops the key back
    out of api_kwargs["extra_body"].
    """

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """No REST catalog — return the static alias list for the picker."""
        return list(_CLAUDE_CLI_MODELS)

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """Stash the live reasoning effort so the client can map it to --effort."""
        reasoning_config = context.get("reasoning_config")
        if not isinstance(reasoning_config, dict):
            return {}
        effort = reasoning_config.get("effort")
        if not isinstance(effort, str) or not effort.strip():
            return {}
        return {"_hermes_claude_effort": effort.strip().lower()}


claude_cli = ClaudeCLIProfile(
    name="claude-cli",
    aliases=("claude-code-cli", "claude-p"),
    api_mode="chat_completions",
    env_vars=(),  # No API key — auth is the user's existing Claude Code OAuth session.
    base_url="acp://claude-cli",
    auth_type="external_process",
    default_aux_model="haiku",
)

register_provider(claude_cli)

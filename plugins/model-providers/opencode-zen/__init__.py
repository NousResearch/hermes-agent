"""OpenCode provider profiles (Zen + Go).

Both use per-model api_mode routing:
  - OpenCode Zen: Claude → anthropic_messages, GPT-5/Codex → codex_responses,
    everything else → chat_completions (this profile)
  - OpenCode Go: MiniMax → anthropic_messages, GLM/Kimi → chat_completions
    (this profile)
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class OpenCodeGoProfile(ProviderProfile):
    """OpenCode Go subscription profile."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model = str(context.get("model") or "").strip().lower()
        if not model.startswith("deepseek-v4-"):
            return {}, {}

        if reasoning_config and isinstance(reasoning_config, dict):
            if reasoning_config.get("enabled") is False:
                return {}, {}
            effort = str(reasoning_config.get("effort") or "medium").strip().lower()
        else:
            effort = "medium"

        if effort == "minimal":
            effort = "low"
        elif effort not in {"low", "medium", "high", "xhigh", "max"}:
            effort = "medium"

        return {}, {"reasoning_effort": effort}


opencode_zen = ProviderProfile(
    name="opencode-zen",
    aliases=("opencode", "opencode_zen", "zen"),
    env_vars=("OPENCODE_ZEN_API_KEY",),
    base_url="https://opencode.ai/zen/v1",
    default_aux_model="gemini-3-flash",
)

opencode_go = OpenCodeGoProfile(
    name="opencode-go",
    aliases=("opencode_go", "go", "opencode-go-sub"),
    env_vars=("OPENCODE_GO_API_KEY",),
    base_url="https://opencode.ai/zen/go/v1",
    default_aux_model="glm-5",
)

register_provider(opencode_zen)
register_provider(opencode_go)

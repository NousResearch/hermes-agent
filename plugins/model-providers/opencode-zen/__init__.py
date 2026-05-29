"""OpenCode provider profiles (Zen + Go).

Both use per-model api_mode routing:
  - OpenCode Zen: Claude → anthropic_messages, GPT-5/Codex → codex_responses,
    everything else → chat_completions (this profile)
  - OpenCode Go: MiniMax → anthropic_messages, GLM/Kimi → chat_completions
    (this profile)
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _flat_model_name(model: str | None) -> str:
    """Return the bare OpenCode model ID, tolerating aggregator prefixes."""
    return (model or "").strip().rsplit("/", 1)[-1].lower()


def _is_kimi_k2_model(model: str | None) -> bool:
    return _flat_model_name(model).startswith("kimi-k2")


def _is_deepseek_thinking_model(model: str | None) -> bool:
    m = _flat_model_name(model)
    if m.startswith("deepseek-v") and not m.startswith("deepseek-v3"):
        return True
    return m == "deepseek-reasoner"


class OpenCodeGoProfile(ProviderProfile):
    """OpenCode Go - model-specific reasoning controls."""

    # Per-model completion-token cap. The opencode-go relay's default is
    # too large for mimo-v2.5-pro — it sends max_tokens=262144 but Xiaomi
    # only supports 131072 completion tokens and 400s the request.
    # Setting an explicit cap here prevents the relay default from being
    # applied. Keys are normalized via _flat_model_name().
    _MODEL_MAX_TOKENS: dict[str, int] = {
        "mimo-v2.5-pro": 131072,
    }

    def get_max_tokens(self, model: str | None) -> int | None:
        cap = self._MODEL_MAX_TOKENS.get(_flat_model_name(model))
        if cap is not None:
            return cap
        return self.default_max_tokens

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if _is_kimi_k2_model(model):
            # Kimi K2 on OpenCode Go uses Moonshot's native wire shape:
            # extra_body.thinking (binary toggle) + top-level reasoning_effort
            # (low|medium|high). Mirrors the KimiProfile (api.moonshot.ai/v1).
            if not isinstance(reasoning_config, dict):
                # No config → leave server defaults alone.
                return extra_body, top_level

            enabled = reasoning_config.get("enabled") is not False
            extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}

            if not enabled:
                return extra_body, top_level

            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort in {"xhigh", "max"}:
                top_level["reasoning_effort"] = "high"
            elif effort in {"low", "medium", "high"}:
                top_level["reasoning_effort"] = effort
            return extra_body, top_level

        if not _is_deepseek_thinking_model(model):
            return extra_body, top_level

        enabled = True
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            enabled = False
        extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}

        if not enabled:
            return extra_body, top_level

        if isinstance(reasoning_config, dict):
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort in {"xhigh", "max"}:
                top_level["reasoning_effort"] = "max"
            elif effort in {"low", "medium", "high"}:
                top_level["reasoning_effort"] = effort

        return extra_body, top_level


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

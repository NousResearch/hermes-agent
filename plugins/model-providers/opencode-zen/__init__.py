"""OpenCode provider profiles (Zen + Go).

Both use per-model api_mode routing:
  - OpenCode Zen: Claude → anthropic_messages, GPT-5/Codex/Grok → codex_responses,
    Muse Spark → codex_responses, everything else → chat_completions (this profile)
  - OpenCode Go: GPT / Grok / Muse Spark → codex_responses, MiniMax/Qwen → anthropic_messages,
    GLM/Kimi/DeepSeek/MiMo → chat_completions (this profile)
"""

from typing import Any

from agent import reasoning_effort as re_
from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile

# Attribution headers (same values as OpenRouter / Vercel / Fireworks); via
# default_headers so they survive model switches and credential rotation.
_ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
    "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
}


def _flat_model_name(model: str | None) -> str:
    """Bare OpenCode model ID, tolerating aggregator prefixes."""
    return (model or "").strip().rsplit("/", 1)[-1].lower()


# Version-less DeepSeek ids that still carry the thinking/effort knobs on this wire: the retired
# ``deepseek-reasoner`` alias and the canonical ``deepseek-flash`` (2026-09 Flash refresh), for
# which the Go relay honours the same top-level ``reasoning_effort``/``thinking`` contract.
_THINKING_CAPABLE_IDS: frozenset[str] = frozenset({"deepseek-reasoner", "deepseek-flash"})


def _is_deepseek_thinking_model(model: str | None) -> bool:
    m = _flat_model_name(model)
    return (m.startswith("deepseek-v") and not m.startswith("deepseek-v3")) or m in _THINKING_CAPABLE_IDS


def _is_glm_5_2_model(model: str | None) -> bool:
    """GLM-5.2 across alias spellings (glm-5.2 / glm-5-2 / glm-5p2)."""
    m = _flat_model_name(model)
    return any(token in m for token in ("glm-5.2", "glm-5-2", "glm-5p2"))


class OpenCodeGoProfile(ProviderProfile):
    """OpenCode Go - model-specific reasoning controls."""

    # The relay's default max_tokens (262144) exceeds what Xiaomi accepts for
    # mimo-v2.5-pro and 400s; keys are normalized via _flat_model_name().
    _MODEL_MAX_TOKENS: dict[str, int] = {"mimo-v2.5-pro": 131072}

    def get_max_tokens(self, model: str | None) -> int | None:
        cap = self._MODEL_MAX_TOKENS.get(_flat_model_name(model))
        return self.default_max_tokens if cap is None else cap

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if _is_glm_5_2_model(model):
            # GLM-5.2 on OpenCode Go uses its native OpenAI-compatible
            # reasoning_effort knob (high/max — declared in
            # agent.reasoning_effort, shared with the zai profile); leave the
            # server default alone when reasoning is disabled or unset.
            if not isinstance(reasoning_config, dict):
                return extra_body, top_level
            if reasoning_config.get("enabled") is False:
                return extra_body, top_level
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if not effort or effort == "none":
                return extra_body, top_level
            from agent.reasoning_effort import (
                GLM52_EFFORTS,
                GLM52_OVERRIDES,
                clamp_effort,
            )

            clamped = clamp_effort(effort, GLM52_EFFORTS, GLM52_OVERRIDES)
            top_level["reasoning_effort"] = (
                clamped if clamped in GLM52_EFFORTS else "high"
            )
            return extra_body, top_level

        if _is_kimi_k2_model(model):
            # Kimi K2 on OpenCode Go uses Moonshot's native wire shape:
            # extra_body.thinking (binary toggle) + top-level reasoning_effort
            # (low|medium|high). Mirrors the KimiProfile (api.moonshot.ai/v1).
            if not isinstance(reasoning_config, dict):
                # No config → leave server defaults alone.
                return extra_body, top_level

            enabled = reasoning_config.get("enabled") is not False
            if not enabled:
                extra_body["thinking"] = {"type": "disabled"}
                return extra_body, top_level

            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort and effort != "none":
                from agent.reasoning_effort import KIMI_K2_EFFORTS, clamp_effort

                clamped = clamp_effort(effort, KIMI_K2_EFFORTS)
                if clamped in KIMI_K2_EFFORTS:
                    top_level["reasoning_effort"] = clamped

            # Avoid "cannot specify both 'thinking' and 'reasoning_effort'" HTTP 400:
            # only send extra_body["thinking"] when no reasoning_effort is set.
            if "reasoning_effort" not in top_level:
                extra_body["thinking"] = {"type": "enabled"}
            return extra_body, top_level

        if not _is_deepseek_thinking_model(model):
            return extra_body, top_level

        enabled = True
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            enabled = False

        if not enabled:
            extra_body["thinking"] = {"type": "disabled"}
            return extra_body, top_level

        if isinstance(reasoning_config, dict):
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort and effort != "none":
                from agent.reasoning_effort import (
                    DEEPSEEK_V4_EFFORTS,
                    DEEPSEEK_V4_OVERRIDES,
                    clamp_effort,
                )

                clamped = clamp_effort(
                    effort, DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES
                )
                if clamped in DEEPSEEK_V4_EFFORTS:
                    top_level["reasoning_effort"] = clamped

        # Avoid "cannot specify both 'thinking' and 'reasoning_effort'" HTTP 400:
        # only send extra_body["thinking"] when no reasoning_effort is set.
        if "reasoning_effort" not in top_level:
            extra_body["thinking"] = {"type": "enabled"}

        return extra_body, top_level


def _build_ox_alpha_reasoning_extras(
    reasoning_config: dict | None, model: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Shared Ox Alpha (x-preview-f-free) reasoning_effort translation.

    Used by both the opencode-zen profile and the opencode-free keyless
    profile — the model is reachable through either provider and the wire
    contract is identical (low/high/max only; anything else 400s).
    """
    if _flat_model_name(model) != "x-preview-f-free":
        return {}, {}
    if not isinstance(reasoning_config, dict):
        return {}, {}
    if reasoning_config.get("enabled") is False:
        return {}, {}

    effort = (reasoning_config.get("effort") or "").strip().lower()
    if not effort or effort == "none":
        return {}, {}

    from agent.reasoning_effort import (
        OX_ALPHA_EFFORTS,
        OX_ALPHA_OVERRIDES,
        clamp_effort,
    )

    clamped = clamp_effort(effort, OX_ALPHA_EFFORTS, OX_ALPHA_OVERRIDES)
    if clamped not in OX_ALPHA_EFFORTS:
        return {}, {}
    return {}, {"reasoning_effort": clamped}


class OpenCodeZenProfile(ProviderProfile):
    """OpenCode Zen - model-specific reasoning controls."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return _build_ox_alpha_reasoning_extras(reasoning_config, model)


opencode_zen = OpenCodeZenProfile(
    name="opencode-zen",
    aliases=("opencode", "opencode_zen", "zen"),
    env_vars=("OPENCODE_ZEN_API_KEY",),
    base_url="https://opencode.ai/zen/v1",
    default_headers=dict(_ATTRIBUTION_HEADERS),
    default_aux_model="gemini-3-flash",
)

opencode_go = OpenCodeGoProfile(
    name="opencode-go", aliases=("opencode_go", "go", "opencode-go-sub"), env_vars=("OPENCODE_GO_API_KEY",),
    base_url="https://opencode.ai/zen/go/v1", default_headers=dict(_ATTRIBUTION_HEADERS),
    default_aux_model="glm-5",
    # The Go relay's upstream validates tool content as a strict string: list-type tool
    # content (native vision embeds) 422s with ``messages.N.tool.content.str Input should
    # be a valid string`` (Console Go, #104731) or 400s ``text is not set`` (MiMo, #47026),
    # and the rejected row stays in history so every later call dies too. Images in user
    # messages are fine, so vision itself keeps working via the text-summary downgrade.
    supports_vision_tool_messages=False,
)

register_provider(opencode_zen)
register_provider(opencode_go)

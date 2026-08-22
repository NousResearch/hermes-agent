"""OpenCode provider profiles (Zen + Go).

Both use per-model api_mode routing:
  - OpenCode Zen: Claude → anthropic_messages, GPT-5/Codex/Grok → codex_responses,
    Muse Spark → codex_responses, everything else → chat_completions (this profile)
  - OpenCode Go: GPT / Grok / Muse Spark → codex_responses, MiniMax/Qwen → anthropic_messages,
    GLM/Kimi/DeepSeek/MiMo → chat_completions (this profile)
"""

from __future__ import annotations

from typing import Any

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile

# Attribution headers sent on every OpenCode request. Same values we send
# to OpenRouter, Vercel AI Gateway, and Fireworks. Going through
# profile.default_headers means they survive model switches and credential
# rotation. Without them OpenCode only sees the OpenAI SDK's generic
# "OpenAI/Python x.y.z" User-Agent and can't tell the traffic is Hermes Agent.
_ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
    "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
}

# Zen's live catalog mixes paid models, currently usable anonymous models, and
# expired promotional IDs. Hermes exposes the usable anonymous subset through
# the dedicated keyless opencode-free provider, so none of these IDs should be
# offered through the credentialed Zen picker.
_ZEN_FREE_TIER_MODELS = frozenset({
    "x-preview-f-free",
    "hy3-free",
    "laguna-s-2.1-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
    "muse-spark-1.2-contributor-free",
    "big-pickle",
    "mimo-v2.5-free",
    "deepseek-v4-flash-free",
})


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


def _is_glm_5_2_model(model: str | None) -> bool:
    """Detect GLM-5.2 across alias spellings (glm-5.2 / glm-5-2 / glm-5p2)."""
    m = _flat_model_name(model)
    return any(token in m for token in ("glm-5.2", "glm-5-2", "glm-5p2"))


class OpenCodeGoProfile(ProviderProfile):
    """OpenCode Go - model-specific reasoning controls."""

    # The relay still advertises these IDs from /v1/models, but rejects every
    # request for them and the current Go documentation no longer lists them.
    _UNAVAILABLE_MODELS = frozenset({
        "hy3-preview",
        "mimo-v2-omni",
        "mimo-v2-pro",
    })

    def fetch_models(self, **kwargs) -> list[str]:
        models = super().fetch_models(**kwargs)
        return [
            model
            for model in models
            if _flat_model_name(model) not in self._UNAVAILABLE_MODELS
        ]

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

    def fetch_models(self, **kwargs) -> list[str]:
        models = super().fetch_models(**kwargs)
        return [
            model
            for model in models
            if _flat_model_name(model) not in _ZEN_FREE_TIER_MODELS
        ]

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
    name="opencode-go",
    aliases=("opencode_go", "go", "opencode-go-sub"),
    env_vars=("OPENCODE_GO_API_KEY",),
    base_url="https://opencode.ai/zen/go/v1",
    default_headers=dict(_ATTRIBUTION_HEADERS),
    default_aux_model="glm-5",
)

register_provider(opencode_zen)
register_provider(opencode_go)

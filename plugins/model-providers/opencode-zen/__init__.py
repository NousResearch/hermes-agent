"""OpenCode provider profiles (Zen + Go).

Both route api_mode per model in core; these profiles carry the
chat_completions reasoning translations (GLM-5.2, Kimi K2, DeepSeek, Ox Alpha).
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


def _is_deepseek_thinking_model(model: str | None) -> bool:
    m = _flat_model_name(model)
    return (m.startswith("deepseek-v") and not m.startswith("deepseek-v3")) or m == "deepseek-reasoner"


def _is_glm_5_2_model(model: str | None) -> bool:
    """GLM-5.2 across alias spellings (glm-5.2 / glm-5-2 / glm-5p2)."""
    m = _flat_model_name(model)
    return any(token in m for token in ("glm-5.2", "glm-5-2", "glm-5p2"))


def _requested_effort(reasoning_config: dict | None) -> str | None:
    """Normalized effort when reasoning is enabled and an effort is set, else None."""
    effort = re_.requested_effort(reasoning_config)
    return None if effort == "none" else effort


def _thinking_toggle_extras(
    reasoning_config: dict | None, efforts, overrides=None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Moonshot/DeepSeek wire shape: extra_body.thinking XOR top-level reasoning_effort
    (sending both is an HTTP 400)."""
    if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
        return {"thinking": {"type": "disabled"}}, {}
    clamped = re_.clamp_effort(_requested_effort(reasoning_config), efforts, overrides)
    if clamped in efforts:
        return {}, {"reasoning_effort": clamped}
    return {"thinking": {"type": "enabled"}}, {}


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
            # Native reasoning_effort knob (high/max); server default when unset/disabled.
            effort = _requested_effort(reasoning_config)
            if effort is None:
                return {}, {}
            clamped = re_.clamp_effort(effort, re_.GLM52_EFFORTS, re_.GLM52_OVERRIDES)
            return {}, {"reasoning_effort": clamped if clamped in re_.GLM52_EFFORTS else "high"}
        if _flat_model_name(model).startswith("kimi-k2"):
            if not isinstance(reasoning_config, dict):
                return {}, {}
            return _thinking_toggle_extras(reasoning_config, re_.KIMI_K2_EFFORTS)
        if _is_deepseek_thinking_model(model):
            return _thinking_toggle_extras(reasoning_config, re_.DEEPSEEK_V4_EFFORTS, re_.DEEPSEEK_V4_OVERRIDES)
        return {}, {}


def _build_ox_alpha_reasoning_extras(
    reasoning_config: dict | None, model: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ox Alpha (x-preview-f-free) reasoning_effort translation, shared with the
    opencode-free profile (low/high/max only; anything else 400s)."""
    if _flat_model_name(model) != "x-preview-f-free":
        return {}, {}
    clamped = re_.clamp_effort(_requested_effort(reasoning_config), re_.OX_ALPHA_EFFORTS, re_.OX_ALPHA_OVERRIDES)
    return ({}, {"reasoning_effort": clamped}) if clamped in re_.OX_ALPHA_EFFORTS else ({}, {})


class OpenCodeZenProfile(ProviderProfile):
    """OpenCode Zen - model-specific reasoning controls."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return _build_ox_alpha_reasoning_extras(reasoning_config, model)


opencode_zen = OpenCodeZenProfile(
    name="opencode-zen", aliases=("opencode", "opencode_zen", "zen"), env_vars=("OPENCODE_ZEN_API_KEY",),
    base_url="https://opencode.ai/zen/v1", default_headers=dict(_ATTRIBUTION_HEADERS),
    default_aux_model="gemini-3-flash",
)

opencode_go = OpenCodeGoProfile(
    name="opencode-go", aliases=("opencode_go", "go", "opencode-go-sub"), env_vars=("OPENCODE_GO_API_KEY",),
    base_url="https://opencode.ai/zen/go/v1", default_headers=dict(_ATTRIBUTION_HEADERS),
    default_aux_model="glm-5",
)

register_provider(opencode_zen)
register_provider(opencode_go)

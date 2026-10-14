"""TokenRhythm provider profile (OpenAI-compatible Chat Completions; no Responses API).

Reasoning controls differ per model family and were verified live against the API, not read from
the catalog's ``reasoningMode`` field. Every model returns reasoning in ``reasoning_content``
(``delta.reasoning_content`` when streaming), and every budget field is rejected with HTTP 400:

* ``toggle_effort`` (DeepSeek): ``thinking.type`` switches it, top-level ``reasoning_effort``
  grades it (low/high/max). ``deepseek-v4-*`` do NOT think when the toggle is omitted, so a
  main-loop turn sends ``thinking: enabled`` explicitly.
* ``toggle`` (Qwen, Doubao Seed): ``thinking.type`` only. ``reasoning_effort`` is accepted but
  ignored (``none`` does not switch it off), so it is never sent.
* ``always`` (GLM-5.x, Kimi, MiniMax, MiMo, LongCat): always thinks with no knob. GLM-5.3,
  kimi-k2.7-code, longcat-2.0 and minimax-m2.7 answer any disable with HTTP 400 ("当前模型必须开启
  深度思考"), so a disable request is dropped, never forwarded.
"""

from typing import Any

from agent.reasoning_effort import (
    TOKENRHYTHM_DEEPSEEK_EFFORTS,
    TOKENRHYTHM_DEEPSEEK_OVERRIDES,
    thinking_toggle_extras,
)
from providers import register_provider
from providers.base import ProviderProfile

_TOGGLE_EFFORT, _TOGGLE, _ALWAYS = "toggle_effort", "toggle", "always"

# Chat models online in TokenRhythm's public catalog (https://tokenrhythm.studio/api/models):
# id -> (context window, accepts images, reasoning control). Every one calls tools and reasons.
# The keyed /v1/models listing appends anything newer at pick time; this table is the offline
# picker and the capability source, since TokenRhythm is not in models.dev. Default model leads.
_MODELS: dict[str, tuple[int, bool, str]] = {
    "deepseek-v4-pro-0813": (1_000_000, False, _TOGGLE_EFFORT),
    "deepseek-v4-flash-0731": (1_000_000, False, _TOGGLE_EFFORT),
    "deepseek-flash": (1_000_000, True, _TOGGLE_EFFORT),
    "glm-5.3": (1_048_576, False, _ALWAYS),
    "glm-5.3-flash": (1_048_576, True, _ALWAYS),
    "glm-5.2": (1_000_000, False, _ALWAYS),
    "glm-5.1": (200_000, False, _ALWAYS),
    "kimi-k2.7-code": (256_000, True, _ALWAYS),
    "kimi-k2.6": (256_000, True, _ALWAYS),
    "qwen3.8-max": (1_000_000, True, _TOGGLE),
    "qwen3.8-flash": (1_000_000, True, _TOGGLE),
    "qwen3.8-27b": (1_000_000, True, _TOGGLE),
    "qwen3.7-max": (1_000_000, False, _TOGGLE),
    "qwen3.7-flash": (1_000_000, True, _TOGGLE),
    "minimax-m2.7": (200_000, False, _ALWAYS),
    "mimo-v2.5-pro": (256_000, False, _ALWAYS),
    "seed-2.1-pro": (262_144, True, _TOGGLE),
    "seed-2.1-turbo": (262_144, True, _TOGGLE),
    "longcat-2.0": (1_000_000, False, _ALWAYS),
}

_DISABLE_EFFORTS = frozenset({"none", "false", "disabled", "off"})


def _reasoning_control(model: str | None) -> str:
    """The model's reasoning control; ids the table does not know get ``always`` (send nothing),
    since a disable sent to a must-think model is a hard 400."""
    entry = _MODELS.get((model or "").strip().rsplit("/", 1)[-1].lower())
    return entry[2] if entry else _ALWAYS


def _disable_requested(reasoning_config: dict) -> bool:
    """``enabled: false``, or Off reported through the effort field (#107238)."""
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    return reasoning_config.get("enabled") is False or effort in _DISABLE_EFFORTS


class TokenRhythmProfile(ProviderProfile):
    """TokenRhythm: ``extra_body.thinking`` toggle, top-level ``reasoning_effort`` for DeepSeek only."""

    def default_reasoning_config(self, model: str | None = None) -> dict | None:
        """Unset effort on a main-loop turn: thinking on at medium (sent as ``high``) for DeepSeek,
        whose v4 models otherwise answer without thinking. The other families think by default.
        Auxiliary calls never take this default, so an unset aux call sends nothing."""
        return {"enabled": True, "effort": "medium"} if _reasoning_control(model) == _TOGGLE_EFFORT else None

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        control = _reasoning_control(model)
        if not isinstance(reasoning_config, dict) or control == _ALWAYS:
            return {}, {}
        if _disable_requested(reasoning_config):
            return {"thinking": {"type": "disabled"}}, {}
        if control == _TOGGLE:
            return {}, {}
        return thinking_toggle_extras(
            reasoning_config, TOKENRHYTHM_DEEPSEEK_EFFORTS, TOKENRHYTHM_DEEPSEEK_OVERRIDES, always_emit_toggle=True
        )


tokenrhythm = TokenRhythmProfile(
    name="tokenrhythm", aliases=("token-rhythm",), display_name="TokenRhythm",
    description="TokenRhythm — OpenAI-compatible gateway to DeepSeek, GLM, Kimi, Qwen, MiniMax and more",
    signup_url="https://tokenrhythm.studio/account/keys",
    env_vars=("TOKENRHYTHM_API_KEY", "TOKENRHYTHM_BASE_URL"),
    base_url="https://tokenrhythm.studio/v1", auth_type="api_key",
    default_aux_model="deepseek-v4-flash-0731",
    fallback_models=tuple(_MODELS),
    model_capabilities={
        model: {"context_window": context, "supports_tools": True, "supports_reasoning": True,
                "supports_vision": vision}
        for model, (context, vision, _control) in _MODELS.items()
    },
)

register_provider(tokenrhythm)

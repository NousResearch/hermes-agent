"""ZAI / GLM provider profiles.

OpenAI-compatible route (``zai``): GLM-4.5+ defaults to thinking ON, so ``reasoning_config``
is translated to ``extra_body.thinking``; GLM-5.2/5.3 also take a native ``reasoning_effort``.
Anthropic Messages route (``zai-coding`` / ``zai-coding-cn``, the Coding Plan endpoints):
thinking + ``output_config.effort`` via the core adapter's GLM branch.
"""

import re
from typing import Any

from agent import reasoning_effort as re_
from providers import register_provider
from providers.base import ProviderProfile

_GLM_VERSION_RE = re.compile(r"^glm-(\d+)(?:\.(\d+))?")
# Alias spellings seen on relays (Fireworks ``glm-5p2``, ``zai-org-glm-5-2``…).
_GLM_5_3_TOKENS = ("glm-5.3", "glm-5-3", "glm-5p3")
_GLM_5_2_TOKENS = ("glm-5.2", "glm-5-2", "glm-5p2") + _GLM_5_3_TOKENS


def _model_supports_thinking(model: str | None) -> bool:
    """GLM thinking-capable model families: glm-4.5 and later (4.5, 4.6, 5…)."""
    match = _GLM_VERSION_RE.match((model or "").strip().lower())
    return bool(match) and (int(match.group(1)), int(match.group(2) or 0)) >= (4, 5)


def _has_token(model: str | None, tokens: tuple[str, ...]) -> bool:
    m = (model or "").strip().lower()
    return any(token in m for token in tokens)


def _glm_5_2_reasoning_effort(reasoning_config: dict | None, *, model: str | None = None) -> str | None:
    """Hermes effort -> GLM vocabulary (5.2: high/max; 5.3: low..max). Below-floor
    efforts clamp to the floor; disabled/unset leaves the server default."""
    effort = re_.requested_effort(reasoning_config)
    if effort is None or effort == "none":
        return None
    if _has_token(model, _GLM_5_3_TOKENS):
        efforts, overrides, floor = re_.GLM53_EFFORTS, re_.GLM53_OVERRIDES, "low"
    else:
        efforts, overrides, floor = re_.GLM52_EFFORTS, re_.GLM52_OVERRIDES, "high"
    clamped = re_.clamp_effort(effort, efforts, overrides)
    return clamped if clamped in efforts else floor


# Per-model capability declarations (canonical model_overrides schema). GLM context windows are
# already covered by DEFAULT_CONTEXT_LENGTHS / the live catalog, so only the fields the generic
# path leaves unknown or gets wrong are declared: reasoning on for GLM-4.5+, vision ONLY for the
# V families — text-only GLM models fail closed so images are never sent to endpoints that
# reject them (the provider-aware media fallback, mirroring ZCode's builtin catalog).
_TEXT_ONLY = {"supports_reasoning": True, "supports_vision": False, "model_family": "glm"}
_VISION = {"supports_vision": True, "model_family": "glm"}
# GLM-5.3-Flash is the natively multimodal member of the 5.3 family: ZCode's dedicated
# ``.*glm-5\.3-flash.*`` rule overrides the text-only family rule with supportsImage (no
# apiTypeMatch, so every wire incl. anthropic-messages), and docs.z.ai/devpack/latest-model
# calls it multimodal. Reasoning + 1M window stay as for the family.
_GLM_53_FLASH = dict(_TEXT_ONLY, supports_vision=True, context_window=1_048_576)

GLM_CAPABILITIES: dict[str, dict[str, Any]] = {
    # ``[1m]`` slugs are the spellings docs.z.ai recommends for the Coding Plan (Claude Code
    # defaults); model_capabilities is an exact-key lookup, so they need their own entries.
    "glm-5.3": dict(_TEXT_ONLY, context_window=1_048_576),
    "glm-5.3[1m]": dict(_TEXT_ONLY, context_window=1_048_576),
    "glm-5.3-flash": dict(_GLM_53_FLASH),
    "glm-5.3-flash[1m]": dict(_GLM_53_FLASH),
    "glm-5.2": dict(_TEXT_ONLY, context_window=1_048_576),
    "glm-5.2[1m]": dict(_TEXT_ONLY, context_window=1_048_576),
    "glm-5.1": dict(_TEXT_ONLY), "glm-5.1-highspeed": dict(_TEXT_ONLY),
    "glm-5": dict(_TEXT_ONLY), "glm-5-turbo": dict(_TEXT_ONLY),
    "glm-4.7": dict(_TEXT_ONLY), "glm-4.7-flash": dict(_TEXT_ONLY), "glm-4.7-flashx": dict(_TEXT_ONLY),
    "glm-4.6": dict(_TEXT_ONLY),
    "glm-4.5": dict(_TEXT_ONLY), "glm-4.5-air": dict(_TEXT_ONLY), "glm-4.5-flash": dict(_TEXT_ONLY),
    "glm-5v-turbo": dict(_VISION),
    "glm-4.6v": dict(_VISION), "glm-4.6v-flash": dict(_VISION), "glm-4.6v-flashx": dict(_VISION),
    "glm-4.1v-thinking-flash": dict(_VISION), "glm-4.1v-thinking-flashx": dict(_VISION),
    "glm-4v-flash": dict(_VISION),
}


class ZaiProfile(ProviderProfile):
    """Z.AI / GLM — extra_body.thinking on/off + GLM-5.2 reasoning_effort.

    The chat_completions projection below applies only to OpenAI-compatible
    routes. The Anthropic Messages route (api_mode="anthropic_messages",
    ``zai-coding`` / ``zai-coding-cn``) maps thinking through the core adapter's
    GLM branch (``agent.anthropic_adapter._glm_thinking_kwargs``), which sends
    ``thinking`` + ``output_config.effort`` instead.
    """

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        if self.api_mode == "anthropic_messages":
            # The Anthropic Messages transport maps reasoning in the adapter; these
            # chat_completions fields would otherwise land as stray extra_body keys.
            return extra_body, top_level
        is_5_2 = _has_token(model, _GLM_5_2_TOKENS)
        if not _model_supports_thinking(model) and not is_5_2:
            return extra_body, top_level
        # Only emit when the user expressed a preference (server default = enabled).
        if isinstance(reasoning_config, dict):
            enabled = reasoning_config.get("enabled") is not False
            if not enabled and _has_token(model, _GLM_5_3_TOKENS):
                # GLM-5.3 rejects disabled thinking; low is its lightest supported mode.
                enabled = True
                top_level["reasoning_effort"] = "low"
            # Preserved Thinking: with thinking on, the standard endpoint drops replayed prior-turn
            # reasoning_content unless clear_thinking is false (docs.z.ai/guides/capabilities/thinking-mode).
            extra_body["thinking"] = {"type": "enabled", "clear_thinking": False} if enabled else {"type": "disabled"}
        if is_5_2:
            effort = _glm_5_2_reasoning_effort(reasoning_config, model=model)
            if effort is not None:
                top_level["reasoning_effort"] = effort
        return extra_body, top_level

    def get_max_tokens(self, model: str | None) -> int | None:
        """Per-model output caps for the Z.AI endpoints (ZCode builtin catalog values):
        GLM-5.2/5.3 support 128K output; 4.6/4.7 and 5V-Turbo 131072; 4.6V-family 32768."""
        m = (model or "").strip().lower()
        if _has_token(m, _GLM_5_2_TOKENS):
            return 128_000
        if "glm-4.6v" in m:
            return 32_768
        if any(token in m for token in ("glm-5v-turbo", "glm-4.7", "glm-4.6")):
            return 131_072
        return self.default_max_tokens

    def default_reasoning_config(self, model: str | None = None) -> dict | None:
        """GLM-4.5+ servers think by default (5.2/5.3 default effort = max): return the explicit
        toggle so an unset Hermes effort records the route's own default instead of looking like
        reasoning-off (the reasoning-rejection ladder and unset-effort UX read this). Effort stays
        unset on the wire, leaving the server default (max) in charge."""
        if _model_supports_thinking(model) or _has_token(model, _GLM_5_2_TOKENS):
            return {"enabled": True}
        return None

    def get_model_context_length(self, model: str) -> int | None:
        """GLM-5.2/5.3 native Z.AI endpoints serve a 1M-token window (docs.z.ai/guides/llm/glm-5.2,
        glm-5.3). The provider-agnostic static table carries OpenRouter's serving numbers
        (5.3 = 1.31M), which over-compress on the native route — this hook is consulted before
        the static defaults, so the fix is scoped to the zai profiles. Other GLM families fall
        through to the generic ladder."""
        if _has_token(model, _GLM_5_3_TOKENS) or _has_token(model, _GLM_5_2_TOKENS):
            return 1_048_576
        return None


zai = ZaiProfile(
    name="zai", aliases=("glm", "z-ai", "z.ai", "zhipu"),
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"), display_name="Z.AI (GLM)",
    description="Z.AI / GLM — Zhipu AI models", signup_url="https://z.ai/",
    fallback_models=("glm-5.2", "glm-5", "glm-4-9b"), base_url="https://api.z.ai/api/paas/v4",
    default_aux_model="glm-4.5-flash",
    model_capabilities=GLM_CAPABILITIES,
)

# GLM Coding Plan Anthropic Messages routes (the Claude Code drop-in endpoint). Keys are sent
# as ANTHROPIC_AUTH_TOKEN (Bearer) by the core's GLM endpoint classification.
zai_coding = ZaiProfile(
    name="zai-coding", aliases=("zai-anthropic", "glm-coding", "glm-anthropic"),
    api_mode="anthropic_messages",
    env_vars=("ZAI_API_KEY", "GLM_API_KEY", "Z_AI_API_KEY"), display_name="Z.AI Coding Plan (GLM)",
    description="Z.AI / GLM Coding Plan via the Anthropic Messages endpoint", signup_url="https://z.ai/",
    fallback_models=("glm-5.3", "glm-5.3-flash", "glm-5.2"),
    base_url="https://api.z.ai/api/anthropic",
    model_capabilities=GLM_CAPABILITIES,
)

zai_coding_cn = ZaiProfile(
    name="zai-coding-cn", aliases=("bigmodel-coding", "bigmodel-anthropic", "glm-coding-cn"),
    api_mode="anthropic_messages",
    env_vars=("BIGMODEL_API_KEY", "ZAI_API_KEY", "GLM_API_KEY"),
    display_name="BigModel Coding Plan (GLM)",
    description="BigModel / GLM Coding Plan via the Anthropic Messages endpoint",
    signup_url="https://open.bigmodel.cn/",
    fallback_models=("glm-5.3", "glm-5.3-flash", "glm-5.2"),
    base_url="https://open.bigmodel.cn/api/anthropic",
    model_capabilities=GLM_CAPABILITIES,
)

register_provider(zai)
register_provider(zai_coding)
register_provider(zai_coding_cn)

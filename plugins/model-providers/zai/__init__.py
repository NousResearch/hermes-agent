"""ZAI / GLM provider profile.

GLM-4.5+ defaults to thinking ON, so ``reasoning_config`` is translated to
``extra_body.thinking``; GLM-5.2/5.3 also take a native ``reasoning_effort``.
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


def _is_glm_5_2(model: str | None) -> bool:
    """Detect GLM-5.2/5.3 (reasoning_effort-capable) across alias spellings.

    Covers the canonical ``glm-5.2``/``glm-5.3`` plus the ``glm-5-2`` /
    ``glm-5p2`` variants seen on relays (Fireworks ``glm-5p2``, etc.) and any
    vendor-prefixed form (``z-ai/glm-5.2``, ``zai-org-glm-5-2``).  GLM-5.3
    uses the same base model as 5.2 (post-training gains only) and exposes
    the same ``reasoning_effort`` knob (verified live 2026-08-14: the
    coding-plan endpoint accepts ``reasoning_effort: high`` for glm-5.3).
    """
    m = (model or "").strip().lower()
    if not m:
        return False
    return any(
        token in m
        for token in ("glm-5.2", "glm-5-2", "glm-5p2", "glm-5.3", "glm-5-3", "glm-5p3")
    )


def _is_glm_5_3(model: str | None) -> bool:
    """Detect GLM-5.3 specifically — it has a wider effort vocabulary.

    5.2 accepts only ``high``/``max``; 5.3 accepts a graded
    ``low``/``medium``/``high``/``max`` scale (verified live, issue #91789),
    so effort mapping must pick the vocabulary per model.
    """
    m = (model or "").strip().lower()
    if not m:
        return False
    return any(token in m for token in ("glm-5.3", "glm-5-3", "glm-5p3"))


def _glm_5_2_reasoning_effort(
    reasoning_config: dict | None, *, model: str | None = None
) -> str | None:
    """Map Hermes reasoning effort onto GLM's native vocabulary.

    GLM-5.2 supports two enabled effort levels (``high``/``max``);
    GLM-5.3 supports the graded ``low``/``medium``/``high``/``max`` scale.
    ``xhigh``/``max``/``ultra`` request the top tier; anything below the
    model's floor clamps to that floor. When reasoning is explicitly
    disabled, or no effort preference is supplied, the server default is
    left untouched.
    """
    if not isinstance(reasoning_config, dict):
        return None
    if reasoning_config.get("enabled") is False:
        return None

    effort = (reasoning_config.get("effort") or "").strip().lower()
    if not effort or effort == "none":
        return None

    # Per-model vocabulary declared in agent.reasoning_effort; xhigh rounds
    # up to max on both. 5.2 cannot think less than high; 5.3 accepts a
    # graded scale down to low (issue #91789).
    from agent.reasoning_effort import (
        GLM52_EFFORTS,
        GLM52_OVERRIDES,
        GLM53_EFFORTS,
        GLM53_OVERRIDES,
        clamp_effort,
    )

    if _is_glm_5_3(model):
        efforts, overrides, floor = GLM53_EFFORTS, GLM53_OVERRIDES, "low"
    else:
        efforts, overrides, floor = GLM52_EFFORTS, GLM52_OVERRIDES, "high"

    clamped = clamp_effort(effort, efforts, overrides)
    return clamped if clamped in efforts else floor


class ZaiProfile(ProviderProfile):
    """Z.AI / GLM — extra_body.thinking on/off + GLM-5.2 reasoning_effort."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        is_5_2 = _has_token(model, _GLM_5_2_TOKENS)
        if not _model_supports_thinking(model) and not is_5_2:
            return extra_body, top_level
        # Only emit when the user expressed a preference (server default = enabled).
        if isinstance(reasoning_config, dict):
            enabled = reasoning_config.get("enabled") is not False
            extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}

        if _is_glm_5_2(model):
            effort = _glm_5_2_reasoning_effort(reasoning_config, model=model)
            if effort is not None:
                top_level["reasoning_effort"] = effort
        return extra_body, top_level


zai = ZaiProfile(
    name="zai", aliases=("glm", "z-ai", "z.ai", "zhipu"),
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"), display_name="Z.AI (GLM)",
    description="Z.AI / GLM — Zhipu AI models", signup_url="https://z.ai/",
    fallback_models=("glm-5.2", "glm-5", "glm-4-9b"), base_url="https://api.z.ai/api/paas/v4",
    default_aux_model="glm-4.5-flash",
)

register_provider(zai)

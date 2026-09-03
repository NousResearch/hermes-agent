"""ZAI / GLM provider profile.

Z.AI's GLM-4.5-and-later chat models default to thinking-mode ON when the
request omits ``thinking``.  Hermes' ``reasoning_config = {"enabled": False}``
was previously a silent no-op on this route — the base profile emits nothing,
so users who turned thinking off (desktop toggle, ``/reasoning none``,
``reasoning_effort: none``/``false`` in config.yaml) kept burning thinking
tokens on every turn.

:meth:`ZaiProfile.build_api_kwargs_extras` translates the Hermes reasoning
config into the wire shape Z.AI's OpenAI-compat endpoint expects:

    {"extra_body": {"thinking": {"type": "enabled" | "disabled"}}}

When no reasoning preference is set (``reasoning_config is None``) the field
is omitted so the server default applies, matching prior behavior.  GLM
models before 4.5 (e.g. ``glm-4-9b``) don't accept ``thinking`` and are left
untouched.

GLM-5.2 additionally exposes a native ``reasoning_effort`` knob with exactly
two enabled levels — ``high`` and ``max`` — on the OpenAI-compatible endpoint
(per Z.AI / BigModel docs).  Hermes' richer effort scale is collapsed onto
those two so the user's effort preference actually reaches the model instead
of being silently dropped.

GLM-5.3 reaches one level lower (``low``/``high``/``max``) but drops the
``thinking`` toggle: it always thinks, and rejects both a disable marker and
any effort word outside that set with ``1210``.  So on 5.3 the toggle is
never sent and an explicit disable resolves to the cheapest level rather
than to nothing — omitting the field would leave the server default, which
is the *most* expensive tier.
"""

from __future__ import annotations

import re
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

_GLM_VERSION_RE = re.compile(r"^glm-(\d+)(?:\.(\d+))?")


def _model_supports_thinking(model: str | None) -> bool:
    """GLM thinking-capable model families: glm-4.5 and later (4.5, 4.6, 5…)."""
    m = (model or "").strip().lower()
    match = _GLM_VERSION_RE.match(m)
    if not match:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return (major, minor) >= (4, 5)


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

    5.2 accepts only ``high``/``max``; 5.3 reaches one level lower with
    ``low``/``high``/``max`` (the server enumerates that set in its own 1210
    rejection), so effort mapping must pick the vocabulary per model. 5.3
    also refuses to turn thinking off at all — see
    :func:`_glm_5_2_reasoning_effort` and :meth:`ZaiProfile.build_api_kwargs_extras`.
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
        # GLM-5.3 has no off switch: the wire rejects a disable with
        # ``1210 ... cannot be disabled``, and omitting the field leaves the
        # server default (max — the most expensive tier), so "off" would cost
        # *more* than any level the user could have asked for. The cheapest
        # supported level is as close as the wire gets. 5.2 has no such trap.
        return "low" if _is_glm_5_3(model) else None

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

        if not _model_supports_thinking(model) and not _is_glm_5_2(model):
            return extra_body, top_level

        # Only emit when the user expressed a preference; omitting the field
        # keeps the server default (enabled) exactly as before. The disable
        # marker is skipped for GLM-5.3, which rejects it with 1210 and would
        # fail every request from a user who turned thinking off; that
        # preference is carried by ``reasoning_effort: low`` instead.
        if isinstance(reasoning_config, dict):
            enabled = reasoning_config.get("enabled") is not False
            if enabled or not _is_glm_5_3(model):
                extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}

        if _is_glm_5_2(model):
            effort = _glm_5_2_reasoning_effort(reasoning_config, model=model)
            if effort is not None:
                top_level["reasoning_effort"] = effort

        return extra_body, top_level


zai = ZaiProfile(
    name="zai",
    aliases=("glm", "z-ai", "z.ai", "zhipu"),
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
    display_name="Z.AI (GLM)",
    description="Z.AI / GLM — Zhipu AI models",
    signup_url="https://z.ai/",
    fallback_models=(
        "glm-5.2",
        "glm-5",
        "glm-4-9b",
    ),
    base_url="https://api.z.ai/api/paas/v4",
    default_aux_model="glm-4.5-flash",
)

register_provider(zai)

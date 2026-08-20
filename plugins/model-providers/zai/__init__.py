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

GLM-5.2 and GLM-5.3 additionally expose a native ``reasoning_effort`` knob
with exactly two enabled levels — ``high`` and ``max`` — on the
OpenAI-compatible endpoint (per Z.AI / BigModel docs).  Hermes' richer effort
scale is collapsed onto those two so the user's effort preference actually
reaches the model instead of being silently dropped.

z.ai also documents "Preserved Thinking" (``clear_thinking: false`` +
reasoning replay, issue #11483), but live probes (2026-08-15) show the
OpenAI-compat endpoint silently drops replayed ``reasoning_content`` from
model attention — only the Anthropic wire honors it — so it is deliberately
NOT emitted here.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

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


def _supports_reasoning_effort(model: str | None) -> bool:
    """Detect GLM models with the native ``reasoning_effort`` dial (5.2, 5.3).

    Covers the canonical ``glm-5.2``/``glm-5.3`` plus the ``glm-5-2`` /
    ``glm-5p2``-style variants seen on relays (Fireworks ``glm-5p2``, etc.)
    and any vendor-prefixed form (``z-ai/glm-5.2``,
    ``accounts/fireworks/models/glm-5p2``, ``zai-org-glm-5-2``).

    Boundary-safe on purpose: a bare substring check would classify
    ``glm-5.30`` or ``notglm-5.3`` as known variants.  The version digit
    must terminate at end-of-string or a non-alphanumeric character, and
    ``glm`` must not be preceded by another word character.
    """
    return (
        _matches_glm_5_minor(model, "2")
        or _matches_glm_5_minor(model, "3")
    )


_GLM_5_VARIANT_RE = re.compile(r"(?<![a-z0-9])glm-5[.\-p]?(\d)(?![0-9a-z])")


def _matches_glm_5_minor(model: str | None, minor: str) -> bool:
    """True when the model id names GLM-5.<minor> in any alias spelling."""
    m = (model or "").strip().lower()
    if not m:
        return False
    return any(
        match.group(1) == minor for match in _GLM_5_VARIANT_RE.finditer(m)
    )


def _is_glm_5_3(model: str | None) -> bool:
    """Detect GLM-5.3 across the alias spellings providers use."""
    return _matches_glm_5_minor(model, "3")


def _reasoning_effort_for_config(reasoning_config: dict | None) -> str | None:
    """Map Hermes reasoning effort onto GLM-5.2/5.3's native ``high``/``max``.

    These models only expose two enabled effort levels. ``xhigh``/``max``/
    ``ultra`` request the top tier; everything else that is enabled requests
    ``high`` (the minimum thinking level). When reasoning is explicitly
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

    if effort in {"xhigh", "max", "ultra"}:
        return "max"
    # low / medium / minimal / high all clamp to the model's minimum: high.
    return "high"


class ZaiProfile(ProviderProfile):
    """Z.AI / GLM — thinking on/off + GLM-5.2/5.3 reasoning_effort."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not _model_supports_thinking(model) and not _supports_reasoning_effort(model):
            return extra_body, top_level

        # Only emit when the user expressed a preference; omitting the field
        # keeps the server default (enabled) exactly as before.
        # GLM-5.3 silently ignores thinking.disabled (no error, but thinking
        # still runs) — don't send a no-op param for it. reasoning_effort
        # (wired below) is 5.3's actual effort control.
        # (No clear_thinking here: see module docstring — z.ai's OpenAI-compat
        # wire drops replayed reasoning_content from model attention.)
        if isinstance(reasoning_config, dict):
            enabled = reasoning_config.get("enabled") is not False
            if enabled or not _is_glm_5_3(model):
                extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}
            else:
                # GLM-5.3 ignores thinking.disabled (no error, thinking
                # still runs server-side).  The user's "disable reasoning"
                # preference is a silent no-op on this wire — say so at
                # debug level so it isn't mistaken for a wiring bug
                # (review point on PR #86433).  reasoning_effort below is
                # 5.3's actual effort control.
                logger.debug(
                    "zai: reasoning disabled for %s but GLM-5.3 ignores "
                    "thinking.disabled on the OpenAI-compat wire; leaving "
                    "server default (thinking on)", model,
                )

        if _supports_reasoning_effort(model):
            effort = _reasoning_effort_for_config(reasoning_config)
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

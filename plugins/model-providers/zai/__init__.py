"""ZAI / GLM provider profile.

Z.AI's GLM-4.5-and-later chat models default to thinking-mode ON when the
request omits ``thinking``.  Hermes' ``reasoning_config = {"enabled": False}``
was previously a silent no-op on this route — the base profile emits nothing,
so users who turned thinking off (desktop toggle, ``/reasoning none``,
``reasoning_effort: none``/``false`` in config.yaml) kept burning thinking
tokens on every turn.

GLM-5.2+ additionally supports an ``effort`` field (``high`` / ``max``) inside
the thinking object.

:meth:`ZaiProfile.build_api_kwargs_extras` translates the Hermes reasoning
config into the wire shape Z.AI's OpenAI-compat endpoint expects:

    {"extra_body": {"thinking": {"type": "enabled" | "disabled"}}}

When no reasoning preference is set (``reasoning_config is None``) the field
is omitted so the server default applies, matching prior behavior.  GLM
models before 4.5 (e.g. ``glm-4-9b``) don't accept ``thinking`` and are left
untouched.
"""

from __future__ import annotations

import re
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

_GLM_VERSION_RE = re.compile(r"^glm-(\d+)(?:\.(\d+))?")


def _strip_vendor_prefix(m: str) -> str:
    for prefix in ("zai/", "z-ai/", "zhipu/"):
        if m.startswith(prefix):
            return m[len(prefix):]
    return m


def _model_supports_thinking(model: str | None) -> bool:
    """GLM thinking-capable model families: glm-4.5 and later (4.5, 4.6, 5…)."""
    m = _strip_vendor_prefix((model or "").strip().lower())
    match = _GLM_VERSION_RE.match(m)
    if not match:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return (major, minor) >= (4, 5)


def _model_supports_effort(model: str | None) -> bool:
    """Return True for GLM models that accept the ``effort`` field.

    GLM-5.2 is currently the only model that supports effort. Match
    ``glm-5.2`` exactly or as a hyphen/suffix boundary (e.g.
    ``glm-5.2-preview``) but not ``glm-5.20``.
    """
    m = _strip_vendor_prefix((model or "").lower().strip())
    return m == "glm-5.2" or m.startswith("glm-5.2-")


class ZaiProfile(ProviderProfile):
    """Z.AI / GLM — extra_body.thinking enabled/disabled + effort for GLM-5.2."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not _model_supports_thinking(model):
            return extra_body, top_level

        # Only emit when the user expressed a preference; omitting the field
        # keeps the server default (enabled) exactly as before.
        if isinstance(reasoning_config, dict):
            enabled = reasoning_config.get("enabled") is not False
            thinking: dict[str, Any] = {"type": "enabled" if enabled else "disabled"}

            # Effort is GLM-5.2+ only. Older models (5.1, 5, 4.x) ignore the
            # field, so don't send it — keeps the wire format clean.
            if enabled and _model_supports_effort(model):
                effort = (reasoning_config.get("effort") or "").strip().lower()
                # Map Hermes effort levels to GLM's high/max.
                # Hermes valid efforts: none, minimal, low, medium, high, xhigh.
                if effort in {"xhigh", "max"}:
                    thinking["effort"] = "max"
                elif effort == "high":
                    thinking["effort"] = "high"
                # Lower efforts (none/minimal/low/medium) → omit, GLM uses server default.

            extra_body["thinking"] = thinking

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

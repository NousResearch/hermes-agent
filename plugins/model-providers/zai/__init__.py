"""ZAI / GLM provider profile.

GLM-5.x is a reasoning (thinking) family: every response carries a
``reasoning_content`` field alongside ``content``, and the API accepts a
top-level ``reasoning_effort`` parameter that controls thinking depth
(``minimal`` disables it entirely; ``low``/``medium``/``high`` scale it).
``xhigh`` is a Hermes-internal level with no Z.AI equivalent, so it is
clamped to ``high``.

Without this profile the agent's ``reasoning_effort`` config is a no-op for
Z.AI: the generic transport gate only forwards reasoning params for
OpenRouter / Nous / GitHub / LM Studio, so the effort level never reaches
the wire and the model keeps its server default regardless of config.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Z.AI accepts these top-level reasoning_effort values. There is no distinct
# ``xhigh`` level — empirically it yields fewer reasoning tokens than ``high``
# — so Hermes' xhigh is clamped to high.
_ZAI_EFFORT_MAP = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
}


def _model_supports_reasoning(model: str | None) -> bool:
    """GLM-5.x thinking-capable model families.

    Earlier GLM generations (glm-4.x, glm-4-9b, glm-4.5-flash, …) are not
    reasoning models and do not accept ``reasoning_effort``.
    """
    m = (model or "").strip().lower()
    return m.startswith("glm-5")


class ZaiProfile(ProviderProfile):
    """Z.AI / GLM — top-level reasoning_effort for GLM-5.x thinking models."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not _model_supports_reasoning(model):
            # glm-4.x and unknowns — leave the wire format untouched.
            return extra_body, top_level

        if not isinstance(reasoning_config, dict):
            return extra_body, top_level

        if reasoning_config.get("enabled") is False:
            # Explicit disable: emit minimal so Z.AI turns thinking off.
            top_level["reasoning_effort"] = "minimal"
            return extra_body, top_level

        effort = (reasoning_config.get("effort") or "").strip().lower()
        if effort:
            top_level["reasoning_effort"] = _ZAI_EFFORT_MAP.get(effort, "medium")

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

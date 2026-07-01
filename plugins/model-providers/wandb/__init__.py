"""W&B (Weights & Biases) inference provider profile.

W&B hosts GLM-5.2 on a vLLM backend at api.inference.wandb.ai. The endpoint
accepts:
  - reasoning_effort as a top-level param ("high" or "max")
  - chat_template_kwargs.enable_thinking as extra_body (false = no thinking)
  - A browser User-Agent is required (Cloudflare blocks urllib default)

GLM-5.2 only recognises "high" and "max" for reasoning_effort; other values
(minimal/low/medium) are silently ignored and the model defaults to Think Max,
so this profile maps unsupported lower efforts to the lightest supported value
("high") instead of omitting the parameter. The z.ai "thinking" parameter is
also ignored by vLLM.

This profile overrides build_api_kwargs_extras to emit the correct params.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _effort_to_glm(effort: str) -> str | None:
    """Map Hermes reasoning effort levels to GLM-5.2's accepted values.

    GLM-5.2 only supports "high" and "max". Unrecognised values silently
    fall back to Think Max (the model default), so we explicitly map:
      xhigh → "max"  (GLM-5.2's deepest thinking)
      high  → "high" (lighter, ~3× faster)
      low/minimal/medium → "high" (lightest supported thinking)
      unset/unknown → None (omit param — let GLM-5.2 use its default)
    """
    e = (effort or "").strip().lower()
    if e in {"xhigh", "max"}:
        return "max"
    if e in {"minimal", "low", "medium", "high"}:
        return "high"
    return None


class WandbProfile(ProviderProfile):
    """W&B inference — GLM-5.2 on vLLM with reasoning_effort + enable_thinking."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not isinstance(reasoning_config, dict):
            return extra_body, top_level

        _enabled = reasoning_config.get("enabled", True)
        _effort = (reasoning_config.get("effort") or "").strip().lower()

        # Thinking disabled → chat_template_kwargs.enable_thinking=false
        # This is the ONLY way to disable thinking on GLM-5.2 vLLM.
        if _effort == "none" or _enabled is False:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            return extra_body, top_level

        # Thinking enabled → emit reasoning_effort as top-level param
        glm_effort = _effort_to_glm(_effort)
        if glm_effort:
            top_level["reasoning_effort"] = glm_effort

        return extra_body, top_level


wandb = WandbProfile(
    name="wandb",
    aliases=("wandb-ai", "weights-and-biases"),
    env_vars=("WANDB_API_KEY",),
    display_name="W&B Inference",
    description="Weights & Biases inference API — GLM-5.2 on vLLM",
    signup_url="https://wandb.ai",
    base_url="https://api.inference.wandb.ai/v1",
    fallback_models=("zai-org/GLM-5.2",),
    default_max_tokens=65536,
    # Cloudflare blocks the default urllib User-Agent; the transport
    # adds a browser UA via default_headers.
    default_headers={"User-Agent": "Mozilla/5.0"},
)

register_provider(wandb)

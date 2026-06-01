"""StepFun provider profile.

Step-3.7-flash (and future reasoning-capable StepFun models) require
top-level ``reasoning_effort`` (low/medium/high) on the Chat Completions
path. See StepFun docs: "The Chat Completions API uses ``reasoning_effort``
to control the effort level".

This profile implements the minimal hook so the native ``stepfun`` provider
actually uses the model's agentic strengths instead of defaulting to the
cheap "low" tier.

Pattern copied from DeepSeekProfile (which does the same top-level
reasoning_effort emission).
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _model_supports_reasoning_effort(model: str | None) -> bool:
    """Return True for StepFun models that accept reasoning_effort.

    Currently only step-3.7-flash (and any future *-reasoning or 3.7+ variants).
    We keep it explicit so non-reasoning StepFun models (if they ever appear)
    stay untouched.
    """
    m = (model or "").strip().lower()
    if not m:
        return False
    # Anchor on the ``step-3.7`` token (not a bare ``3.7`` substring) so we
    # don't accidentally match unrelated version strings like ``step-13.7``.
    # ``stepfun/step-3.7-flash`` still matches since the token is a substring.
    if "step-3.7" in m or "reasoning" in m:
        return True
    return False


class StepFunProfile(ProviderProfile):
    """StepFun Step Plan — emit top-level reasoning_effort for capable models."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not _model_supports_reasoning_effort(model):
            # Older 3.5-flash etc. — leave the wire format exactly as before.
            return extra_body, top_level

        # User explicitly turned reasoning off for this turn.
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            # StepFun does not require a special "disabled" marker the way
            # DeepSeek does, so we simply omit reasoning_effort and let the
            # server use its (low) default. This keeps the request minimal.
            return extra_body, top_level

        # Map Hermes reasoning_config.effort to StepFun values.
        # Hermes uses: low, medium, high, (sometimes xhigh).
        # StepFun accepts exactly: low, medium, high.
        effort = "medium"  # safe default for agent workloads
        if isinstance(reasoning_config, dict):
            raw = (reasoning_config.get("effort") or "").strip().lower()
            if raw in {"low", "medium", "high"}:
                effort = raw
            elif raw in {"xhigh", "max"}:
                effort = "high"

        top_level["reasoning_effort"] = effort
        return extra_body, top_level


stepfun = StepFunProfile(
    name="stepfun",
    aliases=("step", "stepfun-coding-plan"),
    default_aux_model="step-3.7-flash",
    env_vars=("STEPFUN_API_KEY",),
    base_url="https://api.stepfun.ai/step_plan/v1",
)

register_provider(stepfun)

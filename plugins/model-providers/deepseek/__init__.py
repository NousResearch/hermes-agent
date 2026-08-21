"""DeepSeek provider profile.

DeepSeek's V4 family defaults to thinking-mode ON when ``extra_body.thinking``
is unset.  The API then returns ``reasoning_content`` and starts enforcing
the contract that subsequent turns echo it back; combined with how Hermes
replays history this lands on the notorious HTTP 400
``reasoning_content must be passed back`` error after the first tool call
(#15700, #17212, #17825).

This profile overrides :meth:`build_api_kwargs_extras` to mirror the Kimi /
Moonshot wire shape that DeepSeek's OpenAI-compat endpoint expects:

    {"reasoning_effort": "<low|medium|high|max>",
     "extra_body": {"thinking": {"type": "enabled" | "disabled"}}}

Non-thinking models (``deepseek-v3-*`` variants) are left as no-ops so we
don't perturb the V3 wire format.

The legacy aliases ``deepseek-chat`` / ``deepseek-reasoner`` were retired on
2026-07-24.  Use ``deepseek-v4-flash`` or ``deepseek-v4-pro``; Hermes remaps
the retired IDs in ``hermes_cli.model_normalize``.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _model_supports_thinking(model: str | None) -> bool:
    """DeepSeek thinking-capable model families.

    Currently covers the V4 family (``deepseek-v4-pro``, ``deepseek-v4-flash``,
    and any future ``deepseek-v4-*`` variants).  Retired aliases are remapped
    before requests leave Hermes, so they are not listed here.
    """
    m = (model or "").strip().lower()
    if not m:
        return False
    if m.startswith("deepseek-v") and not m.startswith("deepseek-v3"):
        # deepseek-v4-*, deepseek-v5-*, etc. — every V4+ generation has
        # thinking. v3 explicitly excluded.
        return True
    return False


class DeepSeekProfile(ProviderProfile):
    """DeepSeek — extra_body.thinking + top-level reasoning_effort."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not _model_supports_thinking(model):
            # V3 / unknown — leave wire format untouched, current behavior.
            return extra_body, top_level

        # Determine enabled/disabled.  Default is enabled to match DeepSeek's
        # API default; the API requires this to be set explicitly to avoid the
        # reasoning_content echo trap on subsequent turns.
        enabled = True
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            enabled = False

        extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}

        if not enabled:
            return extra_body, top_level

        # Effort mapping via the shared vocabulary in agent.reasoning_effort
        # (DeepSeek V4: low/medium/high/max, xhigh rounds up to max). When no
        # effort is set we omit reasoning_effort so DeepSeek applies its
        # server default (currently high).
        if isinstance(reasoning_config, dict):
            from agent.reasoning_effort import (
                DEEPSEEK_V4_EFFORTS,
                DEEPSEEK_V4_OVERRIDES,
                clamp_effort,
            )

            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort and effort != "none":
                clamped = clamp_effort(
                    effort, DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES
                )
                if clamped in DEEPSEEK_V4_EFFORTS:
                    top_level["reasoning_effort"] = clamped

        return extra_body, top_level


deepseek = DeepSeekProfile(
    name="deepseek",
    aliases=("deepseek-chat",),
    env_vars=("DEEPSEEK_API_KEY",),
    display_name="DeepSeek",
    description="DeepSeek — native DeepSeek API",
    signup_url="https://platform.deepseek.com/",
    fallback_models=(
        "deepseek-v4-pro",
        "deepseek-v4-flash",
    ),
    base_url="https://api.deepseek.com/v1",
    default_aux_model="deepseek-v4-flash",
)

register_provider(deepseek)

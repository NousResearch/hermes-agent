"""DeepSeek provider profile.

V4 defaults to thinking ON when ``extra_body.thinking`` is unset, and then
requires ``reasoning_content`` to be echoed back on later turns (HTTP 400 after
the first tool call otherwise). This profile sets ``thinking`` explicitly and
maps effort onto DeepSeek's ``reasoning_effort``; V3 models are left untouched.
Retired ``deepseek-chat``/``deepseek-reasoner`` IDs are remapped in
``hermes_cli.model_normalize`` before reaching here.
"""

from typing import Any

from agent.reasoning_effort import DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES, thinking_toggle_extras
from providers import register_provider
from providers.base import ProviderProfile


# Version-less canonical ids for thinking-capable DeepSeek models. The 2026-09 Flash
# refresh dropped the ``v<N>`` marker from the public id: ``GET /v1/models`` reports
# ``deepseek-flash`` and the API accepts it directly, so the generation check in
# ``build_api_kwargs_extras`` cannot recognise it.
_THINKING_CAPABLE_IDS: frozenset[str] = frozenset({"deepseek-flash"})


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
    name="deepseek", aliases=("deepseek-chat", "deep-seek"), env_vars=("DEEPSEEK_API_KEY",), display_name="DeepSeek",
    description="DeepSeek — native DeepSeek API", signup_url="https://platform.deepseek.com/",
    fallback_models=("deepseek-v4-pro", "deepseek-flash"), base_url="https://api.deepseek.com/v1",
    default_aux_model="deepseek-flash",
)

register_provider(deepseek)

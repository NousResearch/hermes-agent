"""DeepSeek provider profile.

V4 defaults to thinking ON when ``extra_body.thinking`` is unset, and then
requires ``reasoning_content`` to be echoed back on later turns (HTTP 400 after
the first tool call otherwise). This profile sets ``thinking`` explicitly and
maps effort onto DeepSeek's ``reasoning_effort``; V3 models are left untouched.
Retired ``deepseek-chat``/``deepseek-reasoner`` IDs are remapped in
``hermes_cli.model_normalize`` before reaching here.
"""

from typing import Any

from agent.reasoning_effort import DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES, clamp_effort
from providers import register_provider
from providers.base import ProviderProfile


class DeepSeekProfile(ProviderProfile):
    """DeepSeek — extra_body.thinking + top-level reasoning_effort."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        m = (model or "").strip().lower()
        if not m.startswith("deepseek-v") or m.startswith("deepseek-v3"):  # v4+ only; v3 excluded
            return {}, {}
        rc = reasoning_config if isinstance(reasoning_config, dict) else None
        # Always set thinking explicitly (default enabled, matching the API default)
        # to avoid the reasoning_content echo trap on subsequent turns.
        if rc is not None and rc.get("enabled") is False:
            return {"thinking": {"type": "disabled"}}, {}
        top_level: dict[str, Any] = {}
        # No effort -> omit reasoning_effort so DeepSeek applies its server default.
        effort = (rc.get("effort") or "").strip().lower() if rc is not None else ""
        if effort and effort != "none":
            clamped = clamp_effort(effort, DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES)
            if clamped in DEEPSEEK_V4_EFFORTS:
                top_level["reasoning_effort"] = clamped
        return {"thinking": {"type": "enabled"}}, top_level


deepseek = DeepSeekProfile(
    name="deepseek", aliases=("deepseek-chat",), env_vars=("DEEPSEEK_API_KEY",), display_name="DeepSeek",
    description="DeepSeek — native DeepSeek API", signup_url="https://platform.deepseek.com/",
    fallback_models=("deepseek-v4-pro", "deepseek-v4-flash"), base_url="https://api.deepseek.com/v1",
    default_aux_model="deepseek-v4-flash",
)

register_provider(deepseek)

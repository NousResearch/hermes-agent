"""ClinePass subscription provider.

ClinePass shares Cline's OpenAI-compatible gateway but has a separate quota,
credential, and ``cline-pass/*`` model namespace. Its keys do not support model
discovery, so this profile always returns the curated subscription catalog.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class ClinePassProfile(ProviderProfile):
    """ClinePass wire compatibility with discovery explicitly disabled."""

    def fetch_models(self, **kwargs: Any) -> list[str] | None:
        """Never call ``/models``; ClinePass keys receive 404 there."""
        return list(self.fallback_models)

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        effort = ""
        if (
            isinstance(reasoning_config, dict)
            and reasoning_config.get("enabled") is not False
        ):
            effort = str(reasoning_config.get("effort") or "").strip().lower()
        if "mimo-v2.5" in (model or "").lower():
            effort = {"minimal": "low", "xhigh": "high"}.get(effort, effort)
        top_level = (
            {"reasoning_effort": effort}
            if effort in {"minimal", "low", "medium", "high", "xhigh"}
            else {}
        )
        return {}, top_level


cline_pass = ClinePassProfile(
    name="cline-pass",
    aliases=("clinepass",),
    env_vars=("CLINEPASS_API_KEY", "CLINE_API_KEY"),
    display_name="ClinePass",
    description="ClinePass (monthly subscription quota)",
    signup_url="https://app.cline.bot/dashboard/subscription?personal=true",
    fallback_models=(
        "cline-pass/glm-5.2",
        "cline-pass/kimi-k2.7-code",
        "cline-pass/kimi-k2.6",
        "cline-pass/deepseek-v4-pro",
        "cline-pass/deepseek-v4-flash",
        "cline-pass/mimo-v2.5",
        "cline-pass/mimo-v2.5-pro",
        "cline-pass/minimax-m3",
        "cline-pass/qwen3.7-max",
        "cline-pass/qwen3.7-plus",
    ),
    base_url="https://api.cline.bot/api/v1",
    supports_health_check=False,
    supports_vision=True,
    default_max_tokens=65536,
    default_aux_model="cline-pass/deepseek-v4-flash",
)

register_provider(cline_pass)

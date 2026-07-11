"""Cline pay-as-you-go API-credit provider.

Cline exposes an OpenAI-compatible chat-completions gateway but does not
publish a documented model-list endpoint. The curated fallback contains the
agentic models validated against that gateway and preserves Cline's native
``provider/model`` identifiers verbatim.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class ClineApiProfile(ProviderProfile):
    """Cline API wire compatibility and deterministic model catalog."""

    def fetch_models(self, **kwargs: Any) -> list[str] | None:
        """Return the curated catalog without probing an undocumented endpoint."""
        return list(self.fallback_models)

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        effort = ""
        if (
            isinstance(reasoning_config, dict)
            and reasoning_config.get("enabled") is not False
        ):
            effort = str(reasoning_config.get("effort") or "").strip().lower()
        top_level = (
            {"reasoning_effort": effort}
            if effort in {"minimal", "low", "medium", "high", "xhigh"}
            else {}
        )
        return {}, top_level


cline_api = ClineApiProfile(
    name="cline-api",
    env_vars=("CLINE_API_KEY",),
    display_name="Cline API",
    description="Cline API (pay-as-you-go usage credits)",
    signup_url="https://app.cline.bot/dashboard/account",
    fallback_models=(
        "zai/glm-5.2",
        "moonshotai/kimi-k2.7-code",
        "moonshotai/kimi-k2.6",
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash",
    ),
    base_url="https://api.cline.bot/api/v1",
    supports_health_check=False,
    supports_vision=True,
    default_max_tokens=65536,
    default_aux_model="deepseek/deepseek-v4-flash",
)

register_provider(cline_api)

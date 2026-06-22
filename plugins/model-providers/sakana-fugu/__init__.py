"""Sakana Fugu provider profile."""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class SakanaFuguProfile(ProviderProfile):
    """Provider profile for Sakana AI's Fugu API.

    Fugu exposes an OpenAI-compatible endpoint and recommends the Responses API
    for tooling/agent workflows. The official Fugu model catalog currently
    lists only ``high`` reasoning effort for both Fugu and Fugu Ultra, so Hermes
    clamps all requested efforts to ``high`` and avoids forcing reasoning
    summaries unless Sakana changes the catalog.
    """

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            return {}, {}

        effort = "high"

        return {}, {"reasoning": {"effort": effort}}


sakana_fugu = SakanaFuguProfile(
    name="sakana-fugu",
    aliases=("sakana", "fugu", "sakana-ai", "sakanaai"),
    api_mode="codex_responses",
    display_name="Sakana Fugu",
    description="Sakana Fugu — OpenAI-compatible agentic models via the Responses API",
    signup_url="https://console.sakana.ai/",
    env_vars=("SAKANA_API_KEY", "SAKANA_BASE_URL"),
    base_url="https://api.sakana.ai/v1",
    auth_type="api_key",
    supports_vision=True,
    default_aux_model="fugu",
    fallback_models=("fugu-ultra", "fugu"),
    default_stale_timeout_seconds=7200.0,
)

register_provider(sakana_fugu)

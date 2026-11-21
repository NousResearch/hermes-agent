"""Lanseq provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class LanseqProfile(ProviderProfile):
    """Lanseq OpenAI-compatible provider."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not reasoning_config:
            return {}, {}

        enabled = reasoning_config.get("enabled", True)
        effort = (reasoning_config.get("effort") or "").strip().lower()

        if enabled is False:
            effort = "none"

        effort_map = {
            "none": "none",
            "minimal": "none",
            "low": "low",
            "medium": "medium",
            "high": "medium",
            "xhigh": "xhigh",
            "max": "xhigh",
            "ultra": "xhigh",
        }

        mapped = effort_map.get(effort)
        if not mapped:
            return {}, {}

        return {}, {"reasoning_effort": mapped}


lanseq = LanseqProfile(
    name="lanseq",
    display_name="Lanseq",
    description="OpenAI-compatible inference for open-weight models from Hong Kong / APAC",
    signup_url="https://api.lanseq.cloud/docs",
    env_vars=("LANSEQ_API_KEY", "LANSEQ_BASE_URL"),
    base_url="https://api.lanseq.cloud/v1",
    auth_type="api_key",
    fallback_models=("qwen3.8-27b-int4",),
    default_max_tokens=8192,
)

register_provider(lanseq)

"""Ollama Cloud provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class OllamaCloudProfile(ProviderProfile):
    """Ollama Cloud OpenAI-compatible endpoint.

    Ollama exposes thinking control as top-level ``reasoning_effort`` in the
    OpenAI-compatible chat completions body. Hermes' shared reasoning_config
    uses {"enabled": False} for ``none`` and {"enabled": True, "effort": ...}
    for effort levels, so translate that declarative config here.
    """

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        top_level: dict[str, Any] = {}
        if not isinstance(reasoning_config, dict):
            return {}, top_level

        if reasoning_config.get("enabled") is False:
            top_level["reasoning_effort"] = "none"
            return {}, top_level

        effort = str(reasoning_config.get("effort") or "").strip().lower()
        if effort in {"minimal", "low", "medium", "high", "xhigh"}:
            top_level["reasoning_effort"] = effort
        return {}, top_level


ollama_cloud = OllamaCloudProfile(
    name="ollama-cloud",
    aliases=("ollama_cloud",),
    # Benchmark-derived default for auxiliary/scout use. Routine delegation is
    # configured separately in config.yaml to kimi-k2.6.
    default_aux_model="deepseek-v4-flash",
    env_vars=("OLLAMA_API_KEY",),
    base_url="https://ollama.com/v1",
)

register_provider(ollama_cloud)

"""Kosmik / KosCompute provider profile."""

from typing import Any

try:
    from agent.reasoning_effort import OPENAI_COMPAT_WIRE_EFFORTS, clamp_effort, requested_effort
except ImportError:
    OPENAI_COMPAT_WIRE_EFFORTS = frozenset({"none", "low", "medium", "high"})

    def clamp_effort(effort: str | None, allowed: frozenset[str]) -> str | None:
        return effort if effort in allowed else None

    def requested_effort(config: dict | None) -> str | None:
        if isinstance(config, dict) and config.get("enabled") is False:
            return "none"
        return config.get("effort") if isinstance(config, dict) else None

from providers import register_provider
from providers.base import ProviderProfile


class KosmikProfile(ProviderProfile):
    """Kosmik / KosCompute provider profile (Prague AI compute infrastructure)."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Map Hermes reasoning controls to Kosmik's top-level reasoning_effort."""
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            return {}, {"reasoning_effort": "none"}
        effort = requested_effort(reasoning_config)
        clamped = clamp_effort(effort, OPENAI_COMPAT_WIRE_EFFORTS)
        return ({}, {"reasoning_effort": clamped}) if clamped in OPENAI_COMPAT_WIRE_EFFORTS else ({}, {})

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
        **kwargs: Any,
    ) -> list[str] | None:
        """Fetch live models from Kosmik /v1/models, filtering out audio/TTS endpoints."""
        kwargs_to_pass: dict[str, Any] = {"api_key": api_key, "timeout": timeout, **kwargs}
        if base_url is not None:
            kwargs_to_pass["base_url"] = base_url
        try:
            models = super().fetch_models(**kwargs_to_pass)
        except TypeError:
            kwargs_to_pass.pop("base_url", None)
            models = super().fetch_models(**kwargs_to_pass)
        if models is None:
            return None
        # Kosmik hosts TTS and STT audio models alongside LLMs. Filter them out from the chat picker.
        non_chat_prefixes = ("kosmik/tts", "openai/whisper")
        return [
            m for m in models
            if not any(m.lower().startswith(prefix) for prefix in non_chat_prefixes)
        ]


kosmik = KosmikProfile(
    name="kosmik",
    aliases=("koscompute", "kosmik-ai", "kos"),
    display_name="Kosmik",
    description="Kosmik — European AI compute infrastructure & inference (Prague, CZ)",
    signup_url="https://koscompute.com",
    env_vars=("KOSMIK_API_KEY", "KOSCOMPUTE_API_KEY", "KOSMIK_BASE_URL"),
    base_url="https://api.koscompute.com/v1",
    auth_type="api_key",
    default_aux_model="qwen/qwen3.8-27b",
    fallback_models=(
        "qwen/qwen3.8-27b",
    ),
)

register_provider(kosmik)

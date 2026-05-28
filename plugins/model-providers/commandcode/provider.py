"""CommandCode provider profiles.

CommandCode exposes an OpenAI-compatible endpoint at
``https://api.commandcode.ai/provider/v1`` with a public ``/models`` catalog
that includes ``context_length`` metadata. We keep a static fallback table for
Hermes' offline / CI paths and opportunistically refresh it from the live
catalog when available.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

from .anthropic_shim import build_commandcode_anthropic_profile

logger = logging.getLogger(__name__)

_COMMANDCODE_BASE_URL = "https://api.commandcode.ai/provider/v1"
_COMMANDCODE_MODELS_URL = f"{_COMMANDCODE_BASE_URL}/models"
_COMMANDCODE_ENV_VARS = ("COMMANDCODE_API_KEY",)

# Snapshot taken from CommandCode's public /models catalog (2026-05-28).
# Keep the list intentionally broad so offline users still get the provider's
# main OSS / coding-friendly catalog in the picker.
_STATIC_CONTEXT_LENGTH_OVERRIDES: dict[str, int] = {
    "deepseek/deepseek-v4-pro": 1_000_000,
    "deepseek/deepseek-v4-flash": 1_000_000,
    "Qwen/Qwen3.7-Max": 1_000_000,
    "Qwen/Qwen3.6-Plus": 200_000,
    "Qwen/Qwen3.6-Max-Preview": 200_000,
    "moonshotai/Kimi-K2.6": 256_000,
    "moonshotai/Kimi-K2.5": 256_000,
    "zai-org/GLM-5.1": 200_000,
    "zai-org/GLM-5": 200_000,
    "MiniMaxAI/MiniMax-M2.7": 200_000,
    "MiniMaxAI/MiniMax-M2.5": 200_000,
    "stepfun/Step-3.5-Flash": 1_000_000,
    "xiaomi/mimo-v2.5-pro": 1_000_000,
    "xiaomi/mimo-v2.5": 1_000_000,
    "google/gemini-3.5-flash": 1_000_000,
    "google/gemini-3.1-flash-lite": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-haiku-4-5-20251001": 200_000,
    "gpt-5.5": 200_000,
    "gpt-5.4": 400_000,
    "gpt-5.4-mini": 400_000,
    "gpt-5.3-codex": 400_000,
}
_FALLBACK_MODELS: tuple[str, ...] = tuple(_STATIC_CONTEXT_LENGTH_OVERRIDES.keys())

_MODEL_CACHE: list[str] | None = None
_LIVE_CONTEXT_LENGTH_OVERRIDES: dict[str, int] = {}


def _normalize_model_id(model: str | None) -> str:
    value = str(model or "").strip()
    if not value:
        return ""

    if ":" in value:
        prefix, suffix = value.split(":", 1)
        if prefix.strip().lower() in {
            "commandcode",
            "command-code",
            "commandcode-anthropic",
        }:
            value = suffix.strip()

    lowered = value.lower()
    for prefix in ("commandcode/", "command-code/"):
        if lowered.startswith(prefix):
            value = value[len(prefix):]
            break
    return value.strip()


def _casefold_lookup(mapping: dict[str, int]) -> dict[str, tuple[str, int]]:
    return {key.lower(): (key, value) for key, value in mapping.items()}


class CommandCodeProfile(ProviderProfile):
    """OpenAI-compatible CommandCode provider with live context metadata."""

    @property
    def context_length_overrides(self) -> dict[str, int]:
        merged = dict(_STATIC_CONTEXT_LENGTH_OVERRIDES)
        merged.update(_LIVE_CONTEXT_LENGTH_OVERRIDES)
        return merged

    def resolve_model_id(self, model: str | None) -> str | None:
        normalized = _normalize_model_id(model)
        if not normalized:
            return None

        by_lower = _casefold_lookup(self.context_length_overrides)
        direct = by_lower.get(normalized.lower())
        if direct is not None:
            return direct[0]

        bare = normalized.rsplit("/", 1)[-1].lower()
        suffix_matches = [
            canonical
            for lowered, (canonical, _ctx) in by_lower.items()
            if lowered.rsplit("/", 1)[-1] == bare
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        return normalized

    def get_context_length(self, model: str | None) -> int | None:
        resolved = self.resolve_model_id(model)
        if not resolved:
            return None
        entry = _casefold_lookup(self.context_length_overrides).get(resolved.lower())
        return entry[1] if entry is not None else None

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch the public CommandCode catalog and cache context lengths."""
        global _MODEL_CACHE, _LIVE_CONTEXT_LENGTH_OVERRIDES  # noqa: PLW0603
        if _MODEL_CACHE is not None:
            return list(_MODEL_CACHE)

        request = urllib.request.Request(self.models_url or _COMMANDCODE_MODELS_URL)
        request.add_header("Accept", "application/json")
        request.add_header("User-Agent", _profile_user_agent())
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        for header_name, header_value in self.default_headers.items():
            request.add_header(header_name, header_value)

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
        except Exception as exc:
            logger.debug("fetch_models(commandcode): %s", exc)
            return None

        items = payload if isinstance(payload, list) else payload.get("data", [])
        models: list[str] = []
        live_overrides: dict[str, int] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if not model_id:
                continue
            models.append(model_id)
            context_length = item.get("context_length")
            if isinstance(context_length, int) and context_length > 0:
                live_overrides[model_id] = context_length

        if not models:
            return None

        _MODEL_CACHE = models
        if live_overrides:
            _LIVE_CONTEXT_LENGTH_OVERRIDES = live_overrides
        return list(_MODEL_CACHE)

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pass OpenAI-style reasoning config through for supported models.

        CommandCode speaks a vanilla OpenAI-compatible wire format, so the
        safest provider-level behavior is the same as OpenRouter's generic
        ``extra_body.reasoning`` passthrough.
        """
        if not supports_reasoning:
            return {}, {}
        if reasoning_config is None:
            return {"reasoning": {"enabled": True, "effort": "medium"}}, {}
        return {"reasoning": dict(reasoning_config)}, {}


commandcode = CommandCodeProfile(
    name="commandcode",
    aliases=("command-code", "ccode"),
    env_vars=_COMMANDCODE_ENV_VARS,
    display_name="CommandCode",
    description="CommandCode — OpenAI-compatible unified model gateway",
    signup_url="https://commandcode.ai/",
    base_url=_COMMANDCODE_BASE_URL,
    models_url=_COMMANDCODE_MODELS_URL,
    auth_type="api_key",
    fallback_models=_FALLBACK_MODELS,
    default_aux_model="Qwen/Qwen3.6-Plus",
)

commandcode_anthropic = build_commandcode_anthropic_profile(
    env_vars=_COMMANDCODE_ENV_VARS,
    fallback_models=_FALLBACK_MODELS,
    models_url=_COMMANDCODE_MODELS_URL,
)

register_provider(commandcode)
register_provider(commandcode_anthropic)

__all__ = [
    "CommandCodeProfile",
    "commandcode",
    "commandcode_anthropic",
]

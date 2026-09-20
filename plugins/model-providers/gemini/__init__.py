"""Google Gemini (AI Studio) provider profile.

Reports api_mode="chat_completions" but runs on GeminiNativeClient; this
profile carries auth/endpoint metadata and the thinking_config translation hook.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class GeminiProfile(ProviderProfile):
    """Gemini — translate reasoning_config to thinking_config in extra_body."""

    def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0):
        from urllib.parse import urlparse
        from agent.gemini_model_catalog import fetch_models
        if base_url and urlparse(base_url).hostname != "generativelanguage.googleapis.com":
            return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
        return fetch_models(api_key, timeout=timeout)

    def describe_models(self, *, model_ids: list[str]) -> dict:
        from agent.gemini_catalog_reasoning import describe_thinking_control
        return {model: describe_thinking_control(model) for model in model_ids}

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...]:
        from agent.gemini_catalog_reasoning import supported_efforts
        return supported_efforts(model)

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        """Native: ``thinking_config``; OpenAI-compat /openai subpath:
        ``extra_body.google.thinking_config`` (snake_case)."""
        from agent.transports.chat_completions import (
            _is_gemini_openai_compat_base_url,
            _snake_case_gemini_thinking_config,
        )

        from agent.gemini_catalog_reasoning import build_thinking_config
        raw = build_thinking_config(context.get("model") or "", context.get("reasoning_config"))
        if not raw:
            return {}
        if self.name == "gemini" and _is_gemini_openai_compat_base_url(context.get("base_url") or self.base_url):
            thinking_config = _snake_case_gemini_thinking_config(raw)
            return {"extra_body": {"google": {"thinking_config": thinking_config}}} if thinking_config else {}
        return {"thinking_config": raw}


gemini = GeminiProfile(
    name="gemini", aliases=("google", "google-gemini", "google-ai-studio"), api_mode="chat_completions",
    validate_reasoning_selection=True, model_catalog_authoritative=True,
    env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta", auth_type="api_key",
    default_aux_model="gemini-3.6-flash",
)

register_provider(gemini)

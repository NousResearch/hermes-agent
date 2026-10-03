"""Google Gemini (AI Studio) provider profile.

Reports api_mode="chat_completions" but runs on GeminiNativeClient; this
profile carries auth/endpoint metadata and the thinking_config translation hook.
"""

import logging
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)


class GeminiProfile(ProviderProfile):
    """Gemini — translate reasoning_config to thinking_config in extra_body."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch the live catalog from the native Gemini endpoint.

        The base implementation sends ``Authorization: Bearer <key>`` to
        ``{base_url}/models``. That works for OpenAI-compatible providers, but
        the native ``/v1beta`` endpoint rejects Bearer auth with HTTP 401 — it
        requires an API key via ``x-goog-api-key`` (or the equivalent query
        parameter). Left to the base path, the probe 401s and the picker silently
        falls back to the static list (#62259). Hit the native endpoint with
        header auth and keep only models that support conversational generation.

        Non-native endpoints, including Gemini's OpenAI-compatibility URL and
        custom relays, keep the base implementation's Bearer + ``data[].id``
        contract.
        """
        effective_base = (base_url or self.base_url or "").rstrip("/")
        if not (effective_base and api_key):
            return None

        from agent.gemini_native_adapter import is_native_gemini_base_url

        if not is_native_gemini_base_url(effective_base):
            return super().fetch_models(
                api_key=api_key, base_url=base_url, timeout=timeout
            )

        import json
        import urllib.parse
        import urllib.request

        from agent.models_dev import _NOISE_PATTERNS, _should_hide_from_provider_catalog
        from hermes_cli.urllib_security import open_credentialed_url

        ids: list[str] = []
        page_token = ""
        seen_page_tokens: set[str] = set()
        try:
            while True:
                query = {"pageSize": "1000"}
                if page_token:
                    query["pageToken"] = page_token
                req = urllib.request.Request(
                    f"{effective_base}/models?{urllib.parse.urlencode(query)}"
                )
                req.add_header("x-goog-api-key", api_key)
                req.add_header("Accept", "application/json")
                req.add_header("User-Agent", _profile_user_agent())
                for key, value in self.default_headers.items():
                    req.add_header(key, value)
                with open_credentialed_url(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode())
                if not isinstance(data, dict):
                    return None
                for entry in data.get("models") or []:
                    if not isinstance(entry, dict):
                        continue
                    if "generateContent" not in (entry.get("supportedGenerationMethods") or []):
                        continue
                    model_id = str(entry.get("name") or "").removeprefix("models/")
                    if model_id:
                        ids.append(model_id)
                next_page_token = str(data.get("nextPageToken") or "")
                if not next_page_token:
                    break
                if next_page_token in seen_page_tokens:
                    logger.debug("fetch_models(gemini) failed: repeated page token")
                    return None
                seen_page_tokens.add(next_page_token)
                page_token = next_page_token
        except Exception as exc:
            logger.debug("fetch_models(gemini) failed: %s", type(exc).__name__)
            return None

        conversational = [
            model_id
            for model_id in ids
            if not _NOISE_PATTERNS.search(model_id)
            and not _should_hide_from_provider_catalog("gemini", model_id)
            and not any(family in model_id.lower() for family in _NON_CHAT_FAMILIES)
        ]
        return conversational or None

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        """Native: ``thinking_config``; OpenAI-compat /openai subpath:
        ``extra_body.google.thinking_config`` (snake_case)."""
        from agent.transports.chat_completions import (
            _build_gemini_thinking_config,
            _is_gemini_openai_compat_base_url,
            _snake_case_gemini_thinking_config,
        )

        raw = _build_gemini_thinking_config(context.get("model") or "", context.get("reasoning_config"))
        if not raw:
            return {}
        if self.name == "gemini" and _is_gemini_openai_compat_base_url(context.get("base_url") or self.base_url):
            thinking_config = _snake_case_gemini_thinking_config(raw)
            return {"extra_body": {"google": {"thinking_config": thinking_config}}} if thinking_config else {}
        return {"thinking_config": raw}


_NON_CHAT_FAMILIES = ("lyria", "veo-", "imagen", "nano-banana", "transcribe", "robotics")


gemini = GeminiProfile(
    name="gemini", aliases=("google", "google-gemini", "google-ai-studio"), api_mode="chat_completions",
    env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta", auth_type="api_key",
    default_aux_model="gemini-3.6-flash",
)

register_provider(gemini)

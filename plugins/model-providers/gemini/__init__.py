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
        requires the key as a ``?key=`` query param. Left to the base path, the
        probe 401s and the picker silently falls back to the static list
        (#62259). Hit the native endpoint with query-param auth (the same auth
        the native inference client already uses) and strip the ``models/``
        prefix each entry's ``name`` carries so IDs match what inference expects.

        Gemini's OpenAI-compatibility base URL (``.../openai``) *does* speak
        Bearer + OpenAI-style ``data[].id``, so this native override is gated to
        native endpoints only; the compat base URL delegates to the base
        implementation.
        """
        effective_base = (base_url or self.base_url or "").rstrip("/")
        if not (effective_base and api_key):
            return None

        # The OpenAI-compat endpoint speaks Bearer + data[].id — let the base
        # ProviderProfile handle it rather than forcing native query-param auth.
        from agent.transports.chat_completions import (
            _is_gemini_openai_compat_base_url,
        )

        if _is_gemini_openai_compat_base_url(effective_base):
            return super().fetch_models(
                api_key=api_key, base_url=base_url, timeout=timeout
            )

        import json
        import urllib.parse
        import urllib.request

        url = f"{effective_base}/models?key={urllib.parse.quote(api_key)}"
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())
        for k, v in self.default_headers.items():
            req.add_header(k, v)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            items = data.get("models", []) if isinstance(data, dict) else []
            ids = [
                str(m["name"]).removeprefix("models/")
                for m in items
                if isinstance(m, dict) and m.get("name")
            ]
            return ids or None
        except Exception as exc:
            # Never log the exception value: urllib errors embed the request
            # URL, which carries the api_key in the ``?key=`` query param. Log
            # the exception type only.
            logger.debug("fetch_models(gemini) failed: %s", type(exc).__name__)
            return None

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


gemini = GeminiProfile(
    name="gemini", aliases=("google", "google-gemini", "google-ai-studio"), api_mode="chat_completions",
    env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta", auth_type="api_key",
    default_aux_model="gemini-3.6-flash",
)

register_provider(gemini)

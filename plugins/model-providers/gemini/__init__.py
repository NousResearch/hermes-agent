"""Google Gemini provider profiles.

gemini:            Google AI Studio (API key) — uses GeminiNativeClient
google-gemini-cli: Google Cloud Code Assist (OAuth) — uses GeminiCloudCodeClient

Both report api_mode="chat_completions" but use custom native clients
that bypass the standard OpenAI transport. The profile captures auth
and endpoint metadata for auth.py / runtime_provider.py migration, and
carries the thinking_config translation hook so the transport's profile
path produces the same extra_body shape the legacy flag path did.
"""

import json
import logging
import os
import urllib.request
from typing import Any
from urllib.error import URLError

from providers import register_provider
from providers.base import ProviderProfile


_logger = logging.getLogger(__name__)


class GeminiProfile(ProviderProfile):
    """Gemini — translate reasoning_config to thinking_config in extra_body."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch live model list from Google AI Studio /v1beta/models endpoint.

        Overrides the default Bearer-auth + ``m["id"]`` parser because
        Google's API requires ``?key=`` query-param auth and returns
        ``"name": "models/gemma-4-31b-it"`` (not ``"id"``).
        """
        # Resolve API key: explicit arg > env vars listed in the profile
        if not api_key:
            for var in self.env_vars:
                api_key = os.getenv(var, "").strip()
                if api_key:
                    break
        if not api_key:
            return None

        # Build URL — prefer models_url override, fall back to base_url
        url = (self.models_url or "").strip()
        if not url:
            if not self.base_url:
                return None
            url = self.base_url.rstrip("/")

        # Append models path if not already present, then attach API key
        if "/models" not in url.rsplit("?", 1)[0]:
            url = f"{url.rstrip('/')}/models"
        url = f"{url}?key={api_key}"

        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "hermes-gemini-fetch-models/1.0")

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except URLError as exc:
            # Do NOT log the exception string — it includes the URL with API key
            _logger.debug("fetch_models(gemini): network error (%s)", type(exc).__name__)
            return None
        except Exception as exc:
            _logger.debug("fetch_models(gemini): %s", type(exc).__name__)
            return None

        # Google returns ``{"models": [{"name": "models/gemma-4-31b-it", ...}]}``
        models = data.get("models", [])
        if not models:
            return None

        out: list[str] = []
        for m in models:
            name = m.get("name", "")
            if not isinstance(name, str):
                continue
            # Strip the "models/" prefix
            model_id = name.replace("models/", "", 1) if name.startswith("models/") else name
            if model_id:
                out.append(model_id)
        return out or None

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """Emit extra_body.thinking_config (native) or extra_body.extra_body.google.thinking_config
        (OpenAI-compat /openai subpath), mirroring the legacy path's behavior.
        """
        from agent.transports.chat_completions import (
            _build_gemini_thinking_config,
            _is_gemini_openai_compat_base_url,
            _snake_case_gemini_thinking_config,
        )

        model = context.get("model") or ""
        reasoning_config = context.get("reasoning_config")
        base_url = context.get("base_url") or self.base_url

        raw_thinking_config = _build_gemini_thinking_config(model, reasoning_config)
        if not raw_thinking_config:
            return {}

        body: dict[str, Any] = {}
        if self.name == "gemini" and _is_gemini_openai_compat_base_url(base_url):
            thinking_config = _snake_case_gemini_thinking_config(raw_thinking_config)
            if thinking_config:
                body["extra_body"] = {"google": {"thinking_config": thinking_config}}
        else:
            body["thinking_config"] = raw_thinking_config
        return body


gemini = GeminiProfile(
    name="gemini",
    aliases=("google", "google-gemini", "google-ai-studio"),
    api_mode="chat_completions",
    env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta",
    auth_type="api_key",
    default_aux_model="gemini-3.5-flash",
)

google_gemini_cli = GeminiProfile(
    name="google-gemini-cli",
    aliases=("gemini-cli", "gemini-oauth"),
    api_mode="chat_completions",
    env_vars=(),  # OAuth — no API key
    base_url="cloudcode-pa://google",  # Cloud Code Assist internal scheme
    auth_type="oauth_external",
)

register_provider(gemini)
register_provider(google_gemini_cli)

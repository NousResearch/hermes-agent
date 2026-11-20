"""Google Gemini (AI Studio) provider profile.

Reports api_mode="chat_completions" but supplies its own client (``client.GeminiClient``): each call
goes to the Interactions API (``interactions.py``) or to ``generateContent`` (GeminiNativeClient), per
model. This profile also carries auth/endpoint metadata and the thinking_config translation hook.
"""

import logging
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

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
        from agent.reasoning_effort import gemini_top_thinking_level
        from agent.transports.chat_completions import (
            _build_gemini_thinking_config,
            _is_gemini_openai_compat_base_url,
            _snake_case_gemini_thinking_config,
        )

        raw = _build_gemini_thinking_config(context.get("model") or "", context.get("reasoning_config"))
        if not raw:
            return {}
        raw = gemini_top_thinking_level(context.get("model"), context.get("reasoning_config"), raw)
        base_url = context.get("base_url") or self.base_url
        if self.name == "gemini" and _is_gemini_openai_compat_base_url(base_url):
            thinking_config = _snake_case_gemini_thinking_config(raw)
            return {"extra_body": {"google": {"thinking_config": thinking_config}}} if thinking_config else {}
        from agent.gemini_long_decode import long_decode_max_output_tokens
        from agent.gemini_native_adapter import is_native_gemini_base_url

        # Long decode exists only on native generateContent (not /openai, not proxies).
        budget = long_decode_max_output_tokens(context.get("model") or "", context.get("reasoning_config"))
        return {"thinking_config": raw, **({"max_output_tokens": budget} if budget and is_native_gemini_base_url(base_url) else {})}

    def create_client(self, **client_kwargs: Any) -> Any:
        """The plugin's client (Interactions or generateContent per model) on a native Gemini base; None on
        the OpenAI-compatible ``/openai`` surface and proxies, which keep the standard OpenAI client."""
        from agent.gemini_native_adapter import is_native_gemini_base_url

        base_url = str(client_kwargs.get("base_url") or self.base_url or "")
        if not is_native_gemini_base_url(base_url):
            return None
        from .client import GeminiClient

        allowed = {"api_key", "base_url", "default_headers", "timeout", "http_client"}
        kwargs = {k: v for k, v in client_kwargs.items() if k in allowed}
        if "http_client" not in kwargs and (http_client := _keepalive_http_client(base_url)) is not None:
            kwargs["http_client"] = http_client
        return GeminiClient(**kwargs)


def _keepalive_http_client(base_url: str) -> Any:
    """The keepalive client (and the provider's ``ssl_ca_cert`` / ``ssl_verify``) the core gives a native
    Gemini client; the profile hook receives the client kwargs after the core has consumed the TLS keys."""
    try:
        from agent.process_bootstrap import build_keepalive_http_client
        from agent.ssl_verify import resolve_httpx_verify
        from hermes_cli.config_providers import get_custom_provider_tls_settings

        tls = get_custom_provider_tls_settings(base_url)
        verify = resolve_httpx_verify(ca_bundle=tls.get("ssl_ca_cert"), ssl_verify=tls.get("ssl_verify"), base_url=base_url)
        return build_keepalive_http_client(base_url, verify=verify)
    except Exception:  # GeminiNativeClient then builds its own httpx client
        logger.warning("Gemini: keepalive client unavailable; using a default client", exc_info=True)
        return None


_NON_CHAT_FAMILIES = ("lyria", "veo-", "imagen", "nano-banana", "transcribe", "robotics")


gemini = GeminiProfile(
    name="gemini", aliases=("google", "google-gemini", "google-ai-studio"), api_mode="chat_completions",
    env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta", auth_type="api_key",
    default_aux_model="gemini-3.6-flash",
)

register_provider(gemini)

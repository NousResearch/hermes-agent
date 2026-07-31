"""Google Gemini (AI Studio) provider profile.

Reports api_mode="chat_completions" but supplies its own client (``client.GeminiClient``): each call
goes to the Interactions API (``interactions.py``) or to ``generateContent`` (GeminiNativeClient), per
model. This profile also carries auth/endpoint metadata and the thinking_config translation hook.
"""

import logging
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

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
        """Live model discovery via Gemini's OpenAI-compatible surface.

        Points at the /v1beta/openai subpath (not the native /v1beta root
        the base_url otherwise resolves to) so the base implementation's
        {"data": [{"id": ...}]} parsing applies -- the native /v1beta/models
        endpoint uses a differently-shaped response that would need
        separate parsing.

        Gemini's OpenAI-compat endpoint prefixes returned IDs with
        "models/" (e.g. "models/gemini-2.5-flash") -- native Gemini-API
        convention. Stripped here so callers get the same bare-ID form the
        curated list, user input, and the existing validation path
        (#12532) all use.

        Deliberately does NOT short-circuit with its own curated-list
        merge or early return: returning through fetch_models() here lets
        it flow through the SHARED generic merge in
        hermes_cli.models.provider_model_ids(), which already knows how
        to preserve curated entries when the live catalog is partial or
        stale (review of #75306) -- an earlier revision of this fix
        returned the live result directly from a separate branch,
        bypassing that merge entirely.
        """
        effective_base = (base_url or self.base_url or "").rstrip("/")
        if effective_base.endswith("/v1beta"):
            effective_base += "/openai"
        models = super().fetch_models(
            api_key=api_key, base_url=effective_base or None, timeout=timeout,
        )
        if not models:
            return models
        return [
            m[len("models/"):] if isinstance(m, str) and m.startswith("models/") else m
            for m in models
        ]

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


gemini = GeminiProfile(
    name="gemini", aliases=("google", "google-gemini", "google-ai-studio"), api_mode="chat_completions",
    env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta", auth_type="api_key",
    default_aux_model="gemini-3.6-flash",
)

register_provider(gemini)

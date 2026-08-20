"""Google Gemini provider profiles.

gemini:            Google AI Studio (API key) — uses GeminiNativeClient

Reports api_mode="chat_completions" but uses a custom native client
that bypasses the standard OpenAI transport. The profile captures auth
and endpoint metadata for auth.py / runtime_provider.py migration, and
carries the thinking_config translation hook so the transport's profile
path produces the same extra_body shape the legacy flag path did.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


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
    default_aux_model="gemini-3.6-flash",
)

register_provider(gemini)

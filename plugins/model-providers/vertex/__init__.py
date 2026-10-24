"""Google Vertex AI provider profile: Gemini via the native ``generateContent`` API (GeminiNativeClient).

Auth is OAuth2 (service-account JSON or ADC), not a static key: ``agent/vertex_adapter.py``
mints ``(token, base_url)`` and the token is passed as ``api_key``. ``auth_type="vertex"``
keeps it out of the api_key provider path so a credentials-file path is never mistaken for a key.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class VertexProfile(ProviderProfile):
    """Vertex AI — reuse Gemini's thinking_config translation for extra_body."""

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        """Native ``…/publishers/google`` base: ``thinking_config`` (+ long-decode budget), as the ``gemini``
        provider. An explicit ``…/endpoints/openapi`` base: ``extra_body.google.thinking_config`` (snake_case)."""
        from agent.gemini_long_decode import long_decode_max_output_tokens
        from agent.gemini_native_adapter import is_native_gemini_base_url
        from agent.transports.chat_completions import _build_gemini_thinking_config, _snake_case_gemini_thinking_config

        model = context.get("model") or ""
        raw = _build_gemini_thinking_config(model, context.get("reasoning_config"))
        if not raw:
            return {}
        base_url = context.get("base_url") or self.base_url
        if is_native_gemini_base_url(base_url):
            budget = long_decode_max_output_tokens(model, context.get("reasoning_config"))
            return {"thinking_config": raw, **({"max_output_tokens": budget} if budget else {})}
        thinking_config = _snake_case_gemini_thinking_config(raw)
        return {"extra_body": {"google": {"thinking_config": thinking_config}}} if thinking_config else {}


vertex = VertexProfile(
    name="vertex", aliases=("google-vertex", "vertex-ai", "gcp-vertex", "vertexai"), api_mode="chat_completions",
    env_vars=(),  # OAuth2 via service account / ADC — not a static key env var
    base_url="https://aiplatform.googleapis.com",  # real base_url computed at runtime
    auth_type="vertex", default_aux_model="google/gemini-3.6-flash",
    supports_model_listing=False,  # no publisher /models listing route; setup ships a curated list
)

register_provider(vertex)

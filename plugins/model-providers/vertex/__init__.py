"""Google Vertex AI provider profile: Gemini via Google Cloud's OpenAI-compatible endpoint.

Auth is OAuth2 (service-account JSON or ADC), not a static key: ``agent/vertex_adapter.py``
mints ``(token, base_url)`` and the token is passed as ``api_key``. ``auth_type="vertex"``
keeps it out of the api_key provider path so a credentials-file path is never mistaken for a key.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class VertexProfile(ProviderProfile):
    """Vertex AI — reuse Gemini's thinking_config translation for extra_body."""

    def normalize_model_id(self, model: str, *, api_key: str = "") -> str | None:
        """Vertex's OpenAI-compatible endpoint requires publisher-prefixed IDs (e.g. ``google/gemini-3.8-flash``)."""
        from hermes_cli.model_normalize import _repair_prefix_from_catalogue, _strip_matching_provider_prefix

        raw = _strip_matching_provider_prefix((model or "").strip(), "vertex")
        if not raw:
            return None
        if "/" in raw:
            return raw
        if raw.lower().startswith(("gemini-", "gemma-", "imagen-", "veo-", "chirp-", "text-embedding-", "multimodalembedding")):
            return f"google/{raw}"
        return _repair_prefix_from_catalogue(raw, "vertex")

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        """Emit ``extra_body.google.{thinking_config,safety_settings}`` like the ``gemini`` provider's OpenAI-compat subpath."""
        from agent.gemini_native_adapter import GEMINI_SAFETY_SETTINGS_OFF
        from agent.transports.chat_completions import _build_gemini_thinking_config, _snake_case_gemini_thinking_config

        raw = _build_gemini_thinking_config(context.get("model") or "", context.get("reasoning_config"))
        thinking_config = _snake_case_gemini_thinking_config(raw) if raw else None
        google_extra: dict[str, Any] = {
            "safety_settings": [dict(item) for item in GEMINI_SAFETY_SETTINGS_OFF],
        }
        if thinking_config:
            google_extra["thinking_config"] = thinking_config
        return {"extra_body": {"google": google_extra}}


vertex = VertexProfile(
    name="vertex", aliases=("google-vertex", "vertex-ai", "gcp-vertex", "vertexai"), api_mode="chat_completions",
    env_vars=(),  # OAuth2 via service account / ADC — not a static key env var
    base_url="https://aiplatform.googleapis.com",  # real base_url computed at runtime
    auth_type="vertex", default_aux_model="google/gemini-3.6-flash",
    supports_model_listing=False,  # no /models route on the OpenAI-compat endpoint; setup ships a curated list
)

register_provider(vertex)

"""Google Vertex AI provider profile.

vertex: Gemini models via Google Cloud's OpenAI-compatible endpoint.

Supports two authentication methods:

1. **API Key (Express Mode) — RECOMMENDED.**
   Set ``GOOGLE_VERTEX_API_KEY``, ``GOOGLE_VERTEX_PROJECT``, and optionally
   ``GOOGLE_VERTEX_LOCATION`` in your .env file. No google-auth needed.

2. **OAuth2 / ADC (legacy)**
   Service-account JSON or ADC. Requires ``google-auth``.

Auth selection is automatic — ``get_vertex_config()`` in ``agent/vertex_adapter.py``
picks the right path based on which env vars are set.

``auth_type="vertex"`` marks this as a specially-resolved provider (like
bedrock's ``aws_sdk``) so env-var lookup for a static api_key is not the
only path. The runtime provider resolver (``runtime_provider.py``) handles
API key auth directly without needing the full OAuth2 token flow.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class VertexProfile(ProviderProfile):
    """Vertex AI — reuse Gemini's thinking_config translation for extra_body."""

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """Emit ``extra_body.google.thinking_config`` for the OpenAI-compat
        Vertex surface, mirroring the ``gemini`` provider's behavior.
        """
        from agent.transports.chat_completions import (
            _build_gemini_thinking_config,
            _snake_case_gemini_thinking_config,
        )

        model = context.get("model") or ""
        reasoning_config = context.get("reasoning_config")

        raw_thinking_config = _build_gemini_thinking_config(model, reasoning_config)
        if not raw_thinking_config:
            return {}

        thinking_config = _snake_case_gemini_thinking_config(raw_thinking_config)
        if not thinking_config:
            return {}
        return {"extra_body": {"google": {"thinking_config": thinking_config}}}

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Discover models via Vertex AI's publisher ``models.list`` API.

        Unlike the legacy OAuth2 path (which has no ``/models`` route on the
        OpenAI-compatible endpoint), the API key path can query the publisher
        models endpoint for region-specific model availability.

        If discovery fails (network, auth, or the OAuth2 path), returns None
        and the setup wizard falls back to its curated model list.
        """
        from agent.vertex_adapter import (
            discover_vertex_models,
            resolve_vertex_api_key,
            _resolve_project_override,
            _resolve_region,
        )

        # Only API key auth supports model discovery
        resolved_key = api_key or resolve_vertex_api_key()
        if not resolved_key:
            return None

        project_id = _resolve_project_override()
        if not project_id:
            return None

        region = _resolve_region()
        models = discover_vertex_models(resolved_key, project_id, region, timeout)
        return models if models else None


vertex = VertexProfile(
    name="vertex",
    aliases=("google-vertex", "vertex-ai", "gcp-vertex"),
    api_mode="chat_completions",
    env_vars=(
        "GOOGLE_VERTEX_API_KEY",
        "GOOGLE_VERTEX_PROJECT",
        "GOOGLE_VERTEX_LOCATION",
    ),
    base_url="https://aiplatform.googleapis.com",  # real base_url computed at runtime
    auth_type="vertex",
    default_aux_model="google/gemini-3-flash-preview",
)

register_provider(vertex)

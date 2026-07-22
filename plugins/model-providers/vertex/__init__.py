"""Google Vertex AI provider profile.

vertex: dual-path provider for Google Cloud Vertex AI.

  - **Claude** (Anthropic) -- uses the AnthropicVertex SDK
    (``anthropic_messages`` api_mode).  Model-based routing in
    ``runtime_provider.py`` detects Claude model names and hands off to
    ``hermes_cli.auth.resolve_vertex_anthropic_runtime_credentials()``.

  - **Gemini** (Google) -- uses the OpenAI-compatible endpoint with
    OAuth2 access tokens minted from a service-account JSON or ADC.
    Token resolution and refresh live in ``agent/vertex_adapter.py``.

Auth is always OAuth2 -- never a static API key.  ``auth_type="vertex"``
marks this as an OAuth-token provider (resolved specially, like bedrock's
``aws_sdk``) so it is never treated as an api_key provider that would
mistake a credentials-file path for a key.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class VertexProfile(ProviderProfile):
    """Vertex AI -- Gemini thinking_config + no REST /models endpoint."""

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """Emit ``extra_body.google.thinking_config`` for the OpenAI-compat
        Vertex surface, mirroring the ``gemini`` provider's behavior.

        Only applies to Gemini models; Claude models use the AnthropicVertex
        SDK path which has its own thinking/reasoning support.
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
        """Vertex model listing requires vendor-specific SDKs, not a REST call.
        The setup wizard ships a curated list instead.
        """
        return None


vertex = VertexProfile(
    name="vertex",
    aliases=("google-vertex", "gcp-vertex", "vertex-ai", "vertex-anthropic"),
    api_mode="chat_completions",  # default for Gemini; Claude overrides to anthropic_messages at runtime
    display_name="Google Vertex AI",
    description="Google Vertex AI (Gemini + Claude via GCP; OAuth2 service account or ADC)",
    signup_url="https://cloud.google.com/vertex-ai",
    env_vars=(
        "VERTEX_CREDENTIALS_PATH",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "GOOGLE_CLOUD_PROJECT",
        "CLOUD_ML_REGION",
    ),
    base_url="https://aiplatform.googleapis.com",  # real base_url computed at runtime
    auth_type="vertex",
    supports_health_check=False,
    default_aux_model="google/gemini-3-flash-preview",
)

register_provider(vertex)

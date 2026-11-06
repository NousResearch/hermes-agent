"""Google Vertex AI provider profile: Gemini via Google Cloud's OpenAI-compatible endpoint.

Auth is OAuth2 (service-account JSON or ADC), not a static key: ``agent/vertex_adapter.py``
mints ``(token, base_url)`` and the token is passed as ``api_key``. ``auth_type="vertex"``
keeps it out of the api_key provider path so a credentials-file path is never mistaken for a key.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

#: Request header that opts a Vertex request into Priority PayGo. Supported on the global, us and
#: eu endpoints; it trades a higher per-token rate for steadier service than the shared
#: Standard PayGo pool, which answers capacity 429s when the pool is contended.
PRIORITY_PAYGO_HEADER = "X-Vertex-AI-LLM-Shared-Request-Type"
PRIORITY_PAYGO_VALUE = "priority"


def _priority_paygo_enabled() -> bool:
    """Whether the user opted into Priority PayGo via ``vertex.priority_paygo`` in config.yaml.

    Off unless explicitly enabled: the header changes the per-token rate, so it must never
    ride along silently.
    """
    try:
        from hermes_cli.config import load_config

        section = load_config().get("vertex")
    except Exception:  # health: allow BLE001 -- a config read must never break the request path
        return False
    if not isinstance(section, dict):
        return False
    return bool(section.get("priority_paygo"))


class VertexProfile(ProviderProfile):
    """Vertex AI — reuse Gemini's thinking_config translation for extra_body."""

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        """Emit ``extra_body.google.thinking_config`` like the ``gemini`` provider's OpenAI-compat subpath."""
        from agent.transports.chat_completions import _build_gemini_thinking_config, _snake_case_gemini_thinking_config

        raw = _build_gemini_thinking_config(context.get("model") or "", context.get("reasoning_config"))
        thinking_config = _snake_case_gemini_thinking_config(raw) if raw else None
        return {"extra_body": {"google": {"thinking_config": thinking_config}}} if thinking_config else {}

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Carry the Priority PayGo header when the user opted in.

        Per-request (``extra_headers``) rather than client-level: the OpenAI client is rebuilt
        on credential rotation and provider switches, and a header bound at construction would
        not survive that. Profile-scoped, so it can never reach another provider.
        """
        if not _priority_paygo_enabled():
            return {}, {}
        return {}, {"extra_headers": {PRIORITY_PAYGO_HEADER: PRIORITY_PAYGO_VALUE}}


vertex = VertexProfile(
    name="vertex", aliases=("google-vertex", "vertex-ai", "gcp-vertex", "vertexai"), api_mode="chat_completions",
    env_vars=(),  # OAuth2 via service account / ADC — not a static key env var
    base_url="https://aiplatform.googleapis.com",  # real base_url computed at runtime
    auth_type="vertex", default_aux_model="google/gemini-3.6-flash",
    supports_model_listing=False,  # no /models route on the OpenAI-compat endpoint; setup ships a curated list
)

register_provider(vertex)

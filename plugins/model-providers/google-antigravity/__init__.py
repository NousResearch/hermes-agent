"""Google Antigravity subscription provider profile.

google-antigravity: Gemini models served through Google's Code Assist
endpoint against the user's Antigravity subscription (OAuth), rather than
an AI Studio API key.

Deliberately separate from the ``gemini`` profile: different auth
(OAuth vs API key), different host, different wire path. Selecting this
provider must never change ``gemini`` behavior.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

#: Code Assist internal API root used by the Antigravity/Gemini Code Assist CLI.
ANTIGRAVITY_BASE_URL = "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal"

#: Curated fallback shown when live entitled-model discovery fails.
#: Gemini-only by design — this provider does not surface other vendors.
ANTIGRAVITY_FALLBACK_MODELS = (
    "gemini-pro-agent",
    "gemini-3-flash-agent",
    "gemini-3.1-pro-low",
    "gemini-3.5-flash-low",
)


class AntigravityProfile(ProviderProfile):
    """Antigravity — reuses Gemini thinking-config translation on the native path."""

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """Emit ``extra_body.thinking_config`` like the native Gemini path.

        Code Assist wraps the same ``GenerateContentRequest``, so the
        reasoning knobs translate identically.
        """
        from agent.transports.chat_completions import _build_gemini_thinking_config

        model = context.get("model") or ""
        reasoning_config = context.get("reasoning_config")
        thinking_config = _build_gemini_thinking_config(model, reasoning_config)
        if not thinking_config:
            return {}
        return {"thinking_config": thinking_config}

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Return the live entitled Gemini models, or None to use the fallback.

        The generic OpenAI-style ``/models`` probe on the base class does not
        apply here: Code Assist exposes its catalog through
        ``fetchAvailableModels`` after project resolution via ``loadCodeAssist``.
        Returning None on any failure lets the caller use ``fallback_models``.
        """
        from agent.gemini_cloudcode_adapter import discover_entitled_models

        token = api_key
        project = None
        if not token:
            try:
                from hermes_cli.antigravity_auth import (
                    resolve_antigravity_runtime_credentials,
                )

                creds = resolve_antigravity_runtime_credentials()
                token = creds.get("api_key")
                project = creds.get("project_id")
            except Exception:
                # Not authed yet (or refresh failed) — the picker should still
                # render, so fall through to the curated list.
                return None

        try:
            return discover_entitled_models(
                access_token=token or "",
                project=project,
                base_url=base_url or self.base_url,
                timeout=timeout,
            )
        except Exception:
            return None


antigravity = AntigravityProfile(
    name="google-antigravity",
    aliases=("antigravity", "agy", "google-agy"),
    api_mode="chat_completions",
    display_name="Google Antigravity",
    description="Google Antigravity subscription (Gemini via Code Assist OAuth)",
    signup_url="https://antigravity.google/",
    env_vars=(),
    base_url=ANTIGRAVITY_BASE_URL,
    auth_type="oauth_external",
    # No OpenAI-shaped /models endpoint; fetchAvailableModels is provider-specific.
    supports_health_check=False,
    supports_vision=True,
    fallback_models=ANTIGRAVITY_FALLBACK_MODELS,
    default_aux_model="gemini-3-flash-agent",
)

register_provider(antigravity)

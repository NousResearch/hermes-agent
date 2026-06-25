"""FreeLLMAPI provider profile.

FreeLLMAPI (https://github.com/tashfeenahmed/freellmapi) is a self-hosted
OpenAI-compatible proxy that stacks ~16 free LLM provider tiers behind one
``/v1/chat/completions`` endpoint.  The router handles 429/5xx failover
internally (up to 20 attempts per request) and tracks per-key RPM/RPD/TPM
caps so free tiers stay within limits.

Hermes integration:
  - Point ``model.provider`` at ``freellmapi`` and set ``FREELLMAPI_API_KEY``
    to the unified ``freellmapi-…`` key from the local dashboard.
  - Default base URL: ``http://127.0.0.1:3001/v1`` (Docker / desktop app).
  - Use ``model: auto`` to let the proxy pick the best free model, or pin a
    specific slug from ``GET /v1/models``.
  - ``X-Session-Id`` is forwarded from the Hermes session id so multi-turn
    conversations stay pinned to one upstream model for 30 minutes (sticky
    sessions).  When the proxy exhausts its internal chain it may still
    return 429 — Hermes' own fallback chain then rotates to the next
    configured provider.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:3001/v1"

# Curated agentic models for the /model picker when live catalog fetch fails.
_FALLBACK_MODELS = (
    "auto",
    "gemini-2.5-flash",
    "llama-3.3-70b-versatile",
    "qwen3-235b-a22b",
    "mistral-large-latest",
    "deepseek-chat",
)


def _resolve_base_url(base_url: str | None = None) -> str:
    """Resolve endpoint: explicit arg → env → profile default."""
    for candidate in (
        (base_url or "").strip(),
        os.environ.get("FREELLMAPI_BASE_URL", "").strip(),
    ):
        if candidate:
            return candidate.rstrip("/")
    return _DEFAULT_BASE_URL


class FreeLLMAPIProfile(ProviderProfile):
    """FreeLLMAPI local proxy — sticky sessions + OpenAI chat completions."""

    def build_api_kwargs_extras(
        self,
        *,
        session_id: str | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pin multi-turn affinity via FreeLLMAPI's ``X-Session-Id`` header."""
        top_level: dict[str, Any] = {}
        if session_id:
            top_level["extra_headers"] = {"X-Session-Id": session_id}
        return {}, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch enabled models from the local FreeLLMAPI instance."""
        resolved = _resolve_base_url(base_url)
        try:
            result = super().fetch_models(
                api_key=api_key,
                base_url=resolved,
                timeout=timeout,
            )
            if result:
                # Prefer router auto-pick first when the catalog is reachable.
                models = list(result)
                if "auto" not in models:
                    models.insert(0, "auto")
                return models
        except Exception as exc:
            logger.debug("fetch_models(freellmapi): %s", exc)
        return list(_FALLBACK_MODELS)


freellmapi = FreeLLMAPIProfile(
    name="freellmapi",
    aliases=("freellm", "free-llm-api", "free_llm_api"),
    display_name="FreeLLMAPI",
    description="Self-hosted free-tier router (~1.7B tokens/mo across 16 providers)",
    signup_url="https://github.com/tashfeenahmed/freellmapi",
    env_vars=("FREELLMAPI_API_KEY", "FREELLMAPI_BASE_URL"),
    base_url=_DEFAULT_BASE_URL,
    hostname="127.0.0.1",
    supports_vision=True,
    default_aux_model="gemini-2.5-flash",
    fallback_models=_FALLBACK_MODELS,
)

register_provider(freellmapi)

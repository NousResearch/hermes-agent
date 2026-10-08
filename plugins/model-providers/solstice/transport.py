"""Solstice inference transport: the native Gemini wire on the per-user-quota methods, with the user's
OAuth access token as a Bearer instead of an API key.

Everything else (request building, tool-call ids, thinking config, streaming, error mapping) is the
native adapter's, so a fix there reaches this provider too.

The adapter import lives inside :func:`solstice_client_class`, not at module level: the adapter pulls
``httpx``, and provider discovery imports every bundled plugin in processes that carry no HTTP client
(the PM worker -- a four-package interpreter -- imports ``hermes_cli.config``, which asks
``list_providers()`` for the env vars to inject). An eager import fails the whole plugin there.
"""

from __future__ import annotations

from typing import Any, Dict

INFERENCE_BASE_URL = "https://generativelanguage.googleapis.com/v1alpha"

_CLIENT_CLASS: Any = None


def solstice_client_class() -> Any:
    """``SolsticeClient``, defined on first use so discovery never needs the adapter (and ``httpx``)."""
    global _CLIENT_CLASS
    if _CLIENT_CLASS is None:
        from agent.gemini_native_adapter import GeminiAPIError, GeminiNativeClient, gemini_http_error

        class SolsticeClient(GeminiNativeClient):
            """``GeminiNativeClient`` on ``:generateContentPerUserQuota`` / ``:streamGenerateContentPerUserQuota``."""

            GENERATE_METHOD, STREAM_METHOD = "generateContentPerUserQuota", "streamGenerateContentPerUserQuota"
            MISSING_KEY_ERROR = "Solstice needs a signed-in account. Run `hermes auth add solstice`."

            def __init__(self, *, base_url: Any = None, **kwargs: Any) -> None:
                # No base means this endpoint, never the generic Gemini default (a different API surface).
                super().__init__(base_url=base_url or INFERENCE_BASE_URL, **kwargs)

            def _auth_headers(self) -> Dict[str, str]:
                return {"Authorization": f"Bearer {self.api_key}"}

            def _http_error(self, response: Any, body_text: Any = None) -> GeminiAPIError:
                # The API-key remedies (free tier, Standard key, wrong key surface) do not apply to a bearer.
                return gemini_http_error(response, body_text=body_text, base_url=self.base_url, key_guidance=False)

        _CLIENT_CLASS = SolsticeClient
    return _CLIENT_CLASS


def __getattr__(name: str) -> Any:
    """Keep ``from .transport import SolsticeClient`` working (the adapter loads at that point)."""
    if name == "SolsticeClient":
        return solstice_client_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

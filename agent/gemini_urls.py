"""Pure URL helpers for Gemini and Vertex AI express endpoints.

Shared across core (Vertex Express detection, Gemini TTS, doctor/config probes,
OpenAI-compat subpath detection) and the Google model-provider plugin. Keep
import-light (stdlib only) so callers without the HTTP stack can import it.
"""

from __future__ import annotations

import re
from typing import Any, Optional

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
# A Vertex AI express-mode base, when the user configures one explicitly: aiplatform serves the
# same native API under ``{version}/publishers/google/models/…``; the base carries that prefix so
# every ``{base}/models/{model}:…`` builder (chat, tier probe, TTS) needs no path branching.
# The key itself never decides routing — Google now issues ``AQ.…`` keys for BOTH Google AI Studio
# and Vertex express mode (#115306), so an express key reaches aiplatform only via this base config.
VERTEX_EXPRESS_BASE_URL = "https://aiplatform.googleapis.com/v1beta1/publishers/google"

_API_VERSION_SEGMENT = re.compile(r"^v\d+(?:alpha|beta)?\d*$", re.IGNORECASE)


def is_vertex_express_base_url(base_url: str) -> bool:
    """An aiplatform host speaking the native API with an API key: the bare express surface, or its
    ``…/projects/{p}/locations/{l}`` form, which reaches models served only in ``global`` (Gemini 4).
    The OAuth Vertex provider's ``…/endpoints/openapi`` base is OpenAI-compatible and must stay off
    the native adapter."""
    normalized = str(base_url or "").strip().lower()
    return "aiplatform.googleapis.com" in normalized and "/endpoints/" not in normalized


def normalize_gemini_base_url(base_url: Optional[str]) -> str:
    """Gemini native base URL with the API version segment guaranteed. Google's own client treats the
    base as a host root and appends the version itself, so users configure ``GEMINI_BASE_URL`` (or a
    proxy root like ``http://localhost:4000/gemini``) that way; our request builders expect
    ``{base}/models/{model}:generateContent`` — without ``/v1beta`` that is a guaranteed 404. Trailing
    slashes and an ``/openai`` suffix are stripped; an existing version segment (``v1``, ``v1beta``,
    ``v1alpha``, ...) is kept; empty input returns ``DEFAULT_GEMINI_BASE_URL``. Only the LAST path
    segment is inspected, so ``.../v1beta/extra`` still gets ``/v1beta`` appended; this does not
    decide routing (see ``is_native_gemini_base_url``).

    A Vertex express base (``aiplatform.googleapis.com``, ``…/v1beta1`` or the full
    ``…/v1beta1/publishers/google``) is completed to the ``publishers/google`` form. The key never
    decides routing: Google issues ``AQ.…`` keys for both AI Studio and Vertex express mode
    (#115306), so an express key reaches aiplatform only through this explicit base configuration."""
    trimmed = str(base_url or "").strip().rstrip("/")
    trimmed = re.sub(r"/openai\Z", "", trimmed, flags=re.IGNORECASE).rstrip("/")
    if not trimmed:
        trimmed = DEFAULT_GEMINI_BASE_URL
    if is_vertex_express_base_url(trimmed):
        if trimmed.lower().endswith("/publishers/google"):
            return trimmed
        if "/projects/" not in trimmed.lower() and not _API_VERSION_SEGMENT.match(trimmed.rsplit("/", 1)[-1]):
            trimmed = f"{trimmed}/v1beta1"
        return f"{trimmed}/publishers/google"
    if _API_VERSION_SEGMENT.match(trimmed.rsplit("/", 1)[-1]):
        return trimmed
    return f"{trimmed}/v1beta"


def is_native_gemini_base_url(base_url: str) -> bool:
    """True when the endpoint speaks Gemini's native REST API (not ``/openai``)."""
    normalized = str(base_url or "").strip().rstrip("/").lower()
    if normalized.endswith("/openai"):
        return False
    return "generativelanguage.googleapis.com" in normalized or is_vertex_express_base_url(normalized)


def is_gemini_openai_compat_base_url(base_url: Any) -> bool:
    """True for Google AI Studio's OpenAI-compatible ``…/openai`` subpath."""
    normalized = str(base_url or "").strip().rstrip("/").lower()
    return bool(normalized) and "generativelanguage.googleapis.com" in normalized and normalized.endswith("/openai")


def gemini_accepts_parameters_json_schema(base_url: str) -> bool:
    """``FunctionDeclaration.parametersJsonSchema`` exists only in the ``v1beta`` surface of
    generativelanguage (absent from ``v1`` / ``v1alpha`` content.proto); other versions and
    unknown proxies get the legacy ``parameters`` subset."""
    return str(base_url or "").strip().rstrip("/").lower().endswith("/v1beta")

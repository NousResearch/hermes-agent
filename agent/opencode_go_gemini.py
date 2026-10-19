"""Gemini on the OpenCode Go relay: Google's native wire, on its own root.

OpenCode Go advertises ``gemini-*`` in ``/zen/go/v1/models`` and serves it only from the documented
Google-wire root ``https://opencode.ai/inference/go/v1/models/{model}:generateContent``:

* the legacy ``/zen/go`` root answers 404 for ``:generateContent`` / ``:streamGenerateContent``
* ``/chat/completions``, ``/v1/messages`` and ``/v1/responses`` answer 400
  ``ModelProtocolUnsupported`` for these ids
* every Go path needs a stable ``x-opencode-session`` (else 400 ``MissingSessionID``)
* auth is ``x-goog-api-key`` or a bearer token; the ``?key=`` query form is rejected

Without this, ``opencode_model_api_mode`` leaves Go Gemini ids on ``chat_completions``, the 400 is
classified as a format error, and the turn silently continues on ``fallback_providers``.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlsplit

OPENCODE_GO_GEMINI_BASE_URL = "https://opencode.ai/inference/go/v1"

_GO_ROOT_PATHS = frozenset({"/zen/go", "/zen/go/v1", "/inference/go", "/inference/go/v1"})


def is_opencode_go_base_url(base_url: Optional[str]) -> bool:
    """An official OpenCode Go relay root, in its legacy or its documented spelling."""
    url = urlsplit(str(base_url or "").strip())
    return url.hostname == "opencode.ai" and url.path.rstrip("/").lower() in _GO_ROOT_PATHS


def _is_go_provider(provider: Optional[str]) -> bool:
    name = str(provider or "").strip().lower()
    if name.startswith("custom:"):
        name = name.split(":", 1)[1]
    try:
        from hermes_cli.models import opencode_provider_family

        family = opencode_provider_family(name)
    except Exception:
        family = None
    return family == "opencode-go" if family is not None else False


def is_opencode_go_gemini_route(provider: Optional[str], model: Optional[str], base_url: Optional[str]) -> bool:
    """True when this (provider, model, base_url) must use Google's wire on the Go relay."""
    if not is_opencode_go_base_url(base_url) or not _is_go_provider(provider):
        return False
    bare = str(model or "").strip().rsplit("/", 1)[-1].lower()
    return bare.startswith("gemini-")


def with_session_header(headers, provider: Optional[str], base_url: Optional[str]) -> dict:
    """Copy *headers* and add the conversation's ``x-opencode-session`` for a Go target."""
    merged = dict(headers or {})
    try:
        from agent.opencode_affinity import opencode_session_headers

        merged.update(opencode_session_headers(provider, base_url))
    except Exception:
        pass
    return merged

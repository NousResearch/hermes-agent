"""Azure Foundry endpoint auto-detection.

The detector never crashes on errors (every HTTP call is wrapped in a broad try/except). Callers get
a :class:`DetectionResult` with whatever information could be gathered, and fall back to manual
entry for the rest.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from hermes_cli.urllib_security import open_credentialed_url

logger = logging.getLogger(__name__)

TokenProvider = Optional[Callable[[], str]]

# Azure OpenAI ``api-version`` fallbacks for pre-v1 resources; the v1 GA endpoint accepts requests
# without ``api-version`` entirely, so these are only probed second.
_AZURE_OPENAI_PROBE_API_VERSIONS = (
    "2025-04-01-preview",
    "2024-10-21",  # oldest GA that supports /models
)

# Matches the value ``agent/anthropic_adapter.py`` uses when building the Anthropic client.
_AZURE_ANTHROPIC_API_VERSION = "2025-04-15"


@dataclass
class DetectionResult:
    """Everything auto-detection could gather from a base URL + API key."""

    #: ``"chat_completions"``, ``"anthropic_messages"``, or ``None`` when detection failed.
    api_mode: Optional[str] = None
    #: Deployment / model IDs returned by ``/models`` (best effort; empty when not exposed).
    models: list[str] = field(default_factory=list)
    #: Lowercased host from the base URL (used for display messages).
    hostname: str = ""
    #: Human-readable reason the detector chose ``api_mode`` (shown by the wizard).
    reason: str = ""
    #: ``True`` when ``/models`` returned a valid OpenAI-shaped payload.
    models_probe_ok: bool = False
    #: ``True`` when the URL was determined to be Anthropic-style (path suffix or live probe).
    is_anthropic: bool = False


def _resolve_credential(api_key: Any, token_provider: TokenProvider = None) -> tuple[Optional[str], str]:
    """Coerce wizard inputs into ``(token_or_None, mode)``.

    ``mode`` is ``"entra_id"`` when a callable token provider was supplied (the token is a freshly
    minted bearer JWT, sent ONLY in ``Authorization: Bearer``), else ``"api_key"``.
    """
    # Token-provider path (callable wins when both supplied).
    for provider, label in ((token_provider, "token_provider"), (api_key, "api_key callable")):
        if callable(provider) and not isinstance(provider, str):
            try:
                token = provider()
                return (str(token) if token else None), "entra_id"
            except Exception as exc:
                logger.debug("azure_detect: %s failed: %s", label, exc)
                return None, "entra_id"
    if isinstance(api_key, str) and api_key:
        return api_key, "api_key"
    return None, "api_key"


def _authed_request(url: str, api_key: Any, token_provider, *, method: str = "GET",
                    data: Optional[bytes] = None) -> urllib_request.Request:
    """Build a request carrying the right auth headers for the credential mode."""
    token, mode = _resolve_credential(api_key, token_provider)
    req = urllib_request.Request(url, method=method, data=data)
    if token:
        # Legacy broad-compat behaviour sends both headers so we land on any Azure resource. In
        # entra_id mode send Bearer ONLY — api-key would log a JWT in a slot meant for static keys.
        if mode != "entra_id":
            req.add_header("api-key", token)
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("User-Agent", "hermes-agent/azure-detect")
    return req


def _http_get_json(url: str, api_key: Any, timeout: float = 6.0, *,
                   token_provider: TokenProvider = None) -> tuple[int, Optional[dict]]:
    """GET with auth headers; return ``(status_code, parsed_json_or_None)``. Never raises."""
    req = _authed_request(url, api_key, token_provider)
    try:
        with open_credentialed_url(req, timeout=timeout) as resp:
            body = resp.read()
            try:
                return resp.status, json.loads(body.decode("utf-8", errors="replace"))
            except Exception:
                return resp.status, None
    except HTTPError as exc:
        return exc.code, None
    except (URLError, TimeoutError, OSError) as exc:
        logger.debug("azure_detect: GET %s failed: %s", url, exc)
        return 0, None
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("azure_detect: GET %s unexpected error: %s", url, exc)
        return 0, None


def _strip_trailing_v1(url: str) -> str:
    """Strip trailing ``/v1`` or ``/v1/`` so we can construct sub-paths."""
    return re.sub(r"/v1/?$", "", url.rstrip("/"))


def _looks_like_anthropic_path(url: str) -> bool:
    """True when the path ends in ``/anthropic`` or contains a ``/anthropic/`` segment (Foundry Claude routes)."""
    try:
        path = (urlparse(url).path or "").lower().rstrip("/")
        return path.endswith("/anthropic") or "/anthropic/" in path + "/"
    except Exception:
        return False


def _extract_model_ids(payload: dict) -> list[str]:
    """Model IDs from an OpenAI-shaped ``/models`` response; ``[]`` on any shape mismatch."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data:
        if isinstance(item, dict):
            mid = item.get("id") or item.get("model") or item.get("name")
            if isinstance(mid, str) and mid:
                ids.append(mid)
    return ids


def _probe_openai_models(base_url: str, api_key: Any, *, token_provider: TokenProvider = None) -> tuple[bool, list[str]]:
    """Probe ``<base>/models`` for an OpenAI-shaped response."""
    base_url = base_url.rstrip("/")
    # Azure OpenAI v1 needs no api-version for GA paths, so probe without first; then fall back to
    # explicit api-versions for pre-v1 resources.
    candidates = [f"{base_url}/models"] + [f"{base_url}/models?api-version={v}" for v in _AZURE_OPENAI_PROBE_API_VERSIONS]
    for url in candidates:
        status, body = _http_get_json(url, api_key, token_provider=token_provider)
        if status == 200 and body is not None:
            ids = _extract_model_ids(body)
            if ids:
                logger.info("azure_detect: /models probe OK at %s (%d models)", url, len(ids))
                return True, ids
            # 200 + empty list still counts as "OpenAI shape, no models listed".
            if isinstance(body, dict) and "data" in body:
                return True, []
    return False, []


def _probe_anthropic_messages(base_url: str, api_key: Any, *, token_provider: TokenProvider = None) -> bool:
    """Zero-token POST to ``<base>/v1/messages``: does the endpoint *recognise* the Anthropic shape?

    Any 4xx mentioning ``messages``/``model``, or an Anthropic-shaped error body, counts. Never
    completes a real chat.
    """
    url = f"{_strip_trailing_v1(base_url)}/v1/messages?api-version={_AZURE_ANTHROPIC_API_VERSION}"
    payload = json.dumps({"model": "probe", "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]}).encode("utf-8")
    req = _authed_request(url, api_key, token_provider, method="POST", data=payload)
    req.add_header("anthropic-version", "2023-06-01")
    req.add_header("content-type", "application/json")
    try:
        with open_credentialed_url(req, timeout=6.0) as resp:
            # Should never 200 — "probe" isn't a real deployment — but if it does, it speaks Anthropic.
            return resp.status < 500
    except HTTPError as exc:
        try:
            lowered = exc.read().decode("utf-8", errors="replace").lower()
            if "anthropic" in lowered or '"type"' in lowered and '"error"' in lowered:
                return True
            # Pre-Azure-v1 Foundry returns a plain 404 for Anthropic-style calls on non-Anthropic
            # deployments. A 400 "model not found" IS Anthropic though.
            return exc.code == 400 and ("messages" in lowered or "model" in lowered)
        except Exception:
            return False
    except Exception:  # URLError, TimeoutError, OSError, anything else
        return False


def detect(base_url: str, api_key: Any = "", *, token_provider: TokenProvider = None) -> DetectionResult:
    """Inspect an Azure endpoint and describe its transport + models (advisory — None api_mode means ask the user).

    ``api_key`` may be a string (legacy API-key auth — sends both ``api-key:`` and ``Authorization:
    Bearer``) or a callable returning a bearer JWT (Entra ID auth — sends ONLY ``Authorization:
    Bearer``). ``token_provider`` is an explicit name for the callable form; the callable wins.
    """
    result = DetectionResult()
    try:
        result.hostname = (urlparse(base_url).hostname or "").lower()
    except Exception:
        result.hostname = ""

    # 1. Path sniff: Foundry exposes Anthropic-style deployments under a dedicated /anthropic path.
    if _looks_like_anthropic_path(base_url):
        result.is_anthropic = True
        result.api_mode = "anthropic_messages"
        result.reason = "URL path ends in /anthropic → Anthropic Messages API"
        return result

    # 2. OpenAI-style /models probe — success means the endpoint definitely speaks OpenAI wire.
    ok, models = _probe_openai_models(base_url, api_key, token_provider=token_provider)
    if ok:
        result.models_probe_ok = True
        result.models = models
        result.api_mode = "chat_completions"
        result.reason = (
            f"GET /models returned {len(models)} model(s) — OpenAI-style endpoint" if models
            else "GET /models returned an OpenAI-shaped empty list — OpenAI-style endpoint"
        )
        return result

    # 3. Anthropic Messages probe — slower and more intrusive, so only when /models failed.
    if _probe_anthropic_messages(base_url, api_key, token_provider=token_provider):
        result.is_anthropic = True
        result.api_mode = "anthropic_messages"
        result.reason = "Endpoint accepts Anthropic Messages shape"
        return result

    result.reason = (
        "Could not probe endpoint (private network, missing model list, or "
        "non-standard path) — falling back to manual API-mode selection"
    )
    return result


def lookup_context_length(model: str, base_url: str, api_key: Any = "", *,
                          token_provider: TokenProvider = None) -> Optional[int]:
    """``get_model_context_length`` that returns None when only the fallback default would fire, so
    the wizard can distinguish "we actually know this" from "we guessed".
    """
    model_id = str(model or "").strip()
    if not model_id:
        return None
    try:
        from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT, get_model_context_length
    except Exception:
        return None

    # Resolve the credential once: Entra mode calls the provider; api_key is a string pass-through.
    token, _mode = _resolve_credential(api_key, token_provider)
    try:
        n = get_model_context_length(model_id, base_url=base_url, api_key=token or "")
    except Exception as exc:
        logger.debug("azure_detect: context length lookup failed: %s", exc)
        return None
    return n if isinstance(n, int) and n > 0 and n != DEFAULT_FALLBACK_CONTEXT else None


__all__ = ["DetectionResult", "detect", "lookup_context_length"]

"""Azure Foundry endpoint auto-detection.

Inspect a Microsoft Foundry / Azure OpenAI endpoint to determine:
  - API transport (OpenAI-style ``chat_completions`` vs
    Anthropic-style ``anthropic_messages``)
  - Available models (best effort — Azure does not expose a deployment
    listing via the inference API key, but Azure OpenAI v1 endpoints
    return the resource's model catalog via ``GET /models``)
  - Context length for each discovered/entered model, via the existing
    :func:`agent.model_metadata.get_model_context_length` resolver.

Rationale:

Azure has no pure-API-key deployment-listing endpoint — per Microsoft,
deployment enumeration requires ARM management-plane auth.  Azure
OpenAI v1 endpoints ``{resource}.openai.azure.com/openai/v1`` do return
a ``/models`` list, but it reflects the resource's *available* models
rather than the user's *deployed* deployment names.  In practice it is
still a useful hint — the user picks a familiar model name and we look
up its context length from the catalog.

Authentication modes:
  - ``api_key`` (default): the wizard passes an ``api_key`` string; the
    probe sends both ``api-key:`` and ``Authorization: Bearer`` headers
    so we hit any Azure deployment regardless of which header it expects.
  - ``entra_id``: the wizard passes a ``token_provider`` callable from
    :mod:`agent.azure_identity_adapter`. The probe mints exactly one
    bearer JWT, sends **only** ``Authorization: Bearer <jwt>`` (never
    ``api-key:``), and never persists the token. This matches Microsoft's
    documented contract for keyless inference.

The detector never crashes on errors (every HTTP call is wrapped in a
broad try/except).  Callers get a :class:`DetectionResult` with whatever
information could be gathered, and fall back to manual entry for the
rest.
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

logger = logging.getLogger(__name__)


# Default Azure OpenAI ``api-version`` to probe with.  The v1 GA endpoint
# accepts requests without ``api-version`` entirely, so this is only used
# as a fallback for pre-v1 resources that still require it.
_AZURE_OPENAI_PROBE_API_VERSIONS = (
    "2025-04-01-preview",
    "2024-10-21",  # oldest GA that supports /models
)

# Default Azure Anthropic ``api-version``.  Matches the value used by
# ``agent/anthropic_adapter.py`` when building the Anthropic client.
_AZURE_ANTHROPIC_API_VERSION = "2025-04-15"

_AZURE_DETECT_JSON_BODY_LIMIT_BYTES = 1 * 1024 * 1024
_AZURE_DETECT_ERROR_BODY_LIMIT_BYTES = 8 * 1024


@dataclass
class DetectionResult:
    """Everything auto-detection could gather from a base URL + API key."""

    #: Detected API transport: ``"chat_completions"``,
    #: ``"anthropic_messages"``, or ``None`` when detection failed.
    api_mode: Optional[str] = None

    #: Deployment / model IDs returned by ``/models`` (best effort).
    #: Empty when the endpoint doesn't expose the list with an API key.
    models: list[str] = field(default_factory=list)

    #: Lowercased host from the base URL (used for display messages).
    hostname: str = ""

    #: Human-readable reason the detector chose ``api_mode``.  Useful
    #: for explaining auto-detection to the user in the wizard.
    reason: str = ""

    #: ``True`` when ``/models`` returned a valid OpenAI-shaped payload.
    models_probe_ok: bool = False

    #: ``True`` when the URL was determined to be an Anthropic-style
    #: endpoint (from path suffix or live probe).
    is_anthropic: bool = False


def _resolve_credential(api_key: Any,
                        token_provider: Optional[Callable[[], str]] = None,
                        ) -> tuple[Optional[str], str]:
    """Coerce wizard inputs into a (token, mode) pair.

    Returns ``(token_or_None, mode)`` where ``mode`` is:
      - ``"entra_id"`` when a callable token provider was supplied — the
        returned token is a freshly minted bearer JWT, sent ONLY in
        ``Authorization: Bearer``.
      - ``"api_key"`` when a string key was supplied — the returned token
        is the raw API key, sent in BOTH ``api-key:`` and
        ``Authorization: Bearer`` headers (preserves the original
        broad-compat probe behaviour).
      - ``("", "api_key")`` when neither yields a value.

    Bearer minting failures degrade to ``("", "entra_id")`` so the caller
    can still report "detection incomplete" rather than crashing.
    """
    # Token-provider path (callable wins when both supplied).
    if token_provider is not None and callable(token_provider):
        try:
            token = token_provider()
            return (str(token) if token else None), "entra_id"
        except Exception as exc:
            logger.debug("azure_detect: token_provider failed: %s", exc)
            return None, "entra_id"
    if callable(api_key) and not isinstance(api_key, str):
        try:
            token = api_key()
            return (str(token) if token else None), "entra_id"
        except Exception as exc:
            logger.debug("azure_detect: api_key callable failed: %s", exc)
            return None, "entra_id"
    # API-key path.
    if isinstance(api_key, str) and api_key:
        return api_key, "api_key"
    return None, "api_key"


def _apply_auth_headers(req: urllib_request.Request,
                        token: Optional[str],
                        mode: str) -> None:
    """Attach the right auth headers to ``req`` based on credential mode."""
    if not token:
        return
    if mode == "entra_id":
        # Bearer-only: do NOT also set api-key, which would log a JWT in
        # a header slot intended for static keys.
        req.add_header("Authorization", f"Bearer {token}")
    else:
        # Legacy broad-compat behaviour: send both headers so we land on
        # any Azure resource regardless of which it accepts.
        req.add_header("api-key", token)
        req.add_header("Authorization", f"Bearer {token}")


def _read_response_body_limited(resp: Any, max_bytes: int) -> bytes:
    """Read at most ``max_bytes`` from a urllib response-like object."""
    if max_bytes <= 0:
        return b""
    try:
        body = resp.read(max_bytes + 1)
    except TypeError:
        body = resp.read()
    if not isinstance(body, (bytes, bytearray)):
        body = bytes(body or b"")
    return bytes(body[:max_bytes])


def _http_get_json(url: str,
                   api_key: Any,
                   timeout: float = 6.0,
                   *,
                   token_provider: Optional[Callable[[], str]] = None,
                   ) -> tuple[int, Optional[dict]]:
    """GET a URL with the appropriate auth headers.  Return
    ``(status_code, parsed_json_or_None)``.  Never raises."""
    token, mode = _resolve_credential(api_key, token_provider)
    req = urllib_request.Request(url, method="GET")
    _apply_auth_headers(req, token, mode)
    req.add_header("User-Agent", "hermes-agent/azure-detect")
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = _read_response_body_limited(
                resp,
                _AZURE_DETECT_JSON_BODY_LIMIT_BYTES,
            )
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
    """Return True when the URL's path ends in ``/anthropic`` or
    contains a ``/anthropic/`` segment.  Used by Azure Foundry
    resources that route Claude traffic through a dedicated path."""
    try:
        parsed = urlparse(url)
        path = (parsed.path or "").lower().rstrip("/")
        return path.endswith("/anthropic") or "/anthropic/" in path + "/"
    except Exception:
        return False


def _extract_model_ids(payload: dict) -> list[str]:
    """Extract a list of model IDs from an OpenAI-shaped ``/models``
    response.  Returns ``[]`` on any shape mismatch."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # OpenAI shape: {"id": "gpt-5.4", "object": "model", ...}
        mid = item.get("id") or item.get("model") or item.get("name")
        if isinstance(mid, str) and mid:
            ids.append(mid)
    return ids


def _probe_openai_models(base_url: str,
                         api_key: Any,
                         *,
                         token_provider: Optional[Callable[[], str]] = None,
                         ) -> tuple[bool, list[str]]:
    """Probe ``<base>/models`` for an OpenAI-shaped response.

    Returns ``(ok, models)``.  ``ok`` is True iff the endpoint accepted
    us as an OpenAI-style caller (200 OK + OpenAI-shaped JSON body).
    """
    base_url = base_url.rstrip("/")

    # Azure OpenAI v1: {resource}.openai.azure.com/openai/v1 — no
    # api-version required for GA paths, so probe without first.
    candidates = [f"{base_url}/models"]
    # Fallback: explicit api-version for pre-v1 resources
    for v in _AZURE_OPENAI_PROBE_API_VERSIONS:
        candidates.append(f"{base_url}/models?api-version={v}")

    for url in candidates:
        status, body = _http_get_json(url, api_key, token_provider=token_provider)
        if status == 200 and body is not None:
            ids = _extract_model_ids(body)
            if ids:
                logger.info(
                    "azure_detect: /models probe OK at %s (%d models)",
                    url, len(ids),
                )
                return True, ids
            # 200 + empty list still counts as "OpenAI shape, no models
            # listed" — let the user proceed with manual entry.
            if isinstance(body, dict) and "data" in body:
                return True, []
    return False, []


def _probe_anthropic_messages(base_url: str,
                              api_key: Any,
                              *,
                              token_provider: Optional[Callable[[], str]] = None,
                              ) -> bool:
    """Send a zero-token request to ``<base>/v1/messages`` and check
    whether the endpoint at least *recognises* the Anthropic Messages
    shape (any 4xx that mentions ``messages`` or ``model``, or a 400
    ``invalid_request`` with an Anthropic error shape).  Never completes
    a real chat.
    """
    base = _strip_trailing_v1(base_url)
    url = f"{base}/v1/messages?api-version={_AZURE_ANTHROPIC_API_VERSION}"
    payload = json.dumps({
        "model": "probe",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    }).encode("utf-8")
    req = urllib_request.Request(url, method="POST", data=payload)
    token, mode = _resolve_credential(api_key, token_provider)
    _apply_auth_headers(req, token, mode)
    req.add_header("anthropic-version", "2023-06-01")
    req.add_header("content-type", "application/json")
    req.add_header("User-Agent", "hermes-agent/azure-detect")
    try:
        with urllib_request.urlopen(req, timeout=6.0) as resp:
            # Should never 200 — "probe" isn't a real deployment.  But
            # if it does, the endpoint definitely speaks Anthropic.
            return resp.status < 500
    except HTTPError as exc:
        # 4xx with an Anthropic-shaped error body = Anthropic endpoint.
        try:
            body = _read_response_body_limited(
                exc,
                _AZURE_DETECT_ERROR_BODY_LIMIT_BYTES,
            ).decode("utf-8", errors="replace")
            lowered = body.lower()
            if "anthropic" in lowered or '"type"' in lowered and '"error"' in lowered:
                return True
            # Pre-Azure-v1 Azure Foundry returns a plain 404 for
            # Anthropic-style calls on non-Anthropic deployments.  A
            # 400 "model not found" IS Anthropic though.
            if exc.code == 400 and ("messages" in lowered or "model" in lowered):
                return True
            return False
        except Exception:
            return False
    except (URLError, TimeoutError, OSError):
        return False
    except Exception:  # pragma: no cover
        return False


def detect(base_url: str,
           api_key: Any = "",
           *,
           token_provider: Optional[Callable[[], str]] = None,
           ) -> DetectionResult:
    """Inspect an Azure endpoint and describe its transport + models.

    Call this from the wizard before asking the user to pick an API
    mode manually.  The caller should treat the returned
    :class:`DetectionResult` as *advisory* — if ``api_mode`` is None,
    fall back to asking the user.

    ``api_key`` may be a string (legacy API-key auth — sends both
    ``api-key:`` and ``Authorization: Bearer``) or a callable returning
    a bearer JWT (Entra ID auth — sends ONLY ``Authorization: Bearer``).
    ``token_provider`` is an alternative explicit name for the callable
    form; if both are supplied the callable wins.
    """
    result = DetectionResult()

    try:
        parsed = urlparse(base_url)
        result.hostname = (parsed.hostname or "").lower()
    except Exception:
        result.hostname = ""

    # 1. Path sniff.  Azure Foundry exposes Anthropic-style deployments
    #    under a dedicated ``/anthropic`` path.
    if _looks_like_anthropic_path(base_url):
        result.is_anthropic = True
        result.api_mode = "anthropic_messages"
        result.reason = "URL path ends in /anthropic → Anthropic Messages API"
        return result

    # 2. Try the OpenAI-style /models probe.  If this works, the
    #    endpoint definitely speaks OpenAI wire.
    ok, models = _probe_openai_models(base_url, api_key, token_provider=token_provider)
    if ok:
        result.models_probe_ok = True
        result.models = models
        result.api_mode = "chat_completions"
        result.reason = (
            f"GET /models returned {len(models)} model(s) — OpenAI-style endpoint"
            if models
            else "GET /models returned an OpenAI-shaped empty list — OpenAI-style endpoint"
        )
        return result

    # 3. Fallback: probe the Anthropic Messages shape.  Slower and more
    #    intrusive than /models, so only run it when the OpenAI probe
    #    failed.
    if _probe_anthropic_messages(base_url, api_key, token_provider=token_provider):
        result.is_anthropic = True
        result.api_mode = "anthropic_messages"
        result.reason = "Endpoint accepts Anthropic Messages shape"
        return result

    # Nothing matched.  Caller falls back to manual selection.
    result.reason = (
        "Could not probe endpoint (private network, missing model list, or "
        "non-standard path) — falling back to manual API-mode selection"
    )
    return result


def lookup_context_length(model: str,
                          base_url: str,
                          api_key: Any = "",
                          *,
                          token_provider: Optional[Callable[[], str]] = None,
                          ) -> Optional[int]:
    """Thin wrapper around :func:`agent.model_metadata.get_model_context_length`
    that returns ``None`` when only the fallback default (128k) would
    fire, so the wizard can distinguish "we actually know this" from
    "we guessed.

    For Entra-ID mode pass a callable as ``api_key`` (or via
    ``token_provider=``); the wrapped resolver expects a string, so we
    mint one bearer JWT here for the single lookup. The resolver itself
    only reads catalog metadata over HTTP — no SDK client is built — so
    the minted token is consumed for at most one /models probe.
    """
    model_id = str(model or "").strip()
    if not model_id:
        return None
    try:
        from agent.model_metadata import (
            DEFAULT_FALLBACK_CONTEXT,
            get_model_context_length,
        )
    except Exception:
        return None

    # Resolve the credential once. For Entra mode this calls the token
    # provider; for legacy api_key this is a no-op string pass-through.
    token, mode = _resolve_credential(api_key, token_provider)
    effective_key = token or ""

    try:
        n = get_model_context_length(model_id, base_url=base_url, api_key=effective_key)
    except Exception as exc:
        logger.debug("azure_detect: context length lookup failed: %s", exc)
        return None

    if isinstance(n, int) and n > 0 and n != DEFAULT_FALLBACK_CONTEXT:
        return n
    return None


__all__ = ["DetectionResult", "detect", "lookup_context_length"]

"""Free-route admission, errors, and shared wire helpers."""

from __future__ import annotations

import email.utils
import math
import time
import urllib.request
from datetime import timezone
from typing import Any

_MAX_REQUEST_BODY_BYTES = 1_000_000
_MAX_RESPONSE_BODY_BYTES = 10_000_000
_MAX_CATALOG_BODY_BYTES = 2_000_000
_MAX_SSE_LINE_BYTES = 1_000_000
_MAX_SSE_EVENT_BYTES = 2_000_000
_CATALOG_TTL_SECONDS = 60.0

_ROUTER_MODELS = frozenset({"freemaxxing", "fm", "freemaxxing-auto"})
_PREFERRED_AUTO_MODELS = (
    "deepseek/deepseek-v4-flash-0731",
    "deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-flash",
    "deepseek-v4-flash",
)
_NOUS_FREE_MODEL_IDS = frozenset(_PREFERRED_AUTO_MODELS)


def _open_credentialed(req: urllib.request.Request, *, timeout: float):
    """Open a credentialed URL through Hermes' redirect-safe helper."""
    try:
        from hermes_cli.urllib_security import open_credentialed_url
    except ImportError:  # Standalone test harness only.
        return urllib.request.urlopen(req, timeout=timeout)
    return open_credentialed_url(req, timeout=timeout)


class RateLimitError(Exception):
    def __init__(self, message: str, retry_after: float = 30.0):
        super().__init__(message)
        self.retry_after = retry_after


class TransientError(Exception):
    """Retryable connection, HTTP 5xx, or response-integrity failure."""


class ModelNotFoundError(Exception):
    """The selected backend cannot serve the requested model."""


class AuthError(Exception):
    """The selected backend rejected or lacks a credential."""


class ClientRequestError(Exception):
    """The caller sent a malformed/non-retryable request."""


def _is_router_model(model: str) -> bool:
    return (model or "").strip().lower() in _ROUTER_MODELS


def _backend_kind(backend: Any) -> str:
    return (backend.name or "").strip().lower()


def _accept_catalog_id(backend: Any, model_id: str) -> bool:
    """Admit only routes whose free status is provable for that backend."""
    model = (model_id or "").strip()
    if not model:
        return False
    lowered = model.lower()
    if lowered.endswith(":batch") or lowered.startswith("~"):
        return False

    kind = _backend_kind(backend)
    if kind.startswith("openrouter"):
        return lowered.endswith(":free")
    if kind in {"nous", "nous-portal"}:
        return lowered in _NOUS_FREE_MODEL_IDS
    # Test/private backends are not automatically enrolled by the plugin.  Their
    # operator owns pricing policy, so ordinary ids remain usable in the generic
    # pool implementation.
    return True


def _hermes_user_agent() -> str:
    try:
        from hermes_cli import __version__ as version

        return f"hermes-cli/{version}"
    except Exception:
        return "hermes-cli"


def _parse_retry_after(headers: Any) -> float:
    raw = headers.get("Retry-After") if headers is not None else None
    if raw is None:
        return 30.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        try:
            parsed = email.utils.parsedate_to_datetime(str(raw))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            value = parsed.timestamp() - time.time()
        except Exception:
            return 30.0
    if not math.isfinite(value):
        return 30.0
    return min(max(value, 0.0), 300.0)

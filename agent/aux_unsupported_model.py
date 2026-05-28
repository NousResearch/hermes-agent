"""Detect "model not supported / not found" auxiliary errors.

Split out of ``auxiliary_client.py`` to keep that module from growing past
the 1k-ish line soft ceiling for retry logic.

The detector covers the common provider-side failure modes that arise when
the user's chosen main chat model is forwarded verbatim to a provider that
only supports a subset of its catalog for auxiliary (cheap / fast / side)
tasks. Without detection, the operator sees a confusing ``HTTP 401`` or
``HTTP 404`` warning against a chat model that clearly works for chat.

Signals (any match = True):

* HTTP 404 with a ``model`` + ``not found`` / ``does not exist`` / ``unknown``
  marker in the error text.
* HTTP 401 with an explicit ``not supported`` / ``unknown model`` /
  ``model ... is not`` marker in the error text (some aggregators wrap
  unsupported-model rejections under 401 instead of 404).

Pure auth errors (401 with no model-signal wording) return False -- those
are handled by the auth-refresh chain in ``auxiliary_client.call_llm``.
"""

from __future__ import annotations

from typing import Optional


_UNSUPPORTED_MARKERS_404 = (
    "not found",
    "does not exist",
    "unknown model",
    "no such model",
)

_UNSUPPORTED_MARKERS_GENERAL = (
    "is not supported",
    "not supported",
    "unknown model",
    "unsupported model",
    "model not available",
)


def _err_text(exc: BaseException) -> str:
    """Best-effort lowercased error text from an exception.

    Concatenates ``str(exc)`` with the ``message`` / ``error.message`` fields
    that OpenAI-style client exceptions populate, then lower-cases. Returns
    ``""`` if nothing is extractable.
    """
    parts = [str(exc)]
    msg = getattr(exc, "message", None)
    if isinstance(msg, str):
        parts.append(msg)
    err_obj = getattr(exc, "error", None)
    if isinstance(err_obj, dict):
        m = err_obj.get("message")
        if isinstance(m, str):
            parts.append(m)
    elif hasattr(err_obj, "message"):
        m = getattr(err_obj, "message", None)
        if isinstance(m, str):
            parts.append(m)
    return " ".join(p for p in parts if p).lower()


def is_model_unsupported_error(exc: BaseException) -> bool:
    """Return True iff ``exc`` rejects the requested *model* itself.

    Distinct from:
      * ``_is_auth_error`` -- pure credential failure (401, no model wording).
      * ``_is_payment_error`` -- 402 / quota exhaustion.
      * ``_is_connection_error`` -- DNS / TCP / timeout.
      * ``_is_rate_limit_error`` -- 429.

    Use this to decide whether to fall back to a provider's default
    auxiliary model (``_get_aux_model_for_provider``) instead of raising.
    """
    status: Optional[int] = getattr(exc, "status_code", None)
    text = _err_text(exc)

    # Must mention a model at all -- otherwise "not found" / 404 could
    # route / endpoint misses (different recovery path).
    if "model" not in text:
        return False

    if status == 404:
        return any(m in text for m in _UNSUPPORTED_MARKERS_404)

    if status == 401:
        # Aggregators sometimes classify unsupported-model rejections as
        # 401. Require the explicit "not supported" wording so we don't
        # collide with pure credential failures.
        return any(m in text for m in _UNSUPPORTED_MARKERS_GENERAL)

    # Some providers return 400 / 422 with "not supported".
    if status in {400, 422}:
        return any(m in text for m in _UNSUPPORTED_MARKERS_GENERAL)

    return False

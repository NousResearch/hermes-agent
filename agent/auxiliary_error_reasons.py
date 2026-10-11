"""Error-classification predicates for the auxiliary provider ladder.

Split out of ``agent/auxiliary_client.py``, which is over its line ratchet and may only shrink.

These answer "which fallback reason does this provider error carry" for the request-scoped
failures where the ROUTE cannot serve the request: a dead model id, a model the route does not
support, a malformed 200, or a structured error that arrived without an HTTP status.
``_FALLBACK_REASONS`` in ``auxiliary_client`` pairs each predicate with its label; capacity and
credential failures (auth, payment, rate limit, connection) stay with the ladder that owns their
recovery. The two small helpers below are shared by those predicates and by the ladder.
"""

from __future__ import annotations

from typing import Any


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    """True when any needle is a substring of ``text``."""
    return any(kw in text for kw in needles)


def _exc_http_status(exc: Exception) -> Any:
    """HTTP status on the exception itself or on its ``response`` (None when neither carries one)."""
    return getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)


def _is_model_not_found_error(exc: Exception) -> bool:
    """"Requested model doesn't exist" (404 / invalid model) — typically a long-lived process pinned a
    since-dropped model, or a versioned SKU (`:free` previews) the catalog retired. Excludes billing
    keywords, which :func:`_is_payment_error` owns."""
    status = getattr(exc, "status_code", None)
    err_lower = str(exc).lower()
    if _contains_any(err_lower, (
        "credits", "insufficient funds", "billing", "out of funds", "balance_depleted",
        "no usable credits", "free tier", "free-tier", "not available on the free tier",
    )):
        return False
    if status not in {404, 400, None}:
        return False
    return _contains_any(err_lower, (
        "model does not exist", "does not exist in our configuration", "openrouter catalog",
        "is not a valid model", "no such model", "model not found",
        "the model `",            # OpenAI-style: "The model `X` does not exist"
        "model_not_found", "unknown model",
        # MODEL-LEVEL RETIREMENT. A versioned SKU that leaves the catalog (`:free` previews are
        # retired routinely) answers 404 with a model-scoped body — not the account-level credit
        # wording :func:`_is_payment_error` owns. Left unclassified it is admitted to NO fallback
        # reason, so a retired aux pin fails for days (title generation did in practice).
        "no longer free", "is no longer free",           # Nous free-tier retirement
        "has been retired", "model has been retired", "model is retired", "model retired",
        "model has been discontinued", "model is no longer available", "model no longer available",
    ))


def _is_model_incompatible_error(exc: Exception) -> bool:
    """"This route cannot serve this model" 400 (capability mismatch, e.g. a Codex/ChatGPT-account
    fallback asked to run a non-OpenAI model). Auth/payment predicates don't fire, so this keeps the
    chain going instead of aborting. Excludes billing 400s and not-found 400s."""
    status = getattr(exc, "status_code", None)
    if status not in {400, None}:
        return False
    err_lower = str(exc).lower()
    if _is_model_not_found_error(exc):
        return False
    # Billing keywords checked directly: _is_payment_error is status-gated and misses 400-coded billing bodies.
    if _contains_any(err_lower, (
        "credits", "insufficient funds", "billing", "out of funds", "balance_depleted",
        "no usable credits", "payment required", "free tier", "free-tier",
        "not available on the free tier", "model_not_supported_on_free_tier", "quota",
    )):
        return False
    return _contains_any(err_lower, (
        "is not supported when using",   # codex/ChatGPT-account model gating
        "model is not supported", "not supported with this", "not supported for this account",
        "model_not_supported", "does not support this model", "unsupported model",
    ))


def _is_invalid_aux_response_error(exc: Exception) -> bool:
    """HTTP-200 empty/malformed ChatCompletions — a capability failure routed like model incompatibility."""
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return "auxiliary " in msg and "llm returned invalid response" in msg and "choices[0].message" in msg


def _is_statusless_structured_provider_error(exc: Exception) -> bool:
    """Detect a structured provider failure that has no HTTP status.

    OpenAI-compatible relays may commit SSE with status 200, then send an
    OpenAI-style ``error`` event. The SDK raises a status-less ``APIError`` with
    ``body=data["error"]`` — the INNER error object or a bare string (an
    ``{"error": ...}`` wrapper is accepted too). Any non-empty structured error in
    that status-less shape is a route failure; ordinary HTTP errors keep their
    existing status-based classifiers, and message text alone is insufficient.
    """
    if _exc_http_status(exc) is not None:
        return False
    body = getattr(exc, "body", None)
    err = body.get("error") if isinstance(body, dict) and "error" in body else body
    if isinstance(err, str):
        return bool(err.strip())
    return isinstance(err, dict) and any(err.get(k) for k in ("type", "code", "message"))
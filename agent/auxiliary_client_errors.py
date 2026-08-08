"""Error-predicate leaf family for the auxiliary client router.

Extracted byte-verbatim from ``agent/auxiliary_client.py`` lines 3615-4017
(R2 slice, epic #78647 / #78635): self-contained error-classification
predicates plus the "recently 402'd" unhealthy-provider cache and the
transient-retry constants.

Seam identity: every name in this module is re-exported from
``agent.auxiliary_client`` (``getattr(auxiliary_client, n) is
getattr(auxiliary_client_errors, n)``), so imports, lazy imports, and
monkeypatches that target the godfile keep resolving to these objects.
Keep the two re-export surfaces in sync when editing either side.
"""

import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ── Auxiliary "recently 402'd" unhealthy-provider cache ────────────────────
#
# When an auxiliary provider returns HTTP 402 (Payment Required / credit
# exhaustion), retrying it on every subsequent aux call is wasteful — the
# provider stays depleted for hours or days, but the chain re-tries it as
# the FIRST entry on every compression/title-gen/session-search call,
# burns ~1 RTT, gets 402 again, then falls back. On a long Discord/LCM
# session that adds up to dozens of doomed 402s.
#
# Solution: when ANY caller observes a payment error against a provider,
# mark it unhealthy for ``_AUX_UNHEALTHY_TTL_SECONDS``. ``_resolve_auto``
# Step-2 and ``_try_payment_fallback`` both consult this cache and skip
# unhealthy entries (logging once per skip-reason so the user sees what
# happened). Entries auto-expire so a topped-up account recovers without
# manual intervention.
#
# Failure isolation: the cache is in-process only. A second hermes
# process won't inherit the unhealthy mark — that's intentional, since
# the user might be running two profiles with different OpenRouter keys.

_AUX_UNHEALTHY_TTL_SECONDS = 600  # 10 minutes
_aux_unhealthy_until: Dict[str, float] = {}
_aux_unhealthy_logged_at: Dict[str, float] = {}

# Map provider names that show up in resolved_provider / explicit-config
# back to the chain labels used by _get_provider_chain(). Keep in sync
# with the alias map in _try_payment_fallback below.
_AUX_UNHEALTHY_LABEL_ALIASES = {
    "openrouter": "openrouter",
    "nous": "nous",
    "custom": "local/custom",
    "local/custom": "local/custom",
    "openai-codex": "openai-codex",
    "codex": "openai-codex",
}


def _normalize_chain_label(provider: str) -> str:
    """Normalize a resolved_provider value to a chain label used by
    ``_get_provider_chain()``. Falls back to the lowercased input for
    direct API-key providers (deepseek, alibaba, minimax, etc.) which
    each report their own provider name from the api-key chain.
    """
    if not provider:
        return ""
    p = str(provider).strip().lower()
    return _AUX_UNHEALTHY_LABEL_ALIASES.get(p, p)


def _mark_provider_unhealthy(provider: str, ttl: Optional[float] = None) -> None:
    """Mark ``provider`` as recently-402'd, hidden from chain iteration
    until the TTL expires. Called from the payment-fallback branches in
    ``call_llm`` and ``acall_llm`` after a confirmed payment error.
    """
    label = _normalize_chain_label(provider)
    if not label:
        return
    expires_at = time.time() + (ttl if ttl is not None else _AUX_UNHEALTHY_TTL_SECONDS)
    _aux_unhealthy_until[label] = expires_at
    logger.warning(
        "Auxiliary: marking %s unhealthy for %ds (payment / credit error). "
        "Subsequent auxiliary calls will skip it until %s.",
        label,
        int(ttl if ttl is not None else _AUX_UNHEALTHY_TTL_SECONDS),
        time.strftime("%H:%M:%S", time.localtime(expires_at)),
    )


def _is_provider_unhealthy(label: str) -> bool:
    """True iff ``label`` is in the unhealthy cache and the TTL hasn't expired.
    Lazily evicts expired entries so the cache stays small.
    """
    if not label:
        return False
    expires_at = _aux_unhealthy_until.get(label)
    if expires_at is None:
        return False
    if time.time() >= expires_at:
        _aux_unhealthy_until.pop(label, None)
        _aux_unhealthy_logged_at.pop(label, None)
        return False
    return True


def _log_skip_unhealthy(label: str, task: Optional[str] = None) -> None:
    """Emit a single info-level log per minute when we skip an unhealthy
    provider. Avoids spamming the log on bursty sessions while still
    giving the user a trail.
    """
    now = time.time()
    last = _aux_unhealthy_logged_at.get(label, 0.0)
    if now - last >= 60:
        _aux_unhealthy_logged_at[label] = now
        expires_at = _aux_unhealthy_until.get(label, now)
        logger.info(
            "Auxiliary %s: skipping %s (recently returned payment error, retry in %ds)",
            task or "call", label, max(0, int(expires_at - now)),
        )


def _reset_aux_unhealthy_cache() -> None:
    """Clear the unhealthy cache. Used by tests and by a future explicit
    user trigger (e.g. ``hermes config aux reset``)."""
    _aux_unhealthy_until.clear()
    _aux_unhealthy_logged_at.clear()


def _is_payment_error(exc: Exception) -> bool:
    """Detect payment/credit/quota exhaustion errors.

    Returns True for HTTP 402 (Payment Required) and for 429/other errors
    whose message indicates billing exhaustion or daily quota exhaustion
    rather than transient rate limiting.

    Daily token quota errors (e.g. Bedrock "Too many tokens per day",
    Vertex AI "quota exceeded") are functionally equivalent to credit
    exhaustion — the provider cannot serve the request until the quota
    resets — and should trigger the same provider-fallback logic.
    """
    status = getattr(exc, "status_code", None)
    if status == 402:
        return True
    err_lower = str(exc).lower()
    # OpenRouter and other providers include "credits" or "afford" in 402 bodies,
    # but sometimes wrap them in 429 or other codes.
    # Daily quota exhaustion from Bedrock, Vertex AI, and similar providers
    # uses different language but is semantically identical to credit exhaustion.
    if status in {402, 403, 404, 429, None}:
        if any(kw in err_lower for kw in (
            "credits", "insufficient funds",
            "can only afford", "billing",
            "payment required",
            "out of funds", "run out of funds",
            "balance_depleted", "no usable credits",
            "model_not_supported_on_free_tier",
            "not available on the free tier",
            "requires a subscription", "upgrade for access",
            "upgrade for higher limits", "reached your session usage limit",
            # Daily / monthly / weekly quota exhaustion keywords
            "quota exceeded", "quota_exceeded",
            "too many tokens per day", "daily limit",
            "tokens per day", "daily quota",
            "resource exhausted",  # Vertex AI / gRPC quota errors
            "weekly usage limit", "weekly limit",  # OpenCode Go weekly subscription cap
        )):
            return True
    return False


def _nous_portal_account_has_fresh_paid_access() -> bool:
    """Return True only when the fresh Nous account API says paid access is allowed."""
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        account_info = get_nous_portal_account_info(force_fresh=True)
        return account_info.paid_service_access is True
    except Exception as exc:
        logger.debug("Auxiliary Nous paid-entitlement refresh check failed: %s", exc)
        return False


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect rate-limit errors that warrant provider fallback.

    Returns True for HTTP 429 errors whose message indicates rate limiting
    (as opposed to billing/quota exhaustion, which _is_payment_error handles).
    Also catches OpenAI SDK RateLimitError instances that may not set
    .status_code on the exception object.
    """
    status = getattr(exc, "status_code", None)
    err_lower = str(exc).lower()

    # OpenAI SDK's RateLimitError sometimes omits .status_code —
    # detect by class name so we don't miss these.  (PR #8023 pattern)
    if type(exc).__name__ == "RateLimitError":
        return True

    if status == 429:
        # Distinguish rate-limit from billing: billing keywords are handled
        # by _is_payment_error, everything else on 429 is a rate limit.
        if any(kw in err_lower for kw in (
            "rate limit", "rate_limit", "too many requests",
            "try again", "retry after", "resets in",
        )):
            return True
        # Generic 429 without billing keywords = likely a rate limit
        if not any(kw in err_lower for kw in (
            "credits", "insufficient funds", "billing",
            "payment required", "can only afford",
            "out of funds", "run out of funds",
            "balance_depleted", "no usable credits",
            "model_not_supported_on_free_tier",
            "not available on the free tier",
        )):
            return True
    return False


def _is_timeout_error(exc: Exception) -> bool:
    """Detect a request timeout — the full-budget stall, distinct from a fast
    connection drop.

    A timeout burns the entire configured ``timeout`` before surfacing, so a
    same-provider retry on the critical compression path doubles the
    user-visible wall time (issue #54465). A streaming-close / dropped
    connection, by contrast, fails fast and is cheap to retry — those stay on
    the retry path even for compression.
    """
    try:
        from openai import APITimeoutError
        if isinstance(exc, APITimeoutError):
            return True
    except ImportError:
        pass
    if "Timeout" in type(exc).__name__:
        return True
    return "timed out" in str(exc).lower()


def _is_connection_error(exc: Exception) -> bool:
    """Detect connection/network errors that warrant provider fallback.

    Returns True for errors indicating the provider endpoint is unreachable
    (DNS failure, connection refused, TLS errors, timeouts).  These are
    distinct from API errors (4xx/5xx) which indicate the provider IS
    reachable but returned an error.
    """
    try:
        from openai import APIConnectionError, APITimeoutError
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
    except ImportError:
        pass
    # urllib3 / httpx / httpcore connection errors
    err_type = type(exc).__name__
    if any(kw in err_type for kw in ("Connection", "Timeout", "DNS", "SSL")):
        return True
    err_lower = str(exc).lower()
    if any(kw in err_lower for kw in (
        "connection refused", "name or service not known",
        "no route to host", "network is unreachable",
        "timed out", "connection reset",
        # httpcore / httpx streaming premature-close errors.  These surface
        # when a proxy or provider drops the connection mid-stream and are
        # transient by nature — the request should be retried or rerouted.
        # See issue #18458.
        "incomplete chunked read",
        "peer closed connection",
        "response ended prematurely",
        "unexpected eof",
        "remoteprotocolerror",
        "localprotocolerror",
    )):
        return True
    return False


def _is_transient_transport_error(exc: Exception) -> bool:
    """Return True for a one-off transport blip worth retrying ON the
    same provider before any provider/model fallback.

    Covers connection/streaming-close errors (via the canonical
    ``_is_connection_error`` detector, shared so the two cannot drift) plus a
    pure 5xx/408 HTTP status. Deliberately narrow: this is the "retry the
    same target once" gate, distinct from ``_is_payment_error`` /
    ``_is_auth_error`` / ``_is_rate_limit_error`` which the except-chain
    handles by switching provider, refreshing creds, or rotating the pool.
    """
    if _is_connection_error(exc):
        return True
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    return isinstance(status, int) and (status == 408 or 500 <= status < 600)


_DEFAULT_TRANSIENT_RETRIES = 2
# Base for exponential backoff between transient retries (seconds). Overridable
# so tests can zero it out and not sleep real wall-clock time.
_TRANSIENT_RETRY_BACKOFF_BASE = 1.0


def _transient_retry_count() -> int:
    """Number of same-provider retries for a transient transport blip.

    Read from ``auxiliary.transient_retries`` in config.yaml (default 2 →
    3 total attempts). Clamped to [0, 6] to bound worst-case wall time. A
    connection blip to a pinned auxiliary target (e.g. a MoA reference
    advisor) has no meaningful provider fallback, so a couple of retries with
    backoff is the difference between recovering and silently losing the call.
    Best-effort: any config-read failure falls back to the default.
    """
    try:
        from hermes_cli.config import cfg_get, load_config

        val = cfg_get(load_config(), "auxiliary", "transient_retries")
        if val is None:
            return _DEFAULT_TRANSIENT_RETRIES
        n = int(val)
        return max(0, min(n, 6))
    except Exception:
        return _DEFAULT_TRANSIENT_RETRIES


def _is_auth_error(exc: Exception) -> bool:
    """Detect auth failures that should trigger provider-specific refresh."""
    status = getattr(exc, "status_code", None)
    if status == 401:
        return True
    err_lower = str(exc).lower()
    if "error code: 401" in err_lower or "authenticationerror" in type(exc).__name__.lower():
        return True
    # xAI returns HTTP 403 with "unauthenticated:bad-credentials" when an OAuth2
    # access token has expired or is invalid — semantically a 401 auth failure,
    # even though the status code is 403 (PermissionDenied).
    if status == 403 and "bad-credentials" in err_lower:
        return True
    if "unauthenticated" in err_lower and "bad-credentials" in err_lower:
        return True
    return False


def _is_unsupported_parameter_error(exc: Exception, param: str) -> bool:
    """Detect provider 400s for an unsupported request parameter.

    Different OpenAI-compatible endpoints phrase the same class of error a few
    ways: ``Unsupported parameter: X``, ``unsupported_parameter`` with a
    ``param`` field, ``X is not supported``, ``unknown parameter: X``,
    ``unrecognized request argument: X``.  We match on both the parameter
    name and a generic "unsupported/unknown/unrecognized parameter" marker so
    call sites can reactively retry without the offending key instead of
    surfacing a noisy auxiliary failure.

    Generalizes the temperature-specific detector that originally shipped
    with PR #15621 so the same retry strategy can cover ``max_tokens``,
    ``seed``, ``top_p``, and any future quirk. Credit @nicholasrae (PR #15416)
    for the generalization pattern.
    """
    param_lower = (param or "").lower()
    if not param_lower:
        return False
    err_lower = str(exc).lower()
    if param_lower not in err_lower:
        return False
    return any(marker in err_lower for marker in (
        "unsupported parameter",
        "unsupported_parameter",
        "not supported",
        "does not support",
        "unknown parameter",
        "unrecognized request argument",
        "unrecognized parameter",
        "invalid parameter",
    ))


def _is_unsupported_temperature_error(exc: Exception) -> bool:
    """Back-compat wrapper: detect API errors where the model rejects ``temperature``.

    Delegates to :func:`_is_unsupported_parameter_error`; kept as a separate
    public symbol because existing tests and call sites import it by name.
    """
    return _is_unsupported_parameter_error(exc, "temperature")


def _is_model_not_found_error(exc: Exception) -> bool:
    """Detect "the requested model doesn't exist" errors (404 / invalid model).

    This fires when a resolved model name is no longer served by the endpoint
    — most commonly when a long-lived process pinned a Portal-recommended model
    that has since been dropped from the Nous → OpenRouter catalog. The Nous
    proxy returns 404 with a body like::

        Model 'gpt-5.4-mini' not found. The requested model does not exist
        in our configuration or OpenRouter catalog.

    Distinct from :func:`_is_payment_error` (which also matches some 404s for
    free-tier/credit language) — this one keys on "does not exist / not found /
    not a valid model" phrasing, and explicitly excludes the billing keywords
    that the payment path already owns so the two predicates don't overlap.
    """
    status = getattr(exc, "status_code", None)
    err_lower = str(exc).lower()
    # Billing/quota 404s belong to _is_payment_error — don't claim them here.
    if any(kw in err_lower for kw in (
        "credits", "insufficient funds", "billing", "out of funds",
        "balance_depleted", "no usable credits", "free tier", "free-tier",
        "not available on the free tier",
    )):
        return False
    if status not in {404, 400, None}:
        return False
    return any(kw in err_lower for kw in (
        "model does not exist",
        "does not exist in our configuration",
        "openrouter catalog",
        "is not a valid model",
        "no such model",
        "model not found",
        "the model `",            # OpenAI-style: "The model `X` does not exist"
        "model_not_found",
        "unknown model",
    ))

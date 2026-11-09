"""What the auxiliary recovery ladder does with a failed request, read from ``classify_api_error``'s verdict.

The labels :func:`fallback_label` returns are the ones the logs, unhealthy marks and candidate quarantine
TTLs (agent/auxiliary_health.py) key on.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

from agent.auxiliary_call_outcome import is_cancellation
from agent.error_classifier import (
    POLICY_REFUSAL_REASONS, TRANSIENT_REASONS, ClassifiedError, FailoverReason, classify_api_error,
)

_R = FailoverReason
AUTH_ERROR, PAYMENT_ERROR, MODEL_INCOMPATIBLE = "auth error", "payment error", "model incompatible with route"
MODEL_NOT_FOUND = "model not found"
# The request's own shape failed: another provider only helps once a parameter rung changed the request.
_REQUEST_SHAPE = frozenset({
    _R.format_error, _R.role_alternation, _R.multimodal_tool_content_unsupported, _R.reasoning_mandatory,
    _R.llama_cpp_grammar_pattern, _R.image_too_large, _R.image_corrupt, _R.invalid_encrypted_content,
    _R.thinking_signature, _R.context_overflow, _R.payload_too_large,
    _R.oauth_long_context_beta_forbidden,
})
_LABELS = {
    _R.auth: AUTH_ERROR, _R.auth_permanent: AUTH_ERROR, _R.billing: PAYMENT_ERROR,
    _R.rate_limit: "rate limit", _R.upstream_rate_limit: "rate limit", _R.model_not_found: MODEL_NOT_FOUND,
    _R.model_entitlement: MODEL_INCOMPATIBLE, _R.incomplete_response: "invalid provider response",
    _R.overloaded: "provider overloaded", _R.server_error: "server error", _R.long_context_tier: "long context tier",
}
_PROVIDER_REFUSALS = TRANSIENT_REASONS | {_R.auth, _R.auth_permanent, _R.billing, _R.rate_limit, _R.upstream_rate_limit}


class Failure(NamedTuple):
    """A failed request, its verdict, whether a parameter rung's detector names the rejected field, and the
    client that sent it (whose pool and key a credential verdict is about)."""
    error: Exception
    verdict: ClassifiedError
    rejects_parameter: bool = False
    client: Any = None


def classify_attempt(error: Exception, provider: Optional[str], client: Any = None, model: Any = None,
                     base_url: Optional[str] = "") -> ClassifiedError:
    """``classify_api_error`` on the backend, model, endpoint and credential ``client`` was sent to."""
    slug = str(provider or "").strip().lower()
    import httpx  # late: this module sits on the bootstrap/conftest import chain, which has no httpx

    url, key = getattr(client, "base_url", None), getattr(client, "api_key", None)
    if is_cancellation(error):  # a cancelled task: no retry, no other provider
        return ClassifiedError(_R.unknown, provider=slug, retryable=False, message=str(error),
                               error_context={"interrupted": True})
    return classify_api_error(
        error, provider="" if slug == "auto" else slug, model=model if isinstance(model, str) else "",
        base_url=str(url) if isinstance(url, (str, httpx.URL)) and str(url) else str(base_url or ""),
        api_key=key if isinstance(key, str) else None)


def same_provider_retry(verdict: ClassifiedError) -> bool:
    return verdict.reason in TRANSIENT_REASONS and verdict.retryable


def auth_refresh(verdict: ClassifiedError) -> bool:
    return verdict.reason is _R.auth and verdict.should_rotate_credential


def same_key_retry(verdict: ClassifiedError) -> bool:
    return verdict.reason is _R.rate_limit and verdict.retryable and not verdict.error_context.get("quota_exhausted")


def resend_cannot_help(verdict: ClassifiedError) -> bool:
    """A cancelled task or a policy refusal: no stripped field or stream flag changes the answer."""
    return verdict.reason in POLICY_REFUSAL_REASONS or bool(verdict.error_context.get("interrupted"))


def benches_provider(verdict: ClassifiedError) -> bool:
    return verdict.reason is _R.billing or bool(verdict.error_context.get("quota_exhausted"))


def fallback_label(failure: Failure, *, rung_spent: bool = False) -> Optional[str]:
    """Why another provider should take this request, or None when the failure would follow it anywhere.
    ``rung_spent``: a parameter rung already changed the request and it still failed."""
    verdict = failure.verdict
    reason, context = verdict.reason, verdict.error_context
    structured = "structured provider error" if context.get("statusless_structured") else None
    if context.get("hook_declines_fallback"):
        return None
    if benches_provider(verdict):
        return PAYMENT_ERROR
    if reason is _R.timeout:
        return "connection error" if context.get("transport") == "connection" else "request timed out"
    if reason in _LABELS:
        return _LABELS[reason]
    # Terminal: every provider would get the same request (a safety refusal is of the content itself).
    if reason is _R.content_policy_blocked or not (verdict.retryable or verdict.should_fallback):
        return None
    if failure.rejects_parameter or reason in _REQUEST_SHAPE and not context.get("route_rejects_model"):
        if rung_spent:
            return MODEL_INCOMPATIBLE
        return None if failure.rejects_parameter else structured
    if context.get("route_rejects_model"):
        return MODEL_INCOMPATIBLE
    return reason.value.replace("_", " ") if verdict.should_fallback else structured


def ladder_accepts(failure: Failure, *, rung_spent: bool = False) -> bool:
    """A later rung (credential rotation, provider fallback) acts on this failure; else it re-raises."""
    return failure.verdict.should_rotate_credential or fallback_label(failure, rung_spent=rung_spent) is not None


def stream_negotiation_reraises(verdict: ClassifiedError) -> bool:
    """A failed stream negotiation that is the provider's answer (or a certificate no stream flag repairs):
    re-raised, not re-sent non-streaming."""
    return (verdict.reason in _PROVIDER_REFUSALS or verdict.reason is _R.ssl_cert_verification
            or verdict.status_code == 429 or benches_provider(verdict) or "welcome_refusal" in verdict.error_context
            or resend_cannot_help(verdict))

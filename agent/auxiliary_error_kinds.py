"""What kind of provider failure is this? — the auxiliary client's error predicates.

Split out of ``auxiliary_client.py`` (facade + siblings, see AGENTS.md): these predicates are pure
`Exception -> bool` classifiers plus the phrase tables they read, and every recovery decision in the
facade (retry the same provider, rotate the pool, refresh credentials, drop a request field, walk to
the next fallback) is keyed on one of them.

The phrase tables come from ``agent/error_classifier`` — the main loop's classifier is the single
source for billing/overload markers (#107166); this module only adds the aux-specific phrasings.
"""

import contextlib
from typing import Any, Optional, Tuple

from agent.error_classifier import (
    _BILLING_PATTERNS,
    _OVERLOADED_PATTERNS,
    UNSUPPORTED_PARAM_MARKERS,
    is_reasoning_field_rejection,
    is_reasoning_required_rejection,
)

def _contains_any(text: str, needles: Tuple[str, ...]) -> bool:
    """True when any needle is a substring of ``text``."""
    return any(kw in text for kw in needles)

# Billing-body markers (credit exhaustion wrapped in 402/403/404/429 bodies). The main classifier's
# ``_BILLING_PATTERNS`` is the single source (#107166: the two lists had drifted, so OpenRouter's org
# "Budget limit exceeded" / "Key limit exceeded" 403s were billing for the main loop but not for the aux
# ladder, and aux fallback never fired); the aux list only adds broader phrasings and daily/weekly quota
# exhaustion (functionally credit exhaustion; "resource exhausted" is the Vertex/gRPC quota phrasing —
# also serialized by SDK wrappers and NIM as RESOURCE_EXHAUSTED / ResourceExhausted / resource-exhausted).
_PAYMENT_KEYWORDS = _BILLING_PATTERNS + (
    "credits", "insufficient funds", "can only afford", "billing",
    "isn't available on the free tier", "key limit exceeded", "budget limit",
    "requires a subscription", "upgrade for access", "upgrade for higher limits",
    "reached your session usage limit", "quota exceeded", "quota_exceeded",
    "too many tokens per day", "daily limit", "tokens per day", "daily quota", "resource exhausted",
    "resource_exhausted", "resource-exhausted", "resourceexhausted",
    "weekly usage limit", "weekly limit",
)

def _is_payment_error(exc: Exception) -> bool:
    """Payment/credit/quota exhaustion: HTTP 402, or a billing/quota body on 403/404/429/no-status."""
    status = getattr(exc, "status_code", None)
    return status == 402 or (
        status in {403, 404, 429, None} and _contains_any(str(exc).lower(), _PAYMENT_KEYWORDS)
    )

_RATE_LIMIT_KEYWORDS = (
    "rate limit", "rate_limit", "too many requests", "try again", "retry after", "resets in"
)

_RATE_LIMIT_BILLING_KEYWORDS = (
    "credits", "insufficient funds", "billing", "payment required", "can only afford",
    "out of funds", "run out of funds", "balance_depleted", "no usable credits",
    "model_not_supported_on_free_tier", "not available on the free tier", "isn't available on the free tier",
)

def _is_overloaded_error(exc: Exception) -> bool:
    """Server busy, credential fine ("at capacity upstream … not your API key's rate limit"): back off,
    never bench the credential (#108349). Same phrase table as the main classifier."""
    return _contains_any(str(exc).lower(), _OVERLOADED_PATTERNS)

def _is_rate_limit_error(exc: Exception) -> bool:
    """429 rate limit (not billing/quota, which _is_payment_error owns).

    OpenAI's RateLimitError may omit .status_code — matched by class name. A generic 429 without
    billing keywords counts as a rate limit.
    """
    # (PR #8023 pattern)
    if type(exc).__name__ == "RateLimitError":
        return True
    if getattr(exc, "status_code", None) != 429:
        return False
    err_lower = str(exc).lower()
    return _contains_any(err_lower, _RATE_LIMIT_KEYWORDS) or not _contains_any(err_lower, _RATE_LIMIT_BILLING_KEYWORDS)

def _is_timeout_error(exc: Exception) -> bool:
    """Full-budget request timeout, distinct from a fast connection drop.

    A timeout burns the whole ``timeout`` budget, so a same-provider retry on the compression
    path doubles wall time; fast drops stay on the retry path.
    """
    with contextlib.suppress(ImportError):
        from openai import APITimeoutError
        if isinstance(exc, APITimeoutError):
            return True
    return "Timeout" in type(exc).__name__ or "timed out" in str(exc).lower()

def _is_connection_error(exc: Exception) -> bool:
    """Connection/network errors (endpoint unreachable), as opposed to 4xx/5xx API errors."""
    with contextlib.suppress(ImportError):
        from openai import APIConnectionError, APITimeoutError
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
    if _contains_any(type(exc).__name__, ("Connection", "Timeout", "DNS", "SSL")):
        return True
    return _contains_any(str(exc).lower(), (
        "connection refused", "name or service not known", "no route to host",
        "network is unreachable", "timed out", "connection reset",
        # httpcore/httpx premature stream close — transient, retry/reroute.
        "incomplete chunked read", "peer closed connection", "response ended prematurely",
        "unexpected eof", "remoteprotocolerror", "localprotocolerror",
    ))

def _is_transient_transport_error(exc: Exception) -> bool:
    """One-off transport blip worth retrying on the SAME provider: connection/stream-close errors plus pure 5xx/408.

    Deliberately narrow: payment/auth/rate-limit errors switch provider, refresh creds, or rotate the pool.
    """
    if _is_connection_error(exc):
        return True
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return isinstance(status, int) and (status == 408 or 500 <= status < 600)

_DEFAULT_TRANSIENT_RETRIES = 2

def _transient_retry_count() -> int:
    """Same-provider retries for a transient blip: ``auxiliary.transient_retries``
    (default 2), clamped to [0, 6]; config-read failures fall back to default."""
    try:
        from hermes_cli.config import cfg_get, load_config
        val = cfg_get(load_config(), "auxiliary", "transient_retries")
        return _DEFAULT_TRANSIENT_RETRIES if val is None else max(0, min(int(val), 6))
    except Exception:
        return _DEFAULT_TRANSIENT_RETRIES

def _is_auth_error(exc: Exception) -> bool:
    """Auth failures that should trigger provider-specific refresh."""
    status = getattr(exc, "status_code", None)
    if status == 401:
        return True
    err_lower = str(exc).lower()
    if "error code: 401" in err_lower or "authenticationerror" in type(exc).__name__.lower():
        return True
    # xAI returns 403 "unauthenticated:bad-credentials" for expired OAuth tokens — semantically a 401.
    return "bad-credentials" in err_lower and (status == 403 or "unauthenticated" in err_lower)

def _is_unsupported_parameter_error(exc: Exception, param: str) -> bool:
    """Provider 400 for an unsupported request parameter: the parameter name plus a generic
    unsupported/unknown/unrecognized marker, so call sites can retry without the key."""
    param_lower = (param or "").lower()
    if not param_lower:
        return False
    err_lower = str(exc).lower()
    return param_lower in err_lower and _contains_any(err_lower, UNSUPPORTED_PARAM_MARKERS)

def _is_structured_output_rejection(exc: Exception) -> bool:
    """Provider 400/422 rejecting the structured-output field, on either wire: OpenAI ``response_format``
    (incl. vLLM's ``guided_grammar``/xgrammar failures) or Anthropic ``output_config.format`` ("Extra inputs
    are not permitted"). Callers tolerate an unconstrained reply, so the reaction is one retry without it."""
    status = getattr(exc, "status_code", None)
    if status is not None and status not in {400, 422}:
        return False
    err_lower = str(exc).lower()
    # vLLM grammar-backend failures name the translated parameter, not ours.
    if _contains_any(err_lower, ("guided_grammar", "xgrammar", "compile_grammar_error")):
        return True
    if "extra inputs are not permitted" in err_lower and (
        "response_format" in err_lower or "output_config" in err_lower
    ):
        return True
    if "response_format" in err_lower and "unavailable" in err_lower:
        return True
    # Gateways that validate the request body with a strict pydantic model reject the
    # OBJECT-form json_schema by shape ("str type expected" on response_format.json_schema,
    # 422) rather than by naming the feature. The field is what they refuse; the retry
    # without it is the same remedy, so treat the shape error as a rejection too.
    if "response_format" in err_lower and "json_schema" in err_lower:
        return True
    return _is_unsupported_parameter_error(exc, "response_format") or _is_unsupported_parameter_error(exc, "output_config")

def _without_structured_output_format(kwargs: dict) -> Optional[dict]:
    """Copy *kwargs* without ``response_format`` (top-level and ``extra_body``); None when nothing was
    removed, so call sites don't retry an unchanged request."""
    retry_kwargs = dict(kwargs)
    changed = retry_kwargs.pop("response_format", None) is not None
    extra_body = retry_kwargs.get("extra_body")
    if isinstance(extra_body, dict) and "response_format" in extra_body:
        remaining = {k: v for k, v in extra_body.items() if k != "response_format"}
        if remaining:
            retry_kwargs["extra_body"] = remaining
        else:
            retry_kwargs.pop("extra_body", None)
        changed = True
    return retry_kwargs if changed else None

def _is_reasoning_field_rejection(exc: Exception) -> bool:
    """Provider 400 rejecting a reasoning wire control by name (``reasoning_effort``, ``reasoning``,
    ``thinking``/``think``). Chat-only models behind OpenAI-compatible relays reject the top-level
    ``reasoning_effort: none`` a disabled ``reasoning_config`` projects on the custom profile
    ("Unrecognized request argument supplied: reasoning_effort", #112781); the route default is
    the right answer for such a model, so the reaction is one retry without any reasoning field."""
    status = getattr(exc, "status_code", None)
    if status is not None and status not in {400, 422}:
        return False
    return is_reasoning_field_rejection(str(exc))

def _is_reasoning_required_rejection(exc: Exception) -> bool:
    """Provider 400 refusing to switch reasoning OFF ("Reasoning is mandatory for this endpoint and cannot
    be disabled"): the field is understood, only the disable is refused, so the rung steps the effort up to
    the floor instead of dropping the field (agent/auxiliary_reasoning_floor.py)."""
    status = getattr(exc, "status_code", None)
    if status is not None and status not in {400, 422}:
        return False
    return is_reasoning_required_rejection(str(exc))

def _without_reasoning_fields(kwargs: dict) -> Optional[dict]:
    """Copy *kwargs* without reasoning wire controls (top-level ``reasoning_effort``, the adapter's
    private ``_reasoning_config`` and every ``extra_body`` reasoning key); None when nothing was
    removed, so call sites don't retry an unchanged request."""
    retry_kwargs = dict(kwargs)
    changed = retry_kwargs.pop("reasoning_effort", None) is not None
    changed = retry_kwargs.pop("_reasoning_config", None) is not None or changed
    extra_body = retry_kwargs.get("extra_body")
    if isinstance(extra_body, dict):
        remaining = {k: v for k, v in extra_body.items() if str(k).strip().lower() not in _PROFILE_REASONING_KEYS}
        if len(remaining) != len(extra_body):
            if remaining:
                retry_kwargs["extra_body"] = remaining
            else:
                retry_kwargs.pop("extra_body", None)
            changed = True
    return retry_kwargs if changed else None

def _is_model_not_found_error(exc: Exception) -> bool:
    """"Requested model doesn't exist" (404 / invalid model) — typically a long-lived process pinned a
    since-dropped model. Excludes billing keywords, which :func:`_is_payment_error` owns."""
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

_PROFILE_REASONING_KEYS = {
    "reasoning", "reasoning_effort", "thinking", "thinking_config", "thinkingconfig",
    "thinking_budget", "thinkingbudget", "enable_thinking", "think", "verbosity",
}

def _is_max_tokens_rejection(exc: Exception, client: Any) -> bool:
    err_str = str(exc)
    # ZAI vision models reject max_tokens with code 1210 and a message that never
    # mentions "max_tokens", so detect it explicitly.
    is_zai_param_error = "1210" in err_str and "bigmodel" in str(getattr(client, "base_url", ""))
    return ("max_tokens" in err_str or "unsupported_parameter" in err_str
            or _is_unsupported_parameter_error(exc, "max_tokens") or is_zai_param_error)

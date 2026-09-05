"""API error classification for smart failover and recovery.

A priority-ordered pipeline maps an API exception to a ``ClassifiedError``
whose recovery hints (retry, rotate credential, fallback, compress, abort) the
retry loop in run_agent.py consults instead of re-matching strings itself.
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional, Sequence

logger = logging.getLogger(__name__)

# Synthetic code for the OpenAI SDK rejecting a provider's SSE ``data:`` field
# before any completion chunk arrives; distinct from generic JSON parse errors.
PROVIDER_STREAM_NON_JSON_ERROR_CODE = "provider_stream_non_json_data"


# ── Error taxonomy ──────────────────────────────────────────────────────

class FailoverReason(enum.Enum):
    """Why an API call failed — determines recovery strategy."""
    auth = "auth"                        # Transient auth (401/403) — refresh/rotate
    auth_permanent = "auth_permanent"    # Auth failed after refresh — abort
    billing = "billing"                  # 402 or confirmed credit exhaustion — rotate immediately
    rate_limit = "rate_limit"            # 429 or quota-based throttling — backoff then rotate
    upstream_rate_limit = "upstream_rate_limit"  # Aggregator's upstream model 429 — fallback model, key is healthy
    overloaded = "overloaded"            # 503/529 — provider overloaded, backoff
    server_error = "server_error"        # 500/502 — internal server error, retry
    timeout = "timeout"                  # Connection/read timeout — rebuild client + retry
    ssl_cert_verification = "ssl_cert_verification"  # Deterministic TLS chain failure — fail fast with guidance
    context_overflow = "context_overflow"  # Context too large — compress, not failover
    payload_too_large = "payload_too_large"  # 413 — compress payload
    image_too_large = "image_too_large"   # Native image part exceeds provider's per-image limit — shrink and retry
    image_corrupt = "image_corrupt"       # Provider can't decode image bytes — strip and retry (shrinking won't help)
    model_not_found = "model_not_found"  # 404 or invalid model — fallback to different model
    provider_policy_blocked = "provider_policy_blocked"  # Aggregator account data/privacy policy excluded the only endpoint
    content_policy_blocked = "content_policy_blocked"  # Provider safety filter rejected this prompt — don't retry unchanged
    format_error = "format_error"        # 400 bad request — abort or strip + retry
    invalid_encrypted_content = "invalid_encrypted_content"  # Responses replay blob rejected — strip replay state and retry
    multimodal_tool_content_unsupported = "multimodal_tool_content_unsupported"  # Provider rejected list-type content in tool messages (e.g. Xiaomi MiMo) — downgrade to text and retry
    reasoning_mandatory = "reasoning_mandatory"  # Route rejects reasoning: {enabled: false} — send the disable no more this session and retry

    # Provider-specific
    thinking_signature = "thinking_signature"  # Anthropic thinking block sig invalid
    long_context_tier = "long_context_tier"    # Anthropic "extra usage" tier gate
    oauth_long_context_beta_forbidden = "oauth_long_context_beta_forbidden"  # Anthropic OAuth rejects 1M beta — disable beta and retry
    llama_cpp_grammar_pattern = "llama_cpp_grammar_pattern"  # llama.cpp grammar rejects regex `pattern`/`format` — strip from tools and retry
    unknown = "unknown"                  # Unclassifiable — retry with backoff


@dataclass
class ClassifiedError:
    """Structured classification of an API error with recovery hints."""

    reason: FailoverReason
    status_code: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    message: str = ""
    error_context: Dict[str, Any] = field(default_factory=dict)

    # Recovery hints — the retry loop checks these instead of re-classifying.
    retryable: bool = True
    should_compress: bool = False
    should_rotate_credential: bool = False
    should_fallback: bool = False

    @property
    def is_auth(self) -> bool:
        return self.reason in {FailoverReason.auth, FailoverReason.auth_permanent}

    @property
    def billing_unverified(self) -> bool:
        """True when a ``billing`` verdict rests on an ambiguous body (#82154)."""
        return bool(self.error_context.get("billing_unverified"))


# ── Pattern tables (lowercased substrings) ──────────────────────────────

# Billing exhaustion (not transient rate limit). "out of extra usage" is the
# Anthropic OAuth Pro/Max overage bucket depleted (HTTP 400).
_BILLING_PATTERNS = (
    "insufficient credits", "insufficient_quota", "insufficient balance", "credit balance",
    "credits exhausted", "credits have been exhausted", "requires available credits",
    "account balance is too low", "no usable credits", "top up your credits", "payment required",
    "billing hard limit", "exceeded your current quota", "account is deactivated", "plan does not include",
    "out of extra usage", "out of funds", "run out of funds", "balance_depleted",
    "model_not_supported_on_free_tier", "not available on the free tier",
)

# Not proof of exhaustion: Anthropic returns the same "out of extra usage" body
# for a content-filter rejection (#82154). Verdict stays ``billing`` but is
# marked unverified so surfaces hedge and the pool uses a short cooldown.
_UNVERIFIED_BILLING_PATTERNS = ("out of extra usage",)

# xAI's Grok credit-exhaustion code arrives as HTTP 403, not 402. Provider-
# scoped on purpose: other providers' billing codes on a 403 stay auth failures.
_XAI_SPENDING_LIMIT_ERROR_CODE = "personal-team-blocked:spending-limit"

# Structured codes meaning the account cannot serve paid traffic.
_BILLING_ERROR_CODES = frozenset({
    "insufficient_quota", "billing_not_active", "payment_required", "insufficient_credits",
    "no_usable_credits", "balance_depleted", "model_not_supported_on_free_tier",
    "member_spend_cap_exceeded", _XAI_SPENDING_LIMIT_ERROR_CODE,
})

# Transient rate limiting. Bedrock "Throttling error: Too many tokens" also
# contains an overflow phrase; rate limit is matched first so throttle wins.
_RATE_LIMIT_PATTERNS = (
    "rate limit", "rate_limit", "too many requests", "throttled", "requests per minute",
    "tokens per minute", "requests per day", "try again in", "please retry after",
    "resource exhausted", "resource_exhausted", "resource-exhausted", "resourceexhausted",
    "rate increased too quickly", "throttlingexception", "too many concurrent requests",
    "servicequotaexceededexception", "throttling",
)

# Server busy, credential fine: back off on the same key, never rotate. Z.AI/
# Zhipu reuse HTTP 429 for this, so the 429 path checks these first. Kept narrow
# so a plain "you have been rate-limited" doesn't land here. (#14038, #15297)
_OVERLOADED_PATTERNS = (
    "overloaded", "temporarily overloaded", "service is temporarily overloaded",
    "service may be temporarily overloaded", "server is overloaded", "server overloaded",
    "service overloaded", "service is overloaded", "upstream overloaded", "currently overloaded",
    "at capacity", "over capacity",
)

# Usage-limit patterns that need disambiguation (billing OR rate_limit), and
# the signals that mark such a limit as transient (periodic quota, not billing).
_USAGE_LIMIT_PATTERNS = ("usage limit", "quota", "limit exceeded", "key limit exceeded")
_USAGE_LIMIT_TRANSIENT_SIGNALS = (
    "try again", "retry", "resets at", "reset in", "resets in", "reset after", "available in",
    "wait", "requests remaining", "periodic", "window", "per minute", "per second",
)

# 413 detected from message text (proxies embed the status or re-wrap
# Anthropic's "request_too_large" type without one).
_PAYLOAD_TOO_LARGE_PATTERNS = (
    "request entity too large", "payload too large", "error code: 413", "request_too_large",
    # Normally arrives with an HTTP 413 status (handled by the status path), but aggregators/proxies can
    # re-wrap it into a plain message with no status attribute — route it to the same compression recovery.
    # (port of anomalyco/opencode#37848)
    "request exceeds the maximum size",
)

# Per-image size/dimension 400s (Anthropic 5 MB / 8000 px; MiniMax "media
# exceeds size limit" #76039) — a specific 400 before the request hits 413. A
# non-image media hit is harmless: the shrink pass finds no image parts.
_IMAGE_TOO_LARGE_PATTERNS = (
    "image exceeds", "image too large", "image_too_large", "image size exceeds", "image dimensions exceed",
    "dimensions exceed max allowed size", "max allowed size: 8000", "media exceeds", "media too large",
)

# Undecodable image bytes → strip-and-retry, never shrink. xAI wordings
# (#69078); the last is the full sentence because shorter fragments also match
# non-image download failures.
_IMAGE_CORRUPT_PATTERNS = (
    "invalid png image", "invalid jpeg image", "base64 string of provided image cannot be decoded",
    "downloaded response does not contain a valid jpg, png, webp, or ico image",
)

# 400s rejecting list-type ``content`` in tool messages (Xiaomi MiMo "text is
# not set", Alibaba, OpenAI-compat long tail). Recovery: strip image parts from
# tool messages, remember (provider, model), retry. (#27344)
_MULTIMODAL_TOOL_CONTENT_PATTERNS = (
    "text is not set", "tool message content must be a string", "tool content must be a string",
    "tool message must be a string", "expected string, got list", "expected string, got array",
    "tool_call.content must be string",
)

# Bare "max_tokens" is load-bearing: the output-cap-retry path keys off it;
# empty-response advisories mentioning it are intercepted earlier. Groups:
# generic; vLLM; Ollama; llama.cpp; Chinese; Z.AI (1210); Bedrock; Together.
_CONTEXT_OVERFLOW_PATTERNS = (
    "context length", "context size", "maximum context", "token limit", "too many tokens",
    "reduce the length", "exceeds the limit", "context window", "prompt is too long",
    "prompt exceeds max length", "max_tokens", "maximum number of tokens",
    "exceeds the max_model_len", "max_model_len", "prompt length", "input is too long", "maximum model length",
    "context length exceeded", "truncating input",
    "slot context", "n_ctx_slot",
    "超过最大长度", "上下文长度",
    "tokens in request more than max tokens allowed",
    "input is too long", "max input token", "input token", "exceeds the maximum number of input tokens",
    # Together/Fireworks-style: "Input length 131393 exceeds the maximum allowed input length of 131040
    # tokens."  No other pattern in this list matches that wording. (port of anomalyco/opencode#37848)
    "maximum allowed input length",
)

# Last entry: OpenRouter 404 when no endpoint supports tool calling —
# model_not_found triggers fallback instead of burning retries (#58446).
_MODEL_NOT_FOUND_PATTERNS = (
    "is not a valid model", "invalid model", "model not found", "model_not_found", "does not exist",
    "no such model", "unknown model", "unsupported model", "no endpoints found that support tool use",
)

# Qwen/vLLM chat-template "No user query found". Shared by the invalid-body
# table (→ format_error) and the llama.cpp grammar guard so they cannot drift.
_NO_USER_QUERY_SIGNAL = "no user query found"

# Deterministic rejections of the *transcript* (e.g. a content-less assistant
# stub after a dead stream). NOT overflow — input may be tiny and compression
# cannot invent a missing turn — so fail fast as format_error.
_INVALID_MESSAGE_BODY_PATTERNS = (
    "must have non-empty content", "messages must have non-empty", "invalid_request_body",
    "text content blocks must be non-empty", "content field is required",
    "messages: at least one message is required", _NO_USER_QUERY_SIGNAL,
)

# Malformed request, identical on every retry. Some gateways (codex.nekos.me)
# return these as 5xx, so the 5xx path also checks them.
_REQUEST_VALIDATION_PATTERNS = (
    "unknown parameter", "unsupported parameter", "unrecognized request argument",
    "invalid_request_error", "unknown_parameter", "unsupported_parameter",
)

# Parameters Hermes sends on SOME routes only → hosts where that is deliberate.
# A rejection from any other host means the provider's gateway injected the
# field itself: a server-side flake, not our request shape. prompt_cache_retention
# is only sent for api.meta.ai / bedrock-mantle (agent/transports/codex.py).
_SERVER_INJECTED_PARAM_SENDERS: Dict[str, tuple] = {
    "prompt_cache_retention": ("meta", "muse", "msl", "model-api", "bedrock", "mantle"),
}
_PARAM_REJECTION_WORDS = ("not supported", "unsupported", "unknown", "unrecognized")

# Anthropic thinking-block 400 wordings (see _provider_special_cases).
_THINKING_MUTATION_WORDS = ("signature", "cannot be modified", "must remain as they were")

# Local MoA streaming adapter-shape bugs (see _moa_special_cases).
_MOA_ADAPTER_SHAPE_BUGS = (
    "'types.SimpleNamespace' object is not iterable", "'types.SimpleNamespace' object has no attribute 'index'",
)

# OpenRouter 404 when the account data policy excludes the only endpoint. Not
# model_not_found: the model exists, fallback can't help, body has the fix URL.
_PROVIDER_POLICY_BLOCKED_PATTERNS = (
    "no endpoints available matching your guardrail", "no endpoints available matching your data policy",
    "no endpoints found matching your data policy",
)

# Per-prompt safety-filter blocks: deterministic for the unchanged request, so
# fallback immediately. Each phrase is verbatim from one provider (Codex cyber
# flags #18028, OpenAI moderation, Anthropic safety, Azure token, MiniMax
# #32421) — never a generic word like "policy" that collides with billing/auth.
# "content_filter" deliberately excludes the space variant seen in echoed config.
_CONTENT_POLICY_BLOCKED_PATTERNS = (
    "flagged for possible cybersecurity risk", "trusted access for cyber",
    "violates our usage policies", "violates openai's usage policies", "your request was flagged by",
    "prompt was flagged by our safety", "responses cannot be generated due to safety",
    "content_filter", "responsibleaipolicyviolation", "new_sensitive",
)

# Auth patterns (non-status-code signals).
_AUTH_PATTERNS = (
    "invalid api key", "invalid_api_key", "gateway_auth_failed", "authentication", "unauthorized",
    "forbidden", "invalid token", "token expired", "token revoked", "access denied",
)

# Empty-response advisories (OpenRouter / nano-gpt). Checked before overflow
# because the text often mentions "max_tokens" (caused compression spirals).
_EMPTY_PROVIDER_RESPONSE_PATTERNS = (
    "returned an empty response", "empty response despite retries", "provider returned an empty response",
    "model returning empty responses", "empty response stream",
)

# Timeout wording from generic exception types the type heuristics would miss.
_TIMEOUT_MESSAGE_PATTERNS = (
    "timed out", "turn timed out", "request timed out", "deadline exceeded", "operation timed out",
    "upstream timed out",
)

# Connect/DNS failures from generic exception types with no status. EXCLUDES
# mid-stream disconnects (_SERVER_DISCONNECT_PATTERNS may route large sessions
# to compression; a never-established connection cannot be an overflow).
# Groups: TCP connect; DNS (Python/glibc/macOS/Node); undici bridge; Envoy.
_CONNECTION_MESSAGE_PATTERNS = (
    "connection refused", "econnrefused", "no route to host", "network is unreachable", "network unreachable",
    "name or service not known", "temporary failure in name resolution", "nodename nor servname provided",
    "getaddrinfo failed", "getaddrinfo enotfound", "eai_again",
    "fetch failed", "failed to fetch",
    "upstream connect error",
)

# SSL names keep provider-wrapped SSL errors (chain lost) as transport, not
# unknown; OpenAI SDK errors are not subclasses of Python builtins.
_TRANSPORT_ERROR_TYPES = frozenset({
    "ReadTimeout", "ConnectTimeout", "PoolTimeout", "ConnectError", "RemoteProtocolError",
    "ConnectionError", "ConnectionResetError", "ConnectionAbortedError", "BrokenPipeError",
    "TimeoutError", "ReadError", "ServerDisconnectedError",
    "SSLError", "SSLZeroReturnError", "SSLWantReadError", "SSLWantWriteError", "SSLEOFError", "SSLSyscallError",
    "APIConnectionError", "APITimeoutError",
})

# Ambiguous disconnects (no status): transient hiccup OR a gateway dropping an
# oversized request. A large session + one of these → context-overflow path.
_SERVER_DISCONNECT_PATTERNS = (
    "server disconnected", "peer closed connection", "connection reset by peer", "connection was closed",
    "network connection lost", "unexpected eof", "incomplete chunked read",
)

# Deterministic cert failures (proxy, missing CA, expired/self-signed) — fail
# fast. Checked BEFORE _SSL_TRANSIENT_PATTERNS: these also contain "[SSL:".
_SSL_CERT_VERIFY_PATTERNS = (
    "certificate verify failed", "certificate_verify_failed", "unable to get local issuer certificate",
    "self-signed certificate", "self signed certificate", "certificate has expired",
    "hostname mismatch, certificate is not valid", "unable to verify the first certificate",
)

# Transient SSL alerts: retry but NOT compression (kept apart from disconnects).
# Both space and underscore forms because OpenSSL 3 changed token separators
# (SSLV3_ALERT_... → SSL/TLS_ALERT_...); "[ssl:" is the Python ssl prefix.
_SSL_TRANSIENT_PATTERNS = (
    "bad record mac", "ssl alert", "tls alert", "ssl handshake failure", "tlsv1 alert", "sslv3 alert",
    "bad_record_mac", "ssl_alert", "tls_alert", "tls_alert_internal_error", "[ssl:",
)


# ── Verdicts and rule tables ────────────────────────────────────────────
# A verdict is the ClassifiedError kwargs a stage decided on: ``reason`` plus
# hint overrides (unlisted hints keep dataclass defaults). Rule tables are
# ordered ``(patterns, verdict)`` pairs matched first-hit; ``verdict`` may be
# a callable of the error message.

Verdict = Dict[str, Any]


def _v(reason: FailoverReason, **hints: Any) -> Verdict:
    return {"reason": reason, **hints}


_ROTATE_FALLBACK = {"should_rotate_credential": True, "should_fallback": True}
_ABORT_FALLBACK = {"retryable": False, "should_fallback": True}
_R = FailoverReason

_V_BILLING = _v(_R.billing, retryable=False, **_ROTATE_FALLBACK)
_V_RATE_LIMIT = _v(_R.rate_limit, **_ROTATE_FALLBACK)
_V_AUTH_ROTATE = _v(_R.auth, retryable=False, **_ROTATE_FALLBACK)
_V_AUTH_FALLBACK = _v(_R.auth, **_ABORT_FALLBACK)
_V_MODEL_NOT_FOUND = _v(_R.model_not_found, **_ABORT_FALLBACK)
_V_CONTENT_BLOCKED = _v(_R.content_policy_blocked, **_ABORT_FALLBACK)
_V_FORMAT_ERROR = _v(_R.format_error, **_ABORT_FALLBACK)
_V_POLICY_BLOCKED = _v(_R.provider_policy_blocked, retryable=False)
_V_SSL_CERT = _v(_R.ssl_cert_verification, retryable=False)
_V_CONTEXT_OVERFLOW = _v(_R.context_overflow, should_compress=True)
_V_PAYLOAD_TOO_LARGE = _v(_R.payload_too_large, should_compress=True)
_V_OVERLOADED, _V_SERVER_ERROR, _V_TIMEOUT, _V_UNKNOWN = map(_v, (_R.overloaded, _R.server_error, _R.timeout, _R.unknown))
_V_IMAGE_TOO_LARGE, _V_IMAGE_CORRUPT = _v(_R.image_too_large), _v(_R.image_corrupt)
_V_MULTIMODAL, _V_INVALID_ENCRYPTED = _v(_R.multimodal_tool_content_unsupported), _v(_R.invalid_encrypted_content)
_V_REASONING_MANDATORY = _v(_R.reasoning_mandatory, should_compress=False, should_fallback=False)
# A reasoning-mandatory route answering ``reasoning: {enabled: false}`` (Nous Portal + OpenRouter wording).
_REASONING_MANDATORY_PATTERN = "reasoning is mandatory"


def _billing_hints(error_msg: str) -> Verdict:
    """Billing verdict carrying the #82154 ambiguity marker when applicable."""
    ctx: Dict[str, Any] = {}
    if any(p in error_msg for p in _UNVERIFIED_BILLING_PATTERNS):
        ctx = {"billing_unverified": True, "possible_content_filter": True}
    return {**_V_BILLING, "error_context": ctx}


def _first_match(error_msg: str, rules: Sequence[tuple[Sequence[str], Any]]) -> Optional[Verdict]:
    """Verdict of the first rule whose pattern list hits ``error_msg``."""
    for patterns, verdict in rules:
        if any(p in error_msg for p in patterns):
            return verdict(error_msg) if callable(verdict) else verdict
    return None


# Image/tool-content 400s, ordered: multimodal recovery ≠ image shrink; corrupt
# bytes need strip not shrink; image-shrink is cheaper than context compression.
_IMAGE_TOOL_RULES = (
    (_MULTIMODAL_TOOL_CONTENT_PATTERNS, _V_MULTIMODAL), (_IMAGE_CORRUPT_PATTERNS, _V_IMAGE_CORRUPT),
    (_IMAGE_TOO_LARGE_PATTERNS, _V_IMAGE_TOO_LARGE),
)

# Overflow signals arriving as 5xx (llama.cpp reports overflow as 500; busy /
# model-load OOM as 503). Empty-response advisories must not enter compression.
_OVERFLOW_AS_5XX_RULES = (
    (_EMPTY_PROVIDER_RESPONSE_PATTERNS, _V_SERVER_ERROR), (_CONTEXT_OVERFLOW_PATTERNS, _V_CONTEXT_OVERFLOW),
)

# 404: Nous API surfaces credit depletion as a paid model vanishing from the
# Free Tier (billing, not missing model); policy block before model_not_found.
_404_RULES = (
    (_BILLING_PATTERNS, _V_BILLING), (_PROVIDER_POLICY_BLOCKED_PATTERNS, _V_POLICY_BLOCKED),
    (_MODEL_NOT_FOUND_PATTERNS, _V_MODEL_NOT_FOUND),
)

# 400 tail after the deterministic request-shape checks. Some providers return
# model-not-found / rate-limit / billing as 400 instead of 404/429/402.
_400_TAIL_RULES = _OVERFLOW_AS_5XX_RULES + (
    (_PROVIDER_POLICY_BLOCKED_PATTERNS, _V_POLICY_BLOCKED), (_MODEL_NOT_FOUND_PATTERNS, _V_MODEL_NOT_FOUND),
    (_RATE_LIMIT_PATTERNS, _V_RATE_LIMIT), (_BILLING_PATTERNS, _billing_hints),
)

# Status-less message path, head (before usage-limit disambiguation).
_MESSAGE_HEAD_RULES = ((_PAYLOAD_TOO_LARGE_PATTERNS, _V_PAYLOAD_TOO_LARGE),) + _IMAGE_TOOL_RULES

# Status-less tail. Overload before rate_limit/billing so "overloaded" backs off
# instead of rotating; policy block before model_not_found; timeout/connection
# wording last, classified as transport (never compression).
_MESSAGE_TAIL_RULES = (
    (_OVERLOADED_PATTERNS, _V_OVERLOADED), (_BILLING_PATTERNS, _billing_hints),
    (_RATE_LIMIT_PATTERNS, _V_RATE_LIMIT), (_EMPTY_PROVIDER_RESPONSE_PATTERNS, _V_SERVER_ERROR),
    (_CONTEXT_OVERFLOW_PATTERNS, _V_CONTEXT_OVERFLOW), (_AUTH_PATTERNS, _V_AUTH_ROTATE),
    (_PROVIDER_POLICY_BLOCKED_PATTERNS, _V_POLICY_BLOCKED), (_MODEL_NOT_FOUND_PATTERNS, _V_MODEL_NOT_FOUND),
    (_TIMEOUT_MESSAGE_PATTERNS, _V_TIMEOUT), (_CONNECTION_MESSAGE_PATTERNS, _V_TIMEOUT),
)

# Structured error code → verdict. The error-code rate_limit verdict rotates
# but does not set should_fallback (unlike the message/status paths).
_ERROR_CODE_VERDICTS: Dict[str, Verdict] = {
    **dict.fromkeys(("resource_exhausted", "throttled", "rate_limit_exceeded"),
                    _v(_R.rate_limit, should_rotate_credential=True)),
    **dict.fromkeys(_BILLING_ERROR_CODES, _V_BILLING),
    **dict.fromkeys(("model_not_found", "model_not_available", "invalid_model"), _V_MODEL_NOT_FOUND),
    **dict.fromkeys(("context_length_exceeded", "max_tokens_exceeded"), _V_CONTEXT_OVERFLOW),
    "invalid_encrypted_content": _V_INVALID_ENCRYPTED,
}

# Generic ``invalid_request_error`` is deliberately NOT a 400 validation
# signal — OpenAI stamps it on genuine overflow 400s too.
_400_VALIDATION_CODES = {"unknown_parameter", "unsupported_parameter"}
_5XX_VALIDATION_CODES = _400_VALIDATION_CODES | {"invalid_request_error"}
_400_VALIDATION_PATTERNS = tuple(p for p in _REQUEST_VALIDATION_PATTERNS if p != "invalid_request_error")


# ── Classification pipeline ─────────────────────────────────────────────

@dataclass
class _Ctx:
    """Everything the classifier stages need about one failed call."""

    error: Exception
    status_code: Optional[int]
    body: dict
    msg: str  # lowercased str(error) + body message(s)
    provider: str  # as passed by the caller
    model: str
    approx_tokens: int
    context_length: int
    num_messages: int

    def __post_init__(self) -> None:
        self.error_type = type(self.error).__name__
        self.error_code = _extract_error_code(self.body)
        self.code = self.error_code.lower()
        self.headers = _from_cause_chain(self.error, _headers_of, {})
        self.provider_slug = (self.provider or "").strip().lower()
        self.model_slug = (self.model or "").strip().lower()

    def large_session(self, frac: float, tokens: int, messages: int) -> bool:
        """Absolute thresholds only proxy for smaller context windows."""
        return self.approx_tokens > self.context_length * frac or (
            self.context_length <= 256000 and (self.approx_tokens > tokens or self.num_messages > messages)
        )


def _plugin_verdict(c: _Ctx) -> Optional[Verdict]:
    """First valid plugin classification (runs before the built-in pipeline so a
    provider plugin can add or correct verdicts). invoke_hook isolates callback
    failures; this guard only covers import/dispatch failure."""
    try:
        from hermes_cli.plugins import get_plugin_error_classification
        verdict = get_plugin_error_classification(
            provider=c.provider, model=c.model, status_code=c.status_code, error_type=c.error_type,
            error_code=c.error_code, error_message=c.msg, error_body=c.body, error=c.error,
            approx_tokens=c.approx_tokens, context_length=c.context_length, num_messages=c.num_messages,
        )
    except Exception as exc:
        logger.debug("Plugin error classification unavailable: %s", exc)
        return None
    if verdict is not None:
        logger.info("API error classified by plugin hook: %s (provider=%s, status=%s)",
                    verdict["reason"].value, c.provider, c.status_code)
    return verdict


def _provider_special_cases(c: _Ctx) -> Optional[Verdict]:
    """Highest-priority provider-specific shapes that a status code would misroute."""
    msg, status = c.msg, c.status_code
    # Safety refusal before status classification so a 400 block isn't downgraded
    # to format_error and a status-less block isn't left retryable (#18028).
    if any(p in msg for p in _CONTENT_POLICY_BLOCKED_PATTERNS):
        return _V_CONTENT_BLOCKED
    # Anthropic thinking-block 400s (signature mismatch after transcript
    # mutation). Not gated on provider — OpenRouter proxies Anthropic errors.
    if status == 400 and "thinking" in msg and any(p in msg for p in _THINKING_MUTATION_WORDS):
        return _v(_R.thinking_signature)
    # Anthropic long-context tier gate (429 "extra usage" + "long context").
    if status == 429 and "extra usage" in msg and "long context" in msg:
        return _v(_R.long_context_tier, should_compress=True)
    # Anthropic OAuth rejects the 1M beta header; run_agent retries without it.
    if status == 400 and "long context beta" in msg and "not yet available" in msg:
        return _v(_R.oauth_long_context_beta_forbidden)
    # llama.cpp grammar rejects regex ``pattern``/``format`` in tool schemas; the
    # retry loop strips them. Exclude the Qwen/vLLM "No user query found" error
    # local engines wrap as "Unable to generate parser for this template" —
    # that is a poisoned transcript (→ format_error), not a grammar problem.
    grammar_hit = "error parsing grammar" in msg or "json-schema-to-grammar" in msg or (
        "unable to generate parser" in msg and "template" in msg
    )
    if status == 400 and grammar_hit and _NO_USER_QUERY_SIGNAL not in msg:
        return _v(_R.llama_cpp_grammar_pattern)
    # xAI Grok entitlement as an SSE ``type=error`` frame: no status, matches no
    # pattern list, would otherwise burn max_retries as ``unknown``.
    if "do not have an active grok subscription" in msg or ("out of available resources" in msg and "grok" in msg):
        return _V_AUTH_FALLBACK
    return None


def _moa_special_cases(c: _Ctx) -> Optional[Verdict]:
    # Local MoA streaming adapter-shape bugs are not a provider outage; falling
    # back would silently replace the MoA route with a single model (#55933).
    if c.provider_slug == "moa" and any(s in str(c.error) for s in _MOA_ADAPTER_SHAPE_BUGS):
        return _v(_R.format_error, retryable=False)
    # Persisted MoA preset name that was renamed/deleted — deterministic config error.
    from agent.errors import MoAPresetNotFoundError
    return _v(_R.model_not_found, retryable=False) if isinstance(c.error, MoAPresetNotFoundError) else None


def _by_error_code(c: _Ctx) -> Optional[Verdict]:
    """Structured error codes from the response body."""
    # Request-validation failure as plain-text ``event: error`` SSE data behind
    # HTTP 200: retrying cannot succeed, a configured fallback still may.
    if c.code == PROVIDER_STREAM_NON_JSON_ERROR_CODE and "request validation failed:" in c.msg:
        return _V_FORMAT_ERROR
    return _ERROR_CODE_VERDICTS.get(c.code)


def _by_message(c: _Ctx) -> Optional[Verdict]:
    """Message patterns when no status code settled it; status-less usage
    limits get the same disambiguation as 402."""
    head = _first_match(c.msg, _MESSAGE_HEAD_RULES)
    if head is not None:
        return head
    usage_limit = any(p in c.msg for p in _USAGE_LIMIT_PATTERNS)
    return _classify_402(c.msg, dict) if usage_limit else _first_match(c.msg, _MESSAGE_TAIL_RULES)


def _by_transport(c: _Ctx) -> Optional[Verdict]:
    """SSL, disconnect, circuit-breaker and transport-type heuristics, in that order."""
    msg = c.msg
    # Cert failure → fail fast (checked first: also contains "[ssl:"); transient
    # alert → retry, before disconnects so a flaky handshake never compresses.
    ssl = _first_match(msg, ((_SSL_CERT_VERIFY_PATTERNS, _V_SSL_CERT), (_SSL_TRANSIENT_PATTERNS, _V_TIMEOUT)))
    if ssl is not None:
        return ssl
    # Disconnect + large session → probable overflow rejection, not a hiccup.
    if any(p in msg for p in _SERVER_DISCONNECT_PATTERNS) and not c.status_code:
        # Reasoning models: far more likely the gateway idle-killed a long
        # thinking stream — never compress on a phantom overflow (#52310).
        # Reasoning-model override: a transport disconnect on a reasoning model is much more likely the
        # upstream proxy idle-killing a long thinking stream than a true context overflow — even on large
        # sessions. The default disconnect+large-session routing below would otherwise send the user into
        # the compression branch (should_compress=True) and silently delete conversation history on a
        # phantom context-length error. Reasoning models have multi-minute thinking phases that routinely
        # exceed the cloud gateway's idle window (NVIDIA NIM ~120s — first-party repro at
        # NVIDIA/NemoClaw#4846; OpenAI worker / Anthropic stream-idle similar). The per-reasoning-model
        # stale-timeout floor in agent/reasoning_timeouts.py raises the stale-detector threshold to tolerate
        # long thinking, so a true transport-layer failure here is recoverable via the retry path — not via
        # context compression. Reclassify as timeout. (Part 1 of Fixes #52310.)
        from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
        if get_reasoning_stale_timeout_floor(c.model) is not None:
            return _V_TIMEOUT
        return _V_CONTEXT_OVERFLOW if c.large_session(0.6, 120000, 200) else _V_TIMEOUT
    # Stale-call circuit breaker (_check_stale_giveup RuntimeError before any
    # network call): as ``unknown`` it would burn every retry instantly.
    if c.error_type == "RuntimeError" and "consecutive stale attempts" in msg and "aborting this call" in msg:
        return _v(_R.timeout, **_ABORT_FALLBACK)
    transport = c.error_type in _TRANSPORT_ERROR_TYPES or isinstance(c.error, (TimeoutError, ConnectionError, OSError))
    return _V_TIMEOUT if transport else None


def _by_status(c: _Ctx) -> Optional[Verdict]:
    """HTTP status code with message-aware refinement (unlisted 4xx/5xx → generic)."""
    status = c.status_code
    if status is None:
        return None
    default = _V_FORMAT_ERROR if 400 <= status < 500 else _V_SERVER_ERROR if 500 <= status < 600 else None
    return _STATUS_HANDLERS[status](c) if status in _STATUS_HANDLERS else default


# Stage order: plugin hooks → provider-specific special cases → HTTP status →
# MoA shapes → structured error code → message patterns → SSL → disconnect +
# large session → transport types → unknown (retryable with backoff).
_STAGES: Sequence[Callable[[_Ctx], Optional[Verdict]]] = (
    _plugin_verdict, _provider_special_cases, _by_status, _moa_special_cases,
    _by_error_code, _by_message, _by_transport,
)


def classify_api_error(
    error: Exception, *, provider: str = "", model: str = "",
    approx_tokens: int = 0, context_length: int = 200000, num_messages: int = 0,
) -> ClassifiedError:
    """Classify an API error into a structured recovery recommendation (see ``_STAGES``)."""
    status_code = _extract_status_code(error)
    # Copilot/GitHub Models RateLimitError may not set .status_code; force 429.
    if status_code is None and type(error).__name__ == "RateLimitError":
        status_code = 429
    body = _extract_error_body(error)
    c = _Ctx(
        error, status_code, body, _build_error_msg(error, body), provider, model,
        approx_tokens, context_length, num_messages,
    )
    verdict = next((v for v in (stage(c) for stage in _STAGES) if v is not None), _V_UNKNOWN)
    base = {"status_code": status_code, "provider": provider, "model": model, "message": _extract_message(error, body)}
    return ClassifiedError(**{**base, **verdict})


# ── Status code handlers ────────────────────────────────────────────────

def _status_403(c: _Ctx) -> Verdict:
    # OpenRouter 403 "key limit exceeded" and similar plan/credit exhaustion are billing.
    xai_spend = c.provider_slug == "xai-oauth" and c.code == _XAI_SPENDING_LIMIT_ERROR_CODE
    billing = xai_spend or any(p in c.msg for p in ("key limit exceeded", "spending limit") + _BILLING_PATTERNS)
    return _V_BILLING if billing else _V_AUTH_FALLBACK


def _status_404(c: _Ctx) -> Verdict:
    verdict = _first_match(c.msg, _404_RULES)
    if verdict is not None:
        return verdict
    # Bare id the catalogue only knows prefixed → malformed id (NVIDIA NIM "404
    # page not found", #78796). A generic 404 (wrong path, proxy glitch) stays
    # unknown so the real error surfaces instead of a silent misreported fallback.
    return _V_MODEL_NOT_FOUND if _model_id_missing_known_prefix(c.model_slug, c.provider_slug) else _V_UNKNOWN


def _status_429(c: _Ctx) -> Verdict:
    # Z.AI/Zhipu reuse 429 for server-wide overload: back off on the same
    # key instead of burning the pool (#14038).
    if any(p in c.msg for p in _OVERLOADED_PATTERNS):
        return _V_OVERLOADED
    # OpenRouter-wrapped upstream 429: the key is healthy — fall back, don't bench.
    if _is_openrouter_upstream_error(c.body, c.provider_slug):
        upstream = _extract_upstream_provider_name(c.body)
        ctx = {"upstream_provider": upstream} if upstream else {}
        return _v(_R.upstream_rate_limit, should_fallback=True, error_context=ctx)
    # Quota walls as 429 (Anthropic ``usage_limit_reached``, "quota", billing
    # phrases) are billing ONLY when the body is not itself a rate-limit phrase
    # ("Rate limit exceeded" contains "limit exceeded") and carries no reset/
    # retry signal (#93419, #39441).
    quota_wall = c.code == "usage_limit_reached" or any(
        p in c.msg for p in ("usage_limit_reached",) + _USAGE_LIMIT_PATTERNS + _BILLING_PATTERNS
    )
    explicit_rate_limit = any(p in c.msg for p in _RATE_LIMIT_PATTERNS)
    if quota_wall and not explicit_rate_limit and not _has_usage_limit_transient_signal(c.msg, c.body, c.headers):
        return _V_BILLING
    return _V_RATE_LIMIT


def _status_5xx(c: _Ctx) -> Verdict:
    # Request-validation errors as 5xx (codex.nekos.me) fail fast instead of
    # retry-flooding — unless the parameter was injected server-side.
    validation = any(p in c.msg for p in _REQUEST_VALIDATION_PATTERNS) or c.code in _5XX_VALIDATION_CODES
    if validation and not _is_server_injected_param_rejection(c.msg, c.provider_slug):
        return _V_FORMAT_ERROR
    return _first_match(c.msg, _OVERFLOW_AS_5XX_RULES) or _V_SERVER_ERROR


def _classify_402(error_msg: str, result_fn: Callable[..., Any]) -> Any:
    """Disambiguate 402: "usage limit, try again in 5 minutes" is a periodic quota, not billing."""
    transient = any(p in error_msg for p in _USAGE_LIMIT_PATTERNS) and any(
        p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS
    )
    return result_fn(**(_V_RATE_LIMIT if transient else _V_BILLING))


def _classify_400(c: _Ctx) -> Verdict:
    """400 Bad Request — image/tool shapes, request-shape rejections, overflow, or generic."""
    msg, code = c.msg, c.code
    verdict = _first_match(msg, _IMAGE_TOOL_RULES)
    if verdict is not None:
        return verdict
    # Invalid encrypted reasoning replay blob (OpenAI Responses); before
    # overflow because "encrypted content … could not be verified" trips it.
    if code == "invalid_encrypted_content" or "invalid_encrypted_content" in msg or (
        "encrypted content for item" in msg and "could not be verified" in msg
    ) or "could not decrypt the provided encrypted_content" in msg:
        return _V_INVALID_ENCRYPTED
    # Reasoning-mandatory route rejecting a disable (GLM-5.3 on Nous Portal / OpenRouter). Deterministic
    # for the request shape, but the only bad field is ``reasoning: {enabled: false}`` — the loop drops
    # the disable and retries once. Must precede request-validation, which would abort as format_error.
    if _REASONING_MANDATORY_PATTERN in msg:
        return _V_REASONING_MANDATORY
    # 400 blaming a field this route never sent (Codex OAuth injects then rejects
    # prompt_cache_retention ~20% of the time): transient, retry identical request.
    if _is_server_injected_param_rejection(msg, c.provider_slug):
        return _V_SERVER_ERROR
    # Before overflow: GPT-5's "Unsupported parameter: 'max_tokens'" contains it.
    if any(p in msg for p in _400_VALIDATION_PATTERNS) or code in _400_VALIDATION_CODES:
        return _V_FORMAT_ERROR
    # Malformed message array before overflow: input can be tiny and compression
    # cannot fix it. litellm/Bedrock proxies use errorCode=INVALID_REQUEST_BODY.
    if any(p in msg for p in _INVALID_MESSAGE_BODY_PATTERNS) or code == "invalid_request_body":
        logger.warning(
            "Malformed message array 400 (invalid request body) classified as format_error, NOT context "
            "overflow — failing fast + falling back instead of entering the compression loop. This usually "
            "means an empty-content assistant stub is in the transcript; num_messages=%s approx_tokens=%s. "
            "error=%.200s", c.num_messages, c.approx_tokens, msg,
        )
        return _V_FORMAT_ERROR
    verdict = _first_match(msg, _400_TAIL_RULES)
    if verdict is not None:
        return verdict
    # Generic 400 + large session → probable overflow (Anthropic can return a
    # bare "Error"); proxy shapes are read so a descriptive rejection isn't "bare".
    body_msg = next((m for m in (str(x or "").strip().lower() for x in _body_message_candidates(c.body)) if m), "")
    is_generic = len(body_msg) < 30 or body_msg in {"error", ""}
    if is_generic and c.large_session(0.4, 80000, 80):
        return _V_CONTEXT_OVERFLOW
    return _V_FORMAT_ERROR


# 401 not retryable on its own: rotation/refresh run before the retryability
# check, then the client-error abort path (fallback first) is correct. 408 is
# retry-safe (RFC 9110 §15.5.9; proxies emit it when generation outruns the
# read window). Unlisted 4xx → format_error, 5xx → server_error.
_STATUS_HANDLERS: Dict[int, Callable[[_Ctx], Verdict]] = {
    400: _classify_400, 401: lambda c: _V_AUTH_ROTATE, 402: lambda c: _classify_402(c.msg, dict),
    403: _status_403, 404: _status_404, 408: lambda c: _V_TIMEOUT, 413: lambda c: _V_PAYLOAD_TOO_LARGE,
    429: _status_429, 500: _status_5xx, 502: _status_5xx,
    503: lambda c: _first_match(c.msg, _OVERFLOW_AS_5XX_RULES) or _V_OVERLOADED,
    529: lambda c: _first_match(c.msg, _OVERFLOW_AS_5XX_RULES) or _V_OVERLOADED,
}


# ── Helpers ─────────────────────────────────────────────────────────────

_RESET_FIELDS = ("resets_in_seconds", "resets_at", "reset_at", "retry_after")
_RESET_HEADERS = ("retry-after", "Retry-After", "x-ratelimit-reset", "X-RateLimit-Reset")


def _has_usage_limit_transient_signal(error_msg: str, body: dict, response_headers) -> bool:
    """Whether a usage-limit response identifies a reset window (message, body fields, or headers)."""
    if any(pattern in error_msg for pattern in _USAGE_LIMIT_TRANSIENT_SIGNALS):
        return True
    payloads = [p for p in (body, _error_obj(body)) if isinstance(p, dict)]
    if any(payload.get(f) not in (None, "") for payload in payloads for f in _RESET_FIELDS):
        return True
    if response_headers and hasattr(response_headers, "get"):
        return any(response_headers.get(h) not in (None, "") for h in _RESET_HEADERS)
    return False


def _model_id_missing_known_prefix(model: str, provider: str) -> bool:
    """True when a bare model id is only known to the provider as ``vendor/id``.

    Never guesses: an id absent from the curated catalogue returns False so real
    endpoint problems keep their retryable ``unknown`` classification.
    """
    name = (model or "").strip()
    if not name or "/" in name:
        return False
    try:
        from hermes_cli.model_normalize import suggest_prefixed_model_id
        return bool(suggest_prefixed_model_id((provider or "").strip(), name))
    except Exception:
        return False


def _is_server_injected_param_rejection(error_msg: str, provider: str) -> bool:
    """True when a 400 blames a one-route-only parameter this route never sends.

    Conservative: known parameters only, and only when ``provider`` is not a
    sender, so a genuine bad parameter (``max_tokens`` on GPT-5) stays format_error.
    """
    provider_slug = (provider or "").strip().lower()
    for param, senders in _SERVER_INJECTED_PARAM_SENDERS.items():
        if error_msg and param in error_msg and any(w in error_msg for w in _PARAM_REJECTION_WORDS):
            return not any(sender in provider_slug for sender in senders)
    return False


def _error_obj(body: Any) -> dict:
    """``body["error"]`` when it is a dict, else ``{}``."""
    err = body.get("error") if isinstance(body, dict) else None
    return err if isinstance(err, dict) else {}


def _json_dict(text: Any) -> Optional[dict]:
    """Parse a JSON object string; None for non-strings, blanks, invalid JSON or non-objects."""
    if not (isinstance(text, str) and text.strip()):
        return None
    try:
        inner = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return inner if isinstance(inner, dict) else None


def _openrouter_wrapped_message(err_obj: dict) -> str:
    """Lowercased inner message from OpenRouter's ``error.metadata.raw`` JSON wrapper."""
    metadata = err_obj.get("metadata", {})
    inner = _json_dict(metadata.get("raw")) if isinstance(metadata, dict) else None
    return str(_error_obj(inner).get("message") or "").lower() if inner else ""


def _build_error_msg(error: Exception, body: Any) -> str:
    """Lowercased str(error) + body message + OpenRouter-wrapped upstream message
    (OpenAI SDK's APIStatusError.__str__ omits the body, so it is appended)."""
    raw_msg = str(error).lower()
    body_msg = metadata_msg = ""
    if isinstance(body, dict):
        err_obj = _error_obj(body)
        body_msg = str(err_obj.get("message") or "").lower() or str(body.get("message") or "").lower()
        metadata_msg = _openrouter_wrapped_message(err_obj) if err_obj else ""
    parts = [raw_msg]
    if body_msg and body_msg not in raw_msg:
        parts.append(body_msg)
    if metadata_msg and metadata_msg not in raw_msg and metadata_msg not in body_msg:
        parts.append(metadata_msg)
    return " ".join(parts)


def _body_message_candidates(body: dict) -> Iterator[Any]:
    """Body message fields in priority order (OpenAI, flat, litellm/Bedrock proxy shapes)."""
    yield _error_obj(body).get("message")
    yield body.get("message")
    yield body.get("errorMessage")
    args = body.get("errorArgs")
    yield args.get("reason") if isinstance(args, dict) else None


def _from_cause_chain(error: Exception, pick: Callable[[Any], Any], default: Any) -> Any:
    """First non-None ``pick(exc)`` over the error and its __cause__/__context__ chain (max 5 deep)."""
    current = error
    for _ in range(5):
        found = pick(current)
        if found is not None:
            return found
        cause = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if cause is None or cause is current:
            break
        current = cause
    return default


def _status_of(exc: Any) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    code = getattr(exc, "status", None)  # some SDKs use .status
    return code if isinstance(code, int) and 100 <= code < 600 else None


def _body_of(exc: Any) -> Optional[dict]:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body
    response = getattr(exc, "response", None)
    try:
        json_body = response.json() if response is not None else None
    except Exception:
        return None
    return json_body if isinstance(json_body, dict) else None


def _headers_of(exc: Any) -> Any:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    return headers if headers and hasattr(headers, "get") else None


def _extract_status_code(error: Exception) -> Optional[int]:
    """HTTP status code from the error or its cause chain."""
    return _from_cause_chain(error, _status_of, None)


def _extract_error_body(error: Exception) -> dict:
    """Structured error body from an SDK exception or its cause chain."""
    return _from_cause_chain(error, _body_of, {})


def _code_from_payload(payload: Any, top_keys: Sequence[str], peek_message: bool) -> str:
    """Code/type from ``payload.error`` or a top-level key; ``"400"`` is not a code.
    ``peek_message`` also parses a JSON ``error.message`` for a nested code
    (Responses API surfaces ``invalid_encrypted_content`` this way)."""
    if not isinstance(payload, dict):
        return ""
    error_obj = payload.get("error", {})
    if isinstance(error_obj, dict):
        code = error_obj.get("code") or error_obj.get("type") or ""
        if isinstance(code, str) and code.strip() and code.strip() != "400":
            return code.strip()
        message = error_obj.get("message")
        if peek_message and isinstance(message, str) and message.strip().startswith("{"):
            nested_code = _code_from_payload(_json_dict(message), ("code", "error_code"), False)
            if nested_code:
                return nested_code
    code = next((payload.get(k) for k in top_keys if payload.get(k)), "")
    text = str(code).strip() if isinstance(code, (str, int)) else ""
    return text if text and text != "400" else ""


def _extract_error_code(body: dict) -> str:
    """Extract an error code string from the response body."""
    return _code_from_payload(body, ("code", "error_code", "errorCode"), True) if body else ""


def _extract_message(error: Exception, body: dict) -> str:
    """Extract the most informative error message (structured body first)."""
    msg = next((m for m in _body_message_candidates(body or {}) if isinstance(m, str) and m.strip()), None)
    return (msg.strip() if msg else str(error))[:500]


def _is_openrouter_upstream_error(body: Any, provider: str) -> bool:
    """OpenRouter's "Provider returned error" wrapper: the key is healthy, the
    upstream failed, so credential rotation is the wrong recovery."""
    err = _error_obj(body)
    if str(err.get("message") or "").strip().lower() != "provider returned error":
        return False
    if (provider or "").strip().lower() == "openrouter":
        return True
    # Otherwise require the metadata shape only OpenRouter produces.
    metadata = err.get("metadata")
    return isinstance(metadata, dict) and ("raw" in metadata or "provider_name" in metadata)


def _extract_upstream_provider_name(body: Any) -> Optional[str]:
    """Pull the upstream provider name out of OpenRouter's error metadata."""
    metadata = _error_obj(body).get("metadata")
    name = metadata.get("provider_name") if isinstance(metadata, dict) else None
    return name.strip() if isinstance(name, str) and name.strip() else None

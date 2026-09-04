"""API error classification for smart failover and recovery.

A priority-ordered pipeline maps an API exception to a ``ClassifiedError``
whose recovery hints (retry, rotate credential, fallback, compress, abort) the
retry loop in run_agent.py consults instead of re-matching strings itself.
"""

from __future__ import annotations

import enum
import json
import logging
import time
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
    image_corrupt = "image_corrupt"       # Provider says the image bytes are undecodable — shrinking won't help, strip and retry instead

    # Model / provider policy
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
    # LiteLLM proxies word a hard cap as "hard billing limit" (structured twin:
    # ``terminal_quota_exhausted`` in _BILLING_ERROR_CODES). "terminal billing
    # limit" free text is NOT matched: substring rules can't negate the
    # "non-terminal billing limit" wording, and the structured code covers it.
    "hard billing limit",
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
    "member_spend_cap_exceeded", "terminal_quota_exhausted", _XAI_SPENDING_LIMIT_ERROR_CODE,
    # OpenAI (and OpenAI-compatible aggregators) spend/usage-limit family:
    # a credit balance or an org/project spend or usage cap is exhausted —
    # terminal for this credential until limits are raised.
    "credit_balance_exhausted", "organization_spend_limit_exceeded",
    "organization_usage_limit_exceeded", "project_spend_limit_exceeded",
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
    "server overload", "server_overload",
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

# Patterns confirming usage limit is transient (not billing)
_USAGE_LIMIT_TRANSIENT_SIGNALS = [
    "try again",
    "retry",
    "resets at",
    "reset in",
    "resets in",
    "reset after",
    "available in",
    "wait",
    "requests remaining",
    "periodic",
    "window",
    "per minute",
    "per second",
]

# Payload-too-large patterns detected from message text (no status_code attr).
# Proxies and some backends embed the HTTP status in the error message.
_PAYLOAD_TOO_LARGE_PATTERNS = [
    "request entity too large",
    "payload too large",
    "error code: 413",
    # Anthropic's structured 413 error type.  Normally arrives with an HTTP
    # 413 status (handled by the status path), but aggregators/proxies can
    # re-wrap it into a plain message with no status attribute — route it to
    # the same compression recovery.  (port of anomalyco/opencode#37848)
    "request_too_large",
    "request exceeds the maximum size",
)

# Image-size patterns.  Matched against 400 bodies (not 413) because most
# providers return a 400 with a specific image-too-big message before the
# whole request hits the 413 size limit.  Anthropic's wording is the most
# important here (hard 5 MB per image, returned as
# "messages.N.content.K.image.source.base64: image exceeds 5 MB maximum").
_IMAGE_TOO_LARGE_PATTERNS = [
    "image exceeds",        # Anthropic: "image exceeds 5 MB maximum"
    "image too large",      # generic
    "image_too_large",      # error_code variant
    "image size exceeds",   # variant
    "image dimensions exceed",  # Anthropic: "image dimensions exceed max allowed size: 8000 pixels"
    "dimensions exceed max allowed size",  # Anthropic dimension-cap (wording variant)
    "max allowed size: 8000",  # Anthropic dimension-cap (explicit pixel ceiling)
    # Vendors that reject the same oversized image without using the word
    # "image".  MiniMax's Anthropic-compatible endpoint returns
    # "media exceeds size limit: max 10485760 bytes (2013)" for a native
    # image part above its 10 MB ceiling (#76039).  Matched on the "media"
    # fragment to mirror "image exceeds" above and catch reworded variants.
    # A non-image media rejection (audio/video) that lands here is safe: the
    # shrink pass finds no image parts, returns False, and the caller
    # surfaces the original error unchanged.
    "media exceeds",
    "media too large",
    # "request_too_large" on a request known to contain an image → image is
    # the likely culprit; we still try the shrink path before giving up.
]

# Image-corruption patterns — distinct from _IMAGE_TOO_LARGE_PATTERNS above.
# These fire when the provider can decode the request but not the image
# bytes themselves (e.g. a re-serialized image part in replayed history that
# lost data along the way). Re-encoding/shrinking corrupt bytes does not fix
# corruption, so this list is routed to the strip-and-retry path
# (FailoverReason.image_corrupt), never to the shrink path.
#
# xAI wording: {"code":"invalid-argument","error":"...Invalid PNG image."}
# xAI has a second wording for the same failure class depending on where
# the truncation lands: "Invalid PNG image." for aligned truncation,
# "base64 string of provided image cannot be decoded" for unaligned
# truncation (confirmed by the issue reporter — same root cause, two wire
# messages).
# A third xAI wording covers the URL-image path — the provider downloads
# the image itself and rejects the fetched bytes:
# {"code":"invalid-argument","error":"code: 'Client specified an invalid
# argument', message: \"Downloaded response does not contain a valid JPG,
# PNG, WebP, or ICO image.\""}
# Matched as the full observed sentence on purpose — shorter fragments
# ("downloaded response does not contain a valid") also match non-image
# download failures and would misroute them into strip-and-retry.
# See: https://github.com/NousResearch/hermes-agent/issues/69078
_IMAGE_CORRUPT_PATTERNS = [
    "invalid png image",
    "invalid jpeg image",
    "base64 string of provided image cannot be decoded",
    "downloaded response does not contain a valid jpg, png, webp, or ico image",
]

# Providers that follow the OpenAI spec strictly require tool message
# ``content`` to be a string.  Some (Anthropic native, Codex Responses,
# Gemini native, first-party OpenAI) extend this to accept a content-parts
# list (text + image_url) so screenshots from computer_use survive.  Others
# (Xiaomi MiMo, some Alibaba endpoints, a long tail of OpenAI-compatible
# providers) reject the list with a 400 — the patterns below are the most
# common error shapes we see.  Recovery: strip image parts from tool
# messages in-place, record the (provider, model) for the rest of the
# session so we don't waste another call learning the same lesson, retry.
#
# See: https://github.com/NousResearch/hermes-agent/issues/27344
_MULTIMODAL_TOOL_CONTENT_PATTERNS = [
    # Xiaomi MiMo: {"error":{"code":"400","message":"Param Incorrect","param":"text is not set"}}
    "text is not set",
    # Generic "tool message must be string" shapes
    "tool message content must be a string",
    "tool content must be a string",
    "tool message must be a string",
    # OpenAI-compat servers that reject list-type tool content with a
    # schema-validation message
    "expected string, got list",
    "expected string, got array",
    # Alibaba/DashScope variant
    "tool_call.content must be string",
]

# 400s rejecting list-type ``content`` in tool messages (Xiaomi MiMo "text is
# not set", Alibaba, OpenAI-compat long tail). Recovery: strip image parts from
# tool messages, remember (provider, model), retry. (#27344)
# NVIDIA NIM's Rust gateway never names the field: its serde rejection says the
# body "did not match any variant of untagged enum
# ChatCompletionRequestToolMessageContent", which is the same list-type tool
# content that every other wording here describes (#111231).
_MULTIMODAL_TOOL_CONTENT_PATTERNS = (
    "text is not set", "tool message content must be a string", "tool content must be a string",
    "tool message must be a string", "expected string, got list", "expected string, got array",
    # Console Go / pydantic-v2 relays behind opencode-go (422, param ``messages.N.tool.content.str``, #104731).
    "tool_call.content must be string", "tool.content.str", "input should be a valid string",
    "chatcompletionrequesttoolmessagecontent",
)

# Local-inference memory/resource-ceiling rejections (oMLX/MLX memory guard,
# llama.cpp/vLLM OOM, Metal/CUDA allocation ceilings). The server aborts on a
# prefill memory PEAK, not a window limit, yet its remediation tail says
# "reduce context length" — so without this list the request routes into
# compression, which cannot lower a prefill peak: it burns the compression
# budget, re-hits the wedged server each attempt and ends in a session reset.
# Every token names memory/allocation in BYTES, never a token count, so the
# list is disjoint from _CONTEXT_OVERFLOW_PATTERNS. Must be checked BEFORE
# both overflow AND the usage-limit disambiguation ("memory limit exceeded"
# contains "limit exceeded", which would otherwise read as billing). oMLX
# reworded the accounting sentence in 0.5.7 ("predicted peak would require /
# exceed"); the 0.5.6 wording is still in the field, so both stay. (#52261)
_MEMORY_CEILING_PATTERNS = (
    "memory guard", "memory limit exceeded", "memory_guard_tier", "dynamic ceiling",
    "memory ceiling", "available memory", "out of memory", "insufficient memory",
    "prefill would require", "predicted peak would", "prefill safety cap", "metal_cap",
)

# Structured codes identifying the same rejection at the source, before an
# OpenAI-compatible proxy flattens the body and drops the wording.
_MEMORY_CEILING_ERROR_CODES = frozenset({
    "prefill_memory_exceeded", "prefill_memory_aborted", "omlx_prefill_memory_exceeded",
})

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

# Proxy-side rejection of the model's own tool-call JSON (Ollama "invalid tool call arguments",
# OpenRouter-wrapped "function_call arguments"). Checked before the generic 400 validation and
# overflow heuristics: on a large session the bare message would otherwise read as overflow.
_MALFORMED_TOOL_ARGS_PATTERNS = (
    "invalid tool call arguments", "invalid tool_call arguments", "invalid tool_calls arguments",
    "invalid function call arguments", "invalid function_call arguments",
    "tool call arguments are invalid", "tool_call arguments are invalid",
    "function call arguments are invalid", "function_call arguments are invalid",
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
# A different provider (direct instead of the aggregator; another host's TLS chain) can fix these.
_V_POLICY_BLOCKED = _v(_R.provider_policy_blocked, **_ABORT_FALLBACK)
_V_SSL_CERT = _v(_R.ssl_cert_verification, **_ABORT_FALLBACK)
_V_CONTEXT_OVERFLOW = _v(_R.context_overflow, should_compress=True)
_V_PAYLOAD_TOO_LARGE = _v(_R.payload_too_large, should_compress=True)
_V_OVERLOADED, _V_SERVER_ERROR, _V_TIMEOUT, _V_UNKNOWN = map(_v, (_R.overloaded, _R.server_error, _R.timeout, _R.unknown))
_V_IMAGE_TOO_LARGE, _V_IMAGE_CORRUPT = _v(_R.image_too_large), _v(_R.image_corrupt)
_V_MULTIMODAL, _V_INVALID_ENCRYPTED = _v(_R.multimodal_tool_content_unsupported), _v(_R.invalid_encrypted_content)
_V_REASONING_MANDATORY = _v(_R.reasoning_mandatory, should_compress=False, should_fallback=False)
# The MODEL emitted unparseable tool-call JSON and the proxy (Ollama, OpenRouter) rejected it: no
# other provider can fix that output, so falling back only replays the same broken turn 4-5 times
# (20-60s per occurrence, #12770). Abort this call; the loop's argument repair handles the retry.
_V_MALFORMED_TOOL_ARGS = _v(_R.format_error, retryable=False, should_fallback=False)
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
    (_EMPTY_PROVIDER_RESPONSE_PATTERNS, _V_SERVER_ERROR), (_MEMORY_CEILING_PATTERNS, _V_OVERLOADED),
    (_CONTEXT_OVERFLOW_PATTERNS, _V_CONTEXT_OVERFLOW),
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
_MESSAGE_HEAD_RULES = ((_MEMORY_CEILING_PATTERNS, _V_OVERLOADED),
                       (_PAYLOAD_TOO_LARGE_PATTERNS, _V_PAYLOAD_TOO_LARGE)) + _IMAGE_TOOL_RULES

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
    **dict.fromkeys(_MEMORY_CEILING_ERROR_CODES, _V_OVERLOADED),
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


def _nous_welcome_tier(c: _Ctx) -> Optional[Verdict]:
    """The Nous inference gateway's welcome-tier (free tier) refusals, read from the structured body.

    A 429 carrying a fairshare ``reason`` is either a tier gate (``model_not_free`` /
    ``feature_not_free``: the model or feature is never served on the free tier, so retrying is
    pointless — abort this route and fall back) or capacity (``at_capacity`` / ``admission_closed``
    / ``rate_limited``: honour ``retry_after``, never rotate the free tier's only credential). A
    400/403 whose message names the wrong host or a dark tier is deterministic for the request.
    The parsed refusal rides ``error_context`` so the terminal copy can say what happened.
    """
    from hermes_cli.anon_auth import (
        WELCOME_TIER_GATE_REASONS, parse_welcome_refusal, welcome_route_refusal)
    status = c.status_code
    if status == 429:
        refusal = parse_welcome_refusal(c.body)
        if refusal is None:
            return None
        ctx = {"welcome_refusal": refusal}
        if refusal["reason"] in WELCOME_TIER_GATE_REASONS:
            return _v(_R.model_not_found, retryable=False, should_fallback=True, error_context=ctx)
        if refusal["retry_after"] > 0:
            ctx["reset_at"] = time.time() + refusal["retry_after"]
        return _v(_R.rate_limit, should_fallback=True, error_context=ctx)
    kind = welcome_route_refusal(status, c.msg)
    if kind is None:
        return None
    ctx = {"welcome_route": kind}
    if status == 403:
        return _v(_R.auth_permanent, retryable=False, should_fallback=True, error_context=ctx)
    return _v(_R.format_error, retryable=False, should_fallback=True, error_context=ctx)


def _provider_special_cases(c: _Ctx) -> Optional[Verdict]:
    """Highest-priority provider-specific shapes that a status code would misroute."""
    msg, status = c.msg, c.status_code
    welcome = _nous_welcome_tier(c)
    if welcome is not None:
        return welcome
    # Safety refusal before status classification so a 400 block isn't downgraded
    # to format_error and a status-less block isn't left retryable (#18028).
    if any(p in msg for p in _CONTENT_POLICY_BLOCKED_PATTERNS):
        return _V_CONTENT_BLOCKED
    # ChatGPT Codex masks a rejected encrypted-reasoning replay behind the same bare
    # ``invalid_prompt: Request blocked.`` it uses for real blocks (#92353). Exact envelope
    # + provider only. The verdict keeps format_error's abort-and-fallback hints; the one
    # extra thing it buys is turn_recovery's replay strip, which still requires cached
    # ``codex_reasoning_items`` — a genuine block with nothing to strip behaves as before.
    if _is_codex_masked_replay_rejection(c):
        return _v(_R.invalid_encrypted_content, **_ABORT_FALLBACK)
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
    # A structured billing code is decisive: LiteLLM stamps
    # ``terminal_quota_exhausted`` (a hard cap, not throttling) on 429s, and
    # this handler always returns, so _by_error_code never sees the code.
    if c.code in _BILLING_ERROR_CODES:
        return _V_BILLING
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
    ) or "could not decrypt the provided encrypted_content" in msg or (
        # Custom Responses endpoints wrap a replay rejection in a generic bad_request (#95834).
        "encrypted content could not be decrypted or parsed" in msg
    ) or (
        # OpenCode Zen wraps this OpenAI replay rejection in ``invalid_request_error`` (#111309).
        "encrypted_content" in msg and "was not issued to this caller" in msg
    ) or (
        # Azure Foundry (gpt-6-astra) rejects replayed reasoning from several prior responses this way (#105369).
        "conflicting authenticated continuation identities" in msg
    ):
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
    if any(p in msg for p in _MALFORMED_TOOL_ARGS_PATTERNS):
        return _V_MALFORMED_TOOL_ARGS
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
    # Memory ceiling by code: _by_status runs before _by_error_code, so a
    # 400 whose wording a proxy stripped would fall through to format_error.
    if code in _MEMORY_CEILING_ERROR_CODES:
        return _V_OVERLOADED
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
    422: lambda c: _first_match(c.msg, _IMAGE_TOOL_RULES) or _V_FORMAT_ERROR,
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

_INVALID_MESSAGE_BODY_PATTERNS = [
    "must have non-empty content",
    "messages must have non-empty",
    "invalid_request_body",
    "text content blocks must be non-empty",
    "content field is required",
    "messages: at least one message is required",
    # Qwen / vLLM chat templates raise this when the request has no surviving
    # non-empty user turn (oversized session truncation, compression that
    # dropped the only user message, or a resumed lineage that opens with
    # assistant/tool). Deterministic — compression cannot invent a user
    # query the template already rejected. Fail fast as format_error so we
    # do not thrash the compression loop or mis-route into llama.cpp
    # grammar recovery when local engines wrap the raise_exception as
    # applyPromptTemplate / "Unable to generate parser for this template".
    _NO_USER_QUERY_SIGNAL,
]

# Request-validation patterns — the request is malformed and will fail
# identically on every retry. Some OpenAI-compatible gateways (notably
# codex.nekos.me) return these as 5xx instead of the standard 4xx, which
# makes the generic "5xx → retryable server_error" rule misfire: the retry
# loop hammers the same deterministic rejection 3+ times, then the
# transport-recovery path resets the counter and does it again, producing
# a request flood. When a 5xx body carries one of these unambiguous
# request-validation signals, classify as a non-retryable format_error so
# the loop fails fast and falls back instead of looping.
_REQUEST_VALIDATION_PATTERNS = [
    "unknown parameter",
    "unsupported parameter",
    "unrecognized request argument",
    "invalid_request_error",
    "unknown_parameter",
    "unsupported_parameter",
]

# A reasoning-mandatory route answering ``reasoning: {enabled: false}``
# (Nous Portal + OpenRouter wording; ``error_msg`` is lowercased upstream).
_REASONING_MANDATORY_PATTERN = "reasoning is mandatory"

# Request parameters that Hermes sends on SOME routes only, paired with the
# providers/hosts where sending them is deliberate.
#
# When a host that is NOT in the allowed set rejects one of these fields, the
# client never put it in the body — the provider's own gateway injected it —
# so the 400 is a server-side flake rather than a deterministic request-shape
# error.  See ``_is_server_injected_param_rejection`` and the branch in
# ``_classify_400``.
#
# ``prompt_cache_retention`` is only sent for api.meta.ai and bedrock-mantle
# hosts (agent/transports/codex.py::_default_prompt_cache_retention_for_request).
# The Codex OAuth backend rejects it spontaneously on requests that provably
# never carried it.
_SERVER_INJECTED_PARAM_SENDERS: Dict[str, tuple] = {
    "prompt_cache_retention": ("meta", "muse", "msl", "model-api", "bedrock", "mantle"),
}


def _is_server_injected_param_rejection(error_msg: str, provider: str) -> bool:
    """True when a 400 blames a parameter this route never sends.

    ``error_msg`` is the lowercased, concatenated message text; ``provider`` is
    the lowercased provider slug.  A match means the rejection cannot be
    attributed to our own request shape, so the error is transient and retrying
    the identical request is the correct recovery.

    Deliberately conservative: it fires only for known one-route-only
    parameters AND only when the current provider is not one of the routes that
    actually sends them, so a genuine client-side bad parameter (``max_tokens``
    on a GPT-5 model) still fails fast as a ``format_error``.
    """
    if not error_msg:
        return False
    provider_slug = (provider or "").strip().lower()
    for param, senders in _SERVER_INJECTED_PARAM_SENDERS.items():
        if param not in error_msg:
            continue
        # Require the message to actually be a rejection of that parameter,
        # not an incidental mention.
        if not (
            "not supported" in error_msg
            or "unsupported" in error_msg
            or "unknown" in error_msg
            or "unrecognized" in error_msg
        ):
            continue
        if any(sender in provider_slug for sender in senders):
            # This route sends the field on purpose — a real request error.
            return False
        return True
    return False


# OpenRouter aggregator policy-block patterns.
#
# When a user's OpenRouter account privacy setting (or a per-request
# `provider.data_collection: deny` preference) excludes the only endpoint
# serving a model, OpenRouter returns 404 with a *specific* message that is
# distinct from "model not found":
#
#   "No endpoints available matching your guardrail restrictions and
#    data policy. Configure: https://openrouter.ai/settings/privacy"
#
# We classify this as `provider_policy_blocked` rather than
# `model_not_found` because:
#   - The model *exists* — model_not_found is misleading in logs
#   - Provider fallback won't help: the account-level setting applies to
#     every call on the same OpenRouter account
#   - The error body already contains the fix URL, so the user gets
#     actionable guidance without us rewriting the message
_PROVIDER_POLICY_BLOCKED_PATTERNS = [
    "no endpoints available matching your guardrail",
    "no endpoints available matching your data policy",
    "no endpoints found matching your data policy",
]

# Provider content-policy / safety-filter blocks. Distinct from
# ``provider_policy_blocked`` above (which is an OpenRouter *account*-level
# data/privacy guardrail) — these are *per-prompt* safety decisions made by
# the upstream model provider. They are deterministic for the unchanged
# request, so retrying the same prompt three times just reproduces the same
# block and burns paid attempts on a refusal. The recovery is to switch to a
# configured fallback model/provider immediately, or surface the block to
# the user with actionable guidance if no fallback exists.
#
# Patterns are intentionally narrow — each phrase is a verbatim string from
# a specific provider's safety pipeline, not a generic word like "policy" or
# "violation" that could collide with billing/auth/format errors:
#   • OpenAI Codex cybersecurity refusal (gpt-5.5, the case from #18028)
#   • OpenAI moderation refusal ("violates our usage policies", with
#     "usage policies" disambiguating from billing's "exceeded ... policy")
#   • Anthropic safety refusal ("prompt was flagged by ... safety system")
#   • OpenAI Responses content filter
_CONTENT_POLICY_BLOCKED_PATTERNS = [
    # OpenAI Codex (#18028) — message may arrive without an HTTP status
    "flagged for possible cybersecurity risk",
    "trusted access for cyber",
    # OpenAI moderation — chat completions / responses
    "violates our usage policies",
    "violates openai's usage policies",
    "your request was flagged by",
    # Anthropic safety system
    "prompt was flagged by our safety",
    "responses cannot be generated due to safety",
    # Generic content-filter wording seen on Azure / OpenAI Responses.
    # ``content_filter`` (underscore) is the OpenAI-standard error/finish
    # token surfaced verbatim by their SDKs when a request is blocked.
    # ``responsibleaipolicyviolation`` is Azure OpenAI's error code.
    # Deliberately NOT matching the space variant ("content filter") — it
    # appears in benign config descriptions and tooltip text that providers
    # echo back; the underscore form is provider-specific enough.
    "content_filter",
    "responsibleaipolicyviolation",
    # MiniMax output-layer safety filter. The error string is surfaced
    # verbatim by MiniMax SDK / OpenAI-compatible endpoints, usually in the
    # form "output new_sensitive (1027)" when the model's *output* (often a
    # large tool-call argument block) trips the upstream safety filter and
    # the SSE stream is truncated mid-flight. ``new_sensitive`` is the
    # filter name and is narrow enough that billing / format / auth error
    # strings will not collide. See #32421.
    "new_sensitive",
]

# Auth patterns (non-status-code signals)
_AUTH_PATTERNS = [
    "invalid api key",
    "invalid_api_key",
    "gateway_auth_failed",
    "authentication",
    "unauthorized",
    "forbidden",
    "invalid token",
    "token expired",
    "token revoked",
    "access denied",
]

# Anthropic thinking block signature patterns
_THINKING_SIG_PATTERNS = [
    "signature",  # Combined with "thinking" check
]

# Message-string patterns that indicate a provider-side timeout even when
# the exception type is generic (e.g. RuntimeError from a local shim that
# wraps a subprocess timeout).  Checked before the type-based transport
# heuristics so custom-provider "timed out" errors don't fall through to
# Provider empty-response advisories (OpenRouter / nano-gpt / similar).
# Checked before context-overflow matching because the advisory text often
# mentions "max_tokens" as a possible cause, which historically sat in
# _CONTEXT_OVERFLOW_PATTERNS and sent healthy sessions into a compression
# death spiral ending in "Cannot compress further".
_EMPTY_PROVIDER_RESPONSE_PATTERNS = [
    "returned an empty response",
    "empty response despite retries",
    "provider returned an empty response",
    "model returning empty responses",
    "empty response stream",
]

# the unknown bucket and get misreported as empty responses.
_TIMEOUT_MESSAGE_PATTERNS = [
    "timed out",
    "turn timed out",
    "request timed out",
    "deadline exceeded",
    "operation timed out",
    "upstream timed out",
]

# Connection-establishment / DNS failure message patterns.  These surface
# when the exception TYPE is generic (RuntimeError/Exception from a local
# shim, MCP bridge, subprocess wrapper, or an SDK that re-raises without
# chaining) so the _TRANSPORT_ERROR_TYPES check never fires, and the error
# carries no HTTP status.  Without message-level matching they fall through
# to FailoverReason.unknown, which misses the transport eager-fallback path
# in the retry loop (unknown retries the same dead endpoint for the full
# budget before fallback).  Ported from anomalyco/opencode#40707, which hit
# the same bug shape: serialized midstream errors matched by type only.
#
# Deliberately EXCLUDES mid-stream disconnect strings ("connection reset by
# peer", "peer closed connection", "unexpected eof", "socket hang up") —
# those belong to _SERVER_DISCONNECT_PATTERNS, whose classification step
# runs later and routes large sessions to context-overflow compression.
# A connection that was never established cannot be a server-side overflow
# rejection, so these are safe to classify as plain retryable transport.
_CONNECTION_MESSAGE_PATTERNS = [
    # TCP connect failures
    "connection refused",
    "econnrefused",
    "no route to host",
    "network is unreachable",
    "network unreachable",
    # DNS resolution failures (Python, glibc, macOS, Node bridge phrasings)
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname provided",
    "getaddrinfo failed",
    "getaddrinfo enotfound",
    "eai_again",
    # Node/undici bridge generic network failure (MCP servers, local shims)
    "fetch failed",
    "failed to fetch",
    # Envoy/proxy upstream connect failure (cloud gateways)
    "upstream connect error",
]

# Transport error type names
_TRANSPORT_ERROR_TYPES = frozenset({
    "ReadTimeout", "ConnectTimeout", "PoolTimeout",
    "ConnectError", "RemoteProtocolError",
    "ConnectionError", "ConnectionResetError",
    "ConnectionAbortedError", "BrokenPipeError",
    "TimeoutError", "ReadError",
    "ServerDisconnectedError",
    # SSL/TLS transport errors — transient mid-stream handshake/record
    # failures that should retry rather than surface as a stalled session.
    # ssl.SSLError subclasses OSError (caught by isinstance) but we list
    # the type names here so provider-wrapped SSL errors (e.g. when the
    # SDK re-raises without preserving the exception chain) still classify
    # as transport rather than falling through to the unknown bucket.
    "SSLError", "SSLZeroReturnError", "SSLWantReadError",
    "SSLWantWriteError", "SSLEOFError", "SSLSyscallError",
    # OpenAI SDK errors (not subclasses of Python builtins)
    "APIConnectionError",
    "APITimeoutError",
})

# Server disconnect patterns (no status code, but transport-level).
# These are the "ambiguous" patterns — a plain connection close could be
# transient transport hiccup OR server-side context overflow rejection
# (common when the API gateway disconnects instead of returning an HTTP
# error for oversized requests).  A large session + one of these patterns
# triggers the context-overflow-with-compression recovery path.
_SERVER_DISCONNECT_PATTERNS = [
    "server disconnected",
    "peer closed connection",
    "connection reset by peer",
    "connection was closed",
    "network connection lost",
    "unexpected eof",
    "incomplete chunked read",
]

# SSL certificate verification failures — deterministic, NOT transient.
#
# A failed certificate chain (TLS-inspecting corporate proxy, missing
# custom CA in the trust store, expired certificate, self-signed cert)
# fails identically on every retry. Burning the retry budget before
# surfacing the error hides the actionable fix from the user for minutes.
# Inspired by Claude Code v2.1.199 (July 2026), which made SSL certificate
# errors fail immediately with a fix hint instead of retrying.
#
# Must be checked BEFORE _SSL_TRANSIENT_PATTERNS — "certificate verify
# failed" messages usually also contain "[SSL:" which would otherwise
# match the transient list and retry forever.
_SSL_CERT_VERIFY_PATTERNS = [
    "certificate verify failed",       # Python ssl module canonical text
    "certificate_verify_failed",       # OpenSSL error token
    "unable to get local issuer certificate",
    "self-signed certificate",
    "self signed certificate",
    "certificate has expired",
    "hostname mismatch, certificate is not valid",
    "unable to verify the first certificate",  # Node/undici phrasing (MCP bridges)
]

# SSL/TLS transient failure patterns — intentionally distinct from
# _SERVER_DISCONNECT_PATTERNS above.
#
# An SSL alert mid-stream is almost always a transport-layer hiccup
# (flaky network, mid-session TLS renegotiation failure, load balancer
# dropping the connection) — NOT a server-side context overflow signal.
# So we want the retry path but NOT the compression path; lumping these
# into _SERVER_DISCONNECT_PATTERNS would trigger unnecessary (and
# expensive) context compression on any large-session SSL hiccup.
#
# The OpenSSL library constructs error codes by prepending a format string
# to the uppercased alert reason; OpenSSL 3.x changed the separator
# (e.g. `SSLV3_ALERT_BAD_RECORD_MAC` → `SSL/TLS_ALERT_BAD_RECORD_MAC`),
# which silently stopped matching anything explicit.  Matching on the
# stable substrings (`bad record mac`, `ssl alert`, `tls alert`, etc.)
# survives future OpenSSL format churn without code changes.
_SSL_TRANSIENT_PATTERNS = [
    # Space-separated (human-readable form, Python ssl module, most SDKs)
    "bad record mac",
    "ssl alert",
    "tls alert",
    "ssl handshake failure",
    "tlsv1 alert",
    "sslv3 alert",
    # Underscore-separated (OpenSSL error code tokens, e.g.
    # `ERR_SSL_SSL/TLS_ALERT_BAD_RECORD_MAC`, `SSLV3_ALERT_BAD_RECORD_MAC`)
    "bad_record_mac",
    "ssl_alert",
    "tls_alert",
    "tls_alert_internal_error",
    # Python ssl module prefix, e.g. "[SSL: BAD_RECORD_MAC]"
    "[ssl:",
]


# ── Classification pipeline ─────────────────────────────────────────────

def classify_api_error(
    error: Exception,
    *,
    provider: str = "",
    model: str = "",
    approx_tokens: int = 0,
    context_length: int = 200000,
    num_messages: int = 0,
) -> ClassifiedError:
    """Classify an API error into a structured recovery recommendation.

    Priority-ordered pipeline:
      0. Plugin ``transform_api_error_classification`` hooks (first valid result wins)
      1. Special-case provider-specific patterns (thinking sigs, tier gates)
      2. HTTP status code + message-aware refinement
      3. Error code classification (from body)
      4. Message pattern matching (billing vs rate_limit vs context vs auth)
      5. SSL/TLS transient alert patterns → retry as timeout
      6. Server disconnect + large session → context overflow
      7. Transport error heuristics
      8. Fallback: unknown (retryable with backoff)

    Args:
        error: The exception from the API call.
        provider: Current provider name (e.g. "openrouter", "anthropic").
        model: Current model slug.
        approx_tokens: Approximate token count of the current context.
        context_length: Maximum context length for the current model.

    Returns:
        ClassifiedError with reason and recovery action hints.
    """
    status_code = _extract_status_code(error)
    error_type = type(error).__name__
    # Copilot/GitHub Models RateLimitError may not set .status_code; force 429
    # so downstream rate-limit handling (classifier reason, pool rotation,
    # fallback gating) fires correctly instead of misclassifying as generic.
    if status_code is None and error_type == "RateLimitError":
        status_code = 429
    body = _extract_error_body(error)
    error_code = _extract_error_code(body)
    response_headers = _extract_response_headers(error)


_CODEX_MASKED_REPLAY_MESSAGE = "request blocked."


def _is_codex_masked_replay_rejection(c: "_Ctx") -> bool:
    """HTTP 400 / status-less ``{code: invalid_prompt, message: "Request blocked."}`` from
    ``openai-codex`` — as an SDK error body, a Responses ``error`` SSE frame, or the
    ``response.failed`` text ``"invalid_prompt: Request blocked."``."""
    if c.provider_slug != "openai-codex" or c.status_code not in (None, 400):
        return False
    # The OpenAI SDK unwraps ``body["error"]`` on status errors; stream frames keep the envelope.
    body_msg = next((str(m).strip().lower() for m in _body_message_candidates(c.body or {}) if m), "")
    return (c.code == "invalid_prompt" and body_msg == _CODEX_MASKED_REPLAY_MESSAGE) or (
        c.msg.strip() == f"invalid_prompt: {_CODEX_MASKED_REPLAY_MESSAGE}"
    )


def _error_obj(body: Any) -> dict:
    """``body["error"]`` when it is a dict, else ``{}``."""
    err = body.get("error") if isinstance(body, dict) else None
    return err if isinstance(err, dict) else {}


def _json_dict(text: Any) -> Optional[dict]:
    """Parse a JSON object string; None for non-strings, blanks, invalid JSON or non-objects."""
    if not (isinstance(text, str) and text.strip()):
        return None
    try:
        from hermes_cli.plugins import get_plugin_error_classification
        plugin_classification = get_plugin_error_classification(
            provider=provider,
            model=model,
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            error_message=error_msg,
            error_body=body,
            error=error,
            approx_tokens=approx_tokens,
            context_length=context_length,
            num_messages=num_messages,
        )
    except Exception as exc:
        logger.debug("Plugin error classification unavailable: %s", exc)
        plugin_classification = None
    if plugin_classification is not None:
        reason = plugin_classification.pop("reason")
        logger.info(
            "API error classified by plugin hook: %s (provider=%s, status=%s)",
            reason.value, provider, status_code,
        )
        return _result(reason, **plugin_classification)

    # ── 1. Provider-specific patterns (highest priority) ────────────

    # Provider content-policy / safety-filter block. The provider has made a
    # deterministic refusal decision about THIS prompt — retrying unchanged
    # just reproduces the same refusal and burns paid attempts. Must run
    # before status-based classification so a 400 safety block isn't
    # downgraded to a generic ``format_error`` and a status-less block
    # (OpenAI Codex SDK can raise without one) isn't left in the retryable
    # ``unknown`` bucket. See issue #18028.
    if any(p in error_msg for p in _CONTENT_POLICY_BLOCKED_PATTERNS):
        return _result(
            FailoverReason.content_policy_blocked,
            retryable=False,
            should_fallback=True,
        )

    # Anthropic thinking block recovery (400).  Two distinct failure modes,
    # same recovery (strip all reasoning_details and retry without thinking
    # blocks — see the thinking_signature handler in conversation_loop.py):
    #   1. Signature mismatch: a thinking block is signed against the full
    #      turn content; any upstream mutation (context compression, session
    #      truncation, message merging) invalidates the signature.
    #      Pattern: "signature" + "thinking".
    #   2. Frozen-block mutation: Anthropic rejects any change to the
    #      thinking/redacted_thinking blocks in the *latest* assistant
    #      message — "`thinking` or `redacted_thinking` blocks in the latest
    #      assistant message cannot be modified. These blocks must remain as
    #      they were in the original response."  This carries no "signature"
    #      token, so the original pattern missed it and the turn hard-aborted
    #      as a non-retryable client error instead of self-healing.
    #      Pattern: "thinking" + ("cannot be modified" | "must remain as they were").
    # Don't gate on provider — OpenRouter proxies Anthropic errors, so the
    # provider may be "openrouter" even though the error is Anthropic-specific.
    # The combined patterns are unique enough.
    if (
        status_code == 400
        and "thinking" in error_msg
        and (
            "signature" in error_msg
            or "cannot be modified" in error_msg
            or "must remain as they were" in error_msg
        )
    ):
        return _result(
            FailoverReason.thinking_signature,
            retryable=True,
            should_compress=False,
        )

    # Anthropic long-context tier gate (429 "extra usage" + "long context")
    if (
        status_code == 429
        and "extra usage" in error_msg
        and "long context" in error_msg
    ):
        return _result(
            FailoverReason.long_context_tier,
            retryable=True,
            should_compress=True,
        )

    # Anthropic OAuth subscription rejects the 1M-context beta header.
    # Observed error body: "The long context beta is not yet available for
    # this subscription." Returned as HTTP 400 from native Anthropic when
    # the subscription doesn't include 1M context, even though the request
    # carries ``anthropic-beta: context-1m-2025-08-07``. The recovery path
    # in run_agent.py rebuilds the Anthropic client with the beta stripped
    # and retries once. Pattern is narrow enough that it won't collide with
    # the 429 tier-gate pattern above (different status, different phrase).
    if (
        status_code == 400
        and "long context beta" in error_msg
        and "not yet available" in error_msg
    ):
        return _result(
            FailoverReason.oauth_long_context_beta_forbidden,
            retryable=True,
            should_compress=False,
        )

    # llama.cpp's ``json-schema-to-grammar`` converter (used by its OAI
    # server to build GBNF tool-call parsers) rejects regex escape classes
    # like ``\d``/``\w``/``\s`` and most ``format`` values. MCP servers
    # routinely emit ``"pattern": "\\d{4}-\\d{2}-\\d{2}"`` for date/phone/
    # email params. llama.cpp surfaces this as HTTP 400 with one of a few
    # recognizable phrases; on match we strip ``pattern``/``format`` from
    # ``self.tools`` in the retry loop and retry once. Cloud providers are
    # unaffected — they accept these keywords and we never hit this branch.
    #
    # Exclude Qwen/vLLM template raise_exception("No user query found…")
    # wrapped by some local engines as applyPromptTemplate / "Unable to
    # generate parser for this template". That is a poisoned transcript
    # shape (handled via _INVALID_MESSAGE_BODY_PATTERNS → format_error),
    # not a tool-schema grammar rejection — matching it here strips
    # pattern/format keywords and retries uselessly while the real fix
    # is /new (or a successful compression that preserves a user turn).
    if status_code == 400:
        _llama_cpp_grammar_hit = (
            "error parsing grammar" in error_msg
            or "json-schema-to-grammar" in error_msg
            or (
                "unable to generate parser" in error_msg
                and "template" in error_msg
            )
        )
    else:
        _llama_cpp_grammar_hit = False
    if (
        _llama_cpp_grammar_hit
        and _NO_USER_QUERY_SIGNAL not in error_msg
    ):
        return _result(
            FailoverReason.llama_cpp_grammar_pattern,
            retryable=True,
            should_compress=False,
        )

    # xAI Grok subscription entitlement errors.
    #
    # xAI returns "You have either run out of available resources or do not
    # have an active Grok subscription" through two distinct code paths:
    #
    #   • HTTP 403 — status_code is set; _classify_by_status (step 2) routes
    #     it to FailoverReason.auth correctly, and _is_entitlement_failure
    #     then prevents the credential-refresh loop.
    #
    #   • SSE ``type=error`` frame — surfaced as _StreamErrorEvent with
    #     status_code=None.  _classify_by_status is skipped entirely, and
    #     "grok subscription" / "out of available resources" appear in none
    #     of the message-pattern lists below.  Without this guard the error
    #     falls through to FailoverReason.unknown (retryable=True), burning
    #     max_retries before the agent stops — and _is_entitlement_failure
    #     is never called because it only runs under FailoverReason.auth.
    #
    # Both X Premium+ and SuperGrok subscribers hit this path when their
    # subscription tier does not cover the requested model or feature.
    if (
        "do not have an active grok subscription" in error_msg
        or ("out of available resources" in error_msg and "grok" in error_msg)
    ):
        return _result(
            FailoverReason.auth,
            retryable=False,
            should_fallback=True,
        )

    # ── 2. HTTP status code classification ──────────────────────────

    if status_code is not None:
        classified = _classify_by_status(
            status_code, error_msg, error_code, body,
            provider=provider_lower, model=model_lower,
            approx_tokens=approx_tokens, context_length=context_length,
            num_messages=num_messages,
            response_headers=response_headers,
            result_fn=_result,
        )
        if classified is not None:
            return classified

    # Local MoA streaming compatibility errors are adapter-shape bugs, not a
    # provider outage. Falling back to another model would silently switch the
    # user's selected MoA route to a single-model answer (#55933 follow-up).
    if provider_lower == "moa" and (
        "'types.SimpleNamespace' object is not iterable" in str(error)
        or "'types.SimpleNamespace' object has no attribute 'index'" in str(error)
    ):
        return _result(
            FailoverReason.format_error,
            retryable=False,
            should_fallback=False,
        )

    # Local MoA config drift is deterministic: a persisted session can retain
    # a preset name that was later renamed/deleted. Retrying the same lookup
    # cannot recover and makes a clear config error look like an API outage.
    from agent.errors import MoAPresetNotFoundError

    if isinstance(error, MoAPresetNotFoundError):
        return _result(FailoverReason.model_not_found, retryable=False)

    # ── 3. Error code classification ────────────────────────────────

    if error_code:
        classified = _classify_by_error_code(error_code, error_msg, _result)
        if classified is not None:
            return classified

    # ── 4. Message pattern matching (no status code) ────────────────

    classified = _classify_by_message(
        error_msg, error_type,
        approx_tokens=approx_tokens,
        context_length=context_length,
        result_fn=_result,
    )
    if classified is not None:
        return classified

    # ── 5. SSL certificate verification failures → fail fast ────────
    # A broken certificate chain (TLS-inspecting proxy, missing custom CA,
    # expired/self-signed cert) is deterministic for the host — every retry
    # reproduces the identical handshake failure. Fail immediately with
    # actionable guidance instead of burning the retry budget first.
    # Checked BEFORE the transient-SSL patterns: cert-verify messages also
    # contain "[ssl:" which would otherwise match the transient list.
    # Inspired by Claude Code v2.1.199 (July 2026).
    if any(p in error_msg for p in _SSL_CERT_VERIFY_PATTERNS):
        return _result(
            FailoverReason.ssl_cert_verification,
            retryable=False,
            should_fallback=False,
        )

    # ── 5b. SSL/TLS transient errors → retry as timeout (not compression) ──
    # SSL alerts mid-stream are transport hiccups, not server-side context
    # overflow signals.  Classify before the disconnect check so a large
    # session doesn't incorrectly trigger context compression when the real
    # cause is a flaky TLS handshake.  Also matches when the error is
    # wrapped in a generic exception whose message string carries the SSL
    # alert text but the type isn't ssl.SSLError (happens with some SDKs
    # that re-raise without chaining).
    if any(p in error_msg for p in _SSL_TRANSIENT_PATTERNS):
        return _result(FailoverReason.timeout, retryable=True)

    # ── 6. Server disconnect + large session → context overflow ─────
    # Must come BEFORE generic transport error catch — a disconnect on
    # a large session is more likely context overflow than a transient
    # transport hiccup.  Without this ordering, RemoteProtocolError
    # always maps to timeout regardless of session size.

    is_disconnect = any(p in error_msg for p in _SERVER_DISCONNECT_PATTERNS)
    if is_disconnect and not status_code:
        # Reasoning-model override: a transport disconnect on a reasoning
        # model is much more likely the upstream proxy idle-killing a
        # long thinking stream than a true context overflow — even on
        # large sessions.  The default disconnect+large-session routing
        # below would otherwise send the user into the compression
        # branch (should_compress=True) and silently delete
        # conversation history on a phantom context-length error.
        # Reasoning models have multi-minute thinking phases that
        # routinely exceed the cloud gateway's idle window (NVIDIA
        # NIM ~120s — first-party repro at NVIDIA/NemoClaw#4846;
        # OpenAI worker / Anthropic stream-idle similar).  The
        # per-reasoning-model stale-timeout floor in
        # agent/reasoning_timeouts.py raises the stale-detector
        # threshold to tolerate long thinking, so a true
        # transport-layer failure here is recoverable via the retry
        # path — not via context compression.  Reclassify as timeout.
        # (Part 1 of Fixes #52310.)
        from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
        if get_reasoning_stale_timeout_floor(model) is not None:
            return _result(FailoverReason.timeout, retryable=True)
        # Absolute token/message-count thresholds are only a proxy for smaller
        # context windows.  Large-context sessions can have hundreds of
        # messages while still being far below their actual token budget.
        is_large = approx_tokens > context_length * 0.6 or (
            context_length <= 256000 and (approx_tokens > 120000 or num_messages > 200)
        )
        if is_large:
            return _result(
                FailoverReason.context_overflow,
                retryable=True,
                should_compress=True,
            )
        return _result(FailoverReason.timeout, retryable=True)

    # ── 7b. Stale-call circuit breaker → failover immediately ──────
    # _check_stale_giveup() in agent/chat_completion_helpers.py raises a
    # RuntimeError when the provider has been unresponsive for N
    # consecutive stale attempts (default 5).  The error is NOT a transport
    # timeout — the circuit breaker fires *before* any network call to avoid
    # an indefinite stall.  Without this classification the RuntimeError
    # falls through to FailoverReason.unknown (retryable=True), which burns
    # all max_retries against the same dead provider (each retry hitting the
    # circuit breaker instantly with zero network overhead) before fallback
    # is attempted.  Classify as non-retryable + should_fallback so the
    # retry loop activates the next fallback provider on the first hit.
    if (
        error_type == "RuntimeError"
        and "consecutive stale attempts" in error_msg
        and "aborting this call" in error_msg
    ):
        return _result(
            FailoverReason.timeout,
            retryable=False,
            should_fallback=True,
        )

    # ── 8. Transport / timeout heuristics ───────────────────────────

    if error_type in _TRANSPORT_ERROR_TYPES or isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return _result(FailoverReason.timeout, retryable=True)

    # ── 9. Fallback: unknown ────────────────────────────────────────

    return _result(FailoverReason.unknown, retryable=True)


# ── Status code classification ──────────────────────────────────────────

def _classify_by_status(
    status_code: int,
    error_msg: str,
    error_code: str,
    body: dict,
    *,
    provider: str,
    model: str,
    approx_tokens: int,
    context_length: int,
    num_messages: int = 0,
    response_headers=None,
    result_fn,
) -> Optional[ClassifiedError]:
    """Classify based on HTTP status code with message-aware refinement."""

    if status_code == 401:
        # Not retryable on its own — credential pool rotation and
        # provider-specific refresh (Codex, Anthropic, Nous) run before
        # the retryability check in run_agent.py.  If those succeed, the
        # loop `continue`s.  If they fail, retryable=False ensures we
        # hit the client-error abort path (which tries fallback first).
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if status_code == 403:
        # OpenRouter 403 "key limit exceeded" is actually billing. Other
        # providers also use 403 for account-plan or credit exhaustion.
        if (
            (
                provider == "xai-oauth"
                and error_code.lower() == _XAI_SPENDING_LIMIT_ERROR_CODE
            )
            or "key limit exceeded" in error_msg
            or "spending limit" in error_msg
            or any(p in error_msg for p in _BILLING_PATTERNS)
        ):
            return result_fn(
                FailoverReason.billing,
                retryable=False,
                should_rotate_credential=True,
                should_fallback=True,
            )
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_fallback=True,
        )

    if status_code == 402:
        return _classify_402(error_msg, result_fn)

    if status_code == 404:
        # Nous API currently surfaces HA/NAS credit depletion as a paid model
        # becoming unavailable on the Free Tier, returned as 404 rather than
        # 402. Treat that as entitlement/billing exhaustion, not a missing
        # model, so the retry loop can show credit/top-up guidance.
        if any(p in error_msg for p in _BILLING_PATTERNS):
            return result_fn(
                FailoverReason.billing,
                retryable=False,
                should_rotate_credential=True,
                should_fallback=True,
            )
        # OpenRouter policy-block 404 — distinct from "model not found".
        # The model exists; the user's account privacy setting excludes the
        # only endpoint serving it. Falling back to another provider won't
        # help (same account setting applies).  The error body already
        # contains the fix URL, so just surface it.
        if any(p in error_msg for p in _PROVIDER_POLICY_BLOCKED_PATTERNS):
            return result_fn(
                FailoverReason.provider_policy_blocked,
                retryable=False,
                should_fallback=False,
            )
        if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
            return result_fn(
                FailoverReason.model_not_found,
                retryable=False,
                should_fallback=True,
            )
        # A bare id that the provider's catalogue only knows in prefixed form
        # is a malformed model id, not a routing glitch — NVIDIA NIM answers
        # one with a naked ``404 page not found`` that names nothing, so the
        # generic branch below burns three retries and reports what looks
        # like an outage (#78796). Deterministic: don't retry, and let the
        # model_not_found surface carry the real cause.
        if _model_id_missing_known_prefix(model, provider):
            return result_fn(
                FailoverReason.model_not_found,
                retryable=False,
                should_fallback=True,
            )
        # Generic 404 with no "model not found" signal — could be a wrong
        # endpoint path (common with local llama.cpp / Ollama / vLLM when
        # the URL is slightly misconfigured), a proxy routing glitch, or
        # a transient backend issue.  Classifying these as model_not_found
        # silently falls back to a different provider and tells the model
        # the model is missing, which is wrong and wastes a turn.  Treat
        # as unknown so the retry loop surfaces the real error instead.
        return result_fn(
            FailoverReason.unknown,
            retryable=True,
        )

    if status_code == 413:
        return result_fn(
            FailoverReason.payload_too_large,
            retryable=True,
            should_compress=True,
        )

    if status_code == 429:
        # Already checked long_context_tier above. Some providers (notably
        # Z.AI / Zhipu) reuse HTTP 429 for server-wide overload — same status
        # code as a true per-credential rate limit, but the credential is
        # valid and the correct recovery is "back off and retry the same key",
        # NOT "rotate the credential" (which exhausts the pool while the
        # endpoint is still busy, and does nothing for a single-key user).
        # Disambiguate on the error body so an overload 429 takes the
        # transient-overload path instead of burning the pool. (#14038)
        if any(p in error_msg for p in _OVERLOADED_PATTERNS):
            return result_fn(
                FailoverReason.overloaded,
                retryable=True,
            )
        # Distinguish an OpenRouter-aggregator upstream 429 (an upstream model
        # like DeepSeek rate-limited OpenRouter's aggregate traffic) from an
        # account-level 429 (the user's key is actually throttled). OpenRouter
        # wraps upstream errors with the outer message "Provider returned
        # error" — the user's key is healthy, so marking it exhausted / rotating
        # is wrong and burns the key for ~24min. Fall back to a different model.
        if _is_openrouter_upstream_error(body, provider):
            upstream_provider = _extract_upstream_provider_name(body)
            ctx = {"upstream_provider": upstream_provider} if upstream_provider else {}
            return result_fn(
                FailoverReason.upstream_rate_limit,
                retryable=True,
                should_rotate_credential=False,
                should_fallback=True,
                error_context=ctx,
            )
        # Account/subscription usage exhaustion is a quota wall, not a
        # request-rate throttle. Anthropic returns this as 429, so the generic
        # branch below used to retry it and Desktop rendered a provider error
        # instead of the billing/quota recovery. Preserve periodic quotas when
        # the response supplies an explicit reset/retry signal.
        #
        # The check covers the narrow #93419 core (Anthropic's
        # ``usage_limit_reached``) plus the broader ``_USAGE_LIMIT_PATTERNS``
        # ("quota", "limit exceeded", "key limit exceeded") so other providers'
        # hard quota walls also route to billing — but ONLY when the message is
        # not itself an explicit rate-limit phrase. Without that guard,
        # "Rate limit exceeded" ("limit exceeded" substring) would wrongly
        # promote to non-retryable billing. (broadening + guard credit #39441)
        has_usage_limit = (
            error_code.lower() == "usage_limit_reached"
            or "usage_limit_reached" in error_msg
            or any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
        )
        # Explicit billing phrases in a 429 body are a hard wall regardless of
        # usage-limit wording — a provider that wraps "insufficient credits" in
        # a 429 (rather than 402) was previously retried as a rate limit and
        # burned the pool. (credit #39441)
        has_billing = any(p in error_msg for p in _BILLING_PATTERNS)
        has_explicit_rate_limit = any(
            p in error_msg for p in _RATE_LIMIT_PATTERNS
        )
        has_transient_signal = _has_usage_limit_transient_signal(
            error_msg,
            body,
            response_headers,
        )
        if (
            (has_billing or has_usage_limit)
            and not has_explicit_rate_limit
            and not has_transient_signal
        ):
            return result_fn(
                FailoverReason.billing,
                retryable=False,
                should_rotate_credential=True,
                should_fallback=True,
            )
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if status_code == 400:
        return _classify_400(
            error_msg, error_code, body,
            provider=provider, model=model,
            approx_tokens=approx_tokens,
            context_length=context_length,
            num_messages=num_messages,
            result_fn=result_fn,
        )

    if status_code in {500, 502}:
        # Some OpenAI-compatible gateways return request-validation errors
        # with a 5xx status (codex.nekos.me returns 502 for unknown/
        # unsupported parameters). These are deterministic — every retry
        # gets the identical rejection — so the generic "5xx → retryable
        # server_error" rule turns one bad request into a retry flood.
        # Detect the unambiguous request-validation signals (in either the
        # message text or the structured error code) and fail fast.
        #
        # Exception: a parameter WE never sent on this route was injected by
        # the provider/proxy itself, so the rejection is not deterministic and
        # the generic retryable-5xx handling is correct. Mirrors the guard in
        # _classify_400 — see _is_server_injected_param_rejection.
        if (
            any(p in error_msg for p in _REQUEST_VALIDATION_PATTERNS)
            or error_code.lower() in {"invalid_request_error", "unknown_parameter",
                                      "unsupported_parameter"}
        ) and not _is_server_injected_param_rejection(error_msg, provider):
            return result_fn(
                FailoverReason.format_error,
                retryable=False,
                should_fallback=True,
            )
        # Some local inference servers (notably llama.cpp / llama-server)
        # report context overflow with an HTTP 500 instead of the standard
        # 400/413. The request-validation guard above already ran, so any
        # remaining explicit context-overflow signal routes into the
        # compression-and-retry path (mirroring _classify_400) instead of
        # blind server_error retries that exhaust and drop the turn.
        # Empty-response advisories that mention "max_tokens" must not enter
        # that compression path.
        if any(p in error_msg for p in _EMPTY_PROVIDER_RESPONSE_PATTERNS):
            return result_fn(
                FailoverReason.server_error,
                retryable=True,
                should_compress=False,
            )
        if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
            return result_fn(
                FailoverReason.context_overflow,
                retryable=True,
                should_compress=True,
            )
        return result_fn(FailoverReason.server_error, retryable=True)

    if status_code in {503, 529}:
        # Same overflow-as-5xx variant (server busy / model-load OOM, or a
        # Cloudflare/Tailscale hop relabeling the status). Route explicit
        # overflow bodies into compression; otherwise treat as transient
        # overload and retry.
        if any(p in error_msg for p in _EMPTY_PROVIDER_RESPONSE_PATTERNS):
            return result_fn(
                FailoverReason.server_error,
                retryable=True,
                should_compress=False,
            )
        if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
            return result_fn(
                FailoverReason.context_overflow,
                retryable=True,
                should_compress=True,
            )
        return result_fn(FailoverReason.overloaded, retryable=True)

    # 408 Request Timeout — a transient timing failure the server itself flags
    # as safe to retry (RFC 9110 §15.5.9), not a malformed request. Commonly
    # emitted by reverse proxies sitting in front of self-hosted backends
    # (llama.cpp / Ollama / vLLM) when a long generation outruns the proxy's
    # request-read window. Route to the dedicated ``timeout`` reason (rebuild
    # client + retry) instead of falling through to the generic 4xx bucket
    # below, which would abort the turn on a retry-safe error the same way it
    # aborts a 400 Bad Request.
    if status_code == 408:
        return result_fn(FailoverReason.timeout, retryable=True)

    # Other 4xx — non-retryable
    if 400 <= status_code < 500:
        return result_fn(
            FailoverReason.format_error,
            retryable=False,
            should_fallback=True,
        )

    # Other 5xx — retryable
    if 500 <= status_code < 600:
        return result_fn(FailoverReason.server_error, retryable=True)

    return None


def _has_usage_limit_transient_signal(
    error_msg: str,
    body: dict,
    response_headers,
) -> bool:
    """Return whether a usage-limit response identifies a reset window."""
    if any(pattern in error_msg for pattern in _USAGE_LIMIT_TRANSIENT_SIGNALS):
        return True

    payloads = [body]
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        payloads.append(body["error"])
    reset_fields = ("resets_in_seconds", "resets_at", "reset_at", "retry_after")
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if any(
            payload.get(field) is not None and payload.get(field) != ""
            for field in reset_fields
        ):
            return True

    if response_headers and hasattr(response_headers, "get"):
        for header in (
            "retry-after",
            "Retry-After",
            "x-ratelimit-reset",
            "X-RateLimit-Reset",
        ):
            value = response_headers.get(header)
            if value is not None and value != "":
                return True
    return False


def _classify_402(error_msg: str, result_fn) -> ClassifiedError:
    """Disambiguate 402: billing exhaustion vs transient usage limit.

    The key insight from OpenClaw: some 402s are transient rate limits
    disguised as payment errors.  "Usage limit, try again in 5 minutes"
    is NOT a billing problem — it's a periodic quota that resets.
    """
    # Check for transient usage-limit signals first
    has_usage_limit = any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
    has_transient_signal = any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS)

    if has_usage_limit and has_transient_signal:
        # Transient quota — treat as rate limit, not billing
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Confirmed billing exhaustion
    return result_fn(
        FailoverReason.billing,
        retryable=False,
        should_rotate_credential=True,
        should_fallback=True,
    )


def _classify_400(
    error_msg: str,
    error_code: str,
    body: dict,
    *,
    provider: str,
    model: str,
    approx_tokens: int,
    context_length: int,
    num_messages: int = 0,
    result_fn,
) -> ClassifiedError:
    """Classify 400 Bad Request — context overflow, format error, or generic."""

    # Multimodal tool content rejected from 400.  Must be checked BEFORE
    # image_too_large because the recovery is different (strip image parts
    # from tool messages, mark the model as no-list-tool-content for the
    # rest of the session) and BEFORE context_overflow because some of the
    # patterns ("text is not set") are ambiguous in isolation but become
    # specific when combined with a 400 on a request known to contain
    # multimodal tool content.
    if any(p in error_msg for p in _MULTIMODAL_TOOL_CONTENT_PATTERNS):
        return result_fn(
            FailoverReason.multimodal_tool_content_unsupported,
            retryable=True,
        )

    # Image-corruption from 400 (xAI's undecodable-image check fires this way).
    # Must be checked BEFORE image_too_large: both are image-shaped 400s, but
    # corrupt bytes need strip-and-retry, not shrink-and-retry — shrinking
    # can't repair a truncated/malformed PNG.
    if any(p in error_msg for p in _IMAGE_CORRUPT_PATTERNS):
        return result_fn(
            FailoverReason.image_corrupt,
            retryable=True,
        )

    # Image-too-large from 400 (Anthropic's 5 MB per-image check fires this way).
    # Must be checked BEFORE context_overflow because messages can trip both
    # patterns ("exceeds" + "image") and image-shrink is a cheaper recovery.
    if any(p in error_msg for p in _IMAGE_TOO_LARGE_PATTERNS):
        return result_fn(
            FailoverReason.image_too_large,
            retryable=True,
        )

    # Invalid encrypted reasoning replay blob (OpenAI Responses API).  Must be
    # checked BEFORE context_overflow because some surfaces emit messages that
    # contain context-like phrasing ("encrypted content … could not be
    # verified") which could otherwise trip the context_overflow heuristics.
    # ``error_msg`` is lowercased upstream — match accordingly.
    error_code_lower = (error_code or "").lower()
    if (
        error_code_lower == "invalid_encrypted_content"
        or "invalid_encrypted_content" in error_msg
        or (
            "encrypted content for item" in error_msg
            and "could not be verified" in error_msg
        )
        or "could not decrypt the provided encrypted_content" in error_msg
    ):
        return result_fn(
            FailoverReason.invalid_encrypted_content,
            retryable=True,
            should_fallback=False,
        )

    # Reasoning-mandatory route rejecting a disable (Nous Portal / OpenRouter
    # for GLM-5.3 etc.: "Reasoning is mandatory for this endpoint and cannot
    # be disabled").  Deterministic for the request shape, but the only bad
    # field is ``reasoning: {enabled: false}`` — the conversation_loop drops
    # the disable and retries once.  Must precede the request-validation
    # branch, which would abort the turn as a format_error.
    if _REASONING_MANDATORY_PATTERN in error_msg:
        return result_fn(
            FailoverReason.reasoning_mandatory,
            retryable=True,
            should_compress=False,
            should_fallback=False,
        )

    # Server-injected parameter rejection: a 400 blaming a request field the
    # client never sent.  MUST be checked BEFORE the request-validation branch
    # below, which would otherwise class it as a deterministic format_error and
    # abort the turn.
    #
    # Observed live on the Codex OAuth backend (chatgpt.com/backend-api/codex):
    # it intermittently adds ``prompt_cache_retention`` to its own upstream
    # call and then rejects it, so a byte-identical request succeeds on retry
    # (measured ~20% failure over n=20 on a minimal 1-message request that
    # provably carried no cache parameters).  Retrying is the correct and only
    # recovery; failing fast burnt an entire large-context request per attempt.
    if _is_server_injected_param_rejection(error_msg, provider):
        return result_fn(
            FailoverReason.server_error,
            retryable=True,
            # The request shape was fine — never route this into compression.
            should_compress=False,
        )

    # Request-validation errors (unsupported / unknown parameter) MUST be
    # checked BEFORE context_overflow.  A GPT-5 model rejecting max_tokens
    # returns:
    #   "Unsupported parameter: 'max_tokens' is not supported with this model.
    #    Use 'max_completion_tokens' instead."
    # That string contains the literal substring "max_tokens", which historically
    # sat in _CONTEXT_OVERFLOW_PATTERNS — so without this guard the 400 is
    # misclassified as context_overflow, routed into the compression loop,
    # re-sent with the same bad parameter, and ends in "Cannot compress
    # further".  These errors are deterministic (every retry gets the identical
    # rejection), so classify as a non-retryable format_error and fall back.
    #
    # NOTE: we deliberately do NOT key off the generic ``invalid_request_error``
    # code here — OpenAI stamps that same code on genuine context-overflow 400s,
    # so matching it would mis-route real overflows away from compression. The
    # unambiguous signals are the explicit "unsupported/unknown parameter"
    # message text and the specific parameter-level error codes.
    if (
        any(p in error_msg for p in _REQUEST_VALIDATION_PATTERNS
            if p != "invalid_request_error")
        or error_code_lower in {"unknown_parameter", "unsupported_parameter"}
    ):
        return result_fn(
            FailoverReason.format_error,
            retryable=False,
            should_fallback=True,
        )

    # Malformed message array (empty-content assistant stub, etc.). Must be
    # checked BEFORE context_overflow: the input can be tiny, so the generic
    # "400 + large session" heuristic would otherwise mis-route it into the
    # compression loop and thrash until "Cannot compress further" on every
    # retry (the request is unchanged, so compression cannot fix it). This is
    # a deterministic request-shape rejection — fail fast as a non-retryable
    # format_error and fall back. Checked against the message text AND the
    # structured error code, since proxies (litellm/Bedrock) surface the
    # signal in errorCode=INVALID_REQUEST_BODY.
    if (
        any(p in error_msg for p in _INVALID_MESSAGE_BODY_PATTERNS)
        or error_code_lower == "invalid_request_body"
    ):
        logger.warning(
            "Malformed message array 400 (invalid request body) classified as "
            "format_error, NOT context overflow — failing fast + falling back "
            "instead of entering the compression loop. This usually means an "
            "empty-content assistant stub is in the transcript; num_messages=%s "
            "approx_tokens=%s. error=%.200s",
            num_messages, approx_tokens, error_msg,
        )
        return result_fn(
            FailoverReason.format_error,
            retryable=False,
            should_fallback=True,
        )

    # Empty-provider-response advisories must not enter compression. They
    # often mention "max_tokens" as a possible cause and used to match the
    # bare overflow pattern, then thrash compress until "Cannot compress
    # further" on an otherwise healthy session (custom endpoints / nano-gpt).
    if any(p in error_msg for p in _EMPTY_PROVIDER_RESPONSE_PATTERNS):
        return result_fn(
            FailoverReason.server_error,
            retryable=True,
            should_compress=False,
        )

    # Context overflow from 400
    if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    # Some providers return model-not-found as 400 instead of 404 (e.g. OpenRouter).
    if any(p in error_msg for p in _PROVIDER_POLICY_BLOCKED_PATTERNS):
        return result_fn(
            FailoverReason.provider_policy_blocked,
            retryable=False,
            should_fallback=False,
        )
    if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return result_fn(
            FailoverReason.model_not_found,
            retryable=False,
            should_fallback=True,
        )

    # Some providers return rate limit / billing errors as 400 instead of 429/402.
    # Check these patterns before falling through to format_error.
    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )
    if any(p in error_msg for p in _BILLING_PATTERNS):
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
            # "out of extra usage" on a 400 is ambiguous — it can also be a
            # content-filter rejection (#82154). Mark the verdict unverified
            # so downstream hedges and the pool skips the 1-hour bench.
            error_context=_billing_ambiguity_context(error_msg),
        )

    # Generic 400 + large session → probable context overflow
    # Anthropic sometimes returns a bare "Error" message when context is too large
    err_body_msg = ""
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


# ── Message pattern classification ──────────────────────────────────────

def _classify_by_message(
    error_msg: str,
    error_type: str,
    *,
    approx_tokens: int,
    context_length: int,
    result_fn,
) -> Optional[ClassifiedError]:
    """Classify based on error message patterns when no status code is available."""

    # Payload-too-large patterns (from message text when no status_code)
    if any(p in error_msg for p in _PAYLOAD_TOO_LARGE_PATTERNS):
        return result_fn(
            FailoverReason.payload_too_large,
            retryable=True,
            should_compress=True,
        )

    # Multimodal tool content patterns (from message text when no status_code)
    if any(p in error_msg for p in _MULTIMODAL_TOOL_CONTENT_PATTERNS):
        return result_fn(
            FailoverReason.multimodal_tool_content_unsupported,
            retryable=True,
        )

    # Image-corruption patterns (from message text when no status_code)
    if any(p in error_msg for p in _IMAGE_CORRUPT_PATTERNS):
        return result_fn(
            FailoverReason.image_corrupt,
            retryable=True,
        )

    # Image-too-large patterns (from message text when no status_code)
    if any(p in error_msg for p in _IMAGE_TOO_LARGE_PATTERNS):
        return result_fn(
            FailoverReason.image_too_large,
            retryable=True,
        )

    # Usage-limit patterns need the same disambiguation as 402: some providers
    # surface "usage limit" errors without an HTTP status code.  A transient
    # signal ("try again", "resets at", …) means it's a periodic quota, not
    # billing exhaustion.
    has_usage_limit = any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
    if has_usage_limit:
        has_transient_signal = any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS)
        if has_transient_signal:
            return result_fn(
                FailoverReason.rate_limit,
                retryable=True,
                should_rotate_credential=True,
                should_fallback=True,
            )
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Overloaded / server-busy patterns — must come BEFORE the rate_limit and
    # billing checks so that a message-only "overloaded" (no 503/529 status,
    # e.g. some Anthropic-compatible proxies) classifies as a transient
    # overload (backoff + retry) instead of falling through to `unknown` or
    # incorrectly triggering credential rotation.
    if any(p in error_msg for p in _OVERLOADED_PATTERNS):
        return result_fn(
            FailoverReason.overloaded,
            retryable=True,
        )

    # Billing patterns
    if any(p in error_msg for p in _BILLING_PATTERNS):
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
            # Status-less path: adapters can strip the HTTP status from the
            # Anthropic "out of extra usage" 400, so the same ambiguity
            # marking applies here (#82154).
            error_context=_billing_ambiguity_context(error_msg),
        )

    # Rate limit patterns
    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Empty-provider-response advisories (often mention "max_tokens") must
    # retry without compression — see the matching 400-path guard above.
    if any(p in error_msg for p in _EMPTY_PROVIDER_RESPONSE_PATTERNS):
        return result_fn(
            FailoverReason.server_error,
            retryable=True,
            should_compress=False,
        )

    # Context overflow patterns
    if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    # Auth patterns
    # Auth errors should NOT be retried directly — the credential is invalid and
    # retrying with the same key will always fail.  Set retryable=False so the
    # caller triggers credential rotation (should_rotate_credential=True) or
    # provider fallback rather than an immediate retry loop.
    if any(p in error_msg for p in _AUTH_PATTERNS):
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Provider policy-block (aggregator-side guardrail) — check before
    # model_not_found so we don't mis-label as a missing model.
    if any(p in error_msg for p in _PROVIDER_POLICY_BLOCKED_PATTERNS):
        return result_fn(
            FailoverReason.provider_policy_blocked,
            retryable=False,
            should_fallback=False,
        )

    # Model not found patterns
    if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return result_fn(
            FailoverReason.model_not_found,
            retryable=False,
            should_fallback=True,
        )

    # Timeout message patterns — generic exception types (e.g. RuntimeError)
    # raised by local shims or custom providers that internally wrap a
    # subprocess/HTTP timeout.  Classified as transport timeout so the retry
    # loop rebuilds the client instead of treating the turn as an empty
    # model response.
    if any(p in error_msg for p in _TIMEOUT_MESSAGE_PATTERNS):
        return result_fn(FailoverReason.timeout, retryable=True)

    # Connection-establishment / DNS failure message patterns — same shim
    # problem as the timeout patterns above: the wrapping exception type is
    # generic, so _TRANSPORT_ERROR_TYPES never matches and the error would
    # fall through to FailoverReason.unknown. Classified as timeout (the
    # transport bucket) so the retry loop's eager transport fallback and
    # client rebuild apply. Never routes to compression: a connection that
    # was never established is not a context-overflow signal.
    if any(p in error_msg for p in _CONNECTION_MESSAGE_PATTERNS):
        return result_fn(FailoverReason.timeout, retryable=True)

    return None


# ── Helpers ─────────────────────────────────────────────────────────────

def _extract_status_code(error: Exception) -> Optional[int]:
    """Walk the error and its cause chain to find an HTTP status code."""
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


def _extract_response_headers(error: Exception):
    """Walk the error and its cause chain to find response headers."""
    current = error
    for _ in range(5):
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", None)
        if headers and hasattr(headers, "get"):
            return headers
        cause = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if cause is None or cause is current:
            break
        current = cause
    return {}


def _extract_error_code(body: dict) -> str:
    """Extract an error code string from the response body."""
    if not body:
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

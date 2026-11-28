"""Final-response and provider-error sanitization for gateway chat surfaces.

Extracted from the ``gateway/run.py`` facade: the provider-error reply table, the envelope shape
gates and the last-mile ``_sanitize_gateway_final_response`` boundary share one topic and one test
file, and the facade is over its file-length budget. ``gateway.run`` re-exports every name here, so
``from gateway.run import _sanitize_gateway_final_response`` keeps working.

A turn that terminally fails returns the provider's text as ``final_response``. That text is runtime
copy, not the model's answer, so a failed result widens the envelope gate past the short-envelope cap
that protects ordinary prose — see ``_looks_like_gateway_provider_error``.
"""

import re
from typing import Any, Optional

from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX
from agent.i18n import t

# Surfaces consuming gateway text programmatically must keep RAW status/error text; unknown/empty -> chat.
_GATEWAY_RAW_TEXT_PLATFORMS = frozenset({"local", "api_server", "webhook", "msgraph_webhook"})


def _gateway_surface_passes_raw_text(platform: Any) -> bool:
    """True only for programmatic/local surfaces that must keep raw text."""
    return _gateway_platform_value(platform) in _GATEWAY_RAW_TEXT_PLATFORMS


_GATEWAY_PROVIDER_POLICY_RE = re.compile(
    r"("  # raw provider policy/safety bodies are noisy and may be sensitive
    r"cybersecurity\s+risk"
    r"|security\s+policy"
    r"|safety\s+policy"
    r"|policy\s+violation"
    r"|violat(?:e|es|ed|ion)"
    r"|blocked\s+(?:because|by|under)"
    r"|request\s+(?:was\s+)?(?:blocked|rejected)"
    r"|disallowed"
    r"|moderation"
    r")",
    re.IGNORECASE)

# ``401`` as a status token: not glued to a digit or a timestamp/identifier separator on the left
# (``05:14:15,401``), but trailing punctuation is a real envelope (``HTTP 401: Unauthorized``,
# ``returned 401.``) and must keep matching (#89401).
_GATEWAY_AUTH_ERROR_RE = re.compile(
    r"(provider\s+authentication\s+failed|incorrect\s+api\s+key|invalid\s+api\s+key"
    r"|(?<![\d:,.])401(?!\d))",
    re.IGNORECASE)

_GATEWAY_RATE_LIMIT_RE = re.compile(
    r"(rate\s+limit|rate-limited|\b429\b|quota|usage\s+limit)", re.IGNORECASE)

# Connection-failure markers: the first 8 also anchor the provider-failure envelope shape below.
_CONNECTION_ERROR_MARKERS = (
    r"(?:\w+\.)?(?:api\s*)?connection\s*(?:error|timeout)", r"(?:\w+\.)?connect\s*(?:error|timeout)",
    r"connection\s+refused", r"connection\s+reset", r"connection\s+aborted", r"actively\s+refused",
    r"winerror\s+10061\b", r"errno\s+111\b", r"no\s+route\s+to\s+host", r"network\s+is\s+unreachable",
    r"cannot\s+connect", r"failed\s+to\s+establish", r"could\s+not\s+connect")
_GATEWAY_CONNECTION_ERROR_RE = re.compile("(" + "|".join(_CONNECTION_ERROR_MARKERS) + ")", re.IGNORECASE)

# An ESTABLISHED connection died mid-request. Says nothing about whether the endpoint is up:
# an earlier call in the same turn may already have been answered by it (#26339, #116323).
_CONNECTION_INTERRUPTED_MARKERS = (
    r"connection\s+reset", r"connection\s+aborted", r"errno\s+104\b", r"errno\s+103\b",
    r"broken\s+pipe", r"server\s+disconnected", r"peer\s+closed\s+connection",
    r"connection\s+was\s+closed", r"network\s+connection\s+lost", r"unexpected\s+eof",
    r"incomplete\s+chunked\s+read", r"response\s+ended\s+prematurely", r"socket\s+hang\s+up",
    r"(?:\w+\.)?remoteprotocolerror", r"(?:\w+\.)?readerror")
_GATEWAY_CONNECTION_INTERRUPTED_RE = re.compile(
    "(" + "|".join(_CONNECTION_INTERRUPTED_MARKERS) + ")", re.IGNORECASE)

# Nothing accepted the connection / no path to the host: "the endpoint is not up" IS the diagnosis.
_ENDPOINT_UNREACHABLE_MARKERS = (
    r"connection\s+refused", r"actively\s+refused", r"winerror\s+10061\b", r"errno\s+111\b",
    r"no\s+route\s+to\s+host", r"network\s+is\s+unreachable", r"cannot\s+connect",
    r"failed\s+to\s+establish", r"could\s+not\s+connect", r"(?:\w+\.)?connect\s*(?:error|timeout)")
_GATEWAY_ENDPOINT_UNREACHABLE_RE = re.compile(
    "(" + "|".join(_ENDPOINT_UNREACHABLE_MARKERS) + ")", re.IGNORECASE)


def _gateway_platform_value(platform: Any) -> str:
    """Return a normalized gateway platform value for enums or raw strings."""
    return str(getattr(platform, "value", platform) or "").strip().lower()


def _redact_gateway_user_facing_secrets(text: str) -> str:
    """Secret redaction before text can leave the gateway for a chat platform: the shared egress scrub
    (``force=True`` holds even when ``security.redact_secrets`` is off; fails closed; #23810, #68911)."""
    from agent.redact_phone_policy import redact_for_chat_egress

    return redact_for_chat_egress(text)


# Ordered: rate-limit beats auth beats policy beats connection; first match wins. Rate-limit goes
# first because a quota/429 envelope often also carries an auth-shaped preamble ("Provider
# authentication failed: ... quota exhausted (429) ... Credentials are still valid") and re-auth can
# never fix a quota, so text with both signals must fail safe toward the quota reply (#89401). Copy
# names the slash command the chat user can run; raw provider text stays in the gateway log.
#
# The three connection rows are not interchangeable (#116323): a RESET/EOF on an established
# connection says nothing about whether the endpoint is up (an earlier call in the same turn may have
# been answered by it), a REFUSED/unroutable connect is the endpoint-down case #86570 wrote the
# wording for, and a cause-free SDK ``APIConnectionError: Connection error.`` supports neither
# diagnosis, so the catch-all names the failure without asserting a cause.
# ``(pattern, catalog key)`` — the reply text is resolved with ``t()`` at reply time so the active
# ``display.language`` applies; ``gateway.errors.*`` in ``locales/en.yaml`` holds the wording.
_PROVIDER_ERROR_REPLIES = (
    (_GATEWAY_RATE_LIMIT_RE, "gateway.errors.rate_limited"),
    (_GATEWAY_AUTH_ERROR_RE, "gateway.errors.auth_failed"),
    (_GATEWAY_PROVIDER_POLICY_RE, "gateway.errors.bad_request"),
    (_GATEWAY_CONNECTION_INTERRUPTED_RE, "gateway.errors.connection_interrupted"),
    (_GATEWAY_ENDPOINT_UNREACHABLE_RE, "gateway.errors.unreachable"),
    (_GATEWAY_CONNECTION_ERROR_RE, "gateway.errors.connection_unknown"))


def _context_overflow_reply() -> str:
    """Shared by the failed-turn normalizer and ``run_turn._hmwa_agent_error_reply``; canonical
    commands (/compress, /new) — the /compact and /reset aliases are absent from /help."""
    return t("gateway.errors.context_overflow")


def _rate_limit_reply(text: str) -> str:
    """Name the reset window a quota 429 carries (``resets_in_seconds`` body field, the credential
    pool's ``retry after Ns``, ``resets in 4hr``) so a weekly cap is not sold as "wait a moment"
    (#89401). One grammar table with the retry loop: ``agent.retry_utils.RETRY_DELAY_PATTERNS``."""
    from agent.retry_utils import format_reset_window, reset_delay_from_message
    seconds = reset_delay_from_message(text) or 0
    if seconds < 120:
        return t("gateway.errors.rate_limited")
    return t("gateway.errors.usage_limit_resets", window=format_reset_window(seconds))


def _gateway_provider_error_reply(text: str) -> str:
    """Map raw provider/API errors to a short user-safe Telegram reply."""
    for pattern, reply_key in _PROVIDER_ERROR_REPLIES:
        if pattern.search(text):
            return _rate_limit_reply(text) if pattern is _GATEWAY_RATE_LIMIT_RE else t(reply_key)
    return t("gateway.errors.provider_kept_failing")


# Provider/API failure envelope preambles (not ordinary assistant prose), anchored at line start.
_PROVIDER_ERROR_MARKERS = (
    r"api\s+(?:call\s+)?failed", r"provider\s+authentication\s+failed", r"non-retryable\s+error",
    r"rate\s+limited\s+after\s+\d+\s+(?:retries|attempts?)\b", r"error\s+code\s*:", r"http\s*\d{3}\b",
    r"incorrect\s+api\s+key", r"invalid\s+api\s+key", r"unexpected\s+status\s+\d{3}\s+(?:[A-Za-z][\w' -]{0,40}|<unknown status code>):")
_GATEWAY_PROVIDER_ERROR_SHAPE_RE = re.compile(
    r"^\s*(\W*\s*)?("
    + "|".join(_PROVIDER_ERROR_MARKERS + _CONNECTION_ERROR_MARKERS[:8] + (r"all\s+connection\s+attempts\s+failed",))
    + ")",
    re.IGNORECASE)


def _looks_like_gateway_provider_error(text: str, *, terminal_failure: bool = False) -> bool:
    """True when text is a provider failure envelope, not normal content.

    Must start with the marker, so prose citing a status code misses. The short-envelope cap
    (1-3 lines) keeps prose out wherever the caller cannot prove the turn failed; ``terminal_failure``
    is that proof, and on a failed turn the long form (a multi-line provider body, a quota/billing
    dump) is runtime diagnostic that must not be delivered as the model's answer either."""
    if not text:
        return False
    body = str(text).strip()
    if not terminal_failure and (len(body) > 400 or body.count("\n") > 4):
        return False
    return bool(_GATEWAY_PROVIDER_ERROR_SHAPE_RE.search(body))


def _turn_result_failed(agent_result: Any) -> bool:
    """The turn's structured result says it failed: its text is runtime copy, never an answer."""
    return isinstance(agent_result, dict) and bool(agent_result.get("failed"))


def _is_long_runtime_failure_text(text: str) -> bool:
    """Past the short-envelope cap: a multi-line provider body or a quota/billing dump.

    Short envelopes keep their existing handling (the delivery sanitizer classifies them); this is
    the class that reached chat verbatim because it never fitted the cap.
    """
    body = str(text or "").strip()
    return len(body) > 400 or body.count("\n") > 4


def _sanitize_gateway_final_response(
    platform: Any, text: str, *, agent_result: Optional[dict] = None) -> str:
    """Sanitize final gateway replies for chat surfaces: concise, secret-redacted provider failure
    categories instead of raw HTTP bodies, request IDs, leaked credentials, or policy text.

    ``agent_result`` is the turn's structured result when the caller holds it: a ``failed`` result
    makes this text runtime copy rather than the model's answer, so a long provider diagnostic is
    classified too (see ``_looks_like_gateway_provider_error``)."""
    if not text or _gateway_surface_passes_raw_text(platform):
        return text

    # Lone UTF-16 surrogates make Telegram/Signal ``.encode()`` raise; last defense for legacy/plugin paths.
    # Lone UTF-16 surrogates (U+D800–U+DFFF) in model output crash chat surfaces downstream: Telegram's
    # ``utf16_len`` length check and Signal formatting both ``.encode()`` the reply and raise
    # UnicodeEncodeError before any send (#55143, #55309). The stored-history copy is already sanitized by
    # ``build_assistant_message`` and ``finalize_turn`` scrubs the returned ``final_response``, but this
    # boundary is the last line of defense for every legacy/plugin delivery path that hands us raw text.
    # Raw-text/programmatic surfaces above keep passthrough — their JSON consumers escape surrogates safely.
    from agent.message_sanitization import _sanitize_surrogates

    text = _sanitize_surrogates(str(text))

    # Some OpenAI-compatible providers leak their exact end-of-sequence control token into
    # ``final_response`` even though finish_reason is already ``stop``.  It is transport metadata,
    # not an assistant message; without filtering, chat adapters send a literal ``<|eos|>`` bubble.
    # Reuse the MEDIA boundary's exact, terminal-only recognizer (#111046 / #111348): examples
    # mentioning the token mid-response and non-exact variants remain byte-identical.
    from gateway.platforms.base import _terminal_sentinel_start  # lazy: the adapter tree imports us

    _eos_start = _terminal_sentinel_start(text)
    if _eos_start >= 0:
        text = text[:_eos_start].rstrip()

    # Cancellation metadata, not prose; ACP/TUI already suppress this sentinel, chat surfaces should too.
    # See #7921.
    if str(text).strip().startswith(INTERRUPT_WAITING_FOR_MODEL_PREFIX):
        return ""

    redacted = _redact_gateway_user_facing_secrets(str(text))
    if _looks_like_gateway_provider_error(
            redacted, terminal_failure=_turn_result_failed(agent_result)):
        return _gateway_provider_error_reply(redacted)
    return redacted

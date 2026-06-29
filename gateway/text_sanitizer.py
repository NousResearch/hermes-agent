"""Gateway text sanitization utilities.

Extracted from ``gateway/run.py`` to keep the gateway runner focused on
lifecycle management while this module owns all text filtering, redaction,
and provider-error classification that protects chat surfaces from raw
infrastructure noise and leaked secrets.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Platform classification
# ---------------------------------------------------------------------------

# Surfaces that consume gateway text programmatically (CLI/TUI "local"
# diagnostics, API JSON, webhook payloads) and therefore must keep RAW
# status/error text. EVERY other platform is a human-facing chat surface
# where operational lifecycle/provider-error noise (and any secrets in it)
# must be suppressed or sanitized. Widens #28533's Telegram-only filter to
# all chat gateways (#39293). Fail-closed: unknown/empty platform -> chat.
_GATEWAY_RAW_TEXT_PLATFORMS = frozenset(
    {"local", "api_server", "webhook", "msgraph_webhook"}
)


def _gateway_platform_value(platform: Any) -> str:
    """Return a normalized gateway platform value for enums or raw strings."""
    return str(getattr(platform, "value", platform) or "").strip().lower()


def _gateway_surface_passes_raw_text(platform: Any) -> bool:
    """True only for programmatic/local surfaces that must keep raw text."""
    return _gateway_platform_value(platform) in _GATEWAY_RAW_TEXT_PLATFORMS


# ---------------------------------------------------------------------------
# Provider-error / policy regexes
# ---------------------------------------------------------------------------

_GATEWAY_PROVIDER_ERROR_RE = re.compile(
    r"("  # infrastructure/provider error preambles, not ordinary assistant prose
    r"api\s+(?:call\s+)?failed"
    r"|provider\s+authentication\s+failed"
    r"|non-retryable\s+error"
    r"|rate\s+limited\s+after\s+\d+\s+retries"
    r"|error\s+code\s*:"
    r"|\bhttp\s*\d{3}\b"
    r"|incorrect\s+api\s+key"
    r"|invalid\s+api\s+key"
    r")",
    re.IGNORECASE,
)

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
    re.IGNORECASE,
)

_GATEWAY_AUTH_ERROR_RE = re.compile(
    r"(provider\s+authentication\s+failed|incorrect\s+api\s+key|invalid\s+api\s+key|\b401\b)",
    re.IGNORECASE,
)

_GATEWAY_RATE_LIMIT_RE = re.compile(
    r"(rate\s+limit|rate-limited|\b429\b|quota|usage\s+limit)",
    re.IGNORECASE,
)

_GATEWAY_PROVIDER_ERROR_SHAPE_RE = re.compile(
    r"^\s*(\W*\s*)?("
    r"api\s+(?:call\s+)?failed"
    r"|provider\s+authentication\s+failed"
    r"|non-retryable\s+error"
    r"|rate\s+limited\s+after\s+\d+\s+retries"
    r"|error\s+code\s*:"
    r"|http\s*\d{3}\b"
    r"|incorrect\s+api\s+key"
    r"|invalid\s+api\s+key"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Secret patterns (belt-and-suspenders with agent.redact)
# ---------------------------------------------------------------------------

_GATEWAY_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._\-]{20,}\b"),
)


# ---------------------------------------------------------------------------
# Transient network error classification
# ---------------------------------------------------------------------------

_TRANSIENT_NETWORK_ERROR_NAMES = frozenset({
    "TimedOut",
    "NetworkError",
    "ReadError",
    "WriteError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "RemoteProtocolError",
    "ServerDisconnectedError",
    "ClientConnectorError",
    "ClientOSError",
})


def _non_conversational_metadata(
    metadata: Optional[Dict[str, Any]] = None,
    *,
    platform: Any = None,
) -> Optional[Dict[str, Any]]:
    """Mark Discord lifecycle/status sends without changing other platforms."""
    if _gateway_platform_value(platform) != "discord":
        return metadata
    merged = dict(metadata or {})
    merged["non_conversational"] = True
    return merged


def _is_transient_network_error(exc: BaseException) -> bool:
    """Return True for transient network errors safe to log + swallow.

    The crash class targeted by #31066 / #31110: an unhandled Telegram
    ``TimedOut`` (or peer ``NetworkError`` / ``httpx`` connection error)
    propagating to the event loop and killing the entire gateway
    process. These are by definition transient — the next poll cycle or
    user action recovers — so they must never crash the process.

    Walk the exception cause chain so wrapped errors (e.g. PTB's
    ``NetworkError`` wrapping ``httpx.ConnectError``) are still
    classified. The chain is bounded to avoid pathological cycles.
    """
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    depth = 0
    while cur is not None and depth < 12:
        ident = id(cur)
        if ident in seen:
            break
        seen.add(ident)
        depth += 1
        name = type(cur).__name__
        if name in _TRANSIENT_NETWORK_ERROR_NAMES:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------


def _redact_gateway_user_facing_secrets(text: str) -> str:
    """Secret redaction before text can leave the gateway.

    Delegates to the authoritative ``agent.redact.redact_sensitive_text`` — the
    same Tirith-grade redactor already applied to logs, tool output, and
    approval-command prompts — so the outbound chat path masks the full
    credential set the startup banner promises ("chat responses are scrubbed
    before delivery"), not a divergent subset. ``force=True`` honors redaction
    even when ``security.redact_secrets`` is off, matching the
    ``_redact_approval_command`` reasoning (#23810).

    The narrow ``_GATEWAY_SECRET_PATTERNS`` set runs as a belt-and-suspenders
    second pass so nothing the gateway historically caught can regress, and so
    redaction still degrades gracefully if the import ever fails.
    """
    redacted = str(text or "")
    try:
        from agent.redact import redact_sensitive_text

        redacted = redact_sensitive_text(redacted, force=True)
    except Exception:
        # Fail-soft: fall back to the local pattern pass below rather than
        # letting a redactor import/error leak the raw text to chat.
        pass
    for pattern in _GATEWAY_SECRET_PATTERNS:
        redacted = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", redacted)
    return redacted


def _redact_approval_command(cmd: "str | None") -> str:
    """Redact credentials from a command before it goes into an approval prompt.

    Tirith's *findings* are already redacted, but the gateway approval prompt
    is built from the raw command string, so a credential-shaped value Tirith
    flagged would otherwise be echoed verbatim to the chat platform (#48456).
    Uses ``redact_sensitive_text(force=True)`` — the same Tirith-grade redactor
    — so the prompt honors redaction even when ``security.redact_secrets`` is
    off. Module-level so the wiring is unit-testable (the call site is a deeply
    nested gateway closure that cannot be driven directly).
    """
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(str(cmd or ""), force=True)


# ---------------------------------------------------------------------------
# Provider-error classification and reply mapping
# ---------------------------------------------------------------------------


def _gateway_provider_error_reply(text: str) -> str:
    """Map raw provider/API errors to a short user-safe Telegram reply."""
    if _GATEWAY_AUTH_ERROR_RE.search(text):
        return (
            "⚠️ Provider authentication failed. Check the configured credentials; "
            "raw provider details are in the gateway logs."
        )
    if _GATEWAY_PROVIDER_POLICY_RE.search(text):
        return (
            "⚠️ The model provider rejected the request. I kept the raw provider "
            "error out of chat; check gateway logs for details or try rephrasing."
        )
    if _GATEWAY_RATE_LIMIT_RE.search(text):
        return "⏱️ The model provider is rate-limiting requests. Please wait a moment and try again."
    return (
        "⚠️ The model provider failed after retries. I kept raw provider details "
        "out of chat; check gateway logs for diagnostics."
    )


def _looks_like_gateway_provider_error(text: str) -> bool:
    """True when text is infrastructure/provider failure, not normal content.

    Two heuristics combined so the rewrite only fires on actual provider
    error envelopes, not on assistant prose that happens to mention an
    HTTP status code:

    1. The text is short — real provider errors are 1–3 lines of envelope
       text; assistant answers are usually longer.
    2. AND the error marker appears at the start of the message (optionally
       behind a punctuation/symbol prefix), not buried mid-paragraph in an
       explanation like "HTTP 404 means 'not found' — ...".
    """
    if not text:
        return False
    body = str(text).strip()
    # Provider failure envelopes are short. Assistant answers that happen
    # to mention HTTP status codes ("HTTP 404 means...") tend to be longer.
    if len(body) > 400 or body.count("\n") > 4:
        return False
    return bool(_GATEWAY_PROVIDER_ERROR_SHAPE_RE.search(body))


# ---------------------------------------------------------------------------
# Final response / status sanitization
# ---------------------------------------------------------------------------


def _sanitize_gateway_final_response(platform: Any, text: str) -> str:
    """Sanitize final gateway replies before sending them to chat surfaces.

    Every human-facing chat surface (Telegram, WhatsApp, Discord, Slack,
    Signal, Matrix, plugin platforms, etc.) should receive concise, safe
    provider failure categories with secrets redacted instead of raw HTTP
    bodies, request IDs, leaked credentials, or policy text. Only programmatic
    surfaces in ``_GATEWAY_RAW_TEXT_PLATFORMS`` (CLI/TUI ``local`` diagnostics,
    API JSON, webhook payloads) keep the raw text unchanged.
    """
    if not text:
        return text
    if _gateway_surface_passes_raw_text(platform):
        return text

    redacted = _redact_gateway_user_facing_secrets(str(text))
    if _looks_like_gateway_provider_error(redacted):
        return _gateway_provider_error_reply(redacted)
    return redacted


def _prepare_gateway_status_message(
    platform: Any,
    event_type: str,
    message: str,
    *,
    noisy_pattern: Any = None,
) -> Optional[str]:
    """Filter/sanitize agent status callbacks before platform delivery.

    Local/CLI sessions keep the raw diagnostic stream. Messaging gateway
    surfaces should not receive transient auxiliary/compression chatter.

    Parameters
    ----------
    noisy_pattern:
        Optional compiled regex whose matches suppress the message on
        non-raw-text surfaces.  The gateway runner passes its own
        ``_TELEGRAM_NOISY_STATUS_RE`` here so this module stays free of
        Telegram-specific constants.
    """
    text = str(message or "").strip()
    if not text:
        return None
    if _gateway_surface_passes_raw_text(platform):
        return text

    text = _redact_gateway_user_facing_secrets(text)
    if noisy_pattern is not None and noisy_pattern.search(text):
        return None
    if _looks_like_gateway_provider_error(text):
        return _gateway_provider_error_reply(text)
    return text


# ---------------------------------------------------------------------------
# Notice rendering
# ---------------------------------------------------------------------------


def render_notice_line(notice) -> str:
    """Render an AgentNotice to a single plaintext line for messaging platforms.

    Messaging has no persistent status bar (unlike the TUI), so a notice is a
    one-shot standalone push. The notice policy already bakes the level glyph
    (⚠ / • / ✕ / ✓) into the text, and the TUI + CLI REPL render that text
    verbatim — so we emit it as-is here too. Prepending a per-level glyph would
    DOUBLE it ("⚠ ⚠ Credits 90% used", "⛔ ✕ Credit access paused"). Plaintext
    only — no markdown — so it renders uniformly across Telegram/Discord/Slack/
    SMS without per-platform escaping. Fail-soft: a malformed/empty notice
    degrades to "" rather than raising on the agent's callback path.
    """
    return str(getattr(notice, "text", "") or "").strip()

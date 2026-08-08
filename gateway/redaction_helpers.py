"""Gateway redaction / approval / provider-error helpers.

Extracted from ``gateway/run.py`` (slice 9 of #54962).  These helpers sit at
the gateway boundary: they redact credentials before text leaves the gateway,
render approval-prompt fallbacks, and map raw provider/API errors to short
user-safe chat replies.
"""

import re

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

_GATEWAY_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxapp-\d+-[A-Za-z0-9\-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._\-]{20,}\b"),
)


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


def _format_exec_approval_fallback(
    command: str,
    description: str,
    command_prefix: str,
    *,
    allow_permanent: bool = True,
    allow_session: bool = True,
    smart_denied: bool = False,
) -> str:
    """Render the text fallback from approval capabilities, not platform names."""
    cmd_preview = command[:200] + "..." if len(command) > 200 else command
    heading = "⚠️ **Dangerous command requires approval:**"
    if smart_denied:
        heading = "⚠️ **Smart DENY — owner override for one operation:**"

    choices = [f"Reply `{command_prefix}approve` to execute this one operation"]
    if not smart_denied and allow_session:
        choices.append(
            f"`{command_prefix}approve session` to approve this pattern for the session"
        )
        if allow_permanent:
            choices.append(f"`{command_prefix}approve always` to approve permanently")
    choices.append(f"`{command_prefix}deny` to cancel")
    return (
        f"{heading}\n```\n{cmd_preview}\n```\nReason: {description}\n\n"
        + ", ".join(choices[:-1]) + f", or {choices[-1]}."
    )


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

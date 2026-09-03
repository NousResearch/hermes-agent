"""Mandatory egress guardrail for outbound platform message bodies.

Hermes applies ``agent.redact`` on ingress (tool results into the model) and
on display/persistence — but historically never on the bytes actually sent to
Telegram/Slack/Discord/Matrix. A secret that reaches the assistant's final
response was delivered verbatim. This module closes that gap at the three
outbound choke points:

  * ``gateway/platforms/base.py`` ``_send_with_retry`` (gateway replies)
  * ``gateway/delivery.py`` ``DeliveryTransport.send`` (cron / DeliveryRouter)
  * ``tools/send_message_tool.py`` ``_send_via_adapter`` (agent-initiated)

Behavior:

  1. Redact the body with ``agent.redact.redact_sensitive_text(force=True)``.
     ``force=True`` because this is a safety boundary: the user-facing
     *display* redaction preference must not disable *egress* redaction.
  2. Re-check a normalized form of the body (ANSI escapes and zero-width
     characters stripped, NFKC-folded). This defeats obfuscation bypasses
     where a secret is split by styling sequences, zero-width joiners, or
     written in full-width forms. If the normalized form reveals a secret
     that the raw form hid, the *normalized redacted* body is sent instead —
     losing cosmetic styling is acceptable when it was hiding a credential.
  3. Consult plugin ``outbound_message`` middleware (fail-open per the
     middleware contract), which may further rewrite or block the send.

Steps 1–2 are fail-closed: if redaction itself raises, the send is blocked
(``EgressBlocked``) rather than delivered unexamined.

Opt-out: ``HERMES_EGRESS_GUARDRAIL=false`` disables the whole boundary,
mirroring the ``HERMES_REDACT_SECRETS`` env convention in ``agent/redact.py``.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
# ZWSP, ZWNJ, ZWJ, word joiner, BOM/zero-width no-break space
_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")


def _enabled() -> bool:
    return os.getenv("HERMES_EGRESS_GUARDRAIL", "true").lower() in {
        "1", "true", "yes", "on",
    }


class EgressBlocked(Exception):
    """Raised when the egress boundary vetoes an outbound send."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def normalize_for_detection(text: str) -> str:
    """Strip ANSI/zero-width characters and NFKC-fold for secret detection."""
    stripped = _ANSI_RE.sub("", text)
    stripped = _ZERO_WIDTH_RE.sub("", stripped)
    return unicodedata.normalize("NFKC", stripped)


BLOCK_ERROR = "message vetoed by egress guardrail"
"""Stable error text for blocked sends.

Deliberately free of transport-error vocabulary ("forbidden", "not found",
"unauthorized"...): SendResult errors flow into ``classify_send_error`` /
``DeadTargetRegistry`` substring matching, and a *content* veto must never
mark a delivery *target* dead. Callers should log the detailed reason but
put only this constant in error strings.
"""


def guard_outbound_text(
    text: str,
    *,
    platform: str = "",
    session_key: str = "",
    category: str = "message",
    apply_middleware: bool = True,
) -> str:
    """Redact/inspect an outbound message body immediately before send.

    Returns the (possibly redacted or middleware-rewritten) body to deliver.
    Raises ``EgressBlocked`` when the send must not happen.

    The built-in redaction pass is idempotent, so layered choke points may
    repeat it safely. Plugin ``outbound_message`` middleware is NOT assumed
    idempotent (a footer-appending plugin must not run twice on one body):
    exactly one boundary per delivery path runs with ``apply_middleware=True``
    — the platform-adapter ``send``/``edit_message`` wrapper for in-process
    deliveries, and the relay branch of ``DeliveryTransport.send`` (which
    never reaches a wrapped adapter method). Pre-chunking or defense-in-depth
    call sites pass ``apply_middleware=False`` for redaction only.
    """
    if not text or not isinstance(text, str) or not _enabled():
        return text

    try:
        from agent.redact import redact_sensitive_text

        guarded = redact_sensitive_text(text, force=True)
        normalized = normalize_for_detection(guarded)
        if normalized != guarded:
            # Body contains styling/zero-width/compat characters that could be
            # hiding a split secret. Detect on the normalized form; if that
            # reveals anything new, prefer the de-obfuscated redacted body.
            renormalized = redact_sensitive_text(normalized, force=True)
            if renormalized != normalized:
                logger.warning(
                    "[egress] obfuscated secret detected in outbound %s for %s; "
                    "sending normalized redacted body",
                    category, platform or "unknown platform",
                )
                guarded = renormalized
    except EgressBlocked:
        raise
    except Exception as exc:
        # Fail closed: never deliver a body the redactor could not examine.
        logger.error(
            "[egress] redaction failed for outbound %s on %s; blocking send",
            category, platform or "unknown platform", exc_info=True,
        )
        raise EgressBlocked(f"egress redaction failed: {exc}") from exc

    if not apply_middleware:
        return guarded

    try:
        from hermes_cli.middleware import apply_outbound_message_middleware

        result = apply_outbound_message_middleware(
            guarded,
            platform=platform,
            session_key=session_key,
            category=category,
        )
    except Exception:
        # Plugin middleware is fail-open by contract; a broken dispatch layer
        # must not turn the built-in redaction pass into a delivery outage.
        logger.warning("[egress] outbound_message middleware dispatch failed",
                       exc_info=True)
        return guarded

    if result.blocked:
        raise EgressBlocked(result.block_reason)
    return result.text

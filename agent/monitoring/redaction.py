"""Redaction applied to monitoring data before egress.

Secrets are always redacted, on every export path; no setting disables this.
Wraps ``agent/redact.py::redact_sensitive_text(force=True)`` and fails CLOSED:
if the redactor cannot run, the raw string is never emitted.

``redact_for_export(text, content_mode="pii")`` additionally scrubs e-mail
addresses, phone numbers, and UUID-shaped identifiers — the gateway
diagnostics path always uses this mode, so log-derived messages leave the
process with secrets AND PII already removed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Content-redaction strengths for any content that IS exported.
CONTENT_NONE = "none"   # drop content entirely (structural telemetry only)
CONTENT_PII = "pii"     # codec-aware PII redaction on exported content
CONTENT_MODES = {CONTENT_NONE, CONTENT_PII}

# ── PII patterns (applied only in CONTENT_PII mode, on content that is exported) ──
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# E.164-ish and common separators; conservative to avoid nuking code/IDs.
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?\d{3}[\s.\-]?\d{3,4}(?:[\s.\-]?\d{2,4})?(?!\w)"
)
# Long opaque hex/uuid-ish user identifiers.
_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")


def _secret_redact(text: Optional[str]) -> Optional[str]:
    """Always-on secret redaction. force=True so user config can't disable it."""
    if text is None:
        return None
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(str(text), force=True)
    except Exception:
        # Fail CLOSED: if the redactor can't run, do not emit the raw string.
        return "[redaction-unavailable]"


def _pii_redact(text: str) -> str:
    text = _EMAIL_RE.sub("[email]", text)
    text = _UUID_RE.sub("[id]", text)
    text = _PHONE_RE.sub("[phone]", text)
    return text


def redact_for_export(
    text: Optional[str],
    *,
    content_mode: str = CONTENT_NONE,
) -> Optional[str]:
    """Redact a single content string for export.

    Secrets are ALWAYS stripped. Then PII is stripped when content_mode is 'pii'.
    Callers gate *whether content is exported at all* via telemetry.trajectories
    (see ``content_export_enabled``); this function only scrubs content that the
    caller has already decided to export.
    """
    redacted = _secret_redact(text)
    if redacted is None:
        return None
    if content_mode == CONTENT_PII:
        redacted = _pii_redact(redacted)
    return redacted


__all__ = [
    "CONTENT_NONE",
    "CONTENT_PII",
    "CONTENT_MODES",
    "redact_for_export",
]

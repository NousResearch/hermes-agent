"""Redaction applied to monitoring data before egress.

One unconditional scrub, no modes, no knobs. Every string that leaves the
process passes through ``redact_for_export``:

  * Secrets first — wraps ``agent/redact.py::redact_sensitive_text(force=True)``
    plus bearer/token-shape patterns, and fails CLOSED: if the redactor cannot
    run, the raw string is never emitted.
  * PII second — e-mail addresses, phone numbers, and UUID-shaped identifiers
    are rewritten to ``[email]`` / ``[phone]`` / ``[id]``.

There is deliberately no setting to weaken this. The monitoring plane is
content-free by design: rendered log messages are not exported, and bounded
structured strings are still scrubbed as defense-in-depth. This redactor also
remains available for a future, explicitly gated redacted-message detail mode.
"""

from __future__ import annotations

import re
from typing import Optional

# ── secret shapes (belt-and-suspenders on top of agent/redact.py) ───────────
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+\-/]+=*", re.IGNORECASE)
_TOKEN_RE = re.compile(
    r"\b(xox[baprs]-[A-Za-z0-9-]+|sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,})\b"
)
_SECRET_LITERAL_RE = re.compile(r"\*{3,}")
_BEARER_RESIDUE_RE = re.compile(r"\bBearer\s+\[[^\]]+\]", re.IGNORECASE)

# ── PII shapes ───────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# E.164-ish and common separators; conservative to avoid nuking code/IDs.
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?\d{3}[\s.\-]?\d{3,4}(?:[\s.\-]?\d{2,4})?(?!\w)"
)
# Long opaque hex/uuid-ish user identifiers.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def _secret_redact(text: str) -> str:
    """Always-on secret redaction. force=True so user config can't disable it."""
    try:
        from agent.redact import redact_sensitive_text
        out = redact_sensitive_text(text, force=True)
    except Exception:
        # Fail CLOSED: if the redactor can't run, do not emit the raw string.
        return "[redaction-unavailable]"
    out = _BEARER_RE.sub("[redacted]", out)
    out = _TOKEN_RE.sub("[redacted]", out)
    out = _SECRET_LITERAL_RE.sub("[redacted]", out)
    out = _BEARER_RESIDUE_RE.sub("[redacted]", out)
    return out


def redact_for_export(text: Optional[str]) -> Optional[str]:
    """Scrub a string for egress: secrets, then PII. Unconditional."""
    if text is None:
        return None
    out = _secret_redact(str(text))
    out = _EMAIL_RE.sub("[email]", out)
    out = _UUID_RE.sub("[id]", out)
    out = _PHONE_RE.sub("[phone]", out)
    return out


__all__ = [
    "redact_for_export",
]

"""Redaction applied to monitoring data before egress.

One unconditional scrub, no modes, no knobs. Every string that leaves the process passes
through ``redact_for_export``: secrets via ``agent/redact.py::redact_for_egress`` (the single
pattern source; fails CLOSED so a broken redactor never emits the raw string), then PII
(e-mail, phone, UUID-shaped ids -> ``[email]`` / ``[phone]`` / ``[id]``).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agent.redact import REDACTION_UNAVAILABLE as UNAVAILABLE, redact_for_egress

# ── PII shapes ───────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# E.164-ish and common separators; conservative to avoid nuking code/IDs.
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?\d{3}[\s.\-]?\d{3,4}(?:[\s.\-]?\d{2,4})?(?!\w)"
)
_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")


def redact_for_export(text: Optional[str]) -> Optional[str]:
    """Scrub a string for egress: secrets, then PII. Unconditional."""
    if text is None:
        return None
    out = redact_for_egress(str(text))
    out = _EMAIL_RE.sub("[email]", out)
    out = _UUID_RE.sub("[id]", out)
    out = _PHONE_RE.sub("[phone]", out)
    return out


def redact_bounded(raw: Any, *, limit: int = 500, empty: str = "[redacted]", unavailable: str = UNAVAILABLE) -> str:
    """Redact ``str(raw or "")`` and length-bound it; ``empty`` replaces an empty
    result, ``unavailable`` is returned if redaction itself raises."""
    try:
        return (redact_for_export(str(raw or "")) or empty)[:limit]
    except Exception:
        return unavailable


__all__ = ["redact_for_export", "redact_bounded"]

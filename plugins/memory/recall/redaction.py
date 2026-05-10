"""Conservative redaction helpers for the Recall memory provider."""

from __future__ import annotations

import re

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-proj-[A-Za-z0-9_-]{16,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]+=*", re.IGNORECASE),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(
        r"\b([A-Z0-9_]*(?:API_)?(?:KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)\s*=\s*([^\s]+)",
        re.IGNORECASE,
    ),
)


def redact_text(text: str | None) -> str:
    """Return text with common secret-shaped substrings removed.

    The redactor is intentionally best-effort and conservative. It is used for
    memory previews and recalled context, not as a security boundary.
    """
    if not text:
        return ""
    redacted = str(text)
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.startswith("\\b([A-Z0-9_]"):
            redacted = pattern.sub(lambda m: f"{m.group(1)}=[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted

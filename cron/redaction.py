"""Small, dependency-free credential redaction shared by cron persistence and status."""

from __future__ import annotations

import re
import traceback

_REDACTED_CREDENTIAL = "[redacted credential]"
_CREDENTIAL_PATTERNS = (
    re.compile(r"\b(?:proxy-)?authorization\s*:\s*(?:bearer\s+)?\S+", re.IGNORECASE),
    re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]{8,}\b", re.IGNORECASE),
    re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9_]{8,}\b", re.IGNORECASE),
    re.compile(r"\b(?:sk|xai|anthropic|AIza|gsk|r8|pplx)[-_][A-Za-z0-9_-]{8,}\b", re.IGNORECASE),
    re.compile(r"\bAIzaSy[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b(?:hf|openai)[_-][A-Za-z0-9_-]{8,}\b", re.IGNORECASE),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z0-9 ]*(?:PRIVATE KEY|KEY)-----", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|token|secret|password)\s*[:=]\s*"
        r"(?:bearer\s+)?[A-Za-z0-9._~+/=-]{8,}\b",
        re.IGNORECASE,
    ),
)


def contains_credential(value: str) -> bool:
    """Return whether untrusted free text contains a credential-shaped value."""
    return any(pattern.search(value) for pattern in _CREDENTIAL_PATTERNS)


def redact_credential_text(value: str) -> str:
    """Replace a credential-bearing free-text value before durable/status exposure."""
    return _REDACTED_CREDENTIAL if contains_credential(value) else value


def redact_exception_detail(error: BaseException | str) -> tuple[str, bool]:
    """Return safe exception text and whether its complete traceback is safe to log."""
    detail = str(error)
    trace_text = detail
    if isinstance(error, BaseException):
        trace_text = "".join(
            traceback.TracebackException.from_exception(error, capture_locals=False).format(chain=True)
        )
    return redact_credential_text(detail), not contains_credential(trace_text)

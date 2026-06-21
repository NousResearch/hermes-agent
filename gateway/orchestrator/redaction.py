"""Secret redaction helpers for external-agent outputs."""

from __future__ import annotations

import re

from .registry import AgentSpec

_REDACTED = "[REDACTED]"

# Keep these deliberately conservative: Phase 1~2 prefer false positives over
# leaking user credentials into orchestrator logs or reports.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_.-]{6,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_.-]{6,}"),
    re.compile(r"AKIA[A-Z0-9_.-]{6,}"),
    re.compile(
        r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD|COOKIE)[A-Z0-9_]*)\s*=\s*([^\s'\"]+)"
    ),
    re.compile(r"\b[a-fA-F0-9]{32,}\b"),
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),
)


def _replace_key_value(match: re.Match[str]) -> str:
    if match.lastindex and match.lastindex >= 2:
        return f"{match.group(1)}={_REDACTED}"
    return _REDACTED


def redact_text(text: str | None) -> str:
    """Return ``text`` with common credential/token shapes masked."""

    value = "" if text is None else str(text)
    for pattern in _PATTERNS:
        if pattern.groups >= 2:
            value = pattern.sub(_replace_key_value, value)
        else:
            value = pattern.sub(_REDACTED, value)
    return value


def redact_for(spec: AgentSpec, text: str | None) -> str:
    """Redact output for an agent, suppressing sensitive wrappers entirely."""

    if spec.secrets:
        return "<suppressed>"
    return redact_text(text)

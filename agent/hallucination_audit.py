"""Deterministic guardrails for high-risk operational answers.

This module deliberately avoids an LLM judge.  It catches the failure mode that
hurts most in ops chat: a polished answer about services, cron, trading, Docker,
hardware, or live status with no tool evidence and no admission of uncertainty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


OPERATIONAL_QUESTION_RE = re.compile(
    r"\b("
    r"cron|cronjobs?|timers?|systemd|services?|daemon|process(?:es)?|"
    r"processen?|docker|containers?|freqtrade|freqai|trading|trader|"
    r"grafana|prometheus|"
    r"status|health|gezond|draait|running|actief|online|offline|"
    r"gpu|ram|memory|geheugen|disk|schijf|poort|port|endpoints?|apis?|"
    r"telemetry|telemetrie|news|nieuws"
    r")\b",
    re.IGNORECASE,
)

UNCERTAINTY_MARKER_RE = re.compile(
    r"(geen directe toegang|niet verifi[eë]ren|niet controleren|niet checken|"
    r"weet ik niet zeker|kan ik niet bevestigen|geen bewijs|onzeker|unknown|"
    r"cannot verify|do not have access|don't have access)",
    re.IGNORECASE,
)

EVIDENCE_MARKER_RE = re.compile(
    r"(bewijs\s*:|evidence\s*:|bron\s*:|source\s*:|tool\s*:|"
    r"terminal\s*\(|cronjob\s*\(|session_search|systemctl|journalctl|"
    r"docker\s+ps|pgrep|ps\s+aux|ss\s+-|curl\s+|/health)",
    re.IGNORECASE,
)

OPERATIONAL_CLAIM_RE = re.compile(
    r"("
    r"\b\d+(?:[.,]\d+)?\s*(?:%|gb|mb|ms|s|sec|seconds|min|uur|hour|c|°c)\b|"
    r"\b(?:actief|running|draait|loopt|online|offline|connected|ready|"
    r"nominal|stable|gezond|ok|up|down|operationeel)\b|"
    r"\b(?:docker|container|freqtrade|freqai|grafana|prometheus|"
    r"trinity_tasks|systemd|timer|cronjob|gpu|ram|disk|schijf)\b"
    r")",
    re.IGNORECASE,
)

UNVERIFIED_OPERATIONAL_WARNING = (
    "[Fact-check: niet geverifieerd]\n"
    "Dit Hermes-antwoord bevat operationele claims zonder zichtbare toolbron "
    "of toolgebruik in deze beurt. Behandel het als hypothese; vraag om een "
    "korte `bewijs:`-regel met de gebruikte tool-output.\n\n"
)


@dataclass(frozen=True)
class AuditResult:
    """Result of a deterministic reply audit."""

    text: str
    note: str = ""
    warned: bool = False


def _content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(value)


def _tool_used_after_last_user(messages: Iterable[dict[str, Any]] | None) -> bool:
    """Return True when the current turn includes at least one tool result."""
    if not messages:
        return False
    rows = list(messages)
    last_user_idx = -1
    for idx, row in enumerate(rows):
        if isinstance(row, dict) and row.get("role") == "user":
            last_user_idx = idx
    if last_user_idx < 0:
        return False
    for row in rows[last_user_idx + 1 :]:
        if not isinstance(row, dict):
            continue
        if row.get("role") == "tool":
            return True
        if row.get("role") == "assistant" and row.get("tool_calls"):
            return True
    return False


def audit_operational_reply(
    user_text: Any,
    reply_text: str,
    *,
    messages: Iterable[dict[str, Any]] | None = None,
    enabled: bool = True,
) -> AuditResult:
    """Warn when a high-risk operational answer lacks evidence.

    The function keeps the original answer intact and only prepends a warning.
    It does not try to prove correctness; it prevents unverified operational
    claims from looking authoritative.
    """
    if not enabled:
        return AuditResult(reply_text)

    prompt = _content_to_text(user_text)
    if not OPERATIONAL_QUESTION_RE.search(prompt or ""):
        return AuditResult(reply_text)

    if reply_text.startswith(UNVERIFIED_OPERATIONAL_WARNING):
        return AuditResult(reply_text, note="hallucination_audit_unverified", warned=True)

    if UNCERTAINTY_MARKER_RE.search(reply_text):
        return AuditResult(reply_text, note="hallucination_audit_uncertainty_ok")

    if EVIDENCE_MARKER_RE.search(reply_text):
        return AuditResult(reply_text, note="hallucination_audit_evidence_seen")

    if _tool_used_after_last_user(messages):
        return AuditResult(reply_text, note="hallucination_audit_tool_seen")

    if OPERATIONAL_CLAIM_RE.search(reply_text):
        return AuditResult(
            UNVERIFIED_OPERATIONAL_WARNING + reply_text,
            note="hallucination_audit_unverified",
            warned=True,
        )

    return AuditResult(reply_text, note="hallucination_audit_no_operational_claim")

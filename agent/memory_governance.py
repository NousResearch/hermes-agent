"""Minimal memory governance gate for durable storage decisions.

This module classifies candidate information before anything writes to
persistent memory, Obsidian, Honcho, skills, or project state. It is a review
surface only: callers get a decision and may enqueue that decision for later
human or agent inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write


STORE_VERSION = 1
QUEUE_DIRNAME = "memory_governance"
QUEUE_FILENAME = "review_queue.json"
QUEUE_STATUS_PENDING = "pending_review"

_LOCK = threading.RLock()


class GovernanceLabel(str, Enum):
    """Candidate destination labels."""

    HERMES_MEMORY = "HERMES_MEMORY"
    HONCHO_RUNTIME_ONLY = "HONCHO_RUNTIME_ONLY"
    OBSIDIAN_PROMOTE = "OBSIDIAN_PROMOTE"
    SKILL = "SKILL"
    PROJECT_STATE = "PROJECT_STATE"
    SESSION_ONLY = "SESSION_ONLY"
    REJECT = "REJECT"


@dataclass(frozen=True)
class MemoryGovernanceDecision:
    """Review decision returned by the memory governance classifier."""

    label: GovernanceLabel
    confidence: float
    reason: str
    requires_approval: bool
    destructive: bool
    source_type: str
    candidate_summary: str
    suggested_artifact_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "label": self.label.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "requires_approval": self.requires_approval,
            "destructive": self.destructive,
            "source_type": self.source_type,
            "candidate_summary": self.candidate_summary,
        }
        if self.suggested_artifact_path:
            data["suggested_artifact_path"] = self.suggested_artifact_path
        return data


_SECRET_KEY_RE = re.compile(
    r"\b[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE_KEY|ACCESS_KEY)\b\s*[:=]\s*([^\s,;]+)",
    re.IGNORECASE,
)
_SECRET_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9_]{12,}|xox[baprs]-[A-Za-z0-9-]{12,})\b"
)
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")

_DESTRUCTIVE_RE = re.compile(
    r"\b(remove|delete|erase|purge|prune|compact|compaction|wipe)\b.*\b(memory|memories|USER\.md|MEMORY\.md|entries)\b",
    re.IGNORECASE,
)
_PROJECT_STATE_RE = re.compile(
    r"\b(project state|working state|cursor|last checked|last run|open alert|alert count|checkpoint|progress|current task|next step)\b",
    re.IGNORECASE,
)
_PROCEDURE_RE = re.compile(
    r"\b(checklist|procedure|runbook|workflow|steps?:|playbook|when .* then|before .* then)\b",
    re.IGNORECASE,
)
_OBSIDIAN_RE = re.compile(
    r"\b(adr|canonical project decision|research synthesis|source evidence|evidence:|decision record|architecture decision)\b",
    re.IGNORECASE,
)
_BOOT_CRITICAL_RE = re.compile(
    r"\b(user prefers|umbbi prefers|user is|umbbi is|profile role|maintainer|timezone|boot-critical|default preference|always|never|canonical curated knowledge)\b",
    re.IGNORECASE,
)
_SHORT_REPLY_RE = re.compile(
    r"^(yes|no|ok|okay|sure|go ahead|continue|proceed|run it|do it|그거|계속|진행|확인)\b",
    re.IGNORECASE,
)


def memory_governance_queue_path() -> Path:
    """Return the active profile's memory governance review queue path."""

    return get_hermes_home() / QUEUE_DIRNAME / QUEUE_FILENAME


def classify_memory_candidate(
    candidate: str,
    *,
    source_type: str = "unknown",
    destructive: Optional[bool] = None,
) -> MemoryGovernanceDecision:
    """Classify candidate information before durable storage decisions.

    The classifier is deliberately heuristic and conservative. It does not
    write memory, write Obsidian, modify skills, or update project state.
    """

    text = (candidate or "").strip()
    normalized_source = (source_type or "unknown").strip() or "unknown"
    source_lower = normalized_source.lower()
    summary = _candidate_summary(text)

    if not text:
        return _decision(
            GovernanceLabel.REJECT,
            0.9,
            "Empty candidates cannot be reviewed for durable storage.",
            normalized_source,
            summary,
        )

    if _looks_like_secret(text):
        return _decision(
            GovernanceLabel.REJECT,
            1.0,
            "Secret-looking text or raw credentials must not be stored in durable memory or review artifacts.",
            normalized_source,
            summary,
        )

    destructive_detected = bool(destructive) or bool(_DESTRUCTIVE_RE.search(text))
    if destructive_detected:
        return _decision(
            GovernanceLabel.REJECT,
            0.95,
            "Destructive memory compaction or removal proposals require explicit approval before any action.",
            normalized_source,
            summary,
            requires_approval=True,
            destructive=True,
        )

    if "cron" in source_lower:
        return _decision(
            GovernanceLabel.SESSION_ONLY,
            0.9,
            "Cron follow-up replies are transient execution context, not persistent user memory.",
            normalized_source,
            summary,
        )

    if "session" in source_lower and ("progress" in source_lower or "reply" in source_lower):
        return _decision(
            GovernanceLabel.SESSION_ONLY,
            0.85,
            "Transient session progress is not persistent user memory.",
            normalized_source,
            summary,
        )

    if "money flow radar" in text.lower() or _is_project_state(text, source_lower):
        return _decision(
            GovernanceLabel.PROJECT_STATE,
            0.9,
            "Project working state belongs in project state storage and is not USER memory.",
            normalized_source,
            summary,
            suggested_artifact_path=".omx/state/project-state.json",
        )

    if _is_boot_critical(text, source_lower):
        return _decision(
            GovernanceLabel.HERMES_MEMORY,
            0.85,
            "Stable user preference, profile role, or environment fact appears boot-critical for future Hermes sessions.",
            normalized_source,
            summary,
            requires_approval=True,
        )

    if _PROCEDURE_RE.search(text) or "procedure" in source_lower or "workflow" in source_lower:
        return _decision(
            GovernanceLabel.SKILL,
            0.85,
            "Reusable procedure or checklist belongs in a skill, not persistent user memory.",
            normalized_source,
            summary,
            requires_approval=True,
            suggested_artifact_path="skills/memory-governance/SKILL.md",
        )

    if _OBSIDIAN_RE.search(text) or "research" in source_lower or "synthesis" in source_lower:
        return _decision(
            GovernanceLabel.OBSIDIAN_PROMOTE,
            0.85,
            "Canonical project decisions, research syntheses, and sourced evidence should be promoted to curated Obsidian knowledge.",
            normalized_source,
            summary,
            requires_approval=True,
            suggested_artifact_path="Obsidian/Hermes/Memory Governance.md",
        )

    if "honcho" in source_lower or "runtime" in source_lower:
        return _decision(
            GovernanceLabel.HONCHO_RUNTIME_ONLY,
            0.75,
            "Runtime recall context should stay in Honcho-style runtime memory rather than durable Hermes memory.",
            normalized_source,
            summary,
        )

    if _SHORT_REPLY_RE.search(text) or len(text) <= 120:
        return _decision(
            GovernanceLabel.SESSION_ONLY,
            0.75,
            "Short follow-up answers and transient task context are session-only, not persistent user memory.",
            normalized_source,
            summary,
        )

    return _decision(
        GovernanceLabel.SESSION_ONLY,
        0.55,
        "No durable-storage rule matched; keep the candidate in session context until a clearer artifact destination exists.",
        normalized_source,
        summary,
    )


def enqueue_memory_governance_review(
    decision: MemoryGovernanceDecision,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    created_at: Optional[datetime] = None,
) -> dict[str, Any]:
    """Append a classifier decision to the profile-local review queue."""

    now = created_at or datetime.now(timezone.utc)
    item: dict[str, Any] = {
        "id": f"mgq_{uuid.uuid4().hex[:16]}",
        "created_at": _format_time(now),
        "status": QUEUE_STATUS_PENDING,
        "decision": decision.to_dict(),
    }
    if metadata:
        item["metadata"] = _redact_value(dict(metadata))

    with _LOCK:
        items = _read_queue_unlocked()
        items.append(item)
        _write_queue_unlocked(items)
    return item


def load_memory_governance_review_queue() -> list[dict[str, Any]]:
    """Return all profile-local memory governance review queue items."""

    with _LOCK:
        return _read_queue_unlocked()


def _decision(
    label: GovernanceLabel,
    confidence: float,
    reason: str,
    source_type: str,
    candidate_summary: str,
    *,
    requires_approval: bool = False,
    destructive: bool = False,
    suggested_artifact_path: Optional[str] = None,
) -> MemoryGovernanceDecision:
    return MemoryGovernanceDecision(
        label=label,
        confidence=max(0.0, min(1.0, confidence)),
        reason=reason,
        requires_approval=requires_approval,
        destructive=destructive,
        source_type=source_type,
        candidate_summary=candidate_summary,
        suggested_artifact_path=suggested_artifact_path,
    )


def _is_project_state(text: str, source_lower: str) -> bool:
    if "project_state" in source_lower or (
        "project" in source_lower and ("update" in source_lower or "state" in source_lower)
    ):
        return True
    return bool(_PROJECT_STATE_RE.search(text)) and bool(
        re.search(r"\b(project|money flow radar|state)\b", text, re.IGNORECASE)
    )


def _is_boot_critical(text: str, source_lower: str) -> bool:
    if "preference" in source_lower or "profile" in source_lower or "environment" in source_lower:
        return bool(_BOOT_CRITICAL_RE.search(text))
    return bool(_BOOT_CRITICAL_RE.search(text)) and (
        "user" in text.lower() or "umbbi" in text.lower() or "hermes" in text.lower()
    )


def _looks_like_secret(text: str) -> bool:
    return bool(
        _SECRET_KEY_RE.search(text)
        or _SECRET_TOKEN_RE.search(text)
        or _PRIVATE_KEY_RE.search(text)
    )


def _candidate_summary(text: str, *, limit: int = 240) -> str:
    clean = _redact_text(" ".join((text or "").split()))
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _redact_text(text: str) -> str:
    redacted = _SECRET_KEY_RE.sub(lambda m: m.group(0).replace(m.group(1), "[REDACTED]"), text)
    redacted = _SECRET_TOKEN_RE.sub("[REDACTED]", redacted)
    redacted = _PRIVATE_KEY_RE.sub("-----BEGIN [REDACTED]-----", redacted)
    return redacted


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return _candidate_summary(value, limit=200)
    if isinstance(value, Mapping):
        return {str(k): _redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, tuple):
        return [_redact_value(v) for v in value]
    return value


def _read_queue_unlocked() -> list[dict[str, Any]]:
    path = memory_governance_queue_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("items", [])
    else:
        items = []
    return [dict(item) for item in items if isinstance(item, dict)]


def _write_queue_unlocked(items: list[dict[str, Any]]) -> None:
    atomic_json_write(
        memory_governance_queue_path(),
        {
            "version": STORE_VERSION,
            "items": items,
        },
        sort_keys=True,
    )


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "GovernanceLabel",
    "MemoryGovernanceDecision",
    "classify_memory_candidate",
    "enqueue_memory_governance_review",
    "load_memory_governance_review_queue",
    "memory_governance_queue_path",
]

"""Structured self-evolution ledger for Hermes.

The background review agent can update free-form memory and skills.  This
module adds a deterministic layer for lessons learned from mistakes: record,
deduplicate, recall, and retire actionable patterns without relying on an LLM
to maintain file structure correctly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

LEDGER_VERSION = 1
DEFAULT_LEDGER_RELATIVE_PATH = Path("evolution") / "lessons.jsonl"
_MAX_TEXT_CHARS = 4000
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_\-./]+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvolutionLesson:
    id: str
    mistake: str
    lesson: str
    trigger: str = ""
    fix: str = ""
    tags: list[str] = field(default_factory=list)
    severity: str = "medium"
    evidence: list[str] = field(default_factory=list)
    status: str = "active"
    source: str = "agent"
    confidence: float = 0.7
    occurrences: int = 1
    created_at: str = field(default_factory=_now)
    last_seen_at: str = field(default_factory=_now)
    resolved_at: str = ""
    outcome: str = ""
    version: int = LEDGER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mistake": self.mistake,
            "lesson": self.lesson,
            "trigger": self.trigger,
            "fix": self.fix,
            "tags": list(self.tags),
            "severity": self.severity,
            "evidence": list(self.evidence),
            "status": self.status,
            "source": self.source,
            "confidence": self.confidence,
            "occurrences": self.occurrences,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
            "resolved_at": self.resolved_at,
            "outcome": self.outcome,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvolutionLesson":
        return cls(
            id=str(data.get("id") or ""),
            mistake=_clean_text(data.get("mistake") or ""),
            lesson=_clean_text(data.get("lesson") or ""),
            trigger=_clean_text(data.get("trigger") or ""),
            fix=_clean_text(data.get("fix") or ""),
            tags=_normalize_tags(data.get("tags") or []),
            severity=_normalize_severity(data.get("severity") or "medium"),
            evidence=_normalize_evidence(data.get("evidence") or []),
            status=str(data.get("status") or "active"),
            source=str(data.get("source") or "agent"),
            confidence=_normalize_confidence(data.get("confidence", 0.7)),
            occurrences=max(1, int(data.get("occurrences") or 1)),
            created_at=str(data.get("created_at") or _now()),
            last_seen_at=str(data.get("last_seen_at") or _now()),
            resolved_at=str(data.get("resolved_at") or ""),
            outcome=_clean_text(data.get("outcome") or ""),
            version=int(data.get("version") or LEDGER_VERSION),
        )


def default_ledger_path() -> Path:
    return get_hermes_home() / DEFAULT_LEDGER_RELATIVE_PATH


def record_lesson(
    *,
    mistake: str,
    lesson: str,
    trigger: str = "",
    fix: str = "",
    tags: list[str] | None = None,
    evidence: list[str] | str | None = None,
    severity: str = "medium",
    source: str = "agent",
    confidence: float = 0.7,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Upsert a lesson and return a compact operation result."""
    mistake = _clean_text(mistake)
    lesson = _clean_text(lesson)
    if not mistake or not lesson:
        return {"success": False, "error": "mistake and lesson are required"}

    path = ledger_path or default_ledger_path()
    lessons = _read_lessons(path)
    lesson_id = _fingerprint(mistake=mistake, lesson=lesson, tags=tags or [])
    now = _now()

    existing = next((item for item in lessons if item.id == lesson_id), None)
    if existing is None:
        entry = EvolutionLesson(
            id=lesson_id,
            mistake=mistake,
            lesson=lesson,
            trigger=_clean_text(trigger),
            fix=_clean_text(fix),
            tags=_normalize_tags(tags or []),
            evidence=_normalize_evidence(evidence or []),
            severity=_normalize_severity(severity),
            source=source or "agent",
            confidence=_normalize_confidence(confidence),
            created_at=now,
            last_seen_at=now,
        )
        lessons.append(entry)
        action = "created"
    else:
        existing.occurrences += 1
        existing.last_seen_at = now
        existing.status = "active"
        if trigger:
            existing.trigger = _clean_text(trigger)
        if fix:
            existing.fix = _clean_text(fix)
        existing.tags = sorted(set(existing.tags) | set(_normalize_tags(tags or [])))
        existing.evidence = _merge_limited(
            existing.evidence,
            _normalize_evidence(evidence or []),
            limit=8,
        )
        existing.severity = _max_severity(existing.severity, severity)
        existing.confidence = max(existing.confidence, _normalize_confidence(confidence))
        entry = existing
        action = "updated"

    _write_lessons(path, lessons)
    return {"success": True, "action": action, "lesson": entry.to_dict()}


def recall_lessons(
    *,
    query: str = "",
    tags: list[str] | None = None,
    limit: int = 5,
    include_resolved: bool = False,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Recall lessons by simple token/tag relevance."""
    path = ledger_path or default_ledger_path()
    lessons = _read_lessons(path)
    wanted_tags = set(_normalize_tags(tags or []))
    scored: list[tuple[int, EvolutionLesson]] = []
    for item in lessons:
        if item.status != "active" and not include_resolved:
            continue
        if wanted_tags and not wanted_tags.intersection(item.tags):
            continue
        score = _score_lesson(item, query=query)
        if query and score <= 0:
            continue
        scored.append((score, item))

    scored.sort(
        key=lambda pair: (
            pair[0],
            _severity_rank(pair[1].severity),
            pair[1].occurrences,
            pair[1].last_seen_at,
        ),
        reverse=True,
    )
    max_items = max(1, min(int(limit or 5), 20))
    selected = [item.to_dict() for _, item in scored[:max_items]]
    return {"success": True, "count": len(selected), "lessons": selected}


def list_lessons(
    *,
    status: str = "active",
    limit: int = 20,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    path = ledger_path or default_ledger_path()
    lessons = _read_lessons(path)
    if status != "all":
        lessons = [item for item in lessons if item.status == status]
    lessons.sort(key=lambda item: item.last_seen_at, reverse=True)
    max_items = max(1, min(int(limit or 20), 100))
    return {
        "success": True,
        "count": min(len(lessons), max_items),
        "lessons": [item.to_dict() for item in lessons[:max_items]],
    }


def resolve_lesson(
    *,
    lesson_id: str,
    outcome: str = "",
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    path = ledger_path or default_ledger_path()
    lessons = _read_lessons(path)
    for item in lessons:
        if item.id == lesson_id:
            item.status = "resolved"
            item.resolved_at = _now()
            item.outcome = _clean_text(outcome)
            _write_lessons(path, lessons)
            return {"success": True, "lesson": item.to_dict()}
    return {"success": False, "error": f"lesson not found: {lesson_id}"}


def export_context(
    *,
    query: str = "",
    tags: list[str] | None = None,
    limit: int = 5,
    max_chars: int = 1800,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    recalled = recall_lessons(
        query=query,
        tags=tags,
        limit=limit,
        ledger_path=ledger_path,
    )
    lines = []
    for item in recalled.get("lessons", []):
        line = f"- Avoid: {item['mistake']} | Do instead: {item['lesson']}"
        if item.get("trigger"):
            line += f" | Trigger: {item['trigger']}"
        if item.get("fix"):
            line += f" | Fix: {item['fix']}"
        lines.append(line)
    text = "\n".join(lines)
    cap = max(200, min(int(max_chars or 1800), 8000))
    if len(text) > cap:
        text = text[: cap - 20].rstrip() + "\n...[truncated]"
    return {"success": True, "context": text, "count": len(lines)}


def build_self_evolution_context(
    query: str,
    *,
    tags: list[str] | None = None,
    limit: int = 3,
    max_chars: int = 1600,
    ledger_path: Path | None = None,
) -> str:
    """Return a fenced per-turn reminder block for relevant active lessons."""
    result = export_context(
        query=query,
        tags=tags,
        limit=limit,
        max_chars=max_chars,
        ledger_path=ledger_path,
    )
    context = str(result.get("context") or "").strip()
    if not context:
        return ""
    return (
        "<self-evolution-context>\n"
        "[System note: The following are Hermes' active lessons from prior "
        "mistakes. Treat them as reminders for this turn, not as user input.]\n\n"
        f"{context}\n"
        "</self-evolution-context>"
    )


def evolution_stats(*, ledger_path: Path | None = None) -> dict[str, Any]:
    lessons = _read_lessons(ledger_path or default_ledger_path())
    active = [item for item in lessons if item.status == "active"]
    return {
        "total": len(lessons),
        "active": len(active),
        "resolved": len([item for item in lessons if item.status == "resolved"]),
        "high_severity_active": len(
            [item for item in active if item.severity in {"high", "critical"}]
        ),
        "top_tags": _top_tags(active),
    }


def _read_lessons(path: Path) -> list[EvolutionLesson]:
    if not path.exists():
        return []
    lessons: list[EvolutionLesson] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            lesson = EvolutionLesson.from_dict(data)
            if lesson.id and lesson.mistake and lesson.lesson:
                lessons.append(lesson)
    return lessons


def _write_lessons(path: Path, lessons: list[EvolutionLesson]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for item in lessons
    )
    fd, tmp_name = tempfile.mkstemp(prefix=".lessons-", suffix=".jsonl", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def _fingerprint(*, mistake: str, lesson: str, tags: list[str]) -> str:
    normalized = "\n".join(
        [
            _semantic_key(mistake),
            _semantic_key(lesson),
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _semantic_key(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))[:600]


def _score_lesson(item: EvolutionLesson, *, query: str) -> int:
    if not query:
        return _severity_rank(item.severity) + item.occurrences
    query_tokens = set(_TOKEN_RE.findall(query.lower()))
    if not query_tokens:
        return 0
    haystack = " ".join(
        [
            item.mistake,
            item.lesson,
            item.trigger,
            item.fix,
            " ".join(item.tags),
            " ".join(item.evidence),
        ]
    ).lower()
    score = sum(3 if token in haystack else 0 for token in query_tokens)
    score += len(query_tokens.intersection(item.tags)) * 4
    score += _severity_rank(item.severity)
    score += min(item.occurrences, 5)
    return score


def _clean_text(value: Any) -> str:
    text = str(value or "").replace("\x00", "").strip()
    if len(text) > _MAX_TEXT_CHARS:
        text = text[:_MAX_TEXT_CHARS].rstrip() + " ...[truncated]"
    return text


def _normalize_tags(tags: list[str] | tuple[str, ...] | str) -> list[str]:
    if isinstance(tags, str):
        raw = re.split(r"[,\s]+", tags)
    else:
        raw = [str(tag) for tag in tags]
    cleaned = []
    for tag in raw:
        norm = re.sub(r"[^a-zA-Z0-9_.-]+", "-", tag.strip().lower()).strip("-")
        if norm:
            cleaned.append(norm[:64])
    return sorted(set(cleaned))[:16]


def _normalize_evidence(evidence: list[str] | tuple[str, ...] | str) -> list[str]:
    if isinstance(evidence, str):
        items = [evidence]
    else:
        items = [str(item) for item in evidence]
    return [_clean_text(item)[:1000] for item in items if _clean_text(item)][:8]


def _normalize_severity(value: str) -> str:
    lowered = str(value or "medium").lower().strip()
    return lowered if lowered in {"low", "medium", "high", "critical"} else "medium"


def _normalize_confidence(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.7


def _merge_limited(existing: list[str], new_items: list[str], *, limit: int) -> list[str]:
    merged = []
    seen = set()
    for item in [*existing, *new_items]:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged[-limit:]


def _max_severity(a: str, b: str) -> str:
    return a if _severity_rank(a) >= _severity_rank(b) else _normalize_severity(b)


def _severity_rank(value: str) -> int:
    return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(
        _normalize_severity(value),
        2,
    )


def _top_tags(lessons: list[EvolutionLesson]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in lessons:
        for tag in item.tags:
            counts[tag] = counts.get(tag, 0) + 1
    return [
        {"tag": tag, "count": count}
        for tag, count in sorted(counts.items(), key=lambda pair: pair[1], reverse=True)[:10]
    ]

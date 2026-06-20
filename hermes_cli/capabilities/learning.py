"""Safe learning flow for the Buidl Agent Harness."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from hermes_cli.goal_os import BLIND_PROMPT_PLACEHOLDER, sanitize_blind_prompt_text

LessonStatus = Literal["review", "approved", "rejected", "archived"]

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(token|api[_-]?key|secret|password|authorization)\s*[:=]\s*\S+"),
    re.compile(r"(?i)bearer\s+[a-z0-9._~+\-/=]{12,}"),
    re.compile(r"\b[A-Za-z0-9_\-]{24,}\b"),
)


@dataclass
class LessonCandidate:
    candidate_id: str
    summary: str
    status: LessonStatus = "review"
    source: str = "session-summary"
    created_at: float = field(default_factory=time.time)
    reviewed_at: float | None = None
    reviewer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LessonCandidate":
        return cls(
            candidate_id=str(data.get("candidate_id") or _new_id()),
            summary=str(data.get("summary") or ""),
            status=_valid_status(data.get("status")),
            source=str(data.get("source") or "session-summary"),
            created_at=float(data.get("created_at") or time.time()),
            reviewed_at=data.get("reviewed_at"),
            reviewer=str(data.get("reviewer") or ""),
        )


def _new_id() -> str:
    return f"lesson_{uuid.uuid4().hex[:12]}"


def _valid_status(value: Any) -> LessonStatus:
    text = str(value or "review")
    return text if text in {"review", "approved", "rejected", "archived"} else "review"  # type: ignore[return-value]


def _hermes_home() -> Path:
    import os

    raw = os.environ.get("HERMES_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        return Path.home() / ".hermes"


def sanitize_lesson_text(text: str) -> str:
    sanitized = sanitize_blind_prompt_text(text)
    if sanitized == BLIND_PROMPT_PLACEHOLDER:
        return sanitized
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized.strip()


class LearningManager:
    def __init__(self, store_path: Path | None = None):
        self.store_path = store_path or _hermes_home() / "capabilities" / "memory" / "lesson-candidates.json"

    def _read(self) -> dict[str, Any]:
        if not self.store_path.exists():
            return {"version": 1, "candidates": {}, "archive": {}}
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "candidates": {}, "archive": {}}
        if not isinstance(data, dict):
            return {"version": 1, "candidates": {}, "archive": {}}
        data.setdefault("version", 1)
        data.setdefault("candidates", {})
        data.setdefault("archive", {})
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.store_path)

    def create_candidate(self, summary: str, *, source: str = "session-summary") -> LessonCandidate:
        candidate = LessonCandidate(candidate_id=_new_id(), summary=sanitize_lesson_text(summary), source=source)
        data = self._read()
        data["candidates"][candidate.candidate_id] = candidate.to_dict()
        self._write(data)
        return candidate

    def review_candidate(self, candidate_id: str, *, approve: bool, reviewer: str = "Memory Curator") -> LessonCandidate:
        data = self._read()
        raw = data["candidates"].get(candidate_id)
        if not isinstance(raw, dict):
            raise KeyError(candidate_id)
        candidate = LessonCandidate.from_dict(raw)
        candidate.status = "approved" if approve else "rejected"
        candidate.reviewed_at = time.time()
        candidate.reviewer = reviewer
        if approve:
            data["candidates"][candidate.candidate_id] = candidate.to_dict()
        else:
            candidate.status = "archived"
            data["candidates"].pop(candidate.candidate_id, None)
            data["archive"][candidate.candidate_id] = candidate.to_dict()
        self._write(data)
        return candidate

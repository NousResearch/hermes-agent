"""ValidationResult — LLM judge output for skill-sleep VALIDATE stage."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from typing import Any


@dataclasses.dataclass
class ValidationItem:
    """Per-task judge result."""

    task_index: int
    score: int
    passed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_index": int(self.task_index),
            "score": int(self.score),
            "passed": bool(self.passed),
            "reason": str(self.reason)[:2000],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationItem:
        return cls(
            task_index=int(data.get("task_index", 0)),
            score=int(data.get("score", 0)),
            passed=bool(data.get("passed", False)),
            reason=str(data.get("reason", "")),
        )


@dataclasses.dataclass
class ValidationResult:
    """Aggregated gate result for a candidate diff."""

    generated_at: str
    skill_path: str
    diff_path: str
    gate_type: str
    overall_passed: bool
    total_tasks: int
    passed_tasks: int
    pass_rate: float
    threshold: int
    min_pass_rate: float
    limitation: str
    items: list[ValidationItem]
    rejected_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "skill_path": self.skill_path,
            "diff_path": self.diff_path,
            "gate_type": self.gate_type,
            "overall_passed": bool(self.overall_passed),
            "total_tasks": int(self.total_tasks),
            "passed_tasks": int(self.passed_tasks),
            "pass_rate": round(float(self.pass_rate), 4),
            "threshold": int(self.threshold),
            "min_pass_rate": float(self.min_pass_rate),
            "limitation": self.limitation,
            "items": [it.to_dict() for it in self.items],
            "rejected_reason": self.rejected_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult:
        items = [ValidationItem.from_dict(d) for d in data.get("items", [])]
        return cls(
            generated_at=str(data.get("generated_at", "")),
            skill_path=str(data.get("skill_path", "")),
            diff_path=str(data.get("diff_path", "")),
            gate_type=str(data.get("gate_type", "llm_judge")),
            overall_passed=bool(data.get("overall_passed", False)),
            total_tasks=int(data.get("total_tasks", 0)),
            passed_tasks=int(data.get("passed_tasks", 0)),
            pass_rate=float(data.get("pass_rate", 0.0)),
            threshold=int(data.get("threshold", 70)),
            min_pass_rate=float(data.get("min_pass_rate", 0.6)),
            limitation=str(data.get("limitation", "")),
            items=items,
            rejected_reason=data.get("rejected_reason"),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def __repr__(self) -> str:
        return (
            f"ValidationResult(passed={self.overall_passed}, "
            f"rate={self.pass_rate:.2f} "
            f"[{self.passed_tasks}/{self.total_tasks}], "
            f"threshold={self.threshold})"
        )

"""Proposal — candidate diff dataclass for skill-sleep PROPOSE stage."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from typing import Any


@dataclasses.dataclass
class Proposal:
    """Candidate skill edit produced by the optimizer."""

    generated_at: str
    skill_path: str
    source_task_cards: int
    diff_lines: int
    summary: str
    focused_on: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "skill_path": self.skill_path,
            "source_task_cards": self.source_task_cards,
            "diff_lines": self.diff_lines,
            "summary": self.summary[:2000],
            "focused_on": self.focused_on[:10],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Proposal:
        return cls(
            generated_at=str(data.get("generated_at", "")),
            skill_path=str(data.get("skill_path", "")),
            source_task_cards=int(data.get("source_task_cards", 0)),
            diff_lines=int(data.get("diff_lines", 0)),
            summary=str(data.get("summary", "")),
            focused_on=list(data.get("focused_on", [])),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def __repr__(self) -> str:
        return (
            f"Proposal(skill={self.skill_path}, "
            f"cards={self.source_task_cards}, "
            f"lines={self.diff_lines})"
        )

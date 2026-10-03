"""TaskCard — friction episode dataclass for skill-sleep MINE stage."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class TaskCard:
    """A friction episode that suggests a skill improvement opportunity."""

    skill_name: str
    session_id: str
    user_request: str
    friction_evidence: list[str]
    tool_calls: list[dict[str, Any]]
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "session_id": self.session_id,
            "user_request": self.user_request[:2000],
            "friction_evidence": self.friction_evidence[:5],
            "tool_calls": len(self.tool_calls),
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        short = self.session_id[:24] if self.session_id else "?"
        return (
            f"TaskCard(skill={self.skill_name}, "
            f"session={short}, "
            f"evidence={len(self.friction_evidence)} signals)"
        )

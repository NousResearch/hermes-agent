"""escalation_log.py - EscalationLog and EscalationEvent for recording routing decisions.

Records every request handled by Orchestrator as a structured event trace:
(utterance, routed_skill, confidence, resolved_by, outcome, embedding).
GrowthDreamer replays over this log to evaluate policy shifts.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .needle_ops import embed_text

logger = logging.getLogger(__name__)


@dataclass
class EscalationEvent:
    event_id: str
    utterance: str
    routed_skill_id: Optional[str]
    confidence: float
    resolved_by: str  # "NEEDLE", "BONSAI", "CLOUD_LLM", or "USER"
    outcome: str      # "SUCCESS", "MISFIRE", "ESCALATED", "FAILED"
    resolved_action: Optional[Dict[str, Any]] = None
    embedding: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EscalationEvent:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class EscalationLog:
    """Disk-backed log of escalation events."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._events: List[EscalationEvent] = []
        self.load()

    def load(self) -> None:
        if self.log_path.exists():
            try:
                data = json.loads(self.log_path.read_text(encoding="utf-8"))
                self._events = [EscalationEvent.from_dict(d) for d in data]
            except Exception as e:
                logger.warning("Failed to load escalation log from %s: %s", self.log_path, e)
                self._events = []

    def save(self) -> None:
        data = [e.to_dict() for e in self._events]
        self.log_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def log(
        self,
        utterance: str,
        routed_skill_id: Optional[str],
        confidence: float,
        resolved_by: str,
        outcome: str,
        resolved_action: Optional[Dict[str, Any]] = None,
    ) -> EscalationEvent:
        event_id = f"evt_{int(time.time()*1000)}_{len(self._events)+1}"
        emb = embed_text(utterance)
        event = EscalationEvent(
            event_id=event_id,
            utterance=utterance,
            routed_skill_id=routed_skill_id,
            confidence=confidence,
            resolved_by=resolved_by,
            outcome=outcome,
            resolved_action=resolved_action,
            embedding=emb,
            timestamp=time.time(),
        )
        self._events.append(event)
        self.save()
        return event

    def get_events(
        self,
        skill_id: Optional[str] = None,
        unresolved_only: bool = False,
        min_timestamp: Optional[float] = None,
    ) -> List[EscalationEvent]:
        res = self._events
        if skill_id:
            res = [e for e in res if e.routed_skill_id == skill_id]
        if unresolved_only:
            res = [e for e in res if e.resolved_by in ("BONSAI", "CLOUD_LLM", "USER") or e.outcome == "ESCALATED"]
        if min_timestamp:
            res = [e for e in res if e.timestamp >= min_timestamp]
        return res

    def clear(self) -> None:
        self._events = []
        self.save()

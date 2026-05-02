"""Task Card — structured task definition for Hermes 2.8 Harness Engineering.

Each user request gets a Task Card before execution begins. The Task Card is
the single source of truth for what the task IS, how it should be executed,
and what success looks like.

Design constraints (from engineering plan):
- Task Card holds current state snapshot only. Status history lives in Event Log.
- updated_at refreshes on every write.
- version increments on every write to prevent concurrent overwrites.
- Write failures MUST throw — caller is responsible for writing execution_failed.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2.8.0"

VALID_STATUSES = [
    "pending",
    "running",
    "reviewing",
    "completed",
    "failed",
    "blocked",
    "partial",
]

EXECUTION_MODES = [
    "self_execute",
    "single_agent",
    "pipeline",
    "review_only",
]


@dataclass
class CompiledIntent:
    real_task: str = ""
    task_category: str = "other"
    assumptions: List[str] = field(default_factory=list)
    must_keep: List[str] = field(default_factory=list)
    must_avoid: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    mode: str = "self_execute"
    agents: List[str] = field(default_factory=list)
    delegation_reason: str = ""


@dataclass
class AcceptanceCriteria:
    auto_checkable: List[str] = field(default_factory=list)
    human_judgment: List[str] = field(default_factory=list)
    user_preference_check: List[str] = field(default_factory=list)


@dataclass
class TaskCard:
    """Schema version 2.8.0 — structured task definition.

    Every user request gets a TaskCard before execution. It records:
    - What the user actually wants (compiled_intent)
    - How to execute it (execution_plan)
    - What success looks like (acceptance_criteria)
    - Current status and results

    The TaskCard is stored as JSON at ~/.hermes/task_cards/{task_id}.json.
    State history is NOT stored here — it comes from the Event Log.
    """

    schema_version: str = SCHEMA_VERSION
    task_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    version: int = 0
    raw_user_request: str = ""
    compiled_intent: CompiledIntent = field(default_factory=CompiledIntent)
    execution_plan: ExecutionPlan = field(default_factory=ExecutionPlan)
    acceptance_criteria: AcceptanceCriteria = field(default_factory=AcceptanceCriteria)
    status: str = "pending"
    result_summary: Optional[str] = None
    review_result: Optional[Dict[str, Any]] = None
    session_id: str = ""

    # ── Forward-compatible fields (Sprint 4, 5) ──
    routing_basis: List[str] = field(default_factory=list)
    fallback_used: Optional[str] = None

    @classmethod
    def create(
        cls,
        user_request: str,
        session_id: str = "",
        task_category: str = "other",
    ) -> "TaskCard":
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            task_id=str(uuid.uuid4()),
            created_at=now,
            updated_at=now,
            version=0,
            raw_user_request=user_request,
            compiled_intent=CompiledIntent(
                real_task=user_request,
                task_category=task_category,
            ),
            execution_plan=ExecutionPlan(mode="self_execute"),
            acceptance_criteria=AcceptanceCriteria(),
            status="pending",
            session_id=session_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskCard":
        intent_raw = d.get("compiled_intent", {})
        if isinstance(intent_raw, dict):
            _intent_fields = {f.name for f in fields(CompiledIntent)}
            intent = CompiledIntent(**{k: v for k, v in intent_raw.items() if k in _intent_fields})
        else:
            intent = CompiledIntent()

        plan_raw = d.get("execution_plan", {})
        if isinstance(plan_raw, dict):
            _plan_fields = {f.name for f in fields(ExecutionPlan)}
            plan = ExecutionPlan(**{k: v for k, v in plan_raw.items() if k in _plan_fields})
        else:
            plan = ExecutionPlan()

        ac_raw = d.get("acceptance_criteria", {})
        if isinstance(ac_raw, dict):
            _ac_fields = {f.name for f in fields(AcceptanceCriteria)}
            ac = AcceptanceCriteria(**{k: v for k, v in ac_raw.items() if k in _ac_fields})
        else:
            ac = AcceptanceCriteria()

        return cls(
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            task_id=d.get("task_id", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            version=d.get("version", 1),
            raw_user_request=d.get("raw_user_request", ""),
            compiled_intent=intent,
            execution_plan=plan,
            acceptance_criteria=ac,
            status=d.get("status", "pending"),
            result_summary=d.get("result_summary"),
            review_result=d.get("review_result"),
            session_id=d.get("session_id", ""),
            routing_basis=d.get("routing_basis", []),
            fallback_used=d.get("fallback_used"),
        )

    @classmethod
    def from_json(cls, s: str) -> "TaskCard":
        return cls.from_dict(json.loads(s))


# ── Storage helpers ──


def get_task_cards_dir() -> Path:
    return get_hermes_home() / "task_cards"


def save_task_card(card: TaskCard) -> Path:
    card_dir = get_task_cards_dir()
    card_dir.mkdir(parents=True, exist_ok=True)
    card_path = card_dir / f"{card.task_id}.json"
    card.updated_at = datetime.now(timezone.utc).isoformat()
    card.version += 1
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card.to_json())
    logger.debug("TaskCard saved: %s v%d", card.task_id, card.version)
    return card_path


def load_task_card(task_id: str) -> Optional[TaskCard]:
    card_path = get_task_cards_dir() / f"{task_id}.json"
    if not card_path.exists():
        return None
    with open(card_path, "r", encoding="utf-8") as f:
        return TaskCard.from_json(f.read())

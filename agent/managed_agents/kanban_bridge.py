"""Managed agents bridge to the existing Kanban board."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .review_gate import ReviewRequirement, ReviewResult

KANBAN_STATES = (
    "created",
    "planned",
    "delegated",
    "in_progress",
    "review_pending",
    "changes_requested",
    "approved",
    "done",
    "blocked",
    "failed",
)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    if isinstance(values, (list, tuple, set)):
        return [str(item).strip() for item in values if str(item).strip()]
    value = str(values).strip()
    return [value] if value else []


def _normalize_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    try:
        return int(str(value).strip())
    except Exception:
        return len(_normalize_list(value))


@dataclass(slots=True)
class KanbanBridgeCard:
    card_id: str
    title: str
    state: str = "created"
    assignee: str = ""
    task_id: str = ""
    review_requirement: dict[str, Any] = field(default_factory=dict)
    review_result: dict[str, Any] | None = None
    blocked_reason: str = ""
    failure_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KanbanBridgeRuntime:
    task: Any
    card: KanbanBridgeCard = field(init=False)

    def __post_init__(self) -> None:
        self.card = create_card_from_task(self.task)

    @property
    def state(self) -> str:
        return self.card.state

    def plan(self) -> KanbanBridgeCard:
        return plan_card(self.card)

    def delegate(self, assignee: str) -> KanbanBridgeCard:
        if self.card.state == "created":
            plan_card(self.card)
        return delegate_card(self.card, assignee)

    def sync_execution_plan(self, task_card: Any) -> KanbanBridgeCard:
        mode = str(_get_attr(_get_attr(task_card, "execution_plan", None), "mode", "")).strip()
        agents = _normalize_list(_get_attr(_get_attr(task_card, "execution_plan", None), "agents", []))
        if self.card.state == "created":
            plan_card(self.card)
        if mode == "self_execute":
            return self.card
        if self.card.state == "planned" and agents:
            delegate_card(self.card, agents[0])
        if self.card.state == "delegated":
            start_work(self.card)
        return self.card

    def start(self) -> KanbanBridgeCard:
        if self.card.state == "created":
            plan_card(self.card)
        if self.card.state == "planned":
            delegate_card(self.card, self.card.assignee or str(_get_attr(self.task, "owner_agent", "")))
        return start_work(self.card)

    def request_review(self, reviewers: list[str] | tuple[str, ...]) -> KanbanBridgeCard:
        if self.card.state == "created":
            plan_card(self.card)
        if self.card.state == "planned":
            delegate_card(self.card, self.card.assignee or str(_get_attr(self.task, "owner_agent", "")))
        if self.card.state == "delegated":
            start_work(self.card)
        return request_review(self.card, reviewers)

    def apply_review_result(self, review_result: ReviewResult | Mapping[str, Any]) -> KanbanBridgeCard:
        if self.card.state == "in_progress":
            self.request_review([])
        if self.card.state != "review_pending":
            raise ValueError(f"Cannot complete review from state {self.card.state!r}")
        return complete_review(self.card, review_result)

    def approve(self) -> KanbanBridgeCard:
        if self.card.state == "created":
            plan_card(self.card)
        if self.card.state == "planned":
            delegate_card(self.card, self.card.assignee or str(_get_attr(self.task, "owner_agent", "")))
        if self.card.state == "delegated":
            start_work(self.card)
        if self.card.state == "in_progress":
            request_review(self.card, [])
        if self.card.state == "review_pending":
            complete_review(
                self.card,
                ReviewResult(
                    task_id=self.card.task_id,
                    reviewer="codex",
                    decision="approved",
                ),
            )
        return deliver_card(self.card)

    def changes_requested(self) -> KanbanBridgeCard:
        if self.card.state == "created":
            plan_card(self.card)
        if self.card.state == "planned":
            delegate_card(self.card, self.card.assignee or str(_get_attr(self.task, "owner_agent", "")))
        if self.card.state == "delegated":
            start_work(self.card)
        if self.card.state == "in_progress":
            request_review(self.card, [])
        if self.card.state == "review_pending":
            complete_review(
                self.card,
                ReviewResult(
                    task_id=self.card.task_id,
                    reviewer="codex",
                    decision="changes_requested",
                ),
            )
        return self.card

    def complete(self) -> KanbanBridgeCard:
        if self.card.state == "approved":
            return deliver_card(self.card)
        if self.card.state == "review_pending":
            complete_review(
                self.card,
                ReviewResult(
                    task_id=self.card.task_id,
                    reviewer="codex",
                    decision="approved",
                ),
            )
            return deliver_card(self.card)
        return self.card

    def fail(self, reason: str) -> KanbanBridgeCard:
        return fail_card(self.card, reason=reason)

    def block(self, reason: str) -> KanbanBridgeCard:
        return block_card(self.card, reason=reason)


def should_auto_create_card(task: Any) -> bool:
    execution_plan = _get_attr(task, "execution_plan", None)
    steps = int(_get_attr(task, "steps", _get_attr(task, "step_count", 0)) or 0)
    agent_values = _get_attr(task, "agents", _get_attr(task, "agent_ids", []))
    if not agent_values:
        agent_values = _get_attr(execution_plan, "agents", [])
    agents = _normalize_count(agent_values)
    risk = str(_get_attr(task, "risk", _get_attr(task, "risk_level", "R0")) or "R0").upper()
    needs_review = bool(_get_attr(task, "needs_review", False))
    cross_session = bool(_get_attr(task, "requires_cross_session_recovery", False))
    code = bool(_get_attr(task, "changes_include_code", False))
    test = bool(_get_attr(task, "changes_include_test", False))
    acceptance = bool(_get_attr(task, "changes_include_acceptance", False))

    return any(
        (
            steps > 3,
            agents >= 2,
            risk in {"R2", "R3", "R4"},
            needs_review,
            cross_session,
            code and test and acceptance,
        )
    )


def create_card(task_id: str, *, title: str | None = None, assignee: str = "") -> KanbanBridgeCard:
    return KanbanBridgeCard(card_id=task_id, title=title or task_id, assignee=assignee, task_id=task_id)


def create_card_from_task(task: Any) -> KanbanBridgeCard:
    task_id = str(_get_attr(task, "task_id", "unknown")).strip() or "unknown"
    title = str(_get_attr(task, "title", _get_attr(task, "raw_user_request", task_id)))
    assignee = str(_get_attr(task, "owner_agent", _get_attr(task, "assignee", "")) or "")
    card = create_card(task_id, title=title, assignee=assignee)
    card.metadata["source_task"] = task_id
    return card


def plan_card(card: KanbanBridgeCard) -> KanbanBridgeCard:
    _require_state(card, "created")
    card.state = "planned"
    return card


def delegate_card(card: KanbanBridgeCard, assignee: str) -> KanbanBridgeCard:
    _require_state(card, "planned")
    card.assignee = str(assignee).strip()
    card.state = "delegated"
    return card


def start_work(card: KanbanBridgeCard) -> KanbanBridgeCard:
    _require_state(card, "delegated")
    card.state = "in_progress"
    return card


def request_review(card: KanbanBridgeCard, reviewers: list[str] | tuple[str, ...]) -> KanbanBridgeCard:
    _require_state(card, "in_progress")
    card.review_requirement = _review_requirement_to_dict(reviewers)
    card.state = "review_pending"
    return card


def complete_review(card: KanbanBridgeCard, review_result: ReviewResult | Mapping[str, Any]) -> KanbanBridgeCard:
    _require_state(card, "review_pending")
    payload = _review_result_to_dict(review_result)
    card.review_result = payload
    decision = str(payload.get("decision", "")).strip()
    if decision in {"approved", "pass", "pass_with_notes"}:
        card.state = "approved"
    elif decision == "changes_requested":
        card.state = "changes_requested"
    else:
        raise ValueError(f"Unknown review decision: {decision!r}")
    return card


def resume_work(card: KanbanBridgeCard) -> KanbanBridgeCard:
    _require_state(card, "changes_requested")
    card.state = "in_progress"
    return card


def deliver_card(card: KanbanBridgeCard) -> KanbanBridgeCard:
    _require_state(card, "approved")
    card.state = "done"
    return card


def block_card(card: KanbanBridgeCard, *, reason: str) -> KanbanBridgeCard:
    card.state = "blocked"
    card.blocked_reason = str(reason)
    return card


def fail_card(card: KanbanBridgeCard, *, reason: str) -> KanbanBridgeCard:
    card.state = "failed"
    card.failure_reason = str(reason)
    return card


def serialize_card(card: KanbanBridgeCard) -> dict[str, Any]:
    return {
        "card_id": card.card_id,
        "title": card.title,
        "state": card.state,
        "assignee": card.assignee,
        "task_id": card.task_id,
        "review_requirement": dict(card.review_requirement),
        "review_result": _review_result_to_dict(card.review_result) if card.review_result is not None else None,
        "blocked_reason": card.blocked_reason,
        "failure_reason": card.failure_reason,
        "metadata": dict(card.metadata),
    }


def load_kanban_bridge_config(path: str | Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def build_kanban_bridge(task: Any) -> KanbanBridgeRuntime:
    return KanbanBridgeRuntime(task=task)


def _review_result_to_dict(review_result: ReviewResult | Mapping[str, Any]) -> dict[str, Any]:
    if hasattr(review_result, "to_dict"):
        return dict(review_result.to_dict())  # type: ignore[call-arg]
    if isinstance(review_result, Mapping):
        return dict(review_result)
    return {"decision": str(_get_attr(review_result, "decision", ""))}


def _review_requirement_to_dict(review_requirement: ReviewRequirement | Mapping[str, Any] | list[str] | tuple[str, ...]) -> dict[str, Any]:
    if hasattr(review_requirement, "to_dict"):
        return dict(review_requirement.to_dict())  # type: ignore[call-arg]
    if isinstance(review_requirement, Mapping):
        return dict(review_requirement)
    return {"reviewers": _normalize_list(review_requirement)}


def _require_state(card: KanbanBridgeCard, expected: str) -> None:
    if card.state != expected:
        raise ValueError(f"Cannot { _transition_verb(expected) } a card in state {card.state!r}")


def _transition_verb(expected: str) -> str:
    return {
        "created": "plan",
        "planned": "delegate",
        "delegated": "start work on",
        "in_progress": "request review for",
        "review_pending": "complete review for",
        "changes_requested": "resume work on",
        "approved": "deliver",
    }.get(expected, "transition")

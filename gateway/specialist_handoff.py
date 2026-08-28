"""Transactional, deterministic Kanban handoff for specialist routing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from gateway.configured_board import configured_board_db_path
from gateway.specialist_routing import SpecialistRouteDecision


_ORCHESTRATION_GOAL_MAX_TURNS = 12
_EXAMPLEPROJECT_NAVIGATION_SKILL = "exampleproject-worktree-navigation"


def _required_skills(board: Optional[str]) -> Optional[list[str]]:
    """Pin the project navigation contract on ExampleProject task graphs."""
    if board == "exampleproject-burndown":
        return [_EXAMPLEPROJECT_NAVIGATION_SKILL]
    return None


@dataclass(frozen=True)
class HandoffSource:
    """Trusted source fields needed to create a task notification route."""

    platform: str
    chat_id: str
    chat_type: str
    user_id: Optional[str]
    message_id: str
    guild_id: Optional[str] = None
    thread_id: Optional[str] = None
    user_id_alt: Optional[str] = None
    notifier_profile: Optional[str] = None
    session_id: Optional[str] = None
    delivery_metadata: Optional[dict] = None


@dataclass(frozen=True)
class HandoffResult:
    ok: bool
    task_id: Optional[str] = None
    created: bool = False
    reason: str = ""


def _idempotency_key(source: HandoffSource) -> Optional[str]:
    if not source.message_id or not source.platform or not source.chat_id:
        return None
    return "specialist-routing:" + ":".join(
        (source.platform, source.guild_id or "", source.chat_id, source.thread_id or "", source.message_id)
    )


def _body(*, decision: SpecialistRouteDecision, source: HandoffSource, request: str, router_model: str) -> str:
    return json.dumps(
        {
            "schema": "specialist_routing.v1",
            "request": request[:4_000],
            "profile": decision.profile,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "router_model": router_model or "configured_auxiliary",
            "ingress": {
                "platform": source.platform,
                "guild_id": source.guild_id,
                "chat_id": source.chat_id,
                "thread_id": source.thread_id,
                "message_id": source.message_id,
                "user_id": source.user_id,
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def create_specialist_handoff(*, decision: SpecialistRouteDecision, source: HandoffSource, request: str, router_model: str = "", board: Optional[str] = None) -> HandoffResult:
    """Create a subscribed, durable triage root for specialist orchestration."""
    if not decision.dispatches:
        return HandoffResult(False, reason="non_dispatch_decision")
    if not source.platform or not source.chat_id or not source.message_id:
        return HandoffResult(False, reason="incomplete_source")
    if not isinstance(request, str) or not request.strip():
        return HandoffResult(False, reason="empty_request")
    try:
        from hermes_cli import kanban_db as kb

        key = _idempotency_key(source)
        conn = kb.connect(db_path=configured_board_db_path(board), board=board)
        try:
            existing_id = None
            if key:
                row = conn.execute(
                    "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' ORDER BY created_at DESC LIMIT 1",
                    (key,),
                ).fetchone()
                existing_id = row["id"] if row else None
            with kb.write_txn(conn):
                task_id = kb.create_task(
                    conn, title=decision.title,
                    body=_body(decision=decision, source=source, request=request, router_model=router_model),
                    assignee=decision.profile, created_by="specialist-routing",
                    idempotency_key=key, session_id=source.session_id, board=board,
                    triage=True,
                    goal_mode=True,
                    goal_max_turns=_ORCHESTRATION_GOAL_MAX_TURNS,
                    skills=_required_skills(board),
                )
                kb.add_notify_sub(
                    conn, task_id=task_id, platform=source.platform, chat_id=source.chat_id,
                    chat_type=source.chat_type, thread_id=source.thread_id,
                    user_id=source.user_id, user_id_alt=source.user_id_alt,
                    notifier_profile=source.notifier_profile, delivery_mode="notify",
                    delivery_metadata=source.delivery_metadata, allow_nested=True,
                )
            return HandoffResult(True, task_id=task_id, created=existing_id is None)
        finally:
            conn.close()
    except Exception as exc:
        return HandoffResult(False, reason=f"handoff_error:{type(exc).__name__}")

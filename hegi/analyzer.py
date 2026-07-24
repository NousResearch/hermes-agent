"""Convert structured LLM output into validated meeting minutes."""

from __future__ import annotations

import hashlib
from typing import Any

from .models import (
    ActionItem,
    AgentPosition,
    ConceptDefinition,
    DiscussionStage,
    EvidenceItem,
    MeetingEpisode,
    MeetingMinutes,
)


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _ids(value: Any, valid_ids: set[int]) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            message_id = int(item)
        except (TypeError, ValueError):
            continue
        if message_id in valid_ids and message_id not in result:
            result.append(message_id)
    return result


def _action_id(meeting_id: str, title: str, source_ids: list[int]) -> str:
    digest = hashlib.sha256(
        f"{meeting_id}|{title}|{','.join(map(str, source_ids))}".encode("utf-8")
    ).hexdigest()
    return f"act-{digest[:14]}"


def build_minutes(payload: dict[str, Any], episode: MeetingEpisode) -> MeetingMinutes:
    valid_ids = {message.message_id for message in episode.messages}
    warnings = _strings(payload.get("warnings"))
    discussion_flow: list[DiscussionStage] = []
    for item in payload.get("discussion_flow", []):
        if isinstance(item, dict):
            discussion_flow.append(
                DiscussionStage(
                    heading=str(item.get("heading", "")).strip() or "논의",
                    summary=str(item.get("summary", "")).strip(),
                    source_message_ids=_ids(item.get("source_message_ids"), valid_ids),
                )
            )
    agent_positions: list[AgentPosition] = []
    for item in payload.get("agent_positions", []):
        if isinstance(item, dict):
            agent_positions.append(
                AgentPosition(
                    agent=str(item.get("agent", "")).strip(),
                    position=str(item.get("position", "")).strip(),
                    contributions=_strings(item.get("contributions")),
                    source_message_ids=_ids(item.get("source_message_ids"), valid_ids),
                )
            )
    concepts: list[ConceptDefinition] = []
    for item in payload.get("new_concepts", []):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "proposed"))
        if status not in {"proposed", "agreed", "uncertain"}:
            status = "uncertain"
        concepts.append(
            ConceptDefinition(
                name=str(item.get("name", "")).strip(),
                definition=str(item.get("definition", "")).strip(),
                status=status,  # type: ignore[arg-type]
                source_message_ids=_ids(item.get("source_message_ids"), valid_ids),
            )
        )
    evidence: list[EvidenceItem] = []
    for item in payload.get("evidence_and_sources", []):
        if not isinstance(item, dict):
            continue
        verification = str(item.get("verification", "추가 확인 필요"))
        if verification not in {"검증됨", "추정", "추가 확인 필요"}:
            verification = "추가 확인 필요"
        evidence.append(
            EvidenceItem(
                claim=str(item.get("claim", "")).strip(),
                source=str(item["source"]).strip() if item.get("source") else None,
                verification=verification,  # type: ignore[arg-type]
                source_message_ids=_ids(item.get("source_message_ids"), valid_ids),
            )
        )
    actions: list[ActionItem] = []
    for item in payload.get("action_items", []):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        source_ids = _ids(item.get("source_message_ids"), valid_ids)
        if not title or not source_ids:
            warnings.append(f"근거가 없는 Action Item 제외: {title or '(제목 없음)'}")
            continue
        priority = str(item.get("priority", "medium")).lower()
        if priority not in {"critical", "high", "medium", "low"}:
            priority = "medium"
        actions.append(
            ActionItem(
                action_id=_action_id(episode.meeting_id, title, source_ids),
                title=title,
                description=str(item.get("description", "")).strip(),
                source_message_ids=source_ids,
                owner=str(item["owner"]).strip() if item.get("owner") else None,
                priority=priority,  # type: ignore[arg-type]
                deadline=str(item["deadline"]).strip() if item.get("deadline") else None,
                project_id=(
                    str(item["project_id"]).strip() if item.get("project_id") else None
                ),
                rationale=str(item.get("rationale", "")).strip(),
            )
        )
    try:
        confidence = float(payload.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return MeetingMinutes(
        meeting_id=episode.meeting_id,
        title=str(payload.get("title", "")).strip() or f"연구회의 {episode.meeting_id}",
        background=str(payload.get("background", "")).strip() or "원문에서 배경이 명시되지 않음",
        agenda=_strings(payload.get("agenda")),
        discussion_flow=discussion_flow,
        agent_positions=agent_positions,
        professor_positions=_strings(payload.get("professor_positions")),
        agreements=_strings(payload.get("agreements")),
        disagreements=_strings(payload.get("disagreements")),
        unresolved_questions=_strings(payload.get("unresolved_questions")),
        new_concepts=concepts,
        evidence_and_sources=evidence,
        research_direction=_strings(payload.get("research_direction")),
        action_items=actions,
        memory_evaluation=None,
        confidence=confidence,
        warnings=warnings,
        recommendation=str(payload.get("recommendation", "")).strip(),
    )


def minimal_minutes(episode: MeetingEpisode, error: str) -> MeetingMinutes:
    """Grounded last-resort output when both structured calls fail."""
    positions: list[AgentPosition] = []
    for agent in episode.participants:
        source = [
            message
            for message in episode.messages
            if message.role == "assistant" and message.source_agent == agent
        ]
        positions.append(
            AgentPosition(
                agent=agent,
                position="자동 구조화 분석 실패로 원문 확인 필요",
                contributions=[],
                source_message_ids=[message.message_id for message in source],
            )
        )
    return MeetingMinutes(
        meeting_id=episode.meeting_id,
        title=f"분석 대기 회의 {episode.meeting_id}",
        background="구조화 분석에 실패하여 원문 추적 정보만 보존함",
        agenda=["원문 수동 검토 필요"],
        discussion_flow=[],
        agent_positions=positions,
        professor_positions=[],
        agreements=[],
        disagreements=[],
        unresolved_questions=["회의 구조화 분석 재시도 필요"],
        new_concepts=[],
        evidence_and_sources=[],
        research_direction=[],
        action_items=[],
        memory_evaluation=None,
        confidence=0.0,
        warnings=[f"LLM 분석 실패: {error}"],
        recommendation="자동 전송 전에 재분석할 것",
    )

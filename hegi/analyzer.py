"""Convert structured LLM output into validated meeting minutes."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .models import (
    ActionItem,
    AgentActivity,
    AgentPosition,
    ConceptDefinition,
    DiscussionStage,
    EvidenceItem,
    MeetingEpisode,
    MeetingMinutes,
    TemporalConflict,
)

MEETING_TYPES = {"research_meeting", "operational_incident", "mixed", "other"}
PROFESSOR_NAMES = {"교수", "교수님", "professor", "user", "사용자"}
UNSAFE_MEMORY_ACTION = re.compile(
    r"(Memory\s*Forest|메모리\s*포레스트|기억\s*숲).{0,30}"
    r"(직접\s*(쓰기|저장)|쓰기\s*활성화|자동\s*(커밋|승인)|auto[_ -]?commit)",
    re.IGNORECASE,
)
INCIDENT_WORDS = (
    "부팅 실패",
    "장애",
    "오류",
    "에러",
    "실패",
    "복구",
    "daemon",
    "타임아웃",
    "중단",
)
RESEARCH_WORDS = ("연구", "논문", "개념", "이론", "서지", "해석", "미학")


def _text(value: Any) -> str:
    """Return renderer-safe scalar text without coercing containers to repr."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _text(item))]


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


def classify_meeting_type(payload: dict[str, Any], episode: MeetingEpisode) -> str:
    explicit = _text(payload.get("meeting_type"))
    if explicit in MEETING_TYPES:
        return explicit
    corpus = " ".join(message.content for message in episode.messages).lower()
    incident = sum(word.lower() in corpus for word in INCIDENT_WORDS)
    research = sum(word.lower() in corpus for word in RESEARCH_WORDS)
    if incident and research:
        return "mixed"
    if incident:
        return "operational_incident"
    if research:
        return "research_meeting"
    return "other"


def build_minutes(payload: dict[str, Any], episode: MeetingEpisode) -> MeetingMinutes:
    valid_ids = {message.message_id for message in episode.messages}
    warnings = _strings(payload.get("warnings"))
    professor_positions = _strings(payload.get("professor_positions"))
    discussion_flow: list[DiscussionStage] = []
    for item in payload.get("discussion_flow", []):
        if isinstance(item, dict):
            discussion_flow.append(
                DiscussionStage(
                    heading=_text(item.get("heading")) or "논의",
                    summary=_text(item.get("summary")),
                    source_message_ids=_ids(item.get("source_message_ids"), valid_ids),
                )
            )
    agent_positions: list[AgentPosition] = []
    for item in payload.get("agent_positions", []):
        if isinstance(item, dict):
            agent = _text(item.get("agent"))
            position = _text(item.get("position"))
            if agent.lower() in PROFESSOR_NAMES:
                if position:
                    professor_positions.append(position)
                professor_positions.extend(_strings(item.get("contributions")))
                continue
            agent_positions.append(
                AgentPosition(
                    agent=agent,
                    position=position,
                    contributions=_strings(item.get("contributions")),
                    source_message_ids=_ids(item.get("source_message_ids"), valid_ids),
                )
            )
    activity_log: list[AgentActivity] = []
    for item in payload.get("agent_activity_log", []):
        if not isinstance(item, dict):
            continue
        agent = _text(item.get("agent"))
        if not agent or agent.lower() in PROFESSOR_NAMES:
            continue
        activity_log.append(
            AgentActivity(
                agent=agent,
                activity=_text(item.get("activity")),
                result=_text(item.get("result")) or "결과 미확인",
                source_message_ids=_ids(item.get("source_message_ids"), valid_ids),
            )
        )
    temporal_conflicts: list[TemporalConflict] = []
    for item in payload.get("temporal_conflicts", []):
        if not isinstance(item, dict):
            continue
        status = _text(item.get("resolution_status")) or "uncertain"
        if status not in {"resolved", "unresolved", "superseded", "uncertain"}:
            status = "uncertain"
        temporal_conflicts.append(
            TemporalConflict(
                subject=_text(item.get("subject")) or "상태 충돌",
                earlier_state=_text(item.get("earlier_state")) or "이전 상태 미상",
                current_state=_text(item.get("current_state")) or "현재 상태 미상",
                resolution_status=status,  # type: ignore[arg-type]
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
                name=_text(item.get("name")),
                definition=_text(item.get("definition")),
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
                claim=_text(item.get("claim")),
                source=_text(item.get("source")) or None,
                verification=verification,  # type: ignore[arg-type]
                source_message_ids=_ids(item.get("source_message_ids"), valid_ids),
            )
        )
    actions: list[ActionItem] = []
    for item in payload.get("action_items", []):
        if not isinstance(item, dict):
            continue
        title = _text(item.get("title"))
        source_ids = _ids(item.get("source_message_ids"), valid_ids)
        if not title or not source_ids:
            warnings.append(f"근거가 없는 Action Item 제외: {title or '(제목 없음)'}")
            continue
        action_corpus = " ".join(
            [title, _text(item.get("description")), _text(item.get("rationale"))]
        )
        if UNSAFE_MEMORY_ACTION.search(action_corpus):
            warnings.append(f"안전정책 위반 Action Item 제외: {title}")
            continue
        priority = str(item.get("priority", "medium")).lower()
        if priority not in {"critical", "high", "medium", "low"}:
            priority = "medium"
        actions.append(
            ActionItem(
                action_id=_action_id(episode.meeting_id, title, source_ids),
                title=title,
                description=_text(item.get("description")),
                source_message_ids=source_ids,
                owner=_text(item.get("owner")) or None,
                priority=priority,  # type: ignore[arg-type]
                deadline=_text(item.get("deadline")) or None,
                project_id=(
                    _text(item.get("project_id")) or None
                ),
                rationale=_text(item.get("rationale")),
            )
        )
    try:
        confidence = float(payload.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return MeetingMinutes(
        meeting_id=episode.meeting_id,
        title=_text(payload.get("title")) or f"회의 {episode.meeting_id}",
        background=_text(payload.get("background")) or "원문에서 배경이 명시되지 않음",
        agenda=_strings(payload.get("agenda")),
        discussion_flow=discussion_flow,
        agent_positions=agent_positions,
        professor_positions=professor_positions,
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
        meeting_type=classify_meeting_type(payload, episode),  # type: ignore[arg-type]
        agent_activity_log=activity_log,
        temporal_conflicts=temporal_conflicts,
        recommendation=_text(payload.get("recommendation")),
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
        meeting_type="other",
        recommendation="자동 전송 전에 재분석할 것",
    )

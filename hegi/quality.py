"""Grounding-oriented meeting-minutes quality checks."""

from __future__ import annotations

import re
from dataclasses import fields, is_dataclass
from typing import Any

from .models import MeetingEpisode, MeetingMinutes


_OVERCONFIDENT = re.compile(r"(완벽히|확실히|명백히)\s+(합의|입증|검증)")
_RAW_REPR = re.compile(
    r"(?m)(?:^|[\s:])(?:\[\s*(?:\{['\"]|['\"]|[-+]?\d+\s*,)|"
    r"\{\s*['\"][A-Za-z가-힣_][^}]*['\"]\s*:|"
    r"<[A-Za-z_][\w.]* object at 0x[0-9a-f]+>)",
    re.IGNORECASE,
)
_UNSAFE_MEMORY_WRITE = re.compile(
    r"(Memory\s*Forest|메모리\s*포레스트|기억\s*숲).{0,40}"
    r"(직접\s*(쓰기|저장)|쓰기\s*활성화|자동\s*(커밋|승인)|"
    r"auto[_ -]?commit\s*[:=]\s*true)",
    re.IGNORECASE,
)
_PROFESSOR_ALIASES = {"교수", "교수님", "professor", "user", "사용자"}


class QualityGateError(ValueError):
    """Raised when a final HEGI output violates a hard publication rule."""


def _walk_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if is_dataclass(value):
        result: list[str] = []
        for item in fields(value):
            if item.name == "raw":
                continue
            result.extend(_walk_text(getattr(value, item.name)))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_walk_text(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_walk_text(item))
        return result
    return []


def hard_quality_violations(
    minutes: MeetingMinutes,
    episode: MeetingEpisode,
    *,
    rendered: str = "",
) -> list[str]:
    violations: list[str] = []
    if minutes.meeting_type not in {
        "research_meeting",
        "operational_incident",
        "mixed",
        "other",
    }:
        violations.append(f"잘못된 meeting_type: {minutes.meeting_type}")
    if any(
        message.role == "user" and message.source_agent != "교수"
        for message in episode.messages
    ):
        violations.append("교수 메시지 화자 정규화 누락")
    allowed_agents = {
        message.source_agent
        for message in episode.messages
        if message.role == "assistant"
    } | {name for name in episode.participants if name != "교수"}
    for position in minutes.agent_positions:
        if (
            not position.agent
            or position.agent.lower() in _PROFESSOR_ALIASES
            or position.agent not in allowed_agents
        ):
            violations.append(f"잘못된 에이전트 화자명: {position.agent or '(빈 값)'}")
    for activity in minutes.agent_activity_log:
        if activity.agent not in allowed_agents:
            violations.append(f"잘못된 행동 로그 화자명: {activity.agent or '(빈 값)'}")
    corpus = "\n".join([*_walk_text(minutes), rendered])
    if _RAW_REPR.search(corpus):
        violations.append("raw Python repr 감지")
    if _UNSAFE_MEMORY_WRITE.search(corpus):
        violations.append("Memory Forest 직접 쓰기 안전정책 위반 문구 감지")
    return list(dict.fromkeys(violations))


def enforce_quality_gate(
    minutes: MeetingMinutes,
    episode: MeetingEpisode,
    *,
    rendered: str = "",
) -> None:
    violations = hard_quality_violations(minutes, episode, rendered=rendered)
    if violations:
        raise QualityGateError("; ".join(violations))


def audit_minutes(minutes: MeetingMinutes, episode: MeetingEpisode) -> list[str]:
    warnings: list[str] = []
    valid_ids = {message.message_id for message in episode.messages}
    if not minutes.agenda:
        warnings.append("핵심 의제가 비어 있음")
    if not minutes.discussion_flow:
        warnings.append("논의 전개가 비어 있음")
    if not minutes.agent_positions:
        warnings.append("에이전트별 기여가 비어 있음")
    for stage in minutes.discussion_flow:
        if not stage.source_message_ids:
            warnings.append(f"논의 단계 근거 누락: {stage.heading}")
        elif not set(stage.source_message_ids) <= valid_ids:
            warnings.append(f"논의 단계에 존재하지 않는 message ID: {stage.heading}")
    for item in minutes.action_items:
        if not item.source_message_ids:
            warnings.append(f"Action Item 근거 누락: {item.title}")
        if not item.rationale:
            warnings.append(f"Action Item rationale 누락: {item.title}")
    if minutes.memory_evaluation and not minutes.memory_evaluation.reasons:
        warnings.append("Memory recommendation 이유 누락")
    corpus = " ".join(
        [minutes.background, *minutes.agreements, *minutes.research_direction]
    )
    if _OVERCONFIDENT.search(corpus):
        warnings.append("과도하게 확정적인 표현 검토 필요")
    return warnings

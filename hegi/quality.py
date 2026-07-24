"""Grounding-oriented meeting-minutes quality checks."""

from __future__ import annotations

import re

from .models import MeetingEpisode, MeetingMinutes


_OVERCONFIDENT = re.compile(r"(완벽히|확실히|명백히)\s+(합의|입증|검증)")


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

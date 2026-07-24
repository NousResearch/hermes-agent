from __future__ import annotations

from dataclasses import replace

import pytest

from hegi.analyzer import build_minutes
from hegi.archive import render_markdown
from hegi.collector import deduplicate_messages
from hegi.memory import DraftGate, MemoryEvaluator
from hegi.models import (
    ActionItem,
    AgentActivity,
    AgentPosition,
    MeetingEpisode,
    MeetingMinutes,
    MemoryEvaluation,
    SourceMessage,
    TemporalConflict,
)
from hegi.quality import QualityGateError, enforce_quality_gate
from hegi.state import StateStore


def _message(
    message_id: int,
    *,
    role: str,
    agent: str,
    content: str,
    platform_id: str,
    timestamp: float,
) -> SourceMessage:
    return SourceMessage(
        source_agent=agent,
        source_db=f"/tmp/{agent}.db",
        message_id=message_id,
        session_id=f"s-{agent}",
        platform_message_id=platform_id,
        chat_id="-1",
        chat_type="group",
        role=role,  # type: ignore[arg-type]
        content=content,
        timestamp=timestamp,
    )


def _episode(*, operational: bool = False) -> MeetingEpisode:
    messages = [
        _message(
            1,
            role="user",
            agent="교수",
            content="daemon 부팅 실패를 점검해" if operational else "이 개념을 연구하자",
            platform_id="professor-1",
            timestamp=100,
        ),
        _message(
            2,
            role="assistant",
            agent="헤헤",
            content="설정을 점검했다" if operational else "새 개념을 제안한다",
            platform_id="agent-1",
            timestamp=101,
        ),
        _message(
            3,
            role="assistant",
            agent="헤코",
            content="daemon을 복구했다" if operational else "개념의 범위를 보완한다",
            platform_id="agent-2",
            timestamp=102,
        ),
    ]
    return MeetingEpisode(
        meeting_id="meeting-policy",
        chat_id="-1",
        started_at=100,
        ended_at=200,
        participants=["헤헤", "헤코"],
        messages=messages,
        episode_hash="hash",
        status="quiet",
    )


def _minutes(episode: MeetingEpisode) -> MeetingMinutes:
    return MeetingMinutes(
        meeting_id=episode.meeting_id,
        title="정책 검증",
        background="현재 상태를 검증함",
        agenda=["상태 확인"],
        discussion_flow=[],
        agent_positions=[
            AgentPosition("헤헤", "연구적 판단", ["개념 의견"], [2])
        ],
        professor_positions=["점검 지시"],
        agreements=["복구 완료"],
        disagreements=[],
        unresolved_questions=[],
        new_concepts=[],
        evidence_and_sources=[],
        research_direction=[],
        action_items=[],
        memory_evaluation=MemoryEvaluation(
            searched_queries=["정책 검증"],
            recommendation="merge_existing",
            search_findings=["정책 검증: 1개 후보 검색"],
            duplicate_targets=["기존 정책 (memory-1, 높은 중복)"],
            novelty_basis=["복구 절차만 새로 확인됨"],
            reasons=["기존 기억과 중복됨"],
        ),
        confidence=0.9,
        warnings=[],
        meeting_type="operational_incident",
        agent_activity_log=[
            AgentActivity("헤헤", "설정 점검", "원인 확인", [2]),
            AgentActivity("헤코", "daemon 재시작", "복구 완료", [3]),
        ],
        temporal_conflicts=[
            TemporalConflict(
                "daemon 상태",
                "부팅 실패",
                "정상 실행",
                "resolved",
                [1, 3],
            )
        ],
    )


def test_professor_cross_db_duplicate_is_single_and_normalized():
    first = _message(
        1,
        role="user",
        agent="헤헤",
        content="같은 교수 발언",
        platform_id="professor-1",
        timestamp=100,
    )
    second = replace(first, source_agent="헤코", source_db="/tmp/heco.db", message_id=9)

    merged = deduplicate_messages([first, second])

    assert len(merged) == 1
    assert merged[0].source_agent == "교수"


def test_analyzer_separates_activity_normalizes_professor_and_filters_unsafe_action():
    episode = _episode(operational=True)
    payload = {
        "title": "daemon 장애",
        "background": {"raw": "dict must not render"},
        "agenda": [{"raw": "list item must not render"}, "복구"],
        "discussion_flow": [],
        "agent_positions": [
            {
                "agent": "교수님",
                "position": "복구를 지시함",
                "source_message_ids": [1],
            },
            {
                "agent": "헤헤",
                "position": "원인을 설정 오류로 판단함",
                "source_message_ids": [2],
            },
        ],
        "agent_activity_log": [
            {
                "agent": "헤헤",
                "activity": "설정 확인",
                "result": "오류 발견",
                "source_message_ids": [2],
            }
        ],
        "professor_positions": [],
        "temporal_conflicts": [
            {
                "subject": "daemon",
                "earlier_state": "실패",
                "current_state": "복구",
                "resolution_status": "resolved",
                "source_message_ids": [1, 3],
            }
        ],
        "agreements": ["복구됨"],
        "disagreements": [],
        "unresolved_questions": [],
        "new_concepts": [],
        "evidence_and_sources": [],
        "research_direction": [],
        "action_items": [
            {
                "title": "Memory Forest 직접 쓰기 활성화",
                "description": "자동 저장한다",
                "source_message_ids": [2],
                "priority": "high",
                "rationale": "편의성",
            }
        ],
        "confidence": 0.9,
        "warnings": [],
        "recommendation": "상태 관찰",
    }

    minutes = build_minutes(payload, episode)

    assert minutes.meeting_type == "operational_incident"
    assert minutes.background == "원문에서 배경이 명시되지 않음"
    assert minutes.agenda == ["복구"]
    assert [item.agent for item in minutes.agent_positions] == ["헤헤"]
    assert minutes.professor_positions == ["복구를 지시함"]
    assert minutes.agent_activity_log[0].activity == "설정 확인"
    assert minutes.temporal_conflicts[0].resolution_status == "resolved"
    assert minutes.action_items == []
    assert any("안전정책 위반" in warning for warning in minutes.warnings)


def test_operational_renderer_uses_distinct_template_and_no_python_repr():
    episode = _episode(operational=True)

    rendered = render_markdown(_minutes(episode), episode)

    assert rendered.startswith("# 운영 장애 기록")
    assert "## 5. 에이전트 행동 로그" in rendered
    assert "## 6. 연구적 의견과 해석" in rendered
    assert "이전 `부팅 실패` → 현재 `정상 실행` [resolved]" in rendered
    assert "기존 정책 (memory-1, 높은 중복)" in rendered
    assert "source_message_ids: 1, 2, 3" in rendered
    assert "{'" not in rendered
    assert "[1, 2" not in rendered


def test_quality_gate_blocks_raw_repr_wrong_speaker_and_memory_write_policy():
    episode = _episode()
    minutes = _minutes(episode)
    minutes.meeting_type = "research_meeting"
    minutes.agent_positions = [
        AgentPosition("알 수 없는 화자", "의견", [], [2])
    ]
    minutes.background = "{'unsafe': 'python repr'}"
    minutes.action_items = [
        ActionItem(
            "act-1",
            "Memory Forest 직접 쓰기 활성화",
            "자동 Commit을 켠다",
            [2],
            None,
            "high",
        )
    ]

    with pytest.raises(QualityGateError) as error:
        enforce_quality_gate(minutes, episode)

    message = str(error.value)
    assert "raw Python repr" in message
    assert "잘못된 에이전트 화자명" in message
    assert "Memory Forest 직접 쓰기" in message


class _MemoryBackend:
    def __init__(self):
        self.drafts: list[dict] = []

    def search(self, query: str, limit: int = 5):
        return {"results": []}

    def create_draft(self, arguments: dict):
        self.drafts.append(arguments)
        return {"state": "pending"}


def test_operational_incident_uses_no_memory_and_cannot_create_draft(tmp_path):
    episode = _episode(operational=True)
    minutes = _minutes(episode)
    backend = _MemoryBackend()
    evaluation = MemoryEvaluator(backend).evaluate(minutes)
    state = StateStore(tmp_path / "state.db")
    state.save_episode(
        minutes.meeting_id,
        episode.episode_hash,
        {"meeting_id": minutes.meeting_id},
        "reported",
    )
    gate = DraftGate(state, backend, professor_user_ids=["42"])
    gate.approve(
        meeting_id=minutes.meeting_id,
        text="기억해",
        user_id="42",
        platform_message_id="approval-1",
    )

    assert evaluation.recommendation == "no_memory"
    assert evaluation.search_findings
    with pytest.raises(PermissionError, match="no_memory"):
        gate.create_draft_after_recheck(
            minutes,
            evaluation,
            project="media_aesthetics",
        )
    assert backend.drafts == []

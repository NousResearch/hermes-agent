"""Tests for Context Continuity Manager handoff packet generation."""

from agent.context_continuity import (
    ContextContinuityStatus,
    build_handoff_packet,
    recommend_continuity_action,
    should_defer_automatic_compression,
)
from hermes_cli.commands import resolve_command


def test_handoff_packet_is_structured_for_new_session_resume():
    messages = [
        {"role": "user", "content": "Build Context Continuity Manager for long sessions."},
        {"role": "assistant", "content": "Plan: add a handoff packet first."},
        {"role": "tool", "name": "read_file", "content": "read cli.py"},
        {"role": "assistant", "content": "Implemented tests for the handoff generator."},
    ]

    packet = build_handoff_packet(
        messages,
        session_id="sess-123",
        context_tokens=75000,
        context_length=100000,
        current_step="Add CLI /handoff command",
        created_at="2026-05-13T16:30:00Z",
    )

    assert packet.startswith("[이동 준비: 새 세션 이어가기 안내]")
    for heading in [
        "## 운영 상태 체크리스트",
        "## 현재 상태",
        "## 계승된 운영 요약",
        "## 이번 세션 visible 메시지 outline",
        "## 이어가기 지시",
    ]:
        assert heading in packet
    assert "원본 세션: sess-123" in packet
    assert "source_session_id: sess-123" in packet
    assert "packet_scope: operational_summary_plus_bounded_visible_outline" in packet
    assert "packet_limit_note: not_full_raw_transcript" in packet
    assert "message_count: 4" in packet
    assert "created_at: 2026-05-13T16:30:00Z" in packet
    hash_line = next(line for line in packet.splitlines() if line.startswith("packet_hash: "))
    assert hash_line.startswith("packet_hash: sha256:")
    assert len(hash_line.removeprefix("packet_hash: sha256:")) == 64
    assert "75%" in packet
    assert "Build Context Continuity Manager" in packet
    assert "Add CLI /handoff command" in packet
    assert "먼저 실제 파일/상태를 확인" in packet


def test_handoff_packet_is_concise_and_does_not_claim_automatic_move():
    messages = [
        {"role": "user", "content": "Continue the WebUI session move implementation."},
        {"role": "assistant", "content": "Verified the MVP and found the next slice."},
    ]

    packet = build_handoff_packet(messages, session_id="sess-source")

    assert len(packet.splitlines()) <= 50
    assert "대상 세션" not in packet
    assert "전달 완료" not in packet
    assert "자동 생성" not in packet
    assert "target_session_id" not in packet
    assert "packet_scope: operational_summary_plus_bounded_visible_outline" in packet
    assert "전체 원문" in packet
    assert "message_count: 2" in packet


def test_recommend_continuity_action_prefers_handoff_before_compression():
    status = recommend_continuity_action(
        ContextContinuityStatus(
            context_tokens=85000,
            context_length=100000,
            remaining_todos=3,
            compression_count=0,
            high_risk_task=True,
        )
    )

    assert status.level == "strong_handoff"
    assert status.recommended_action == "handoff"
    assert "새 세션" in status.reason


def test_automatic_compression_is_deferred_for_continuity_handoff_policy():
    status = ContextContinuityStatus(
        context_tokens=85_000,
        context_length=100_000,
        remaining_todos=2,
        high_risk_task=True,
    )

    decision = should_defer_automatic_compression(status)

    assert decision.defer is True
    assert decision.recommended_action == "handoff"
    assert "/m" in decision.reason
    assert "/h" in decision.reason
    assert "자동 압축" not in decision.reason


def test_automatic_compression_is_deferred_even_before_handoff_threshold():
    status = ContextContinuityStatus(
        context_tokens=55_000,
        context_length=100_000,
    )

    decision = should_defer_automatic_compression(status)

    assert decision.defer is True
    assert decision.recommended_action == "continue"
    assert "자동 압축" not in decision.reason
    assert "/m" in decision.reason


def test_handoff_command_registered():
    cmd = resolve_command("handoff")
    assert cmd is not None
    assert cmd.name == "handoff"
    assert "이어가기 안내" in cmd.description

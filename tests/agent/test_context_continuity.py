"""Tests for Context Continuity Manager handoff packet generation."""

from agent.context_continuity import (
    ContextContinuityStatus,
    build_handoff_packet,
    recommend_continuity_action,
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
    )

    assert packet.startswith("[세션 이동 인계문]")
    for heading in [
        "## 목표",
        "## 현재 상태",
        "## 완료한 것",
        "## 중요 결정",
        "## 변경/검증 상태",
        "## 다음 작업",
        "## 완료 기준",
    ]:
        assert heading in packet
    assert "원본 세션: sess-123" in packet
    assert "기준: 현재 세션 전체 요약" in packet
    hash_line = next(line for line in packet.splitlines() if line.startswith("본문 해시: "))
    assert hash_line.startswith("본문 해시: sha256:")
    assert len(hash_line.removeprefix("본문 해시: sha256:")) == 64
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

    assert len(packet.splitlines()) <= 42
    assert "대상 세션" not in packet
    assert "전달 완료" not in packet
    assert "자동 생성" not in packet
    assert "message_count" not in packet


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


def test_handoff_command_registered():
    cmd = resolve_command("handoff")
    assert cmd is not None
    assert cmd.name == "handoff"
    assert "이어가기 안내" in cmd.description

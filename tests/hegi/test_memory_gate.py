from __future__ import annotations

import pytest

from hegi.memory import DraftGate, MemoryEvaluator, parse_approval_command
from hegi.models import MeetingMinutes
from hegi.state import StateStore


class FakeMemory:
    def __init__(self):
        self.searches: list[str] = []
        self.drafts: list[dict] = []

    def search(self, query: str, limit: int = 5):
        self.searches.append(query)
        return {"results": []}

    def create_draft(self, arguments):
        self.drafts.append(arguments)
        return {"ok": True, "draft_id": "MF-1", "state": "pending"}


def _minutes() -> MeetingMinutes:
    return MeetingMinutes(
        meeting_id="meeting",
        title="새 개념 회의",
        background="배경",
        agenda=["의제"],
        discussion_flow=[],
        agent_positions=[],
        professor_positions=["검토"],
        agreements=["새 원칙을 잠정 채택"],
        disagreements=[],
        unresolved_questions=[],
        new_concepts=[],
        evidence_and_sources=[],
        research_direction=["후속 논문에 반영"],
        action_items=[],
        memory_evaluation=None,
        confidence=0.8,
        warnings=[],
    )


def test_approval_parser_recognizes_supported_phrases():
    assert parse_approval_command("@헤기 초안 만들어") == "draft"
    assert parse_approval_command("@헤기 기억하지 마") == "reject"
    assert parse_approval_command("그냥 확인") is None


def test_draft_is_blocked_until_authorized_approval_and_rechecks(tmp_path):
    state = StateStore(tmp_path / "state.db")
    state.save_episode("meeting", "hash", {"meeting_id": "meeting"}, "reported")
    backend = FakeMemory()
    evaluation = MemoryEvaluator(backend).evaluate(_minutes())
    gate = DraftGate(state, backend, professor_user_ids=["42"])

    with pytest.raises(PermissionError):
        gate.create_draft_after_recheck(
            _minutes(), evaluation, project="media_aesthetics"
        )
    with pytest.raises(PermissionError):
        gate.approve(
            meeting_id="meeting",
            text="@헤기 초안 만들어",
            user_id="7",
            platform_message_id="m-1",
        )

    assert (
        gate.approve(
            meeting_id="meeting",
            text="@헤기 초안 만들어",
            user_id="42",
            platform_message_id="m-2",
        )
        == "draft"
    )
    before = len(backend.searches)
    result = gate.create_draft_after_recheck(
        _minutes(), evaluation, project="media_aesthetics"
    )
    assert result["state"] == "pending"
    assert len(backend.searches) > before
    assert backend.drafts[0]["status"] == "provisional"


def test_reject_approval_never_creates_draft(tmp_path):
    state = StateStore(tmp_path / "state.db")
    state.save_episode("meeting", "hash", {"meeting_id": "meeting"}, "reported")
    backend = FakeMemory()
    evaluation = MemoryEvaluator(backend).evaluate(_minutes())
    gate = DraftGate(state, backend, professor_user_ids=["42"])
    gate.approve(
        meeting_id="meeting",
        text="@헤기 기억하지 마",
        user_id="42",
        platform_message_id="m-3",
    )
    with pytest.raises(PermissionError):
        gate.create_draft_after_recheck(
            _minutes(), evaluation, project="media_aesthetics"
        )
    assert backend.drafts == []

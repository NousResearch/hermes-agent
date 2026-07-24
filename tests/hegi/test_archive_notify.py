from __future__ import annotations

from pathlib import Path

import pytest

from hegi.archive import ArchiveManager
from hegi.models import MeetingEpisode, MeetingMinutes, MemoryEvaluation
from hegi.notify import TelegramReporter
from hegi.state import StateStore


def _minutes() -> MeetingMinutes:
    return MeetingMinutes(
        meeting_id="meeting",
        title="연구 회의",
        background="배경",
        agenda=["의제"],
        discussion_flow=[],
        agent_positions=[],
        professor_positions=[],
        agreements=["합의"],
        disagreements=[],
        unresolved_questions=[],
        new_concepts=[],
        evidence_and_sources=[],
        research_direction=[],
        action_items=[],
        memory_evaluation=MemoryEvaluation(
            recommendation="no_memory", reasons=["일시적 정보"]
        ),
        confidence=0.8,
        warnings=[],
        metadata={"hegi_version": "2.0.0"},
    )


def _episode() -> MeetingEpisode:
    return MeetingEpisode(
        meeting_id="meeting",
        chat_id="-1",
        started_at=100,
        ended_at=200,
        participants=["헤헤", "헤코"],
        messages=[],
        episode_hash="hash",
        status="quiet",
    )


def test_archive_preserves_revisions_and_spools_when_nas_offline(tmp_path):
    manager = ArchiveManager(tmp_path / "spool", tmp_path / "missing-nas")
    first = manager.archive(_minutes(), _episode())
    second = manager.archive(_minutes(), _episode())
    assert Path(first["markdown"]).is_file()
    assert ".r2.md" in str(second["markdown"])
    assert first["nas_markdown"] is None


def test_telegram_resume_skips_already_sent_parts(tmp_path):
    state = StateStore(tmp_path / "state.db")
    calls: list[str] = []

    def flaky(_token, _chat_id, content, **_kwargs):
        calls.append(content)
        if content.startswith("2/4"):
            raise OSError("network")
        return {"message_id": len(calls)}

    reporter = TelegramReporter(
        state, token="not-a-real-token", chat_id="-1", sender=flaky, attempts=1
    )
    with pytest.raises(OSError):
        reporter.send(_minutes())
    assert state.delivered_parts("meeting") == {1}

    resumed_calls: list[str] = []

    def healthy(_token, _chat_id, content, **_kwargs):
        resumed_calls.append(content)
        return {"message_id": len(resumed_calls) + 10}

    TelegramReporter(
        state, token="not-a-real-token", chat_id="-1", sender=healthy
    ).send(_minutes())
    assert len(resumed_calls) == 3
    assert all(not content.startswith("1/4") for content in resumed_calls)
    assert state.delivered_parts("meeting") == {1, 2, 3, 4}

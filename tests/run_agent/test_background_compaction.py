"""Regression tests for stale-safe background compaction jobs."""
from __future__ import annotations

from agent.background_compaction import BackgroundCompactionManager, CompactionSnapshot
from threading import Event


def test_background_compaction_stores_fresh_result_without_blocking_active_turn():
    manager = BackgroundCompactionManager()
    snapshot = CompactionSnapshot(
        session_id="child-session",
        parent_session_id="parent-session",
        generation=1,
        message_count=3,
        message_hash="abc123",
        focus_topic="memorymunch",
    )
    calls = []
    release_builder = Event()

    def build_summary(snap: CompactionSnapshot) -> str:
        calls.append(snap.session_id)
        release_builder.wait(timeout=2)
        return "RICH_BACKGROUND_SUMMARY"

    job = manager.schedule(
        snapshot,
        build_summary,
        is_current=lambda snap: snap.generation == 1,
    )

    assert job.done is False
    release_builder.set()
    result = manager.wait(job.job_id, timeout=2)

    assert calls == ["child-session"]
    assert result is not None
    assert result.applied is True
    assert result.stale is False
    assert result.summary == "RICH_BACKGROUND_SUMMARY"
    assert manager.latest_for_session("child-session").summary == "RICH_BACKGROUND_SUMMARY"


def test_background_compaction_discards_stale_result():
    manager = BackgroundCompactionManager()
    snapshot = CompactionSnapshot(
        session_id="child-session",
        parent_session_id="parent-session",
        generation=1,
        message_count=3,
        message_hash="oldhash",
    )

    job = manager.schedule(
        snapshot,
        lambda snap: "STALE_SUMMARY_SHOULD_NOT_APPLY",
        is_current=lambda snap: False,
    )
    result = manager.wait(job.job_id, timeout=2)

    assert result is not None
    assert result.applied is False
    assert result.stale is True
    assert manager.latest_for_session("child-session") is None

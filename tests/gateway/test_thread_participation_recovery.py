"""A temporary publication failure must not strand an active thread only in memory."""

import asyncio
import json

import pytest

from gateway.platforms.helpers import ThreadParticipationTracker


@pytest.mark.parametrize("async_mark", [False, True])
def test_failed_thread_publication_is_retried_after_storage_recovers(tmp_path, monkeypatch, async_mark):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state_path = tmp_path / "discord_threads.json"
    state_path.mkdir()  # A real I/O obstruction at the snapshot destination.
    tracker = ThreadParticipationTracker("discord")
    thread_id = "discussion-1"

    def mark():
        if async_mark:
            asyncio.run(tracker.mark_async(thread_id))
        else:
            tracker.mark(thread_id)

    with pytest.raises(OSError):
        mark()
    assert thread_id in tracker  # Inbound gating must remain immediate.
    state_path.rmdir()
    mark()

    assert json.loads(state_path.read_text(encoding="utf-8")) == [thread_id]
    assert thread_id in ThreadParticipationTracker("discord")

"""Refused idle commits must not extract memory from an uncommitted boundary."""

import time
from unittest.mock import MagicMock

import pytest

from agent import conversation_compression as cc
from hermes_state_idle import IdleCompactionSuperseded


@pytest.mark.parametrize("error, cooldown_expected", [
    (IdleCompactionSuperseded("idle compaction fence lost"), False),
    (OSError("database write failed"), True),
])
def test_stale_idle_commit_has_no_memory_extraction(monkeypatch, error, cooldown_expected):
    agent = MagicMock()
    agent.session_id = "session"
    agent._post_reply_idle_claim = (1, 4, "worker")
    agent._session_db.archive_and_compact.side_effect = error
    agent._persist_user_message_idx = None
    lease = MagicMock(holder="compressor", watermark=4)
    attempt = MagicMock(started_at=time.monotonic())
    messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "reply"}]
    compressed = [{"role": "user", "content": "summary"}]
    monkeypatch.setattr(cc, "_salvage_or_refuse_grown_transcript", lambda *args, **kwargs: (compressed, None))
    monkeypatch.setattr(cc, "_held_watermark", lambda *args, **kwargs: 4)
    monkeypatch.setattr(cc, "_restore_prune_rearm_tokens", lambda *args: None)
    from agent import conversation_compression_archive as archive
    monkeypatch.setattr(archive, "coverage_for_commit", lambda *args: ([], []))

    result = cc._commit_compaction(
        agent, messages, compressed, in_place=True, lease=lease,
        new_system_prompt="pinned", system_message="pinned", compressed_user_turn_outcome="",
        messages_before_compression=list(messages), made_progress=True, attempt=attempt,
    )
    assert result.session_commit_succeeded is False
    agent.commit_memory_session.assert_not_called()
    if cooldown_expected:
        agent.context_compressor._record_compression_failure_cooldown.assert_called_once()
    else:
        agent.context_compressor._record_compression_failure_cooldown.assert_not_called()

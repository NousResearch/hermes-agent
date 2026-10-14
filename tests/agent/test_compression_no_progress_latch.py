"""A compression that does not shrink must not latch awaiting_real_usage."""

from types import SimpleNamespace

from agent.conversation_compression import _salvage_or_refuse_grown_transcript


def test_unchanged_transcript_does_not_latch_usage():
    compressor = SimpleNamespace(
        awaiting_real_usage_after_compression=True,
        _last_compression_made_progress=True,
        rejected=False,
    )

    def record_rejected_compaction():
        compressor.rejected = True

    compressor.record_rejected_compaction = record_rejected_compaction
    agent = SimpleNamespace(
        session_id="sess",
        context_compressor=compressor,
        _cached_system_prompt="system",
    )
    messages = [{"role": "user", "content": "same photo turn"}]
    compressed, _prompt = _salvage_or_refuse_grown_transcript(
        agent,
        messages,
        [dict(messages[0])],
        system_message="system",
        attempt_started_at=0.0,
        attempt_snapshot={},
    )
    assert compressed is None
    assert compressor.awaiting_real_usage_after_compression is False
    assert compressor._last_compression_made_progress is False
    assert compressor.rejected is True

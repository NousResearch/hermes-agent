"""A same-count noop/sanitized compress is housekeeping, not a summary attempt.

Counting it toward the per-turn cap locks the session out (attempts_exhausted)
while the provider prompt is still over the threshold. A real summary — fewer
messages, or a compacted status — still proceeds to commit.
"""

from types import SimpleNamespace

from agent.conversation_compression import (
    _candidate_rejected,
    compression_attempt_was_housekeeping,
)


def _agent(status: str):
    compressor = SimpleNamespace(
        _last_compression_status=status,
        _last_compress_aborted=False,
        _last_compression_made_progress=True,
        awaiting_real_usage_after_compression=True,
        _compression_attempt_generation=0,
    )
    return SimpleNamespace(
        context_compressor=compressor,
        session_id="session",
        _compression_attempt_noop=None,
    ), compressor


def test_same_count_sanitized_pass_is_housekeeping_and_restores_the_transcript():
    agent, compressor = _agent("sanitized")
    before = [
        {"role": "user", "content": "keep"},
        {"role": "assistant", "content": "keep"},
    ]
    live = [dict(row) for row in before]
    live[1]["content"] = "stubbed in place"
    sanitized = [dict(row) for row in live]

    rejected = _candidate_rejected(
        agent, sanitized, live, before, attempt_generation=0, attempt_started_at=0.0,
    )

    assert rejected is True
    assert compression_attempt_was_housekeeping(agent) is True
    assert live == before
    assert compressor._last_compression_made_progress is False
    assert compressor.awaiting_real_usage_after_compression is False


def test_same_count_noop_is_housekeeping():
    agent, _compressor = _agent("noop")
    before = [{"role": "user", "content": "only"}]
    rejected = _candidate_rejected(
        agent, list(before), list(before), before,
        attempt_generation=0, attempt_started_at=0.0,
    )
    assert rejected is True
    assert compression_attempt_was_housekeeping(agent) is True


def test_shorter_compacted_transcript_is_not_housekeeping():
    agent, _compressor = _agent("compacted")
    before = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "recent"},
    ]
    compressed = [{"role": "user", "content": "summary"}, {"role": "user", "content": "recent"}]

    rejected = _candidate_rejected(
        agent, compressed, list(before), before,
        attempt_generation=0, attempt_started_at=0.0,
    )

    assert rejected is False
    assert compression_attempt_was_housekeeping(agent) is False


def test_missing_status_does_not_look_like_housekeeping():
    """Built-in compressor has no ``_last_compression_status``. A MagicMock
    auto-attribute must not classify a real summary as housekeeping."""
    agent, _compressor = _agent("sanitized")
    del agent.context_compressor._last_compression_status
    before = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    compressed = [{"role": "user", "content": "summary"}]

    rejected = _candidate_rejected(
        agent, compressed, list(before), before,
        attempt_generation=0, attempt_started_at=0.0,
    )

    assert rejected is False
    assert compression_attempt_was_housekeeping(agent) is False

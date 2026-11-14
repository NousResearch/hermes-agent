"""Pruning is recovery evidence, not a successful summary or a new transcript shape."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import compress_context
from agent.turn_usage import record_response_usage
from tests.agent.test_compression_prune_only_commit import _messages
from tests.agent.test_turn_usage_log_line import _agent


@pytest.mark.parametrize("prompt", [None, 32_768, 40_000, 10_000])
def test_committed_pruning_is_summary_neutral_and_only_real_headroom_rearms(tmp_path, monkeypatch, prompt):
    agent = _agent(tmp_path, monkeypatch)
    compressor = ContextCompressor(model="test/model", config_context_length=65_536,
                                   threshold_tokens_cap=32_768, quiet_mode=True)
    agent.context_compressor = compressor
    agent._compression_feasibility_checked = True
    compressor._fallback_compression_streak = 2
    try:
        with patch.object(compressor, "_generate_summary", side_effect=AssertionError("summary must not run")):
            messages, _ = compress_context(agent, _messages(), "system", approx_tokens=74_000)
        assert compressor._fallback_compression_streak == 2
        assert compressor._verify_compaction_cleared_threshold
        usage = None if prompt is None else SimpleNamespace(prompt_tokens=prompt, completion_tokens=1,
                                                            total_tokens=prompt + 1)
        outcome = record_response_usage(agent, SimpleNamespace(usage=usage), messages=messages,
            api_call_count=1, api_duration=0.1, compression_attempts=3, max_compression_attempts=3)
        assert outcome.rearmed == (prompt == 10_000)
        assert outcome.compression_attempts == (0 if prompt == 10_000 else 3)
        assert compressor._fallback_compression_streak == 2
    finally:
        agent.close()


def test_structural_guard_survives_rebind_and_compression_lineage_but_not_new_session(tmp_path):
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-a", source="cli")
    db.create_session("session-b", source="cli", parent_session_id="session-a")
    compressor = ContextCompressor(model="test/model", config_context_length=65_536, quiet_mode=True)
    compressor.bind_session_state(db, "session-a")
    compressor.compress([{"role": "user", "content": "short"}], current_tokens=60_000)
    deadline = compressor._structural_no_op_backoff_until
    assert deadline > 0
    for session, kwargs in [("session-a", {}), ("session-b", {"boundary_reason": "compression", "old_session_id": "session-a"})]:
        compressor.on_session_start(session, **kwargs)
        assert compressor._structural_no_op_backoff_until == deadline
        assert compressor._structural_no_op_len == 1
        assert not compressor.should_compress(60_000)
    compressor.on_session_start("unrelated")
    assert compressor._structural_no_op_backoff_until == 0
    db.close()

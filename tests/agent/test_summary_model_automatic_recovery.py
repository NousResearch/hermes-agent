"""Automatic recovery of an old benched session through the durable commit path."""

from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import finalize_context_engine_compression_notification
from hermes_state import SessionDB
from tests.agent.test_compression_rotation_state import _build_agent_with_db


def _compressor(db, session_id="protected"):
    compressor = ContextCompressor(
        model="test/model", config_context_length=100_000,
        threshold_percent=0.85, protect_first_n=2, protect_last_n=2,
        quiet_mode=True, proactive_prune_tokens=0,
    )
    compressor.bind_session_state(db, session_id)
    return compressor


def _messages():
    messages = [{"role": "system", "content": "stable system"}]
    for i in range(20):
        messages.extend([
            {"role": "user", "content": f"synthetic question {i} " + "a " * 2000},
            {"role": "assistant", "content": f"synthetic answer {i} " + "b " * 2000},
        ])
    return messages


@pytest.mark.parametrize("in_place", [True, False])
@pytest.mark.parametrize("defer_notification", [True, False])
def test_automatic_bench_commit_persists_probe_for_resumed_conversation(tmp_path, in_place, defer_notification):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("protected", "desktop", model="test/model")
    db.set_compression_fallback_streak("protected", 2)
    db.record_compression_failure_cooldown("protected", 900.0, "synthetic historical failure")
    previous_failure = db.get_compression_failure_cooldown_row("protected")
    compressor = _compressor(db)
    agent = _build_agent_with_db(db, "protected", platform="desktop")
    agent.context_compressor = compressor
    agent.compression_in_place = in_place
    agent._compression_feasibility_checked = True
    messages = _messages()
    for message in messages:
        db.append_message("protected", message["role"], message["content"])
    with patch("agent.context_compressor.time.time", return_value=1000.0), \
            patch.object(compressor, "_fallback_streak_skip", wraps=compressor._fallback_streak_skip) as bench, \
            patch.object(compressor, "_generate_summary", return_value="synthetic healthy summary") as generate:
        returned, _ = agent._compress_context(
            messages, "stable system", approx_tokens=120_000,
            defer_context_engine_notification=defer_notification,
        )
        if defer_notification:
            assert finalize_context_engine_compression_notification(agent, committed=True)
    bench.assert_called_once()
    generate.assert_not_called()
    assert len(returned) < len(messages)
    session_id = agent.session_id
    assert db.get_compression_fallback_streak(session_id) == 2
    assert db.get_session_model_config_value(session_id, "_fallback_probe_at") == (
        1000.0 + compressor._ANTI_THRASH_RECOVERY_SECONDS
    )
    assert db._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", ("protected",)).fetchone()[0] >= len(messages)
    assert db.get_compression_lock_holder("protected") is None
    assert db.get_compression_failure_cooldown_row("protected") == previous_failure

    resumed = _compressor(db, session_id)
    agent.context_compressor = resumed
    with patch("agent.context_compressor.time.time", return_value=1000.0 + resumed._ANTI_THRASH_RECOVERY_SECONDS + 1), \
            patch.object(resumed, "_generate_summary", return_value="synthetic healthy summary") as generate:
        agent._compress_context(_messages(), "stable system", approx_tokens=120_000)
    generate.assert_called_once()
    assert db.get_compression_fallback_streak(agent.session_id) == 0
    assert db.get_session_model_config_value(agent.session_id, "_fallback_probe_at", 0.0) == 0.0
    db.close()


def test_unsuccessful_probe_keeps_streak_and_rebenches_next_attempt(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("protected", "desktop", model="test/model")
    db.set_compression_fallback_streak("protected", 2)
    db.patch_session_model_config("protected", {"_fallback_probe_at": 900.0})
    compressor = _compressor(db)
    agent = _build_agent_with_db(db, "protected", platform="desktop")
    agent.context_compressor = compressor
    agent.compression_in_place = True
    agent._compression_feasibility_checked = True
    with patch("agent.context_compressor.time.time", return_value=1000.0), \
            patch.object(compressor, "_generate_summary", return_value=None) as generate:
        agent._compress_context(_messages(), "stable system", approx_tokens=120_000)
    generate.assert_called_once()
    assert db.get_compression_fallback_streak("protected") == 3
    for now in (1001.0, 1002.0):
        resumed = _compressor(db)
        agent.context_compressor = resumed
        with patch("agent.context_compressor.time.time", return_value=now), \
                patch.object(resumed, "_generate_summary", return_value="must not be called") as generate:
            agent._compress_context(_messages(), "stable system", approx_tokens=120_000)
        generate.assert_not_called()
        assert db.get_compression_fallback_streak("protected") == 3
        assert db.get_session_model_config_value("protected", "_fallback_probe_at") == 1301.0
    db.close()

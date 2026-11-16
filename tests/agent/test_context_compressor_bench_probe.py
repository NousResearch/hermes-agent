"""The summary-model bench (fallback streak >= 2) and its persisted probe window."""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from hermes_state import SessionDB


def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(model="test/model", threshold_percent=0.85, protect_first_n=2, protect_last_n=2, quiet_mode=True)
        _ = c.context_length
        return c


def _messages(n_pairs=10):
    msgs = [{"role": "system", "content": "system prompt"}]
    for i in range(n_pairs):
        msgs += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": "short reply"}]
    return msgs


def test_bench_probe_deadline_survives_a_compressor_rebuild(tmp_path):
    """The gateway rebuilds the agent on cache eviction: a fresh compressor bound to the same
    session must resume the persisted bench window, not restart it (else the probe starves)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", "cli")
    db.set_compression_fallback_streak("s1", 2)
    compressor = _compressor()
    compressor.bind_session_state(db, "s1")
    window = compressor._ANTI_THRASH_RECOVERY_SECONDS

    with patch("agent.context_compressor.time.time", return_value=1000.0), \
            patch.object(compressor, "_generate_summary", return_value="LLM summary") as mock_gen:
        compressor.compress(_messages(), force=False)  # benched, arms the window
    mock_gen.assert_not_called()

    rebuilt = _compressor()
    rebuilt.bind_session_state(db, "s1")
    with patch("agent.context_compressor.time.time", return_value=1000.0 + window + 1), \
            patch.object(rebuilt, "_generate_summary", return_value="LLM summary") as mock_gen:
        rebuilt.compress(_messages(), force=False)
    mock_gen.assert_called_once()

    # Rotating compaction (in_place: false) binds a fresh child row: the deadline must follow it.
    with patch("agent.context_compressor.time.time", return_value=2000.0), \
            patch.object(rebuilt, "_generate_summary", return_value="LLM summary"):
        rebuilt.compress(_messages(), force=False)  # benched again, re-arms the window
    db.create_session("s2", "cli", parent_session_id="s1")
    rebuilt.on_session_start("s2", boundary_reason="compression", old_session_id="s1", session_db=db)
    with patch("agent.context_compressor.time.time", return_value=2000.0 + window + 1), \
            patch.object(rebuilt, "_generate_summary", return_value="LLM summary") as mock_gen:
        rebuilt.compress(_messages(), force=False)
    mock_gen.assert_called_once()

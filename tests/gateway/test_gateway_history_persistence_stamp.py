from types import SimpleNamespace
from agent.session_persistence import _db_flush_collect
from gateway.run import _build_gateway_agent_history


def test_db_flush_collect_does_not_recollect_gateway_history_rows():
    """Verify tool turn flushes do not re-append rebuilt user/assistant rows to state.db (#123462)."""
    stored = [
        {"role": "user", "content": "first", "_db_persisted": True, "timestamp": 1.0},
        {"role": "assistant", "content": "reply", "_db_persisted": True},
    ]
    history, _ = _build_gateway_agent_history(stored)
    new = {"role": "user", "content": "new"}

    agent = SimpleNamespace(
        session_id="s",
        _flushed_db_message_ids=None,
        _last_flushed_db_idx=0,
        _db_flush_scan_prefix=None,
        _persist_user_message_idx=None,
        _pending_cli_user_message=None,
    )
    rows, msgs = _db_flush_collect(agent, list(history) + [new], None)

    assert msgs == [new]

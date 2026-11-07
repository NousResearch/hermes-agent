"""Agent teardown releases engine resources once and detaches an owned session DB."""

import logging
import threading
from unittest.mock import Mock, patch

import pytest

from run_agent import AIAgent


def _bare_agent(**attrs):
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "sid"
    agent._memory_manager = None
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent._context_engine_shutdown_lock = threading.Lock()
    for key, value in attrs.items():
        setattr(agent, key, value)
    return agent


@pytest.mark.parametrize("cleanup", ["close", "gateway", "cli"])
def test_context_engine_shutdown_once(cleanup):
    engine = Mock()
    agent = _bare_agent(context_compressor=engine)
    agent.commit_memory_session([])
    engine.shutdown.assert_not_called()
    engine.on_session_end.reset_mock()

    if cleanup == "close":
        agent.close()
    else:
        agent.shutdown_memory_provider([])
    if cleanup == "cli":
        agent.shutdown_memory_provider([])
    else:
        agent.close()

    engine.on_session_end.assert_called_once_with("sid", [])
    engine.shutdown.assert_called_once_with()


def test_context_engine_shutdown_survives_earlier_teardown_error():
    engine = Mock()
    agent = _bare_agent(context_compressor=engine)
    with patch("run_agent._notify_context_engine_session_end", side_effect=RuntimeError("session end failed")):
        with pytest.raises(RuntimeError, match="session end failed"):
            agent.shutdown_memory_provider([])
    agent.close()
    engine.shutdown.assert_called_once_with()


def test_context_engine_shutdown_error_does_not_break_close(caplog):
    engine = Mock()
    engine.shutdown.side_effect = RuntimeError("engine cleanup failed")
    db = Mock()
    agent = _bare_agent(context_compressor=engine, _session_db=db, _owns_session_db=True)
    with caplog.at_level(logging.DEBUG):
        agent.close()
        agent.close()
    engine.shutdown.assert_called_once_with()
    db.close.assert_called_once_with()
    assert any(r.levelno == logging.DEBUG and "Context engine shutdown failed" in r.message
               for r in caplog.records)


def test_double_close_does_not_use_released_session_db(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="sid", source="desktop")
    agent = _bare_agent(_session_db=db, _owns_session_db=True)
    with patch.object(db, "end_session", wraps=db.end_session) as end_session:
        try:
            agent.close()
            agent.close()
            end_session.assert_called_once_with("sid", "agent_close")
            assert agent._session_db is None
            assert db._conn is None
        finally:
            db.close()

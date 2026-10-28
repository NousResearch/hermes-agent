"""AIAgent(temporary=True): no memory or recall, no auto-title, and every row it writes is a hidden temporary row."""

from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


def _tool_defs(*names):
    return [{"type": "function", "function": {"name": n, "description": n, "parameters": {"type": "object", "properties": {}}}}
            for n in names]


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _agent(db, session_id, **kwargs):
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, session_db=db, session_id=session_id, platform="cli", **kwargs)
    agent.client = MagicMock()
    return agent


def test_temporary_agent_turns_off_memory_recall_and_review(db):
    agent = _agent(db, "t-1", temporary=True)
    assert agent._memory_store is None and agent._memory_manager is None
    assert agent.skip_background_review is True
    assert {"memory", "session_search"} <= set(agent.disabled_toolsets or [])

    normal = _agent(db, "n-1", skip_memory=True)
    assert "session_search" not in set(normal.disabled_toolsets or [])


def test_temporary_agent_row_is_hidden_and_temporary(db):
    agent = _agent(db, "t-2", temporary=True)
    agent._ensure_db_session()
    row = db.get_session("t-2")
    assert row["hidden"] == 1 and db.is_session_temporary("t-2")

    normal = _agent(db, "n-2", skip_memory=True)
    normal._ensure_db_session()
    assert db.get_session("n-2")["hidden"] == 0 and not db.is_session_temporary("n-2")


def test_temporary_agent_never_adopts_a_saved_conversation(db):
    db.create_session("saved", source="cli")
    db.append_messages_batch("saved", [{"role": "user", "content": "keep this"}])
    agent = _agent(db, "saved", temporary=True)
    agent._ensure_db_session()
    assert not db.is_session_temporary("saved")
    assert db.get_session("saved")["hidden"] == 0


def test_temporary_agent_is_never_auto_titled(db):
    from agent.turn_context import _maybe_title_session_at_turn_start
    agent = _agent(db, "t-3", temporary=True)
    with patch("agent.title_generator.maybe_auto_title") as auto_title:
        _maybe_title_session_at_turn_start(agent, [{"role": "user", "content": "plan my surprise party"}])
    auto_title.assert_not_called()


def test_temporary_agent_never_saves_a_trajectory(db):
    agent = _agent(db, "t-traj", temporary=True)
    agent.save_trajectories = True
    with patch("agent.session_persistence._save_trajectory_to_file") as save:
        agent._save_trajectory([{"role": "user", "content": "secret"}], "secret", True)
    save.assert_not_called()


def test_temporary_turn_log_line_never_carries_the_message():
    from types import SimpleNamespace

    from agent.temporary_chat import turn_log_preview
    assert turn_log_preview(SimpleNamespace(_temporary=False), "hello\nthere") == "hello there"
    assert "x" * 80 + "..." == turn_log_preview(SimpleNamespace(_temporary=False), "x" * 200)
    assert turn_log_preview(SimpleNamespace(_temporary=True), "my secret plans") == "<temporary chat>"

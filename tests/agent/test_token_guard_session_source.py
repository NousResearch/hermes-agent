"""The token-accounting guard never mints an anonymous session row (#111999).

When every row create of a turn loses to the SQLite lock, the queued token delta's
"ensure the row exists" guard becomes the session's first writer. It must stamp the agent's
real surface: the turn lease treats an existing row as proof the create happened, so the
creator never returns to repair a ``source='unknown'`` placeholder.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB
from run_agent import AIAgent

SID = "20260913_210721_c89ac8"


def _response():
    msg = SimpleNamespace(content="partial answer", tool_calls=None, reasoning=None, reasoning_content=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150,
                            prompt_tokens_details=None, completion_tokens_details=None)
    return SimpleNamespace(choices=[choice], usage=usage, model="grok-4.6", id="x")


def test_guard_first_writer_carries_the_agent_source(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
            skip_context_files=True, skip_memory=True, session_id=SID, session_db=db, platform="desktop",
        )
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = _response()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False

    # The whole first turn runs under contention: every create_session attempt is refused.
    def locked(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")
    db.create_session = locked
    try:
        agent.run_conversation("first prompt")
        assert db.flush_token_counts()
        row = db.get_session(SID)
        assert row is not None, "the queued token delta must materialize the row"
        assert row["source"] == "desktop"
        assert db.list_sessions_rich(source="unknown", limit=50) == []
    finally:
        agent.close()
        db.close()

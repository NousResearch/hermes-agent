"""Regression coverage for steer display identity across compactions (#117137)."""

from unittest.mock import patch

import pytest

from agent import message_metadata
from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
from agent.prompt_builder import steer_user_row
from agent.turn_iteration_prep import _inject_steer_after_newest_tool_result
from hermes_state import SessionDB
from run_agent import AIAgent


@pytest.mark.parametrize("insertion", ["after-tool-batch", "before-api-call"])
def test_delayed_flush_keeps_steer_before_its_reply(tmp_path, monkeypatch, insertion):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("steered", source="cli")
        messages = [
            {"role": "user", "content": "Check the numbers", "timestamp": 100.0},
            {"role": "assistant", "content": "Checking", "timestamp": 101.0},
            {"role": "tool", "content": "12", "timestamp": 102.0},
        ]
        monkeypatch.setattr(message_metadata, "wall_time", lambda: 103.0)
        if insertion == "after-tool-batch":
            with patch.object(AIAgent, "_drain_pending_steer", return_value="Use the new total"):
                apply_pending_steer_to_tool_results(object.__new__(AIAgent), messages, 1)
        else:
            _inject_steer_after_newest_tool_result(None, messages, "Use the new total")
        message_metadata.append_message(
            messages, {"role": "assistant", "content": "The new total is 12"}, timestamp=104.0
        )

        db.append_messages_batch("steered", messages)

        steer, reply = db.get_messages("steered")[-2:]
        assert steer["display_kind"] == "steer"
        assert (steer["timestamp"], reply["timestamp"]) == (103.0, 104.0)


def test_flushed_steer_keeps_one_display_identity_across_compactions(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    session_id = "steer-compaction"
    with (
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._ensure_db_session()
    steer = steer_user_row("focus on the persistence failure")

    agent._flush_messages_to_session_db([steer], [])

    durable_timestamp = db.get_messages(session_id)[0]["timestamp"]
    assert steer["timestamp"] == durable_timestamp

    for _ in range(2):
        db.archive_and_compact(session_id, [dict(steer)])

    physical = db.get_messages(session_id, include_inactive=True)
    assert len([row for row in physical if row["display_kind"] == "steer"]) == 3
    visible = db.get_messages(session_id, include_compacted=True)
    assert [(row["display_kind"], row["content"]) for row in visible] == [
        ("steer", steer["content"]),
    ]

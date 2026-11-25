"""Regression coverage for steer display identity across compactions (#117137)."""

from unittest.mock import patch

from agent.prompt_builder import steer_user_row
from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView
from run_agent import AIAgent
from tests.fixtures.storage_read import transcript_rows
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def test_flushed_steer_keeps_one_display_identity_across_compactions(tmp_path):
    db = storage_handle(tmp_path)
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
            persistence=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._ensure_db_session()
    steer = steer_user_row("focus on the persistence failure")

    agent._flush_messages_to_session_db([steer], [])

    durable_timestamp = transcript_rows(db, session_id)[0]["timestamp"]
    assert steer["timestamp"] == durable_timestamp

    for _ in range(2):
        db.messages.compact(session_id, [NewMessage.from_live(dict(steer))])

    physical = transcript_rows(db, session_id, TranscriptRead(view=TranscriptView.ALL))
    assert len([row for row in physical if row["display_kind"] == "steer"]) == 3
    visible = transcript_rows(db, session_id, TranscriptRead(view=TranscriptView.DISPLAY))
    assert [(row["display_kind"], row["content"]) for row in visible] == [
        ("steer", steer["content"]),
    ]

"""Regression coverage for steer display identity across compactions (#117137)."""

from unittest.mock import patch

from agent.prompt_builder import steer_user_row
from hermes_state import SessionDB
from run_agent import AIAgent


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


def test_flushed_steer_row_keeps_content_under_stale_persist_override(tmp_path):
    """#132165 defense-in-depth: even if a stale ``_persist_user_message_idx`` points
    at a steer row, the opening-prompt override must not replace the correction's
    stored content (or displace it into api_content)."""
    import os

    db = SessionDB(tmp_path / "state.db")
    session_id = "steer-stale-override"
    with (
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key=os.environ["OPENROUTER_API_KEY"],
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
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "the opening prompt"

    agent._flush_messages_to_session_db([steer], [])

    row = db.get_messages(session_id)[0]
    assert row["display_kind"] == "steer"
    assert row["content"] == steer["content"]
    assert row.get("api_content") is None

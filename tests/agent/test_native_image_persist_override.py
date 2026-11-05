"""Regression coverage for the persist override on native multimodal turns (#132971).

A gateway turn carries a plain-string persist override — the authored text after
``strip_discord_triggering_note`` peels off the model-facing routing note. For a
text-only turn ``_override_replaces_content`` lets it replace the live content, so
the durable row holds only what the user wrote. For a turn whose image is routed
natively the live content is an OpenAI-style part list, and the string override
was silently skipped: the routing note and the transient attachment-cache path
landed in the durable ``content`` verbatim (transcripts, FTS, memory providers —
the pollution #71304 / #114719 describe, reopened on the multimodal path).

These tests pin the fix: the string override replaces the text part only, the
media parts survive as ``[screenshot]`` placeholders, and rows without an
override (or merged with a compaction summary) keep their full projection.
"""

import os

from unittest.mock import patch

from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY
from hermes_state import SessionDB
from run_agent import AIAgent


NOTE = (
    "[Triggering message id: `1305567224559165440` — use as `message_id` for reply/react/pin "
    "via the discord tools.]"
)
AUTHORED = "[User] TEST - read this later"
CACHE_HINT = "[Image attached at: /Users/x/.hermes/cache/images/img_abc123.png]"


def _make_agent(tmp_path, session_id):
    db = SessionDB(tmp_path / "state.db")
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
    return agent, db


def _native_image_user_row(text: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aGVsbG8="},
            },
        ],
    }


def test_native_image_turn_durable_row_keeps_override_text_and_media_placeholder(
    tmp_path,
):
    agent, db = _make_agent(tmp_path, "native-image-persist")
    live_text = f"{NOTE}\n\n{AUTHORED}\n\n{CACHE_HINT}"
    agent._persist_user_message_override = AUTHORED
    agent._persist_user_message_idx = 0

    agent._flush_messages_to_session_db([_native_image_user_row(live_text)], [])

    rows = db.get_messages(agent.session_id)
    assert len(rows) == 1
    assert rows[0]["content"] == f"{AUTHORED}\n[screenshot]"
    assert "Triggering message id" not in rows[0]["content"]
    assert "img_abc123" not in rows[0]["content"]


def test_text_only_part_list_collapses_to_override(tmp_path):
    agent, db = _make_agent(tmp_path, "native-image-text-parts-only")
    agent._persist_user_message_override = AUTHORED
    agent._persist_user_message_idx = 0

    agent._flush_messages_to_session_db(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{NOTE}\n\n{AUTHORED}"}],
            }
        ],
        [],
    )

    rows = db.get_messages(agent.session_id)
    assert len(rows) == 1
    assert rows[0]["content"] == AUTHORED


def test_without_override_the_native_projection_is_unchanged(tmp_path):
    agent, db = _make_agent(tmp_path, "native-image-no-override")
    live_text = f"{NOTE}\n\n{AUTHORED}\n\n{CACHE_HINT}"

    agent._flush_messages_to_session_db([_native_image_user_row(live_text)], [])

    rows = db.get_messages(agent.session_id)
    assert len(rows) == 1
    assert rows[0]["content"] == f"{live_text}\n[screenshot]"


def test_summary_merged_multimodal_row_is_not_replaced(tmp_path):
    agent, db = _make_agent(tmp_path, "native-image-summary-merge")
    live_text = f"{NOTE}\n\n{AUTHORED}"
    msg = _native_image_user_row(live_text)
    msg[COMPRESSED_SUMMARY_METADATA_KEY] = True
    agent._persist_user_message_override = AUTHORED
    agent._persist_user_message_idx = 0

    agent._flush_messages_to_session_db([msg], [])

    rows = db.get_messages(agent.session_id)
    assert len(rows) == 1
    assert rows[0]["content"] == f"{live_text}\n[screenshot]"

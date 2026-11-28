"""Regression for a staged attachment variant during proactive prune."""

from __future__ import annotations

from types import SimpleNamespace

from agent.conversation_compression_archive import held_message
from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.turn_context import _stage_turn_user_message
from persistence.conversation.messages import NewMessage
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_session

DURABLE = "Please look at this\n@image:/tmp/shot.png"
LIVE = (
    "[The user attached an image: shot.png]\n"
    "[Examine it with the vision_analyze tool using image_url: /tmp/shot.png]\n\n"
    "Please look at this"
)


def _staged_turn(tmp_path):
    store = open_persistence(profile_scope(home=tmp_path))
    session_id = "PRUNE_SUBMIT_ADOPTION"
    seed_session(store, session_id, "desktop")
    row_id = store.messages.append(session_id, NewMessage(role="user", content=DURABLE))
    staged = {"role": "user", "content": DURABLE, "_row_id": int(row_id), _DB_PERSISTED_MARKER: True}
    agent = SimpleNamespace(_pending_cli_user_message=staged)
    user_msg, _ = _stage_turn_user_message(agent, LIVE, DURABLE, None, None, None, None)
    assert user_msg is staged and user_msg["content"] == LIVE
    return store, session_id, row_id, [user_msg]


def test_own_api_variant_is_not_a_rewrite(tmp_path):
    store, session_id, _row_id, held = _staged_turn(tmp_path)
    record = held_message(held[0])
    assert record.held_durable_content == DURABLE
    assert store.messages.rewritten_held_row_ids(session_id, (record,)) == ()
    store.close()


def test_a_concurrent_user_row_rewrite_is_still_stale(tmp_path):
    store, session_id, row_id, held = _staged_turn(tmp_path)
    store.messages.set_user_message_content(session_id, row_id, "someone else's text")
    assert store.messages.rewritten_held_row_ids(session_id, (held_message(held[0]),)) == (row_id,)
    store.close()


def test_without_the_recorded_durable_form_a_variant_is_still_stale(tmp_path):
    store, session_id, row_id, held = _staged_turn(tmp_path)
    message = dict(held[0])
    message.pop("_held_durable_content")
    assert store.messages.rewritten_held_row_ids(session_id, (held_message(message),)) == (row_id,)
    store.close()


def test_unchanged_adoption_records_nothing():
    staged = {"role": "user", "content": "plain", "_row_id": 1, _DB_PERSISTED_MARKER: True}
    agent = SimpleNamespace(_pending_cli_user_message=staged)
    user_msg, _ = _stage_turn_user_message(agent, "plain", None, None, None, None, None)
    assert "_held_durable_content" not in user_msg

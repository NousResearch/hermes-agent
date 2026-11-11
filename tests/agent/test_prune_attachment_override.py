"""Proactive prune must not treat the turn's own API-facing user variant as a lost race (local 2026-11-06).

Desktop writes the user row at submit (``_write_submit_user_row``); for an ``@image:``/``@file:`` turn
``_adopt_submit_user_row`` rewrites it to the clean persisted form and ``_stage_turn_user_message`` then
swaps the live dict's ``content`` to the model-facing variant ("[The user attached an image: …]"),
keeping ``_row_id`` and the persist marker. The lease-less staleness probe compared live ``content`` with
stored ``content``, reported the row "rewritten in place", and every proactive prune for the rest of the
turn was skipped ("another compaction already committed this session"): 9 sessions in 3 weeks, every one
with an attachment; session 20261106_142504_63c19a grew to 157k tokens unpruned.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent.context_compressor import _DB_PERSISTED_MARKER
from hermes_state import SessionDB

DURABLE = "Please look at this\n@image:/tmp/shot.png"
LIVE = ("[The user attached an image: shot.png]\n[Examine it with the vision_analyze tool using image_url: "
        "/tmp/shot.png]\n\nPlease look at this")


def _staged_turn(tmp_path: Path):
    """The real adoption path: a submit-written row adopted by _stage_turn_user_message."""
    from agent.turn_context import _stage_turn_user_message
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PRUNE_SUBMIT_ADOPTION"
    db.create_session(sid, source="desktop")
    rid = db.append_message(sid, "user", content=DURABLE)
    staged = {"role": "user", "content": DURABLE, "_row_id": rid, _DB_PERSISTED_MARKER: True}
    agent = SimpleNamespace(_pending_cli_user_message=staged)
    user_msg, _ = _stage_turn_user_message(agent, LIVE, DURABLE, None, None, None, None)
    assert user_msg is staged and user_msg["content"] == LIVE
    return db, sid, rid, [user_msg]


def test_own_api_variant_is_not_a_rewrite(tmp_path: Path) -> None:
    db, sid, _rid, held = _staged_turn(tmp_path)
    assert held[0]["_held_durable_content"] == DURABLE
    assert db.rewritten_held_row_ids(sid, held) == []


def test_a_concurrent_user_row_rewrite_is_still_stale(tmp_path: Path) -> None:
    db, sid, rid, held = _staged_turn(tmp_path)
    db.set_user_message_content(sid, rid, "someone else's text")
    assert db.rewritten_held_row_ids(sid, held) == [rid]


def test_without_the_recorded_durable_form_a_variant_is_still_stale(tmp_path: Path) -> None:
    db, sid, rid, held = _staged_turn(tmp_path)
    held[0].pop("_held_durable_content")
    assert db.rewritten_held_row_ids(sid, held) == [rid]


def test_unchanged_adoption_records_nothing() -> None:
    from agent.turn_context import _stage_turn_user_message
    staged = {"role": "user", "content": "plain", "_row_id": 1, _DB_PERSISTED_MARKER: True}
    agent = SimpleNamespace(_pending_cli_user_message=staged)
    user_msg, _ = _stage_turn_user_message(agent, "plain", None, None, None, None, None)
    assert "_held_durable_content" not in user_msg

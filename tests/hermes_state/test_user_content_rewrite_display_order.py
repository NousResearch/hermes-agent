"""A content rewrite must not leave the user row without a display slot (#128468).

The turn prologue rewrites a submit-time-persisted user row in place when the
prompt gains multimodal parts — native image attachments are the reported path
(``set_user_message_content`` via ``_append_multimodal_context``). That UPDATE
fires ``messages_display_identity_update``, which nulls ``display_identity`` and
``display_order`` (the identity is the content hash, so a content change really
does invalidate them), but nothing rebuilt them on the write path: image-bearing
user rows sat at ``display_order = NULL`` until some ``include_compacted``
reader backfilled the session, and every indexed display projection dropped the
row for exactly that window — on the desktop, the assistant reply above it
rendered twice, once above the tool blocks that produced it.

The rewrite now re-folds the display index in the same transaction, so the row
keeps a continuous display slot. These tests pin the durable rows directly: the
display projection itself backfills before reading, so it cannot see the window
this fix closes.
"""

from hermes_state import SessionDB

IMAGE_TURN = [
    {"type": "text", "text": "what is in this image?"},
    {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
]

REWRITTEN_TURN = IMAGE_TURN + [{"type": "text", "text": "[gateway note]"}]


def _persist_session(db, sid):
    """A session whose newest turn is an image-bearing user row: the submit-time
    persist, before the turn prologue would rewrite it in place."""
    ts = 1727000000.0
    db.create_session(sid, source="desktop")
    db.append_message(sid, "user", "check my config", timestamp=ts)
    db.append_message(sid, "assistant", "looks fine", timestamp=ts + 1)
    db.append_message(sid, "user", IMAGE_TURN, timestamp=ts + 2)
    return db._read_one(
        "SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 1", (sid,))[0]


def _display_slots(db, sid):
    return db._read_all(
        "SELECT id, display_order, display_identity IS NULL AS ident_null FROM messages "
        "WHERE session_id = ? ORDER BY id", (sid,))


def test_rewrite_rebuilds_the_display_slot(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    row_id = _persist_session(db, "s1")

    assert db.set_user_message_content("s1", row_id, REWRITTEN_TURN) == 1

    slots = {r["id"]: r for r in _display_slots(db, "s1")}
    # A fresh identity (new content) with no twin starts its own slot at its row id.
    assert slots[row_id]["display_order"] == row_id
    assert slots[row_id]["ident_null"] == 0


def test_rewrite_leaves_no_null_slot_in_the_session(tmp_path):
    """The identity trigger's blast radius is every display-visible row sharing the
    OLD identity (a compaction twin); the re-fold must clear every NULL it caused."""
    db = SessionDB(tmp_path / "state.db")
    ts = 1727000000.0
    db.create_session("s1", source="desktop")
    db.append_message("s1", "user", "check my config", timestamp=ts)
    db.append_message("s1", "assistant", "looks fine", timestamp=ts + 1)
    db.append_message("s1", "user", IMAGE_TURN, timestamp=ts + 2)
    # The archived twin of the image turn: same content, same timestamp, same identity.
    db.append_message("s1", "user", IMAGE_TURN, timestamp=ts + 2)
    twin_id = db._read_one(
        "SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 1", ("s1",))[0]
    row_id = twin_id - 1

    assert db.set_user_message_content("s1", row_id, REWRITTEN_TURN) == 1

    slots = {r["id"]: r for r in _display_slots(db, "s1")}
    assert all(slot["display_order"] is not None for slot in slots.values())
    assert all(slot["ident_null"] == 0 for slot in slots.values())


def test_rewrite_keeps_the_projection_whole(tmp_path):
    """After the rewrite the transcript projects every logical message exactly once,
    in transcript order — no dropped image turn, no duplicated reply."""
    db = SessionDB(tmp_path / "state.db")
    row_id = _persist_session(db, "s1")

    db.set_user_message_content("s1", row_id, REWRITTEN_TURN)

    projected = db.get_messages("s1", include_compacted=True)
    assert [m["role"] for m in projected] == ["user", "assistant", "user"]


def test_rewrite_of_missing_row_is_still_a_noop(tmp_path):
    """The guards (unknown row, stale id) keep returning 0 untouched; the re-fold
    only ever runs inside a rewrite that actually changed a row."""
    db = SessionDB(tmp_path / "state.db")
    _persist_session(db, "s1")

    assert db.set_user_message_content("s1", 99999, REWRITTEN_TURN) == 0
    assert db.set_user_message_content("", 1, REWRITTEN_TURN) == 0
    assert all(slot["display_order"] is not None for slot in _display_slots(db, "s1"))

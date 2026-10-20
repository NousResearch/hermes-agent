"""A user-row rewrite must not leave the session half-indexed (issue #136914).

The turn prologue rewrites a submit-time user row in place (``set_user_message_content``:
@-file expansion, native image parts) and ``set_latest_matching_message_display_kind``
stamps presentation metadata on the fresh row. Both update display-identity columns, and
the ``messages_display_identity_update`` trigger NULLs the row's ``display_order`` /
``display_identity`` in response. Only the compacted display read backfills, so a session
served through the timeline probes keeps matching the missing-index probe forever and every
read takes the legacy path, whose sort key differs — the reported viewport jump.

These tests drive both rewrites on an indexed session and assert the probe the timeline
uses never flips it back to legacy, and the rewritten row keeps a correct display order.
"""

from hermes_state import SessionDB
from hermes_state_messages import _DISPLAY_INDEX_MISSING_SQL


def _indexed(db, sid):
    return db._read_one(_DISPLAY_INDEX_MISSING_SQL, (sid,)) is None


def _make_session(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    sid = "20261010_103045_9e7950"
    ts = 1728000000.0
    db.create_session(sid, source="desktop")
    db.append_message(sid, "user", "check this @file:notes.txt", timestamp=ts)
    db.append_message(sid, "assistant", "ok", timestamp=ts + 1)
    db.append_message(sid, "user", "and one more", timestamp=ts + 2)
    assert _indexed(db, sid)
    return db, sid, ts


def test_user_content_rewrite_keeps_the_session_indexed(tmp_path):
    db, sid, _ = _make_session(tmp_path)
    user_id = db._read_one(
        "SELECT id FROM messages WHERE session_id = ? AND role = 'user' AND content LIKE '%@file:%'",
        (sid,))[0]

    assert db.set_user_message_content(sid, user_id, "check this [notes.txt expanded]") == 1

    assert _indexed(db, sid), "content rewrite left the session on the legacy display path"
    order = db._read_one("SELECT display_order FROM messages WHERE id = ?", (user_id,))[0]
    assert order == user_id, "rewritten row lost its display_order group anchor"


def test_display_kind_stamp_keeps_the_session_indexed(tmp_path):
    db, sid, _ = _make_session(tmp_path)

    stamped = db.set_latest_matching_message_display_kind(
        sid, role="assistant", content="ok", display_kind="composite")

    assert stamped
    assert _indexed(db, sid), "display_kind stamp left the session on the legacy display path"
    # The stamped row still projects exactly once, at its original position.
    projected = db.get_messages(sid, include_compacted=True)
    assert [m["role"] for m in projected] == ["user", "assistant", "user"]


def test_rewrite_then_compaction_projection_stays_stable(tmp_path):
    db, sid, _ = _make_session(tmp_path)
    user_id = db._read_one(
        "SELECT id FROM messages WHERE session_id = ? AND role = 'user' AND content LIKE '%@file:%'",
        (sid,))[0]

    db.set_user_message_content(sid, user_id, "check this [notes.txt expanded]")

    live = db.get_messages_as_conversation(sid)
    db.archive_and_compact(sid, [{"role": "user", "content": "[summary]"}] + live[-2:], tail_count=2)
    assert _indexed(db, sid)
    projected = db.get_messages(sid, include_compacted=True)
    # Display history keeps the archived original (the rewritten user row), then the summary row
    # and the two protected-tail re-clones: four rows, each exactly once.
    assert [m["content"] for m in projected] == [
        "check this [notes.txt expanded]", "[summary]", "ok", "and one more"], \
        "rewrite + compaction lost or duplicated a row"

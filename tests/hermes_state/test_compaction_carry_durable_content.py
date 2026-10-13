"""A compaction's carried tail must not edit the messages it carries.

A resumed agent holds user/assistant text as the loader decoded it (sanitized and
stripped). When in-place compaction re-inserts that tail, a message whose stored
text ended in whitespace (a ``--query-file`` body ending in a newline, a pasted
block) used to come back without it: the user's text was changed, the copy no
longer folded into its archived original, and every display read listed the
message twice.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _stored_rows(db, sid, content_prefix):
    return [m for m in db.get_messages(sid, include_compacted=True) if str(m["content"]).startswith(content_prefix)]


def test_carried_copy_keeps_the_stored_text_and_displays_once(db):
    db.create_session("chat", source="cli")
    db.append_message("chat", "user", "first ask")
    db.append_message("chat", "assistant", "first reply")
    db.append_message("chat", "user", "keep me exactly\n")
    db.append_message("chat", "assistant", "kept reply")

    live = db.get_messages_as_conversation("chat")  # what a resumed agent compacts
    assert live[-2]["content"] == "keep me exactly"  # the load lens stripped the newline
    db.archive_and_compact("chat", [{"role": "user", "content": "[summary]"}] + live[-2:], tail_count=2)

    shown = _stored_rows(db, "chat", "keep me exactly")
    assert [m["content"] for m in shown] == ["keep me exactly\n"]
    assert db.get_messages_as_conversation("chat")[-2]["content"] == "keep me exactly"


def test_a_real_edit_of_the_carried_text_is_kept(db):
    db.create_session("chat", source="cli")
    db.append_message("chat", "user", "draft one\n")
    db.append_message("chat", "assistant", "reply")

    live = db.get_messages_as_conversation("chat")
    live[0]["content"] = "draft two"  # not just the load lens: the compressor changed it
    db.archive_and_compact("chat", [{"role": "user", "content": "[summary]"}] + live, tail_count=2)

    active = [m["content"] for m in db.get_messages_as_conversation("chat")]
    assert "draft two" in active
    assert "draft one\n" not in [m["content"] for m in db.get_messages("chat")]

"""A row whose ``display_order`` is NULL must survive the display page query (#118996).

``set_user_message_content`` rewrites a user row after insert; the
``messages_display_identity_update`` trigger then clears ``display_order`` and
``display_identity``. The display projection groups by ``display_order`` and re-finds
each group's representative with an equality join, and SQL's ``NULL = NULL`` is never
true -- so the NULL group's representative is never found and the row silently
disappears from the transcript.

The heal that would backfill those NULLs runs in a different transaction from the page
read (and a read-only handle skips it entirely), so a re-NULLed row can still be on
disk when the page query runs. Ordering may degrade for such a row; dropping it must
not happen.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed_rewritten_user_row(db, sid="s1"):
    """A submit-time user row whose content the turn prologue rewrites, clearing its sort key."""
    db.create_session(sid, source="cli")
    db.append_message(sid, "assistant", "before")
    row_id = db.append_message(sid, "user", "raw keystrokes @file:/tmp/x.txt")
    db.append_message(sid, "assistant", "after")
    db.set_user_message_content(sid, row_id, "expanded prompt")
    return row_id


def test_set_user_message_content_clears_the_display_sort_key(db):
    sid = "s1"
    row_id = _seed_rewritten_user_row(db, sid)

    with db._read_ctx() as conn:
        display_order, display_identity = conn.execute(
            "SELECT display_order, display_identity FROM messages WHERE id = ?", (row_id,)
        ).fetchone()

    assert display_order is None and display_identity is None


def test_null_display_order_row_is_not_dropped_from_the_display_page(db, monkeypatch):
    sid = "s1"
    _seed_rewritten_user_row(db, sid)
    # Heal and page are separate transactions: a writer re-NULLing the row in that window
    # (or a read-only handle, which skips the heal) leaves the NULL group in the page
    # query. Returning True keeps the healed projection path itself under test.
    monkeypatch.setattr(SessionDB, "_ensure_display_order", lambda self, session_id: True)

    messages = db.get_messages(sid, include_compacted=True)
    contents = [m["content"] for m in messages]

    assert [m["role"] for m in messages].count("user") == 1, contents
    assert "expanded prompt" in contents
    assert {"before", "after"} <= set(contents)

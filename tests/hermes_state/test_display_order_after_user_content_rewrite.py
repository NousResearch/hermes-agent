"""A prompt the turn prologue rewrites must keep a usable display sort key.

An attachment turn (``@image:``/``@url:`` expansion, native image parts) is the
one prompt shape written twice: ``_adopt_submit_user_row`` persists the raw
keystrokes at submit time, then the prologue rewrites that same row through
``set_user_message_content`` so the durable transcript replays what the model
was actually sent.

That second write fires ``messages_display_identity_update``, which clears
``display_identity`` and ``display_order`` for the row so the identity can be
recomputed. Nothing reassigned them, so the row reached the next display read
with a NULL sort key -- and the deduped display page joins candidates with
``candidate.display_order = page.display_order``, which is never true for NULL
in SQL. Every NULL row collapsed into a single un-selectable page group, so the
rewritten turn AND the rows sharing its group vanished from the transcript the
user sees, while remaining on disk.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _display_ids(db, sid):
    return [m.get("id") for m in db.get_messages(sid, include_compacted=True)]


def _null_order_rows(db, sid):
    row = db._read_one(
        "SELECT COUNT(*) AS n FROM messages WHERE session_id = ? "
        "AND display_order IS NULL AND (active = 1 OR compacted = 1)",
        (sid,))
    return row["n"] if row else 0


def _seed(db, sid="s1"):
    db.create_session(sid, source="cli")
    db.append_messages_batch(sid, [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "@image:/tmp/shot.png what is this"},
        {"role": "assistant", "content": "a2"},
    ])
    return sid


def test_rewriting_a_user_row_leaves_no_null_display_order(db):
    """The prologue's rewrite must not strand the row without a sort key."""
    sid = _seed(db)
    db.get_messages(sid, include_compacted=True)  # backfill display order
    assert _null_order_rows(db, sid) == 0

    target = db._read_one(
        "SELECT id FROM messages WHERE session_id = ? AND role = 'user' "
        "AND active = 1 ORDER BY id DESC LIMIT 1", (sid,))["id"]
    assert db.set_user_message_content(sid, target, "expanded image parts") == 1

    assert _null_order_rows(db, sid) == 0


def test_rewritten_turn_stays_visible_in_the_display_read(db):
    """No message may disappear from the transcript because of the rewrite.

    An invariant check rather than a regression reproduction: ``get_messages``
    backfills before reading, so it repairs the stranded row on the way out.
    The rows only vanish for a reader that does not (or cannot) backfill first
    -- which is why the fix restores the sort key at write time, not just at
    read time.
    """
    sid = _seed(db)
    before = _display_ids(db, sid)

    target = db._read_one(
        "SELECT id FROM messages WHERE session_id = ? AND role = 'user' "
        "AND active = 1 ORDER BY id DESC LIMIT 1", (sid,))["id"]
    db.set_user_message_content(sid, target, "expanded image parts")

    after = _display_ids(db, sid)
    assert after == before
    assert target in after


def test_display_page_join_is_null_safe(db):
    """The deduped display page must resolve a representative for a NULL group.

    Guards the ``IS`` (NULL-safe) comparison in the page join. ``get_messages``
    backfills before reading, so the NULL group is unreachable through the
    public API once the backfill works; this exercises the join directly, which
    is the shape that silently dropped rows when written with ``=``.
    """
    sid = _seed(db)
    db.get_messages(sid, include_compacted=True)
    db._write_sql(
        "UPDATE messages SET display_order = NULL WHERE session_id = ? "
        "AND role = 'assistant' AND active = 1", (sid,))

    page_join = """WITH page AS (
            SELECT display_order FROM messages
            WHERE session_id = ? AND (active = 1 OR compacted = 1)
            GROUP BY display_order
        )
        SELECT COUNT(*) AS n FROM page JOIN messages AS chosen ON chosen.id = (
            SELECT candidate.id FROM messages AS candidate
            WHERE candidate.session_id = ?
              AND candidate.display_order {op} page.display_order
              AND (candidate.active = 1 OR candidate.compacted = 1)
            ORDER BY candidate.active DESC, candidate.id DESC LIMIT 1
        )"""
    groups = db._read_one(
        "SELECT COUNT(*) AS n FROM (SELECT display_order FROM messages "
        "WHERE session_id = ? AND (active = 1 OR compacted = 1) GROUP BY display_order)",
        (sid,))["n"]

    null_safe = db._read_one(page_join.format(op="IS"), (sid, sid))["n"]
    equality = db._read_one(page_join.format(op="="), (sid, sid))["n"]

    assert null_safe == groups
    assert equality < groups  # the regression this guards against

"""#82956: ``replace_messages(archive_dropped=True)`` must archive only the dropped suffix.

The archive-mode rewind used to archive EVERY live row and re-insert the kept
prefix as fresh rows, so the ``active=0/compacted=0`` archive grew by the whole
transcript on every rewind/edit with nothing pruning it.  Invariants pinned:

- inactive rows == turns the user genuinely dropped (not dropped + prefix × rounds);
- the kept prefix keeps its row ids across rounds (stable ids prove no re-insert).
"""

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import (
    append_message,
    get_messages,
    get_messages_as_conversation,
    replace_messages,
)
from persistence.sqlite.conversation._store.sessions import create_session, get_session


def _turn(i: int, tag: str) -> dict:
    return {"role": "user" if i % 2 == 0 else "assistant", "content": f"{tag} {i}"}


def test_archive_mode_rewind_archives_only_the_dropped_suffix(tmp_path):
    db = StoreHolder(tmp_path / "state.db")
    sid = "rewind-suffix-only"
    create_session(db, sid, "test")
    history = [_turn(i, "msg") for i in range(50)]
    for m in history:
        append_message(db, sid, m["role"], m["content"])

    def live_ids():
        return [m["id"] for m in get_messages(db, sid)]

    prefix_ids = live_ids()[:40]
    for k in range(3):
        replace_messages(db, sid, [dict(m) for m in history[:40]], active_only=True, archive_dropped=True)
        assert live_ids() == prefix_ids, f"round {k}: kept prefix was re-inserted under new ids"
        history = history[:40] + [_turn(i, f"new {k}") for i in range(10)]
        for m in history[40:]:
            append_message(db, sid, m["role"], m["content"])

    rows = get_messages(db, sid, include_inactive=True)
    inactive = [m for m in rows if not m["active"]]
    assert len(inactive) == 30, f"expected the 3×10 dropped turns archived, got {len(inactive)}"
    assert len(rows) == 80
    assert len([m for m in rows if m["active"]]) == 50
    assert get_session(db, sid)["message_count"] == 50

    # A rewrite that changes turn 30 of 40: rows 0-29 keep their ids, everything from the
    # divergent row on is archived (50 - 30 = 20 more), and the 10 rewritten turns are new rows.
    rewritten = [dict(m) for m in history[:30]] + [_turn(30, "edited")] + [dict(m) for m in history[31:40]]
    replace_messages(db, sid, rewritten, active_only=True, archive_dropped=True)
    ids = live_ids()
    assert ids[:30] == prefix_ids[:30]
    assert len(ids) == 40 and set(ids[30:]).isdisjoint(prefix_ids)
    assert len([m for m in get_messages(db, sid, include_inactive=True) if not m["active"]]) == 50
    assert get_session(db, sid)["message_count"] == 40


def test_reloaded_session_prefix_still_matches_its_rows(tmp_path):
    """Loading strips/sanitizes user+assistant text; a rewind issued from that loaded view must still
    recognise the stored rows as the same prefix, or every post-reload rewind falls back to archive-all."""
    db = StoreHolder(tmp_path / "state.db")
    sid = "rewind-after-reload"
    create_session(db, sid, "test")
    for i in range(6):
        append_message(db, sid, "user" if i % 2 == 0 else "assistant", f"turn {i}   \n")

    loaded = get_messages_as_conversation(db, sid)
    prefix_ids = [m["id"] for m in get_messages(db, sid)][:4]

    replace_messages(db, sid, [dict(m) for m in loaded[:4]], active_only=True, archive_dropped=True)

    assert [m["id"] for m in get_messages(db, sid)] == prefix_ids
    assert len([m for m in get_messages(db, sid, include_inactive=True) if not m["active"]]) == 2

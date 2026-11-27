"""Visible archived messages keep the same once-only reaction contract."""

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import (
    append_message,
    archive_and_compact,
    get_messages,
    get_messages_as_conversation,
    rewind_to_message,
    set_message_reaction,
    take_unseen_reactions,
)
from persistence.sqlite.conversation._store.sessions import create_session


@pytest.mark.parametrize("react_before_compaction", [False, True])
def test_compacted_reactions_are_delivered_once_without_reviving_rewound_rows(
    tmp_path, monkeypatch, react_before_compaction,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "state.db"
    db = StoreHolder(db_path=db_path)
    try:
        sid = create_session(db, "compacted-reactions", "test")
        append_message(db, sid, "user", "remember this answer")
        target = append_message(db, sid, "assistant", "the original answer")
        append_message(db, sid, "user", "continue")
        tail = get_messages_as_conversation(db, sid)[-1]
        if react_before_compaction:
            set_message_reaction(db, sid, target, "👍")
        archive_and_compact(
            db, sid, [{"role": "user", "content": "summary"}, tail], tail_count=1,
        )
        visible = get_messages(db, sid, include_compacted=True)
        assert target in {row["id"] for row in visible}
        if not react_before_compaction:
            assert set_message_reaction(db, sid, target, "👍")

        rewind_target = append_message(db, sid, "user", "discard this turn")
        discarded = append_message(db, sid, "assistant", "discarded answer")
        set_message_reaction(db, sid, discarded, "👎")
        rewind_to_message(db, sid, rewind_target)
        before = get_messages_as_conversation(db, sid)

        pending = take_unseen_reactions(db, sid)
        assert [(item["row_id"], item["emoji"], item["text"]) for item in pending] == [
            (target, "👍", "the original answer"),
        ]
        assert take_unseen_reactions(db, sid) == []
        assert get_messages_as_conversation(db, sid) == before
        archived = {row["id"]: row for row in get_messages(db, sid, include_compacted=True)}
        assert archived[target]["content"] == "the original answer"
        assert discarded not in archived
    finally:
        db.close()

    reopened = StoreHolder(db_path=db_path)
    try:
        assert take_unseen_reactions(reopened, sid) == []
    finally:
        reopened.close()

"""Visible archived messages keep the same once-only reaction contract."""

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("react_before_compaction", [False, True])
def test_compacted_reactions_are_delivered_once_without_reviving_rewound_rows(
    tmp_path, monkeypatch, react_before_compaction,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        sid = db.create_session("compacted-reactions", "test")
        db.append_message(sid, "user", "remember this answer")
        target = db.append_message(sid, "assistant", "the original answer")
        db.append_message(sid, "user", "continue")
        tail = db.get_messages_as_conversation(sid)[-1]
        if react_before_compaction:
            db.set_message_reaction(sid, target, "👍")
        db.archive_and_compact(
            sid, [{"role": "user", "content": "summary"}, tail], tail_count=1,
        )
        visible = db.get_messages(sid, include_compacted=True)
        assert target in {row["id"] for row in visible}
        if not react_before_compaction:
            assert db.set_message_reaction(sid, target, "👍")

        rewind_target = db.append_message(sid, "user", "discard this turn")
        discarded = db.append_message(sid, "assistant", "discarded answer")
        db.set_message_reaction(sid, discarded, "👎")
        db.rewind_to_message(sid, rewind_target)
        before = db.get_messages_as_conversation(sid)

        pending = db.take_unseen_reactions(sid)
        assert [(item["row_id"], item["emoji"], item["text"]) for item in pending] == [
            (target, "👍", "the original answer"),
        ]
        assert db.take_unseen_reactions(sid) == []
        assert db.get_messages_as_conversation(sid) == before
        archived = {row["id"]: row for row in db.get_messages(sid, include_compacted=True)}
        assert archived[target]["content"] == "the original answer"
        assert discarded not in archived
    finally:
        db.close()

    reopened = SessionDB(db_path=db_path)
    try:
        assert reopened.take_unseen_reactions(sid) == []
    finally:
        reopened.close()

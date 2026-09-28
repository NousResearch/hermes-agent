"""Admission excludes its staged input before alternation repair."""
from hermes_state import SessionDB


def test_exclude_staged_row_before_live_replay_repair(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="desktop")
    db.append_message("s", "user", "previous")
    db.append_message("s", "assistant", "answer")
    db.append_message("s", "user", "external")
    own = db.append_message("s", "user", "staged next input")
    db.append_message("s", "assistant", "external answer after staging")
    try:
        history = db.get_messages_as_conversation("s", repair_alternation=True,
                                                  include_row_ids=True, exclude_row_ids={own})
        assert [m["content"] for m in history] == ["previous", "answer", "external", "external answer after staging"]
        assert len(db.get_messages("s")) == 5
    finally:
        db.close()

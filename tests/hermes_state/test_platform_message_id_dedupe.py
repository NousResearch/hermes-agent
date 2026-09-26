"""One platform message id stays one active row across a second insert."""

from hermes_state import SessionDB


def test_second_insert_of_the_same_platform_message_is_skipped(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sess", source="telegram")
    first = db.append_message(
        "sess", role="user", content="photo", platform_message_id="42", timestamp=1700000000.0,
    )
    second = db.append_message(
        "sess", role="user", content="photo", platform_message_id="42", timestamp=1700000000.0,
    )
    active = [row for row in db.get_messages("sess") if row.get("role") == "user"]
    assert first
    assert second == 0
    assert len(active) == 1
    db.close()

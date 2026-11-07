"""Opening a terminal reserves a session, but must not add an empty chat to history."""

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch


@pytest.mark.parametrize("recent", [False, True])
def test_unsubmitted_tui_drafts_do_not_consume_history_pages(tmp_path, recent):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("conversation", source="tui")
        db.append_message("conversation", "user", "hello")
        for index in range(3):
            db.create_session(f"draft-{index}", source="tui")
        db.end_session("draft-0", "tui_shutdown")

        rows = db.list_sessions_rich(limit=1, order_by_last_active=recent)
        assert [row["id"] for row in rows] == ["conversation"]
        assert db.session_count(exclude_children=True) == 1
        assert db.session_count_by_source(exclude_children=True) == {"tui": 1}
        assert {row["id"] for row in db.search_sessions()} == {"conversation"}
        with SessionDB(tmp_path / "state.db", read_only=True) as reader:
            assert [row["id"] for row in reader.list_sessions_rich(limit=1)] == ["conversation"]
            assert reader.session_count(exclude_children=True) == 1
        # The reserved identity remains usable by the terminal on reconnect.
        assert db.get_session("draft-1") is not None
        db.append_message("draft-1", "user", "first prompt")
        assert db.session_count(exclude_children=True) == 2
        assert "draft-1" in {row["id"] for row in db.list_sessions_rich()}


@pytest.mark.parametrize("status", ["queued", "unknown"])
def test_first_admission_and_explicitly_kept_sessions_remain_visible(tmp_path, status):
    with SessionDB(tmp_path / "state.db") as db:
        for sid, source in [("submitted", "tui"), ("named", "tui"),
                            ("pinned", "tui"), ("other-surface", "gui")]:
            db.create_session(sid, source=source)
        db.set_session_title("named", "Planned work")
        db._execute_write(lambda conn: conn.execute("UPDATE sessions SET pinned=1 WHERE id='pinned'"))
        epoch = begin_runtime_epoch(db, instance_id="owner")
        admit_session_input(db, epoch=epoch, principal_id="human", session_id="submitted",
                            request_id="first", payload={"text": "pending prompt"})
        db._execute_write(lambda conn: conn.execute(
            "UPDATE session_admissions SET status=? WHERE target_session_id='submitted'", (status,)))
        assert db.get_session("submitted")["message_count"] == 0
        expected = {"submitted", "named", "pinned", "other-surface"}
        assert {row["id"] for row in db.list_sessions_rich()} == expected
        assert db.session_count(exclude_children=True) == len(expected)

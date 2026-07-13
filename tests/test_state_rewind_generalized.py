import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def test_rewind_requires_user_role_by_default(db):
    db.create_session("s1", source="cli")
    db.append_message("s1", "user", "u")
    assistant = db.append_message("s1", "assistant", "a")

    with pytest.raises(ValueError, match="must be a 'user'"):
        db.rewind_to_message("s1", assistant)


def test_rewind_can_target_assistant_when_role_requirement_disabled(db):
    db.create_session("s1", source="cli")
    user = db.append_message("s1", "user", "u")
    assistant = db.append_message("s1", "assistant", "a")

    result = db.rewind_to_message("s1", assistant, require_user_role=False)

    assert result["rewound_ids"] == [assistant]
    assert result["rewound_count"] == 1
    assert result["new_head_id"] == user
    assert [m["id"] for m in db.get_messages("s1")] == [user]


def test_rewound_ids_include_only_rows_flipped_from_active_to_inactive(db):
    db.create_session("s1", source="cli")
    db.append_message("s1", "user", "u1")
    target = db.append_message("s1", "assistant", "a1")
    inactive_hole = db.append_message("s1", "user", "u2")
    active_tail = db.append_message("s1", "assistant", "a2")
    db._conn.execute("UPDATE messages SET active = 0 WHERE id = ?", (inactive_hole,))

    result = db.rewind_to_message("s1", target, require_user_role=False)

    assert result["rewound_ids"] == [target, active_tail]
    assert inactive_hole not in result["rewound_ids"]
    all_rows = db.get_messages("s1", include_inactive=True)
    assert {r["id"]: r["active"] for r in all_rows}[inactive_hole] == 0


def test_rewind_raise_guard_prevents_tool_orphan(db):
    db.create_session("s1", source="cli")
    db.append_message("s1", "user", "u")
    tool = db.append_message("s1", "tool", "early", tool_call_id="c1")
    assistant = db.append_message("s1", "assistant", None, tool_calls=[{"id": "c1"}])

    with pytest.raises(ValueError, match=f"id={tool}"):
        db.rewind_to_message("s1", assistant, require_user_role=False)

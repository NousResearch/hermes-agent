import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def test_membership_moves_between_groups_and_preserves_sessions(db):
    db.create_session("grouped-a", source="cli")
    db.create_session("grouped-b", source="cli")
    first = db.create_session_group("Startup", color="blue")
    second = db.create_session_group("Job Hunt")

    assert db.assign_sessions_to_group(first["id"], ["grouped-a", "grouped-b"]) == 2
    assert db.session_ids_for_group("startup") == {"grouped-a", "grouped-b"}

    assert db.assign_sessions_to_group(second["id"], ["grouped-b"]) == 1
    assert db.session_ids_for_group(first["id"]) == {"grouped-a"}
    assert db.session_ids_for_group("Job Hunt") == {"grouped-b"}

    assert db.remove_sessions_from_group("Job Hunt", ["grouped-b"]) == 1
    assert db.delete_session_group("Startup") is True
    assert db.get_session("grouped-a") is not None
    assert db.get_session("grouped-b") is not None


def test_assignment_rejects_unknown_session_atomically(db):
    db.create_session("known", source="cli")
    db.create_session_group("Project")

    with pytest.raises(ValueError, match="session not found: missing"):
        db.assign_sessions_to_group("Project", ["known", "missing"])

    assert db.session_ids_for_group("Project") == set()


def test_duplicate_group_name_is_case_insensitive(db):
    db.create_session_group("Startup")

    with pytest.raises(ValueError, match="already exists"):
        db.create_session_group("startup")

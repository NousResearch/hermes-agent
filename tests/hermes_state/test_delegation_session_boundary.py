import time

from hermes_state import SessionDB


def _insert_group(db: SessionDB, work_id: str, session_id: str) -> None:
    now = time.time()
    db._conn.execute(
        "INSERT INTO async_delegation_work_groups "
        "(work_id,origin_session,parent_session_id,owner_turn_id,state,created_at,updated_at,"
        "closeout_claim,closeout_claimed_at,closeout_turn_id,closeout_owner_pid,"
        "closeout_owner_started_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            work_id, session_id, session_id, "owner", "closing", now, now,
            "claim", now, "turn", 123, 456,
        ),
    )
    db._conn.commit()


def test_delete_session_drops_only_its_groups_in_exact_profile_db(
    tmp_path, monkeypatch
):
    ambient = tmp_path / "ambient"
    profile = tmp_path / "profile"
    ambient.mkdir()
    profile.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(ambient))

    profile_db = SessionDB(db_path=profile / "state.db")
    ambient_db = SessionDB(db_path=ambient / "state.db")
    try:
        for db in (profile_db, ambient_db):
            db.create_session("session-a", source="cli")
            _insert_group(db, "work-a", "session-a")
        profile_db.create_session("session-b", source="cli")
        _insert_group(profile_db, "work-b", "session-b")

        assert profile_db.delete_session("session-a") is True

        closed = profile_db._conn.execute(
            "SELECT * FROM async_delegation_work_groups WHERE work_id='work-a'"
        ).fetchone()
        assert closed["state"] == "closed"
        assert closed["terminal_disposition"] == "dropped"
        assert closed["terminal_diagnostics"] == "owning session deleted"
        for column in (
            "closeout_claim", "closeout_claimed_at", "closeout_turn_id",
            "closeout_owner_pid", "closeout_owner_started_at",
        ):
            assert closed[column] is None
        assert profile_db._conn.execute(
            "SELECT state FROM async_delegation_work_groups WHERE work_id='work-b'"
        ).fetchone()[0] == "closing"
        assert ambient_db._conn.execute(
            "SELECT state FROM async_delegation_work_groups WHERE work_id='work-a'"
        ).fetchone()[0] == "closing"
        assert profile_db.delete_session("session-a") is False
    finally:
        profile_db.close()
        ambient_db.close()

"""Regression for #90569: ``SessionDB(db_path=<str>)`` crashed on ``str.parent``."""

from hermes_state import SessionDB


def test_str_and_path_spellings_open_the_same_database(tmp_path):
    target = tmp_path / "state.db"

    with SessionDB(db_path=str(target)) as db:
        assert db.db_path == target
        db.create_session("s1", "cli")

    with SessionDB(db_path=target, read_only=True) as db:
        assert db.get_session("s1") is not None


def test_str_path_opens_read_only(tmp_path):
    target = tmp_path / "state.db"
    SessionDB(db_path=target).close()

    with SessionDB(db_path=str(target), read_only=True) as db:
        assert db.db_path == target

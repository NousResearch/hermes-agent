"""Read-only inspection needs no writable lifecycle metadata and cannot bind a successor."""
from pathlib import Path
import shutil

import pytest

import hermes_state
from hermes_cli.profile_incarnation import ensure_profile_incarnation, write_fresh_profile_incarnation
from hermes_state import SessionDB


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    home = root / "profiles" / "reader"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    ensure_profile_incarnation(home)
    with SessionDB(home / "state.db") as db:
        db.create_session("original", "tui")
    return home


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("existing_locks", [False, True])
def test_readonly_profile_does_not_require_writable_locks(home, existing_locks):
    locks = home.parent / ".locks"
    if existing_locks:
        for path in locks.iterdir():
            path.chmod(0o400)
        locks.chmod(0o500)
    else:
        shutil.rmtree(locks)
    home.parent.chmod(0o500)
    try:
        with SessionDB(home / "state.db", read_only=True) as db:
            assert db.get_session("original")["id"] == "original"
        assert locks.exists() is existing_locks
    finally:
        home.parent.chmod(0o700)
        if locks.exists():
            locks.chmod(0o700)
            for path in locks.iterdir():
                path.chmod(0o600)


@pytest.mark.parametrize("marked", [False, True])
def test_readonly_open_rejects_generation_replaced_before_connect(home, tmp_path, monkeypatch, marked):
    if not marked:
        (home / ".profile-incarnation").unlink()
    # Both fixture databases are quiescent before this simulated publication.
    successor = tmp_path / "successor"
    successor.mkdir()
    with SessionDB(successor / "state.db") as db:
        db.create_session("successor", "tui")
    real_connect = SessionDB._connect_read_only
    opened = []

    def replace_then_connect(self, timeout):
        home.rename(tmp_path / "retired")
        successor.rename(home)
        write_fresh_profile_incarnation(home)
        connection = real_connect(self, timeout)
        opened.append(connection)
        return connection

    with monkeypatch.context() as patch:
        patch.setattr(SessionDB, "_connect_read_only", replace_then_connect)
        with pytest.raises(FileNotFoundError):
            SessionDB(home / "state.db", read_only=True)
    assert opened
    import sqlite3
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
    with SessionDB(home / "state.db", read_only=True) as fresh:
        assert fresh.get_session("successor")["id"] == "successor"
        assert fresh.get_session("original") is None

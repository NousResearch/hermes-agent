"""SessionDB must still open on a filesystem that refuses chmod.

Object-storage NFS gateways, ``all_squash`` exports and their virtiofs pass-through pin every
file to a fixed owner/mode and answer ``chmod`` with EPERM even on a freshly created file.
``_secure_state_db_files`` tightens state.db and its sidecars to 0600 before every open; that
hardening must not decide whether the session store opens at all.
"""
import logging
import sqlite3
import stat
import sys

import pytest

import hermes_state


@pytest.fixture
def chmod_refused(monkeypatch):
    calls = []

    def refuse(path, mode, *args, **kwargs):
        calls.append((str(path), mode))
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(hermes_state.os, "chmod", refuse)
    monkeypatch.setattr(hermes_state, "_CHMOD_REFUSED_WARNED", False)
    return calls


def test_securing_a_fresh_state_db_survives_eperm(tmp_path, chmod_refused):
    db = tmp_path / "state.db"
    hermes_state._secure_state_db_files(db, create_main=True)
    assert db.exists()
    assert chmod_refused, "chmod is still attempted; best-effort must not degrade into never"


def test_securing_existing_sidecars_survives_eperm(tmp_path, chmod_refused):
    for name in ("state.db", "state.db-wal", "state.db-shm"):
        (tmp_path / name).write_bytes(b"")
    hermes_state._secure_state_db_files(tmp_path / "state.db")
    assert len(chmod_refused) == 3


def test_session_db_opens_and_creates_schema_when_chmod_is_refused(tmp_path, chmod_refused, caplog):
    path = tmp_path / "state.db"
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        db = hermes_state.SessionDB(db_path=path)
        try:
            assert db.db_path == path
        finally:
            db.close()
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    finally:
        con.close()
    assert "sessions" in tables
    assert chmod_refused
    assert len([r for r in caplog.records if "refused chmod" in r.getMessage()]) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
def test_mode_is_still_tightened_where_the_filesystem_allows_it(tmp_path):
    db = tmp_path / "state.db"
    db.write_bytes(b"")
    db.chmod(0o644)
    hermes_state._secure_state_db_files(db)
    assert stat.S_IMODE(db.stat().st_mode) == 0o600


def test_other_oserrors_still_propagate(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    db.write_bytes(b"")

    def boom(path, mode, *args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(hermes_state.os, "chmod", boom)
    with pytest.raises(OSError):
        hermes_state._secure_state_db_files(db)

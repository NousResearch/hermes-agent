"""Tests for SQLCipher whole-database encryption (state.db / kanban.db).

The ``hermes_state`` module decides whether to use SQLCipher at import time,
so the "open an encrypted database" half of each test runs in a fresh
subprocess that imports the module with encryption already active.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("sqlcipher3")

from hermes_constants import get_hermes_home  # noqa: E402
from hermes_crypto import dbcrypt, migrate  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_in_subprocess(script: str) -> subprocess.CompletedProcess:
    """Run *script* in a fresh interpreter with the test HERMES_HOME + keyfile."""
    import os

    env = dict(os.environ)
    env["HERMES_HOME"] = str(get_hermes_home())
    env["PYTHONPATH"] = str(_REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


# ── dbcrypt unit tests (in-process — no module rebind needed) ───────────────


def test_dbcrypt_encrypt_decrypt_round_trip():
    import sqlite3

    db = get_hermes_home() / "scratch.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t(x TEXT)")
    conn.execute("INSERT INTO t VALUES('hello')")
    conn.commit()
    conn.close()

    dek = bytes(range(32))
    assert dbcrypt.is_plaintext_sqlite(db)

    dbcrypt.encrypt_database(db, dek)
    assert dbcrypt.is_not_plaintext_sqlite(db)
    assert not dbcrypt.is_plaintext_sqlite(db)

    enc = dbcrypt.connect_encrypted(db, dek)
    assert enc.execute("SELECT x FROM t").fetchone()[0] == "hello"
    enc.close()

    dbcrypt.decrypt_database(db, dek)
    assert dbcrypt.is_plaintext_sqlite(db)


def test_is_not_plaintext_sqlite_heuristic():
    home = get_hermes_home()
    missing = home / "missing.db"
    empty = home / "empty.db"
    junk = home / "junk.db"

    assert not dbcrypt.is_not_plaintext_sqlite(missing)
    empty.write_bytes(b"")
    assert not dbcrypt.is_not_plaintext_sqlite(empty)
    junk.write_bytes(b"not a sqlite file")
    assert dbcrypt.is_not_plaintext_sqlite(junk)


def test_dbcrypt_wrong_key_rejected():
    import sqlite3

    db = get_hermes_home() / "scratch2.db"
    sqlite3.connect(str(db)).close()
    dbcrypt.encrypt_database(db, bytes(range(32)))
    with pytest.raises(Exception):
        dbcrypt.connect_encrypted(db, bytes(32))


# ── End-to-end: encrypt state.db, then open it from a fresh process ─────────


def test_state_db_encrypts_and_stays_searchable():
    import hermes_state

    # 1. Build a plaintext state.db with a searchable message. An explicit
    #    path is required because hermes_state.DEFAULT_DB_PATH is bound at
    #    import time, while the hermetic conftest gives each test a fresh
    #    HERMES_HOME.
    state_path = get_hermes_home() / "state.db"
    db = hermes_state.SessionDB(state_path)
    db.create_session("s1", "cli", model="m")
    db.append_message("s1", "user", content="encrypted database pineapple token")
    db.close()
    assert dbcrypt.is_plaintext_sqlite(state_path)

    # 2. Encrypt it (keyfile mode — a subprocess can unlock with no prompt).
    migrate.enable("keyfile", encrypt_databases=True)
    assert dbcrypt.is_not_plaintext_sqlite(state_path)

    # 3. A fresh process must open it via SQLCipher and keep FTS working.
    result = _run_in_subprocess(
        """
        import hermes_state
        assert hermes_state._DB_ENCRYPTED, "expected SQLCipher rebind"
        db = hermes_state.SessionDB()
        sess = db.get_session("s1")
        assert sess is not None and sess["model"] == "m", sess
        hits = db.search_messages("pineapple")
        assert hits, "FTS5 search must work on the encrypted database"
        db.append_message("s1", "user", content="second write works too")
        db.close()
        print("DB-OK")
        """
    )
    assert "DB-OK" in result.stdout, (result.stdout, result.stderr)


def test_disable_decrypts_state_db():
    import hermes_state

    state_path = get_hermes_home() / "state.db"
    db = hermes_state.SessionDB(state_path)
    db.create_session("s2", "cli")
    db.close()

    migrate.enable("keyfile", encrypt_databases=True)
    assert dbcrypt.is_not_plaintext_sqlite(state_path)

    migrate.disable()
    assert dbcrypt.is_plaintext_sqlite(state_path)

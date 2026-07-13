"""The state.db health-check and schema-repair paths must key the connection.

``hermes_state`` rebinds the module-level ``sqlite3`` name to
``sqlcipher3.dbapi2`` when database encryption is on, so a connection opened
without ``PRAGMA key`` raises ``file is not a database`` on its first statement.
``_db_opens_cleanly`` and ``repair_state_db_schema`` used to open exactly such
unkeyed connections, which meant:

* a **healthy** encrypted state.db was reported corrupt, every single time —
  and ``hermes sessions repair`` then offered destructive ``writable_schema``
  surgery on it; and
* a **genuinely** corrupt encrypted state.db could never be repaired.

The rebind is decided at import time, so every "open an encrypted DB" assertion
below runs in a fresh interpreter — same pattern as ``test_database_encryption``.
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

FAST_ARGON2 = {"time_cost": 1, "memory_cost_kib": 8, "parallelism": 1}

def _run_in_subprocess(script: str) -> subprocess.CompletedProcess:
    """Run *script* in a fresh interpreter with the test HERMES_HOME.

    The passphrase is scrubbed from the environment and stdin is an empty,
    closed pipe, so a passphrase-mode keystore genuinely cannot unlock: no env
    var to read, and ``sys.stdin.isatty()`` is False so ``get_data_key()``
    raises instead of prompting.

    ``input=""`` rather than ``stdin=DEVNULL`` on purpose — on Windows DEVNULL
    is the ``NUL`` *character* device, for which ``isatty()`` returns **True**.
    The child would then reach ``getpass``, which on Windows reads the console
    directly via ``msvcrt`` (ignoring the redirection) and blocks forever.
    """
    import os

    env = dict(os.environ)
    env["HERMES_HOME"] = str(get_hermes_home())
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env.pop("HERMES_ENCRYPTION_PASSPHRASE", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        input="",
        timeout=120,
    )


def _build_state_db(*, sessions: int = 1) -> Path:
    import hermes_state

    state_path = get_hermes_home() / "state.db"
    db = hermes_state.SessionDB(state_path)
    for i in range(sessions):
        db.create_session(f"s{i}", "cli", model="m")
        db.append_message(f"s{i}", "user", content=f"encrypted pineapple {i}")
    db.close()
    assert dbcrypt.is_plaintext_sqlite(state_path)
    return state_path


def _encrypted_state_db(*, sessions: int = 1) -> Path:
    """Build a healthy state.db, then encrypt it (keyfile — unlocks silently)."""
    state_path = _build_state_db(sessions=sessions)
    migrate.enable("keyfile", encrypt_databases=True)
    assert dbcrypt.is_not_plaintext_sqlite(state_path)
    return state_path


def test_healthy_encrypted_db_opens_cleanly():
    """The core bug: a healthy encrypted state.db must not read as corrupt."""
    _encrypted_state_db()

    result = _run_in_subprocess(
        """
        import hermes_state
        from hermes_constants import get_hermes_home

        assert hermes_state._DB_ENCRYPTED, "expected SQLCipher rebind"
        p = get_hermes_home() / "state.db"
        reason = hermes_state._db_opens_cleanly(p)
        assert reason is None, f"healthy encrypted DB reported corrupt: {reason}"
        print("CLEAN-OK")
        """
    )
    assert "CLEAN-OK" in result.stdout, (result.stdout, result.stderr)


def test_sessions_repair_reports_clean_on_healthy_encrypted_db():
    """'hermes sessions repair' must not offer surgery on a healthy encrypted DB."""
    _encrypted_state_db()

    result = _run_in_subprocess(
        """
        import sys
        import hermes_cli.main as main

        sys.argv = ["hermes", "sessions", "repair", "--check-only"]
        try:
            main.main()
        except SystemExit:
            pass
        """
    )
    assert "opens cleanly" in result.stdout, (result.stdout, result.stderr)
    assert "does not open cleanly" not in result.stdout, result.stdout


def test_corrupt_encrypted_db_is_repaired():
    """A genuinely corrupt encrypted DB is repairable — and stays encrypted."""
    _encrypted_state_db()

    result = _run_in_subprocess(
        """
        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_crypto import dbcrypt

        p = get_hermes_home() / "state.db"

        # Inject the duplicate messages_fts row that produces "malformed
        # database schema (messages_fts)" — the same corruption
        # tests/test_state_db_malformed_repair.py builds, but written through a
        # *keyed* connection so it lands inside the ciphertext.
        conn = hermes_state.connect_state_db(p, isolation_level=None)
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute(
            "INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) "
            "SELECT type, name, tbl_name, rootpage, sql FROM sqlite_master "
            "WHERE name='messages_fts'"
        )
        conn.execute("PRAGMA writable_schema=OFF")
        conn.commit()
        conn.close()

        assert hermes_state._db_opens_cleanly(p) is not None, "corruption did not take"

        report = hermes_state.repair_state_db_schema(p)
        assert report["repaired"], report

        # Repair strategy 2 runs VACUUM, which rewrites the whole file — it
        # must not silently drop the database back to plaintext SQLite.
        assert dbcrypt.is_not_plaintext_sqlite(p), "repair decrypted the database"

        db = hermes_state.SessionDB(p)
        assert db.get_session("s0") is not None
        db.close()
        print("REPAIR-OK")
        """
    )
    assert "REPAIR-OK" in result.stdout, (result.stdout, result.stderr)


def test_locked_keystore_is_not_reported_as_corruption():
    """A keystore we cannot unlock says nothing about the file's integrity.

    Guards the misdiagnosis this fix could easily have introduced: surfacing
    "cannot unlock" as a repair *reason* would invite the user to run
    writable_schema surgery on a perfectly healthy encrypted database.
    """
    state_path = _build_state_db()
    # Passphrase mode: the subprocess gets no passphrase and no TTY, so the
    # keystore genuinely cannot unlock there.
    migrate.enable(
        "passphrase", passphrase="pw", argon2_params=FAST_ARGON2, encrypt_databases=True
    )
    assert dbcrypt.is_not_plaintext_sqlite(state_path)
    before = state_path.read_bytes()

    result = _run_in_subprocess(
        """
        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_crypto.errors import HermesCryptoError

        p = get_hermes_home() / "state.db"

        try:
            hermes_state._db_opens_cleanly(p)
        except HermesCryptoError:
            pass
        else:
            raise AssertionError("locked keystore must not yield a corruption verdict")

        report = hermes_state.repair_state_db_schema(p)
        assert report["repaired"] is False, report
        assert "keystore" in (report["error"] or "").lower(), report
        assert report["backup_path"] is None, "must not back up a DB it cannot read"
        print("LOCKED-OK")
        """
    )
    assert "LOCKED-OK" in result.stdout, (result.stdout, result.stderr)
    # Nothing touched the database, and no stray backup was left behind.
    assert state_path.read_bytes() == before
    assert not list(state_path.parent.glob("*.malformed-backup-*"))

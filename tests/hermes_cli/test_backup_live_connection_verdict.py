"""verify_sqlite_integrity: a live in-process connection is indeterminate, never corruption."""

import sqlite3
from contextlib import closing


def test_integrity_verdict_is_indeterminate_not_corrupt_while_a_connection_is_live(tmp_path):
    """A tracked live connection must not read as database corruption.

    ``read_header_bytes_preopen`` refuses a byte-level read while a connection is
    live (a raw ``close()`` would cancel that connection's POSIX advisory locks).
    It signals refusal by returning ``None`` — the same value it returns for a
    genuinely unreadable file — so ``verify_sqlite_integrity`` used to answer
    ``valid=False, "cannot read header"`` and the updater declared a healthy
    state.db corrupt, then refused its own repair with the same live connection.

    Refusal is not evidence of corruption: the verdict must be indeterminate, and
    a genuinely unreadable file must still be reported invalid.
    """
    from hermes_cli.backup import verify_sqlite_integrity
    from hermes_cli.sqlite_safe_read import connect_tracked

    db = tmp_path / "state.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("CREATE TABLE t (a INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()

    healthy = verify_sqlite_integrity(db, check_header=True, run_pragma=False)
    assert healthy["valid"] is True

    live = connect_tracked(db)
    try:
        assert verify_sqlite_integrity(db, check_header=True, run_pragma=False)["valid"] is not False, (
            "a live connection was reported as corruption"
        )
    finally:
        live.close()

    # Refusal must not become a blanket pass either: a file that is not a database
    # is still invalid once nothing is holding it.
    db.write_bytes(b"this is not a SQLite database" * 64)
    broken = verify_sqlite_integrity(db, check_header=True, run_pragma=False)
    assert broken["valid"] is False

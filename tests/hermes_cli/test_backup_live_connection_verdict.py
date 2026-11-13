"""verify_sqlite_integrity: a live in-process connection is indeterminate, never corruption,
and never hides deeper corruption."""

import sqlite3
from contextlib import closing
from pathlib import Path


def _multi_page_db(path: Path) -> Path:
    """A real multi-page database: enough rows that the b-tree owns interior pages."""
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA page_size=1024")
        conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, blob TEXT)")
        conn.executemany(
            "INSERT INTO sessions (blob) VALUES (?)", [(f"{'x' * 900}{i}",) for i in range(200)])
        conn.commit()
    return path


def _wreck_interior_page(path: Path, page: int) -> None:
    """Overwrite one whole interior page, leaving page 1 — and its header — intact.

    The result still *opens*, so ``connect_tracked`` succeeds on it: interior damage is
    the corruption class that a live connection must not be able to hide.
    """
    page_size = int.from_bytes(path.read_bytes()[16:18], "big")
    raw = bytearray(path.read_bytes())
    assert len(raw) > (page + 1) * page_size, "fixture must span more pages than it wrecks"
    for offset in range(page * page_size, (page + 1) * page_size):
        raw[offset] = 0xFF
    path.write_bytes(bytes(raw))
    assert path.read_bytes()[:16] == b"SQLite format 3\0", "the header must survive the wreck"


def test_live_connection_leaves_only_the_header_verdict_indeterminate(tmp_path):
    """A tracked live connection costs the byte probe, not the whole verdict.

    ``read_header_bytes_preopen`` refuses a byte-level read while a connection is
    live (a raw ``close()`` would cancel that connection's POSIX advisory locks) and
    signals refusal with ``None`` — the same value it returns for an unreadable file.
    Reading that as a verdict made a healthy state.db look corrupt, so the updater
    declared it so and then refused its own repair with that same connection.

    The size ceiling and ``PRAGMA integrity_check`` open their own read-only
    connection and are unaffected, so they must still decide. ``None`` is reserved
    for the one case where skipping the byte probe left nothing to run.
    """
    from hermes_cli.backup import verify_sqlite_integrity
    from hermes_cli.sqlite_safe_read import connect_tracked

    db = _multi_page_db(tmp_path / "state.db")
    assert verify_sqlite_integrity(db)["valid"] is True

    live = connect_tracked(db)
    try:
        # Header-only check, header unavailable: no verdict was reached, and that is
        # not a corruption claim.
        assert verify_sqlite_integrity(db, check_header=True, run_pragma=False)["valid"] is None
        # The pragma walks every page: it decides, and it says the file is fine.
        assert verify_sqlite_integrity(db, check_header=True, run_pragma=True)["valid"] is True
        # The size-ceiling probe replaces the pragma for huge files and decides too.
        ceiling = verify_sqlite_integrity(db, check_header=True, run_pragma=True, max_bytes=1)
        assert ceiling["valid"] is True
        # ...and it must not credit a check that never ran: the same verdict is reached by a
        # different set of checks once nothing is live, so the message has to say which.
        assert "header check skipped" in ceiling["message"]
        assert "header + schema probe passed" not in ceiling["message"]
    finally:
        live.close()

    # Indeterminate must not decay into a blanket pass: a file that is not a database
    # is still invalid once nothing is holding it.
    db.write_bytes(b"this is not a SQLite database" * 64)
    assert verify_sqlite_integrity(db, check_header=True, run_pragma=False)["valid"] is False


def test_live_connection_does_not_mask_corruption_the_deeper_check_finds(tmp_path):
    """The reason the deeper checks must survive the skipped byte probe.

    Deep corruption is exactly the class of damage that still *opens*, so it is also
    the class a live connection coincides with. Bailing out with an indeterminate
    verdict on the byte probe would silently launder that corruption: the guard would
    have had a database in hand whose ``PRAGMA integrity_check`` fails and reported
    nothing. Whatever the byte probe could not answer, the checks that follow it must.
    """
    from hermes_cli.backup import verify_sqlite_integrity
    from hermes_cli.sqlite_safe_read import connect_tracked

    db = _multi_page_db(tmp_path / "state.db")
    _wreck_interior_page(db, page=3)

    live = connect_tracked(db)
    try:
        result = verify_sqlite_integrity(db, check_header=True, run_pragma=True)
    finally:
        live.close()

    assert result["valid"] is False, f"a live connection hid real corruption: {result['message']}"
    # The corruption is real, not an artifact of the live handle: the same call
    # reaches the same verdict once the connection is closed.
    assert verify_sqlite_integrity(db, check_header=True, run_pragma=True)["valid"] is False

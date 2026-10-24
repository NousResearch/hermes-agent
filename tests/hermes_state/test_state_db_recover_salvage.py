"""Regression for #141757: torn BASE-TABLE b-tree pages are beyond every in-place repair
strategy — the final VACUUM has to read every base-table row — so ``hermes doctor --fix`` could
only punt to a manual ``sqlite3 state.db ".recover"`` the field rarely runs. The salvage strategy
runs that remedy automatically and drops the .recover FTS5 reconstruction (raw shadow rows the
engine never built re-corrupt the write path). Design and field-tested approach: the #141757
reporter.
"""

import hashlib
import shutil
import sqlite3
import struct
import subprocess
import uuid
from pathlib import Path

import pytest

import hermes_state_repair
from hermes_state import SessionDB
from hermes_state_repair import repair_state_db_schema


def _sqlite3_cli_version() -> tuple[int, ...]:
    """Version tuple of the ``sqlite3`` CLI on PATH; ``(0,)`` when absent or unparseable."""
    if shutil.which("sqlite3") is None:
        return (0,)
    try:
        first = subprocess.run(
            ["sqlite3", "--version"], capture_output=True, check=True,
        ).stdout.decode().split()[0]
        return tuple(int(part) for part in first.split("."))
    except Exception:
        return (0,)


# Salvaging torn pages needs the modern .recover (it walks readable leaf pages and skips the
# corrupt ones); 3.40-era CLIs abort on them, so those hosts keep today's fall-through behaviour.
needs_modern_sqlite3_cli = pytest.mark.skipif(
    _sqlite3_cli_version() < (3, 45),
    reason="salvage needs sqlite3 CLI >= 3.45 (modern .recover page walking); "
           "older CLIs and stock Windows fall through unchanged",
)

MESSAGE_COUNT = 6
CORRUPTED_MARKER = b"MARKER-0003"


def _page_size(path: Path) -> int:
    """SQLite page size from the file header (1 encodes 65536)."""
    value = struct.unpack(">H", path.read_bytes()[16:18])[0]
    return 65536 if value == 1 else value


def _build_db_one_message_per_page(db_path: Path) -> str:
    """A healthy db whose messages each occupy their own leaf page, so corrupting one page
    destroys exactly one message and nothing else."""
    session_id = str(uuid.uuid4())
    db = SessionDB(db_path=db_path)
    db.create_session(session_id=session_id, source="cli")
    db.close()
    # Size each row past half a page after the file (and its page size) exists: no two rows
    # share a leaf page and the row never overflows (still under the usable page budget).
    content_len = _page_size(db_path) // 2 + 512
    db = SessionDB(db_path=db_path)
    for i in range(MESSAGE_COUNT):
        marker = f"MARKER-{i:04d}"
        filler = (f"filler-{i}-" * (content_len // 11 + 1))[:content_len]
        db.append_message(session_id, role="user", content=f"{marker} {filler} {marker}")
    db.close()
    return session_id


def _corrupt_page_containing(path: Path, needle: bytes) -> None:
    """Overwrite the single page holding *needle* with garbage, preserving file length — the
    torn-write shape: structure gone, every other page intact."""
    data = bytearray(path.read_bytes())
    page_size = _page_size(path)
    hits = [
        start for start in range(0, len(data), page_size)
        if needle in data[start:start + page_size]
    ]
    assert len(hits) == 1, f"fixture expects {needle!r} on exactly one page, found {len(hits)}"
    # The unicode61 tokenizer splits the marker at '-', and the trigram index stores 3-grams, so
    # only the raw messages row can hold the verbatim string — the page is a messages leaf.
    data[hits[0]:hits[0] + page_size] = b"\x5a" * page_size
    path.write_bytes(bytes(data))


def _recovered_markers(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        contents = [row[0] or "" for row in conn.execute("SELECT content FROM messages")]
    finally:
        conn.close()
    return {c[:11] for c in contents if c.startswith("MARKER-")}


@needs_modern_sqlite3_cli
def test_base_table_corruption_is_salvaged(tmp_path):
    """A torn messages page ends every in-place strategy; the salvage recovers every other
    message and the db reopens clean, with the untrustworthy FTS reconstruction dropped."""
    db_path = tmp_path / "state.db"
    _build_db_one_message_per_page(db_path)
    _corrupt_page_containing(db_path, CORRUPTED_MARKER)
    assert hermes_state_repair._db_opens_cleanly(db_path) is not None, "fixture precondition: torn page detected"

    report = repair_state_db_schema(db_path, backup=False)

    assert report["repaired"] is True, report
    assert report["strategy"] == "recover_salvage"
    assert hermes_state_repair._db_opens_cleanly(db_path) is None
    conn = sqlite3.connect(str(db_path))
    try:
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        fts_schema = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'messages_fts%'").fetchone()[0]
    finally:
        conn.close()
    assert sessions == 1
    assert fts_schema == 0, "the .recover FTS reconstruction must be dropped, not trusted"
    markers = _recovered_markers(db_path)
    expected = {f"MARKER-{i:04d}" for i in range(MESSAGE_COUNT)} - {CORRUPTED_MARKER.decode()}
    assert markers == expected, "every message on an intact page survives; the torn page's row does not"
    leftovers = sorted(p.name for p in db_path.parent.glob("*.recover.sql"))
    assert leftovers == [], "the salvage SQL script is transient"

    # The next open rebuilds the FTS indexes from messages and the write path works again.
    db = SessionDB(db_path=db_path)
    try:
        hits = db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH '\"MARKER-0000\"'").fetchone()[0]
        assert hits >= 1, "FTS rebuilt from the salvaged messages on reopen"
    finally:
        db.close()


def test_missing_sqlite3_cli_falls_through_without_touching_the_file(tmp_path, monkeypatch):
    """No sqlite3 CLI (stock Windows): the report falls through unchanged and the damaged file
    is left byte-identical — the scratch discipline the repair path promises."""
    db_path = tmp_path / "state.db"
    _build_db_one_message_per_page(db_path)
    _corrupt_page_containing(db_path, CORRUPTED_MARKER)
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()

    monkeypatch.setattr(hermes_state_repair.shutil, "which", lambda _name: None)
    report = repair_state_db_schema(db_path, backup=False)

    assert report["repaired"] is False
    assert "recover salvage failed" in (report["error"] or "")
    assert "sqlite3 CLI not found" in (report["error"] or "")
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before


@needs_modern_sqlite3_cli
def test_salvage_does_not_resurrect_a_garbage_file(tmp_path):
    """A file with no salvageable structure must not count as repaired: the clean probe needs
    a readable hermes schema, so the attempt ledger (#86747) still sees failure."""
    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"SQLite format 3\x00" + b"\x5a" * 4096)

    report = repair_state_db_schema(db_path, backup=False)

    assert report["repaired"] is False
    assert report["strategy"] is None

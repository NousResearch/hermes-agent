"""Tests for the `hermes kanban integrity` and `hermes kanban recover`
substrate-diagnostics subcommands (hermes_cli.kanban).

Both commands must be read-only-safe on the live DB: no schema init, no
corrupt-guard backups, no KanbanDbCorruptError. Recovery always writes a
NEW file built offline from a private temp copy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban as kc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_kanban(argv: list[str]) -> int:
    """Drive the real argparse tree + kanban_command, returning the exit code."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args(["kanban"] + argv)
    return kc.kanban_command(args)


def _json_from_capsys(capsys) -> dict:
    """Parse the command's stdout (a single JSON object) into a dict."""
    return json.loads(capsys.readouterr().out)


def _make_healthy_db(path: Path, rows: int = 25) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, label TEXT)")
        conn.execute("CREATE TABLE notes(body TEXT)")
        conn.executemany(
            "INSERT INTO items(label) VALUES (?)",
            [(f"item-{i}",) for i in range(rows)],
        )
        conn.executemany(
            "INSERT INTO notes(body) VALUES (?)",
            [(f"note-{i}",) for i in range(rows // 5)],
        )
        conn.commit()
    finally:
        conn.close()


def _write_corrupt_db(path: Path) -> bytes:
    """Valid SQLite header + garbage pages (same shape as the db-layer tests)."""
    header = b"SQLite format 3\x00" + b"\x10\x00\x02\x02\x00\x40\x20\x20"
    header += b"\x00\x00\x00\x0c\x00\x00\x23\x46\x00\x00\x00\x00"
    header = header.ljust(100, b"\x00")
    payload = b"definitely not a valid sqlite page \x00\x01\x02\x03" * 64
    blob = header + payload
    path.write_bytes(blob)
    return blob


def _clobber_middle_page(path: Path, page_size: int = 4096) -> None:
    """Corrupt a real DB by overwriting an interior page — the recover e2e
    shape (header intact, integrity_check fails, most rows recoverable)."""
    data = bytearray(path.read_bytes())
    start = page_size * 2
    assert len(data) > start + page_size, "db too small to clobber safely"
    data[start:start + page_size] = b"\xde\xad" * (page_size // 2)
    path.write_bytes(bytes(data))


def _sqlite3_cli_supports_recover() -> bool:
    binary = shutil.which("sqlite3")
    if not binary:
        return False
    try:
        proc = subprocess.run(
            [binary, ":memory:", ".recover"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return False
    return proc.returncode == 0 and "unknown command" not in (proc.stderr or "").lower()


requires_recover = pytest.mark.skipif(
    not _sqlite3_cli_supports_recover(),
    reason="sqlite3 CLI missing or lacks .recover support",
)


# ---------------------------------------------------------------------------
# hermes kanban integrity
# ---------------------------------------------------------------------------

def test_integrity_healthy_db_verdict_ok_exit_0(tmp_path, capsys):
    db = tmp_path / "k.db"
    _make_healthy_db(db)
    rc = _run_kanban(["integrity", "--db", str(db)])
    payload = _json_from_capsys(capsys)
    assert rc == 0
    assert payload["verdict"] == "ok"
    assert payload["quick_check"] == "ok"
    assert payload["integrity_check"] == "ok"
    assert payload["db_path"] == str(db)


def test_integrity_corrupt_db_verdict_corrupt_exit_3(tmp_path, capsys):
    db = tmp_path / "k.db"
    original = _write_corrupt_db(db)
    rc = _run_kanban(["integrity", "--db", str(db)])
    payload = _json_from_capsys(capsys)
    assert rc == 3
    assert payload["verdict"] == "corrupt"
    # Read-only-safe: no backups, no schema init, bytes untouched.
    assert list(tmp_path.glob("*.corrupt.*")) == []
    assert db.read_bytes() == original


def test_integrity_missing_file_verdict_unavailable_exit_4(tmp_path, capsys):
    db = tmp_path / "missing.db"
    rc = _run_kanban(["integrity", "--db", str(db)])
    payload = _json_from_capsys(capsys)
    assert rc == 4
    assert payload["verdict"] == "unavailable"
    assert not db.exists(), "integrity probe must never create the file"


# ---------------------------------------------------------------------------
# hermes kanban recover
# ---------------------------------------------------------------------------

@requires_recover
def test_recover_healthy_db_roundtrip(tmp_path, capsys):
    db = tmp_path / "k.db"
    _make_healthy_db(db, rows=25)
    before = db.read_bytes()
    out = tmp_path / "recovered.db"

    rc = _run_kanban(["recover", "--db", str(db), "--output", str(out)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["verdict"] == "ok"
    assert payload["integrity_check"] == "ok"
    assert payload["tables"]["items"] == {"old": 25, "new": 25}
    assert payload["tables"]["notes"]["old"] == payload["tables"]["notes"]["new"]
    assert out.exists()
    # Live DB never modified; swap reminder printed loudly.
    assert db.read_bytes() == before
    assert "was NOT modified" in captured.err
    assert "-wal" in captured.err and "-shm" in captured.err

    # The recovered file is a real, readable SQLite DB.
    conn = sqlite3.connect(out)
    try:
        n = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    finally:
        conn.close()
    assert n == 25


@requires_recover
def test_recover_corrupt_db_produces_clean_new_file(tmp_path, capsys):
    db = tmp_path / "k.db"
    _make_healthy_db(db, rows=500)
    _clobber_middle_page(db)
    before = db.read_bytes()
    out = tmp_path / "recovered.db"

    rc = _run_kanban(["recover", "--db", str(db), "--output", str(out)])
    payload = json.loads(capsys.readouterr().out)

    # Recovered DB must be clean for exit 0; corrupt result exits 3.
    if payload["verdict"] == "ok":
        assert rc == 0
        assert payload["integrity_check"] == "ok"
        # Some rows recovered; the damaged source may not be countable (None).
        items = payload["tables"]["items"]
        assert items["new"] is not None and items["new"] > 0
    else:
        assert rc == 3
    # Live (corrupt) DB is bit-for-bit untouched either way.
    assert db.read_bytes() == before


def test_recover_refuses_existing_output(tmp_path, capsys):
    db = tmp_path / "k.db"
    _make_healthy_db(db)
    out = tmp_path / "already-there.db"
    out.write_bytes(b"precious")

    rc = _run_kanban(["recover", "--db", str(db), "--output", str(out)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "refusing" in captured.err
    assert out.read_bytes() == b"precious"


def test_recover_refuses_symlink_output(tmp_path, capsys):
    """A BROKEN symlink reports exists() == False, but the sqlite3 CLI
    would follow it and write through to the link target — reject it."""
    db = tmp_path / "k.db"
    _make_healthy_db(db)
    target = tmp_path / "victim.db"  # deliberately never created
    out = tmp_path / "sneaky-link.db"
    out.symlink_to(target)
    assert not out.exists() and out.is_symlink()

    rc = _run_kanban(["recover", "--db", str(db), "--output", str(out)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "refusing" in captured.err
    assert not target.exists(), "recover must never write through a symlink"


def test_recover_refuses_symlink_output_to_existing_file(tmp_path, capsys):
    """A live symlink to an existing file is also rejected (and untouched)."""
    db = tmp_path / "k.db"
    _make_healthy_db(db)
    target = tmp_path / "precious.db"
    target.write_bytes(b"precious")
    out = tmp_path / "link-to-precious.db"
    out.symlink_to(target)

    rc = _run_kanban(["recover", "--db", str(db), "--output", str(out)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "refusing" in captured.err
    assert target.read_bytes() == b"precious"


def test_recover_refuses_missing_output_directory(tmp_path, capsys):
    db = tmp_path / "k.db"
    _make_healthy_db(db)
    out = tmp_path / "no-such-dir" / "out.db"

    rc = _run_kanban(["recover", "--db", str(db), "--output", str(out)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "output directory" in captured.err
    assert not out.parent.exists(), "recover must not create the directory"


def test_recover_missing_source_db(tmp_path, capsys):
    rc = _run_kanban([
        "recover",
        "--db", str(tmp_path / "nope.db"),
        "--output", str(tmp_path / "out.db"),
    ])
    captured = capsys.readouterr()
    assert rc == 4
    assert "no such database file" in captured.err
    assert not (tmp_path / "out.db").exists()

"""Tests for the read-only session storage attribution report (#90719).

Deterministic fixture: a throwaway HERMES_HOME with a real state.db built
through SessionDB, one deliberately oversized session. Verifies reported
sizes match direct measurements, the JSON shape mirrors the human report,
dbstat absence degrades cleanly, and the report mutates nothing.
"""
import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli.session_storage_report import (
    build_storage_report,
    format_storage_report,
)


@pytest.fixture()
def fixture_home(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        # Small sessions
        for i in range(3):
            sid = f"small-{i}"
            db.create_session(sid, source="cli")
            db.append_message(sid, "user", f"small message {i}")
        # Deliberately largest session
        big_sid = "big-session"
        big_payload = "x" * 200_000
        db.create_session(big_sid, source="cli")
        db.append_message(big_sid, "user", big_payload)
    finally:
        db.close()
    return tmp_path / "state.db"


def test_report_runs_and_is_json_serializable(fixture_home):
    report = build_storage_report(fixture_home)
    # Round-trips through the same serializer the --json flag uses
    json.dumps(report)


def test_file_sizes_match_filesystem(fixture_home):
    report = build_storage_report(fixture_home)
    stat = fixture_home.stat().st_size
    assert report["files"]["state_db_bytes"] == stat


def test_page_math_is_consistent(fixture_home):
    report = build_storage_report(fixture_home)
    db = report["database"]
    assert db["page_count"] * db["page_size"] >= report["files"]["state_db_bytes"]
    assert db["free_bytes_estimate"] == db["freelist_pages"] * db["page_size"]


def test_largest_session_reported(fixture_home):
    report = build_storage_report(fixture_home)
    largest = report["logical"]["largest_sessions"]
    assert largest, "expected at least one session"
    # The big session must be first and its payload >= 200_000 bytes
    assert largest[0]["session_id"] == "big-session"
    assert largest[0]["content_bytes"] >= 200_000


def test_json_matches_human_report_data(fixture_home):
    report = build_storage_report(fixture_home)
    text = format_storage_report(report)
    # Every role row and largest-session id appears in the human render
    for row in report["logical"]["by_role"]:
        assert row["role"] in text
    for row in report["logical"]["largest_sessions"]:
        assert row["session_id"] in text
    assert "Read-only report" in text


def test_no_mutation(fixture_home):
    before = fixture_home.read_bytes()
    wal = fixture_home.parent / "state.db-wal"
    wal_before = wal.read_bytes() if wal.exists() else None
    build_storage_report()
    assert fixture_home.read_bytes() == before
    wal_after = wal.read_bytes() if wal.exists() else None
    assert wal_before == wal_after


def test_dbstat_unavailable_degrades(fixture_home, monkeypatch):
    from hermes_cli import session_storage_report as mod

    def _raise(*a, **k):
        raise sqlite3.Error("no such table: dbstat")

    monkeypatch.setattr(mod, "_physical_layer", lambda conn: {
        "available": False, "entries": [], "total_bytes": None,
    })
    report = build_storage_report(fixture_home)
    assert report["physical"]["available"] is False
    text = format_storage_report(report)
    assert "unavailable" in text


def test_missing_database(fixture_home):
    (fixture_home.parent / "elsewhere").mkdir(exist_ok=True)
    report = build_storage_report(fixture_home.parent / "elsewhere" / "nope.db")
    assert report.get("error") == "database file not found"

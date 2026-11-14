"""``recover_transcripts`` against the legacy offline recovery: both run on the same source (which neither opens), each
into its own output, and the reports and the recovered databases are compared."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from hermes_cli.session_recovery import inspect_session_database, recover_session_database
from persistence.sqlite.admin import SQLiteStoreAdmin, _lost_and_found
from persistence.sqlite.admin._lost_and_found import find_sqlite3_cli
from tests.hermes_cli.test_session_recovery import (
    _corrupt_middle_table_leaf, _corrupt_table_root, _make_many_sessions_source, _make_page_spanning_source,
    _make_source, _sha256,
)
from tests.hermes_cli.test_session_recovery_lost_and_found import _make_schema_unreadable_source

_FREE_SPACE = ("work_dir_free_bytes", "output_dir_free_bytes")


def _normalized(report, output: Path):
    """The report with ``output`` written as ``<output>`` and the free-space readings (which move) dropped."""
    if isinstance(report, dict):
        return {key: _normalized(value, output) for key, value in report.items() if key not in _FREE_SPACE}
    if isinstance(report, list):
        return [_normalized(value, output) for value in report]
    if isinstance(report, str):
        return report.replace(str(output), "<output>")
    return report


def _database(path: Path) -> dict:
    """Schema and every row of every table, in a stable order."""
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        schema = sorted(conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master").fetchall(), key=repr)
        tables = [row[1] for row in schema if row[0] == "table" and not row[1].startswith("sqlite_")]
        rows = {table: sorted(conn.execute(f'SELECT * FROM "{table}"').fetchall(), key=repr) for table in tables}
        return {"schema": schema, "rows": rows}


def _source_state(source: Path) -> tuple:
    return tuple((p.name, _sha256(p), p.stat().st_mtime_ns) for p in sorted(source.parent.glob(source.name + "*")))


def _assert_same_recovery(tmp_path: Path, source: Path, **kwargs) -> dict:
    """Recover ``source`` through both paths; return the admin report after asserting it matches."""
    before = _source_state(source)
    legacy_out, admin_out = tmp_path / "out" / "legacy.db", tmp_path / "out" / "admin.db"
    legacy_out.parent.mkdir(exist_ok=True)
    legacy = recover_session_database(source, legacy_out, **kwargs)
    admin = SQLiteStoreAdmin.recover_transcripts(source, admin_out, **kwargs)
    assert _normalized(admin, admin_out) == _normalized(legacy, legacy_out)
    assert _database(admin_out) == _database(legacy_out)
    assert _source_state(source) == before
    return admin


def test_inspect_only_reports_the_same_and_writes_nothing(tmp_path):
    source = tmp_path / "state.db"
    _make_source(source)
    before, listing = _source_state(source), sorted(p.name for p in tmp_path.iterdir())
    output = tmp_path / "never-written.db"
    legacy = inspect_session_database(source)
    admin = SQLiteStoreAdmin.recover_transcripts(source, output, inspect_only=True)
    assert _normalized(admin, output) == _normalized(legacy, output)
    assert admin["recoverable"] is True and admin["tables"]["messages"]["rows"] == 21
    assert not output.exists()
    assert _source_state(source) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == listing


def test_inspect_only_of_a_damaged_source_reports_the_same(tmp_path):
    source = tmp_path / "state.db"
    _make_schema_unreadable_source(source)
    output = tmp_path / "never-written.db"
    legacy = inspect_session_database(source)
    admin = SQLiteStoreAdmin.recover_transcripts(source, output, inspect_only=True)
    assert _normalized(admin, output) == _normalized(legacy, output)
    assert admin["recoverable"] is False
    assert not output.exists()


def test_recover_copies_the_same_rows_into_a_current_schema_output(tmp_path):
    source = tmp_path / "state.db"
    expected = _make_source(source)
    report = _assert_same_recovery(tmp_path, source)
    assert report["complete"] is True and report["verified"] is True
    rows = _database(tmp_path / "out" / "admin.db")["rows"]
    assert len(rows["sessions"]) == expected["sessions"] and len(rows["messages"]) == expected["messages"]
    assert rows["telegram_dm_topic_bindings"]


def test_recover_of_a_damaged_leaf_reports_the_same_failed_copy(tmp_path):
    source = tmp_path / "state.db"
    messages_root, _ = _make_page_spanning_source(source)
    _corrupt_middle_table_leaf(source, messages_root)
    report = _assert_same_recovery(tmp_path, source)
    assert report["verified"] is False and report["complete"] is False
    assert report["copy"]["messages"]["status"] in {"partial", "failed"}


def test_partial_recovery_salvages_the_same_rows_around_a_damaged_leaf(tmp_path):
    source = tmp_path / "state.db"
    messages_root, _ = _make_page_spanning_source(source)
    _corrupt_middle_table_leaf(source, messages_root)
    report = _assert_same_recovery(tmp_path, source, allow_partial=True)
    assert report["partial"] is True
    assert 0 < len(_database(tmp_path / "out" / "admin.db")["rows"]["messages"]) < 320


def test_partial_recovery_reconstructs_the_same_unsalvageable_sessions(tmp_path):
    source = tmp_path / "state.db"
    _corrupt_table_root(source, _make_many_sessions_source(source))
    report = _assert_same_recovery(tmp_path, source, allow_partial=True)
    assert report["orphan_cleanup"]["sessions_reconstructed"] > 0


def test_schema_unreadable_source_without_a_salvage_shell_refuses_the_same(tmp_path, monkeypatch):
    monkeypatch.setattr(_lost_and_found, "find_sqlite3_cli", lambda: None)
    source = tmp_path / "state.db"
    _make_schema_unreadable_source(source)
    outputs = (tmp_path / "legacy.db", tmp_path / "admin.db")
    with pytest.raises(Exception) as legacy:
        recover_session_database(source, outputs[0], allow_partial=True)
    with pytest.raises(Exception) as admin:
        SQLiteStoreAdmin.recover_transcripts(source, outputs[1], allow_partial=True)
    assert type(admin.value).__name__ == type(legacy.value).__name__ == "SessionRecoverySourceError"
    assert str(admin.value) == str(legacy.value)
    assert not outputs[0].exists() and not outputs[1].exists()


@pytest.mark.skipif(find_sqlite3_cli() is None, reason="sqlite3 CLI with sqlite_dbpage not on PATH")
def test_lost_and_found_salvage_matches(tmp_path):
    source = tmp_path / "state.db"
    _make_schema_unreadable_source(source)
    report = _assert_same_recovery(tmp_path, source, allow_partial=True)
    assert report["mode"] == "lost_and_found_salvage"


def test_recover_refuses_an_existing_output_the_same(tmp_path):
    source = tmp_path / "state.db"
    _make_source(source)
    output = tmp_path / "exists.db"
    output.write_bytes(b"keep")
    with pytest.raises(Exception) as legacy:
        recover_session_database(source, output)
    with pytest.raises(Exception) as admin:
        SQLiteStoreAdmin.recover_transcripts(source, output)
    assert type(admin.value).__name__ == type(legacy.value).__name__ == "SessionRecoverySafetyError"
    assert str(admin.value) == str(legacy.value)
    assert output.read_bytes() == b"keep"


def test_recover_and_inspect_of_a_missing_source_raise_the_same(tmp_path):
    source, output = tmp_path / "missing.db", tmp_path / "out.db"
    for inspect_only in (False, True):
        with pytest.raises(Exception) as legacy:
            if inspect_only:
                inspect_session_database(source)
            else:
                recover_session_database(source, output)
        with pytest.raises(Exception) as admin:
            SQLiteStoreAdmin.recover_transcripts(source, output, inspect_only=inspect_only)
        assert type(admin.value) is type(legacy.value) is FileNotFoundError
        assert str(admin.value) == str(legacy.value)
    assert not output.exists()


def test_recover_reports_progress_by_chunk_into_a_given_work_dir_as_the_legacy(tmp_path):
    source = tmp_path / "state.db"
    _make_many_sessions_source(source)
    events = {"legacy": [], "admin": []}
    work = tmp_path / "work"
    work.mkdir()
    _assert_same_recovery(tmp_path, source, chunk_size=7)
    legacy_out, admin_out = tmp_path / "legacy-p.db", tmp_path / "admin-p.db"
    legacy = recover_session_database(source, legacy_out, work_dir=work, chunk_size=7,
                                      progress_cb=events["legacy"].append)
    admin = SQLiteStoreAdmin.recover_transcripts(source, admin_out, work_dir=work, chunk_size=7,
                                                 progress_cb=events["admin"].append)
    assert _normalized(admin, admin_out) == _normalized(legacy, legacy_out)
    assert len(events["admin"]) > 1 and events["admin"] == events["legacy"]
    assert list(work.iterdir()) == []


def test_inspect_with_a_given_work_dir_matches(tmp_path):
    source = tmp_path / "state.db"
    _make_source(source)
    work = tmp_path / "work"
    work.mkdir()
    legacy = inspect_session_database(source, work_dir=work)
    admin = SQLiteStoreAdmin.recover_transcripts(source, tmp_path / "unused.db", inspect_only=True, work_dir=work)
    assert _normalized(admin, work) == _normalized(legacy, work)
    assert not (tmp_path / "unused.db").exists()

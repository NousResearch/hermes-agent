"""Preflight + audit for the deliberate archive cascade (#70185).

``set_session_archived`` flips a whole compression lineage in one recursive-CTE statement:
one sidebar archive can hide days of work with no confirmation and no trace. The read-only
preflight (:meth:`SessionDB.preview_session_archive_lineage`) gives API surfaces the blast
radius BEFORE committing so they can demand explicit confirmation; the JSONL audit append
records what each successful call covered (trigger, timestamp, ids) so a surprise cascade
leaves a recoverable trail outside state.db.
"""

import json
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _lineage(db: SessionDB) -> None:
    """root --compression--> mid --compression--> tip (oldest first)."""
    base = time.time() - 1000
    db.create_session("root", source="desktop")
    db.create_session("mid", source="desktop", parent_session_id="root")
    db.create_session("tip", source="desktop", parent_session_id="mid")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression' "
        "WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression' "
        "WHERE id = 'mid'",
        (base + 20, base + 30),
    )
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'tip'", (base + 40,))
    db._conn.commit()


def test_preview_counts_the_whole_lineage(db):
    _lineage(db)

    preview = db.preview_session_archive_lineage("tip", archived=True)

    assert preview["cascade_count"] == 3
    assert preview["cascade_extra"] == 2  # rows beyond the targeted one
    assert set(preview["affected_ids"]) == {"root", "mid", "tip"}
    assert preview["oldest_started_at"] < preview["newest_started_at"]
    # Read-only preflight: nothing flipped.
    assert db.get_session("tip")["archived"] == 0


def test_preview_single_session_has_no_cascade(db):
    db.create_session("solo", source="desktop")

    preview = db.preview_session_archive_lineage("solo", archived=True)

    assert preview["cascade_count"] == 1
    assert preview["cascade_extra"] == 0
    assert preview["affected_ids"] == ["solo"]


def test_preview_unknown_session_is_empty(db):
    assert db.preview_session_archive_lineage("no-such-row", archived=True)["cascade_count"] == 0


def test_preview_fully_archived_lineage_is_empty(db):
    """An idempotent re-archive flips nothing, so the gate must let it through."""
    _lineage(db)
    assert db.set_session_archived("tip", True)

    assert db.preview_session_archive_lineage("tip", archived=True)["cascade_count"] == 0
    # ...while the unarchive direction still sees the full lineage.
    assert db.preview_session_archive_lineage("tip", archived=False)["cascade_count"] == 3


def test_archive_appends_audit_record(db):
    _lineage(db)

    assert db.set_session_archived("mid", True, trigger="web_dashboard")

    log_path = db.db_path.parent / "logs" / "archives.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["trigger"] == "web_dashboard"
    assert record["target"] == "mid"
    assert record["archived"] is True
    assert record["cascade_count"] == 3
    assert set(record["affected_ids"]) == {"root", "mid", "tip"}


def test_unarchive_appends_audit_record(db):
    _lineage(db)
    db.set_session_archived("tip", True, trigger="web_dashboard")

    assert db.set_session_archived("tip", False, trigger="web_dashboard")

    log_path = db.db_path.parent / "logs" / "archives.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["archived"] is False
    assert records[-1]["cascade_count"] == 3


def test_audit_failure_never_blocks_the_archive(db):
    _lineage(db)
    # A `logs` regular file makes the audit dir creation fail — the archive must still land.
    (db.db_path.parent / "logs").write_text("not a directory", encoding="utf-8")

    assert db.set_session_archived("tip", True) is True
    assert db.get_session("root")["archived"] == 1

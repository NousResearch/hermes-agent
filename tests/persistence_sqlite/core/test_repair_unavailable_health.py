"""An unavailable validation must not escalate repair or consume its retry budget."""

import contextlib
import gc
import sqlite3
from pathlib import Path

from persistence.sqlite._state import repair
from tests.persistence_sqlite.core import test_fts_cjk_index_retirement as retirement
from tests.persistence_sqlite.core.test_state_db_malformed_repair import _corrupt_duplicate_fts

cjk_db = retirement.cjk_db


def _contents(path):
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA writable_schema=ON")
        schema = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
        messages = conn.execute("SELECT * FROM messages").fetchall()
        return schema, messages


def test_unavailable_post_strategy_probe_preserves_source_and_retry_budget(cjk_db, monkeypatch):
    # The legacy fixture's outside-write probe leaves a raw connection for GC.
    gc.collect()
    _corrupt_duplicate_fts(cjk_db)
    before = _contents(cjk_db)
    reason = repair._db_opens_cleanly(cjk_db)
    assert reason is not None and "already exists" in reason
    assert not repair._live_writer_holds_db(cjk_db)
    connect = repair._connect_repair_durable
    constrained_probes = []

    def constrained_connect(path, *args, **kwargs):
        conn = connect(path, *args, **kwargs)
        # Repair staging fits; the health probe's separate temporary destination does not.
        if Path(path).parent.name.startswith("hermes-cjk-probe-"):
            constrained_probes.append(conn.execute("PRAGMA max_page_count=5").fetchone()[0])
        return conn

    monkeypatch.setattr(repair, "_connect_repair_durable", constrained_connect)
    report = repair.repair_state_db_schema(cjk_db, backup=False)

    assert constrained_probes == [5]
    assert not report["repaired"] and report["strategy"] is None
    assert "could not check repaired snapshot" in report["error"]
    assert _contents(cjk_db) == before
    assert not cjk_db.with_name(cjk_db.name + ".repair-attempts.json").exists()
    assert not cjk_db.with_name(cjk_db.name + ".repair-scratch").exists()

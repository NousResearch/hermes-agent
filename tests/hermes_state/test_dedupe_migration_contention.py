"""The v25 dedupe migration must degrade gracefully on a contended DB.

Enterprise field report (2026-08-14): with state.db locked by another process,
``_dedupe_legacy_system_prompts`` raised ``sqlite3.OperationalError``
mid-loop (only the initial SELECT was guarded), which aborted schema init,
left the schema version below 25, and made EVERY subsequent
``SessionDB.__init__`` re-enter the same blocking migration — the fuel of
the gateway's watchdog crash loop.

Contract:
- A write failure mid-migration returns gracefully (no raise). Partial
  migration is safe by design: the legacy ``system_prompt`` column is kept
  as a read fallback for unmigrated rows.
- Rows migrated before the failure stay migrated; unmigrated rows remain
  readable and are picked up by a later successful run.
"""

import sqlite3

import pytest

from hermes_state import SessionDB


def _make_legacy_db(tmp_path, n_rows=5):
    """Open a real SessionDB, then regress it to a pre-v25 shape."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.close()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    for i in range(n_rows):
        cur.execute(
            "INSERT OR IGNORE INTO sessions (id, source, started_at) "
            "VALUES (?, 'test', 1.0)",
            (f"sess-{i}",),
        )
        cur.execute(
            "UPDATE sessions SET system_prompt = ?, system_prompt_hash = NULL "
            "WHERE id = ?",
            (f"legacy prompt {i}", f"sess-{i}"),
        )
    conn.commit()
    inserted = cur.execute(
        "SELECT COUNT(*) FROM sessions WHERE id LIKE 'sess-%'"
    ).fetchone()[0]
    conn.close()
    assert inserted == n_rows, f"fixture only created {inserted}/{n_rows} rows"
    return db_path


class _FailAfterN:
    """Cursor proxy: UPDATE statements start failing after N successes."""

    def __init__(self, cursor, fail_after):
        self._cursor = cursor
        self._updates = 0
        self._fail_after = fail_after

    def execute(self, sql, *args, **kwargs):
        if sql.lstrip().upper().startswith("UPDATE"):
            if self._updates >= self._fail_after:
                raise sqlite3.OperationalError("database is locked")
            self._updates += 1
        return self._cursor.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._cursor, name)


def test_mid_loop_lock_error_returns_instead_of_raising(tmp_path):
    db_path = _make_legacy_db(tmp_path)
    db = SessionDB(db_path=db_path)
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        raw = conn.cursor()
        proxy = _FailAfterN(raw, fail_after=2)
        # Must NOT raise even though the third UPDATE hits "database is locked".
        db._dedupe_legacy_system_prompts(proxy)
        conn.commit()

        rows = raw.execute(
            "SELECT id, system_prompt, system_prompt_hash FROM sessions "
            "WHERE id LIKE 'sess-%' ORDER BY id"
        ).fetchall()
        migrated = [r for r in rows if r["system_prompt"] is None]
        legacy = [r for r in rows if r["system_prompt"] is not None]
        assert migrated, "no rows migrated before the simulated lock"
        assert legacy, "expected unmigrated remainder after the failure"
        # Migrated rows carry a hash; legacy rows keep their readable prompt.
        assert all(r["system_prompt_hash"] for r in migrated)
        assert all(r["system_prompt"].startswith("legacy prompt") for r in legacy)
        conn.close()
    finally:
        db.close()


def test_later_run_completes_the_remainder(tmp_path):
    db_path = _make_legacy_db(tmp_path)
    db = SessionDB(db_path=db_path)
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        raw = conn.cursor()
        db._dedupe_legacy_system_prompts(_FailAfterN(raw, fail_after=2))
        conn.commit()
        # Second run with no failures finishes the job.
        db._dedupe_legacy_system_prompts(raw)
        conn.commit()
        remaining = raw.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE id LIKE 'sess-%' AND system_prompt IS NOT NULL"
        ).fetchone()[0]
        assert remaining == 0
        conn.close()
    finally:
        db.close()

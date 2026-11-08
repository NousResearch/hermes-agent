"""A data backfill skipped by lock contention must be retried on the next open.

``_run_data_migrations`` stamps ``schema_version`` at the end whether or not its
best-effort DATA backfills ran. A backfill gated on ``current_version < N`` alone
therefore runs exactly once: when a sibling process holds the write lock at that
instant (schema init runs on a 1s-timeout connection and gives up), the backfill was
swallowed and the version still advanced — a permanent, DEBUG-only loss of the v18
gateway-metadata backfill and the v25 prompt dedupe.

Contract:
- Contention leaves the backfill PENDING (no completion marker), so the next open
  re-runs it and the data actually lands.
- A backfill this runtime can never perform (no JSON1) is settled, so the chain does
  not re-attempt it on every open.
"""

import sqlite3
import threading
import time

from hermes_state import SessionDB
from hermes_state_schema import SCHEMA_VERSION

# The marker keys are spelled out rather than imported: the contract under test is the
# BEHAVIOUR (a contended backfill resumes on the next open), which must fail on a tree
# that has no markers at all rather than error at import.
_V18 = "data_migration_v18_gateway_metadata"
_V25 = "data_migration_v25_prompt_dedupe"

_LOCKED = sqlite3.OperationalError("database is locked")


def _regress_version(db, version):
    db._conn.execute("UPDATE schema_version SET version = ?", (version,))
    db._conn.commit()


def _legacy_prompt_rows(tmp_path, n_rows=3):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    for i in range(n_rows):
        db._conn.execute(
            "INSERT OR IGNORE INTO sessions (id, source, started_at) VALUES (?, 'test', 1.0)",
            (f"legacy-{i}",),
        )
        db._conn.execute(
            "UPDATE sessions SET system_prompt = ?, system_prompt_hash = NULL WHERE id = ?",
            (f"legacy prompt {i}", f"legacy-{i}"),
        )
    db._conn.commit()
    _regress_version(db, 24)
    db.close()
    return db_path


def test_contended_v18_backfill_retries_on_next_open(tmp_path, monkeypatch):
    """The v18 gateway-metadata backfill: swallowed by contention, then completed."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _regress_version(db, 17)
    db.close()

    calls = []
    real_backfill = SessionDB._backfill_gateway_metadata_from_sessions_json

    def _contended_once(self, cursor):
        calls.append("v18")
        if len(calls) == 1:
            raise _LOCKED
        return real_backfill(self, cursor)

    monkeypatch.setattr(
        SessionDB, "_backfill_gateway_metadata_from_sessions_json", _contended_once
    )

    contended = SessionDB(db_path=db_path)
    try:
        assert calls == ["v18"], "the backfill never ran"
        # schema_version still advanced (unchanged behaviour) …
        assert contended._conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()[0] == SCHEMA_VERSION
        # … but the contended backfill is NOT recorded as done.
        assert contended.get_meta(_V18) != "done"
    finally:
        contended.close()

    resumed = SessionDB(db_path=db_path)
    try:
        assert calls == ["v18", "v18"], "the next open did not retry the backfill"
        assert resumed.get_meta(_V18) == "done"
    finally:
        resumed.close()


def test_contended_v25_dedupe_resumes_on_next_open(tmp_path, monkeypatch):
    """v25's own warning promises "the next schema init resumes the migration"."""
    db_path = _legacy_prompt_rows(tmp_path)

    store_calls = []
    real_store = SessionDB._store_system_prompt

    def _contended_once(conn, prompt):
        store_calls.append(prompt)
        if len(store_calls) == 1:
            raise _LOCKED
        return real_store(conn, prompt)

    monkeypatch.setattr(SessionDB, "_store_system_prompt", staticmethod(_contended_once))

    paused = SessionDB(db_path=db_path)
    try:
        assert store_calls, "the dedupe never ran"
        assert paused.get_meta(_V25) != "done"
        # Partial migration stays readable: the legacy column is the fallback.
        remaining = paused._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE id LIKE 'legacy-%' AND system_prompt IS NOT NULL"
        ).fetchone()[0]
        assert remaining > 0, "expected unmigrated rows after the pause"
    finally:
        paused.close()

    resumed = SessionDB(db_path=db_path)
    try:
        assert resumed.get_meta(_V25) == "done"
        assert resumed._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE id LIKE 'legacy-%' AND system_prompt IS NOT NULL"
        ).fetchone()[0] == 0, "the remainder was never migrated"
    finally:
        resumed.close()


def test_unperformable_backfill_is_settled_not_retried_forever(tmp_path, monkeypatch):
    """A non-contention failure can never succeed here, so it must not stay pending."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _regress_version(db, 17)
    db.close()

    calls = []

    def _no_json1(self, cursor):
        calls.append("v18")
        raise sqlite3.OperationalError("no such function: json_set")

    monkeypatch.setattr(
        SessionDB, "_backfill_gateway_metadata_from_sessions_json", _no_json1
    )

    first = SessionDB(db_path=db_path)
    try:
        assert calls == ["v18"]
        assert first.get_meta(_V18) == "done"
    finally:
        first.close()

    second = SessionDB(db_path=db_path)
    try:
        assert calls == ["v18"], "an impossible backfill was retried on every open"
    finally:
        second.close()


def test_real_sibling_write_lock_defers_backfill_until_patience_retry(tmp_path):
    """The central claim against a REAL sibling writer, not an injected OperationalError.

    A second connection holds the state.db write lock, so BOTH the v18 backfill and its
    marker write contend: the whole init raises and ``_connect_and_init_with_lock_patience``
    retries the open. Once the sibling releases, the same open completes the backfill and
    stamps the version. (The injected-error tests above prove the marker state machine; this
    proves the propagate-and-retry branch end to end. Review evidence note, #134671.)
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _regress_version(db, 17)
    db.close()

    sibling = sqlite3.connect(db_path, timeout=5.0, isolation_level=None)
    try:
        sibling.execute("BEGIN IMMEDIATE")
        sibling.execute("INSERT OR IGNORE INTO state_meta (key, value) VALUES ('sibling-hold', '1')")

        opened = []
        failure = {}

        def _open():
            try:
                opened.append(SessionDB(db_path=db_path))
            except BaseException as exc:  # pragma: no cover — surfaced via `failure`
                failure["error"] = exc

        thread = threading.Thread(target=_open, name="sibling-locked-open")
        thread.start()
        time.sleep(1.5)  # first init attempt contends (1s busy timeout) and propagates
        sibling.execute("COMMIT")  # release: the patience retry completes the open
        thread.join(timeout=25.0)
        assert not thread.is_alive(), "the open never finished within write patience"
        assert not failure, f"open failed instead of retrying: {failure.get('error')!r}"

        contended = opened[0]
        try:
            assert contended.get_meta(_V18) == "done", "the deferred backfill never landed"
            assert contended._conn.execute(
                "SELECT version FROM schema_version"
            ).fetchone()[0] == SCHEMA_VERSION
        finally:
            contended.close()
    finally:
        sibling.close()

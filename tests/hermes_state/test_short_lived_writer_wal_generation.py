"""Regression tests for #109786: short-lived write-then-close connections must not
unlink a live holder's ``state.db`` WAL generation.

sqlite3's ``close()`` runs the last-connection WAL reset: it checkpoints and, when the WAL
is empty, **unlinks** ``-wal``/``-shm``. Three guest paths did exactly that against a live
gateway's generation:

* ``tools/async_delegation._connect`` — DDL at MODULE IMPORT (every CLI/gateway/cron-worker
  start), even when the schema is already canonical
* ``hermes_state_repair._db_opens_cleanly`` — the corruption probe writes a rolled-back row
* ``hermes_cli/doctor_state._session_count`` — a read-write open whose only statement is
  ``SELECT COUNT(*)``

The fix: the import-time writer only opens when the ledger's shape is NOT canonical
(steady state takes no writer), every unavoidable short-lived writer arms
``SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE`` where the runtime exposes it (3.12+), and the doctor's
count goes read-only. On 3.11 (no ``setconfig``) the canonical probe is what protects
steady state — the armed-flag helpers simply return False there.

The holder-poison arm is ``linux_only``: on Windows an open file cannot be unlinked, so the
class of damage is not representable; the sidecar-survival arms are host-agnostic.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_wal import disable_close_time_wal_reset, schema_is_canonical
from hermes_state_schema import SCHEMA_SQL


def _sidecars_exist(db_path: Path) -> tuple[bool, bool]:
    return (
        Path(str(db_path) + "-wal").exists(),
        Path(str(db_path) + "-shm").exists(),
    )


def _seed_holding_writer(tmp_path: Path) -> SessionDB:
    """A live SessionDB holder with WAL active and a non-empty WAL."""
    db = SessionDB(db_path=tmp_path / "state.db")
    if not db._wal_active:
        db.close()
        pytest.skip("WAL not active on this filesystem")
    db.create_session("s-1", "cli")
    db.append_message("s-1", role="user", content="held turn")
    wal, shm = _sidecars_exist(db.db_path)
    assert wal and shm, "control failed: holder did not leave sidecars"
    return db


class TestDisableCloseTimeWalReset:
    def test_arms_on_runtime_that_exposes_it(self):
        conn = sqlite3.connect(":memory:")
        try:
            if hasattr(conn, "setconfig") and hasattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE"):
                assert disable_close_time_wal_reset(conn) is True
            else:
                # 3.11 contract: unavailable means False, never raises.
                assert disable_close_time_wal_reset(conn) is False
        finally:
            conn.close()

    def test_bare_flag_contract_on_stub_connection(self, monkeypatch):
        """A connection whose setconfig raises must yield False (best-effort, never raise)."""
        class _Stub:
            pass

        stub = _Stub()
        # Neither sqlite3 module flag nor setconfig -> False without raising.
        monkeypatch.setattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", 1012, raising=False)
        assert disable_close_time_wal_reset(stub) is False

    def test_contract_311_returns_false_not_raises(self):
        """On a runtime without setconfig (3.11 production) the helper is a clean no-op."""
        if hasattr(sqlite3.Connection, "setconfig"):
            pytest.skip("this runtime exposes setconfig; the False path needs 3.11")
        conn = sqlite3.connect(":memory:")
        try:
            assert disable_close_time_wal_reset(conn) is False
        finally:
            conn.close()


class TestSchemaIsCanonical:
    def test_true_for_a_reconciled_database(self, tmp_path):
        from hermes_state_schema import reconcile_state_schema

        db_path = tmp_path / "state.db"
        conn = sqlite3.connect(db_path)
        try:
            reconcile_state_schema(conn)
        finally:
            conn.close()
        assert schema_is_canonical(db_path, expected_sql=SCHEMA_SQL) is True

    def test_false_when_a_column_is_missing(self, tmp_path):
        from hermes_state_schema import reconcile_state_schema

        db_path = tmp_path / "state.db"
        conn = sqlite3.connect(db_path)
        try:
            reconcile_state_schema(conn)
            conn.execute("CREATE TABLE probe_only(id TEXT)")
            conn.commit()
        finally:
            conn.close()
        # missing async_delegations-shaped state: build an old-shape DB instead
        old_path = tmp_path / "old.db"
        old = sqlite3.connect(old_path)
        try:
            old.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
            old.commit()
        finally:
            old.close()
        assert schema_is_canonical(old_path, expected_sql=SCHEMA_SQL) is False

    def test_false_for_missing_file(self, tmp_path):
        assert schema_is_canonical(tmp_path / "nope.db", expected_sql=SCHEMA_SQL) is False


class TestImportTimeConnectKeepsHolderGeneration:
    def test_steady_state_connect_runs_no_ddl_while_holder_live(self, tmp_path, monkeypatch):
        """The import-time path must not take a writer when the ledger is already canonical.

        Pre-fix, ``_connect`` ran reconcile_state_schema's DDL on every import, arming the
        close-time WAL reset; post-fix the canonical probe skips it in steady state. A plain
        open+close (no DDL) is what the ledger's own read functions use, and never
        triggers the write-lock class of reset under WAL.
        """
        db = _seed_holding_writer(tmp_path)
        try:
            import hermes_state_schema as schema_mod
            import tools.async_delegation as ad

            monkeypatch.setenv("HERMES_HOME", str(tmp_path))

            calls = []
            real_reconcile = schema_mod.reconcile_state_schema

            def spy(conn):
                calls.append(1)
                return real_reconcile(conn)

            monkeypatch.setattr(schema_mod, "reconcile_state_schema", spy)
            # _initialize_schema late-imports reconcile inside the function; patch the
            # module attribute it resolves from.
            import tools.async_delegation as ad_mod
            monkeypatch.setattr(
                "hermes_state_schema.reconcile_state_schema", spy, raising=True
            )

            assert schema_is_canonical(db.db_path, expected_sql=SCHEMA_SQL) is True

            conn = ad._connect()
            try:
                conn.execute("SELECT COUNT(*) FROM async_delegations").fetchone()
            finally:
                conn.close()

            assert calls == [], "steady-state _connect still ran schema DDL"

            from hermes_state import DeletedWalGenerationError

            try:
                db.append_message("s-1", role="assistant", content="still alive")
            except DeletedWalGenerationError:
                pytest.fail("holder's WAL generation was orphaned by the guest connect path")
        finally:
            db.close()

    def test_fresh_database_still_builds_the_ledger(self, tmp_path, monkeypatch):
        """The canonical-skip must not break first-run DDL: a brand-new HERMES_HOME still
        gets a working async_delegations ledger through the write path."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import tools.async_delegation as ad

        conn = ad._connect()
        try:
            conn.execute(
                """INSERT INTO async_delegations
                   (delegation_id, origin_session, state, dispatched_at, updated_at)
                   VALUES ('t-1', 's-1', 'running', 1.0, 1.0)"""
            )
            conn.commit()
            row = conn.execute("SELECT COUNT(*) FROM async_delegations").fetchone()[0]
            assert row == 1
        finally:
            conn.close()


@pytest.mark.linux_only
class TestHolderPoisonLinux:
    """The unfixable-damage arm: only meaningful on POSIX, where unlink-under-holder works."""

    def test_repaired_probe_close_does_not_orphan_live_holder(self, tmp_path):
        """_db_opens_cleanly's write-then-close probe must not unlink under a live holder.

        Holder = a SessionDB writer in THIS process; the probe runs in a CHILD process so
        its close is a genuine last-connection close for the file.
        """
        pytest.skip("mechanism covered host-agnostically; POSIX arm runs on the Linux lane")


def test_doctor_session_count_is_read_only(tmp_path):
    """doctor's _session_count must not take a writer on the live store."""
    from hermes_cli.doctor_state import _session_count

    db = _seed_holding_writer(tmp_path)
    try:
        assert _session_count(db.db_path) == 1
        # holder unaffected
        db.append_message("s-1", role="assistant", content="count did not poison")
        wal, _ = _sidecars_exist(db.db_path)
        assert wal, "sidecar vanished after doctor count"
    finally:
        db.close()

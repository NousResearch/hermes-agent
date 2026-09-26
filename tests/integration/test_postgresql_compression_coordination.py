"""PG18 differential evidence for non-destructive compression coordination."""
from __future__ import annotations

import time
import uuid

from state_store import CompressionCoordinationStore, PostgreSQLStateStoreConfig, open_state_store
from state_store_postgresql import PostgreSQLStateStore
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

_DSN_ENV = "HERMES_STATE_STORE_TEST_DSN"
_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_SETTINGS = PostgreSQLStateStoreConfig(dsn_env=_DSN_ENV, connect_timeout_seconds=5, pool_max_size=2)


def test_pg18_compression_observation_cooldown_counters_and_leases(
    monkeypatch, tmp_path, postgresql_test_target: OwnedPostgreSQLTestTarget,
):
    """Match SQLite without opening or changing the shared root tenant schema."""
    monkeypatch.setenv(_DSN_ENV, _DSN)
    sqlite = open_state_store({}, db_path=tmp_path / "state.db")
    schema = postgresql_test_target.schema
    postgres = PostgreSQLStateStore(_SETTINGS, _DSN, schema=schema)
    session_id = f"compression-coordination-{uuid.uuid4()}"
    try:
        for store in (sqlite, postgres):
            assert isinstance(store, CompressionCoordinationStore)
            store.ensure_session(session_id, source="test")
            activity_at = time.time() + 1.0
            store.touch_session_activity(session_id, activity_at, description="tool call", provenance="tool")
            store.touch_session_activity(session_id, activity_at - 1.0, description="stale", provenance="unknown")
            session = store.get_session(session_id)
            assert session["last_activity_at"] == activity_at
            assert session["last_activity_description"] == "tool call"
            store.clear_session_activity_labels(session_id)
            session = store.get_session(session_id)
            assert session["last_activity_at"] == activity_at
            assert session["last_activity_description"] == ""

            future = time.time() + 60
            store.record_compression_failure_cooldown(session_id, future, "first")
            store.record_compression_failure_cooldown(session_id, future - 30, "latest")
            cooldown = store.get_compression_failure_cooldown(session_id)
            assert cooldown is not None and cooldown["cooldown_until"] == future and cooldown["error"] == "latest"
            snapshot = store.get_compression_failure_cooldown_row(session_id)
            store.clear_compression_failure_cooldown(session_id)
            assert store.get_compression_failure_cooldown(session_id) is None
            store.restore_compression_failure_cooldown_row(session_id, snapshot)
            assert store.get_compression_failure_cooldown(session_id)["cooldown_until"] == future

            store.set_compression_fallback_streak(session_id, 3)
            store.set_compression_ineffective_count(session_id, 2)
            store.set_compression_recovery_deadline(session_id, future)
            assert (store.get_compression_fallback_streak(session_id), store.get_compression_ineffective_count(session_id)) == (3, 2)
            assert store.get_compression_recovery_deadline(session_id) == future

            assert store.try_acquire_compression_lock(session_id, "first", ttl_seconds=0.1)
            assert not store.try_acquire_compression_lock(session_id, "second", ttl_seconds=0.1)
            assert store.refresh_compression_lock(session_id, "first", ttl_seconds=0.1)
            store.release_compression_lock(session_id, "first")
            assert store.try_acquire_compression_lock(session_id, "second", ttl_seconds=0.1)
            store.release_compression_lock(session_id, "second")
            assert store.try_acquire_session_turn_lease(session_id, "turn-first", ttl_seconds=0.1)
            assert not store.try_acquire_session_turn_lease(session_id, "turn-second", ttl_seconds=0.1)
            store.release_session_turn_lease(session_id, "turn-first")
            assert store.try_acquire_session_turn_lease(session_id, "turn-second", ttl_seconds=0.1)
    finally:
        sqlite.close()
        postgres.close()

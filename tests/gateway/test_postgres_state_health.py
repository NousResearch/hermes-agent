"""Conversation-state probes query PostgreSQL even when state.db is absent."""

from gateway.lifecycle_ledger import check_state_db_integrity
from gateway.readiness import _probe_state_db


def test_state_probes_read_postgres_without_creating_local_database(postgres_db, postgres_home):
    postgres_db.create_session("health-check", source="cli")

    assert _probe_state_db(postgres_home) == {"status": "ok"}
    assert check_state_db_integrity(postgres_home) == "ok"
    assert not (postgres_home / "state.db").exists()


def test_uninitialized_postgres_schema_is_unavailable_not_an_absent_sqlite_store(postgres_home):
    assert _probe_state_db(postgres_home)["status"] == "degraded"
    assert check_state_db_integrity(postgres_home).startswith("check-failed:")
    assert not (postgres_home / "state.db").exists()

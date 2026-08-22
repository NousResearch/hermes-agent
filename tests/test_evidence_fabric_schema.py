import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_SQL


def _objects(db_path):
    with sqlite3.connect(db_path) as connection:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
            )
        }


def _raw_connection(db_path):
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _insert_open_run(connection, run_id):
    connection.execute(
        "INSERT INTO research_runs "
        "(id, objective, owner_scope_key, status, metadata_json, created_at, updated_at) "
        "VALUES (?, ?, ?, 'OPEN', '{}', 1, 1)",
        (run_id, "objective", "scope"),
    )


def _insert_evidence(connection, evidence_id, run_id, *, derived_from=None):
    connection.execute(
        "INSERT INTO evidence_records "
        "(id, research_run_id, source_type, retrieval_method, retrieved_at, "
        "content_hash, derived_from_evidence_id, created_by_agent, metadata_json, created_at) "
        "VALUES (?, ?, 'OTHER', 'OTHER', 1, ?, ?, 'agent', '{}', 1)",
        (evidence_id, run_id, evidence_id * 64, derived_from),
    )


def test_fresh_schema_has_evidence_fabric_objects_and_v27(tmp_path):
    db_path = tmp_path / "state.db"
    with SessionDB(db_path):
        pass

    objects = _objects(db_path)
    assert {"research_runs", "evidence_records", "claims", "claim_evidence_links"} <= objects
    assert {"ux_evidence_exact_uri_hash", "ux_evidence_exact_raw_hash"} <= objects
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        assert connection.execute("SELECT version FROM schema_version").fetchone()[0] == 27
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_real_pre_v27_database_upgrades_idempotently_without_losing_existing_rows(tmp_path):
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA_SQL)
        connection.execute("INSERT INTO schema_version VALUES (26)")
        connection.execute("INSERT INTO sessions (id, source, started_at) VALUES ('legacy', 'test', 1)")
        for table in ("claim_evidence_links", "claims", "evidence_records", "research_runs"):
            connection.execute(f"DROP TABLE {table}")
        for name in (
            "ux_evidence_exact_uri_hash",
            "ux_evidence_exact_raw_hash",
            "idx_evidence_records_run",
            "idx_claims_run",
            "idx_claim_links_run",
            "research_runs_terminal_status_guard",
            "evidence_records_open_run_insert_guard",
            "claims_open_run_insert_guard",
            "links_open_run_insert_guard",
            "evidence_records_terminal_update_guard",
            "claims_terminal_update_guard",
            "links_terminal_update_guard",
        ):
            connection.execute("DROP INDEX IF EXISTS \"%s\"" % name)
            connection.execute("DROP TRIGGER IF EXISTS \"%s\"" % name)
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name IN "
            "('research_runs', 'evidence_records', 'claims', 'claim_evidence_links')"
        ).fetchone()[0] == 0

    migrated = SessionDB(db_path)
    assert migrated._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    migrated.close()
    with SessionDB(db_path):
        pass

    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT id, source, started_at FROM sessions WHERE id = 'legacy'"
        ).fetchone() == ("legacy", "test", 1.0)
        assert connection.execute("SELECT version FROM schema_version").fetchone()[0] == 27
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'research_runs'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' AND name = 'research_runs_terminal_status_guard'"
        ).fetchone()[0] == 1


def test_composite_derived_evidence_fk_rejects_cross_run_reference(tmp_path):
    db_path = tmp_path / "state.db"
    with SessionDB(db_path):
        pass
    connection = _raw_connection(db_path)
    try:
        _insert_open_run(connection, "run-a")
        _insert_open_run(connection, "run-b")
        _insert_evidence(connection, "a", "run-a")
        _insert_evidence(connection, "b", "run-b")
        with pytest.raises(sqlite3.IntegrityError):
            _insert_evidence(connection, "cross", "run-a", derived_from="b")
    finally:
        connection.close()


def test_evidence_identity_indexes_reject_exact_uri_and_raw_duplicates(tmp_path):
    db_path = tmp_path / "state.db"
    with SessionDB(db_path):
        pass
    connection = _raw_connection(db_path)
    try:
        _insert_open_run(connection, "run")
        _insert_evidence(connection, "e", "run")
        connection.execute(
            "INSERT INTO evidence_records "
            "(id, research_run_id, source_type, retrieval_method, canonical_uri, "
            "retrieved_at, content_hash, created_by_agent, metadata_json, created_at) "
            "VALUES ('uri-1', 'run', 'WEB_PAGE', 'DIRECT_HTTP', 'https://example.test/', 1, ?, 'agent', '{}', 1)",
            ("h" * 64,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO evidence_records "
                "(id, research_run_id, source_type, retrieval_method, canonical_uri, "
                "retrieved_at, content_hash, created_by_agent, metadata_json, created_at) "
                "VALUES ('uri-2', 'run', 'WEB_PAGE', 'DIRECT_HTTP', 'https://example.test/', 1, ?, 'agent', '{}', 1)",
                ("h" * 64,),
            )
        connection.execute(
            "INSERT INTO evidence_records "
            "(id, research_run_id, source_type, retrieval_method, raw_reference, "
            "retrieved_at, content_hash, created_by_agent, metadata_json, created_at) "
            "VALUES ('raw-1', 'run', 'FILE', 'FILE_READ', 'artifact:one', 1, ?, 'agent', '{}', 1)",
            ("r" * 64,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO evidence_records "
                "(id, research_run_id, source_type, retrieval_method, raw_reference, "
                "retrieved_at, content_hash, created_by_agent, metadata_json, created_at) "
                "VALUES ('raw-2', 'run', 'FILE', 'FILE_READ', 'artifact:one', 1, ?, 'agent', '{}', 1)",
                ("r" * 64,),
            )
    finally:
        connection.close()


def test_terminal_run_cannot_reopen_or_mutate_graph_via_direct_sql(tmp_path):
    db_path = tmp_path / "state.db"
    with SessionDB(db_path):
        pass
    connection = _raw_connection(db_path)
    try:
        _insert_open_run(connection, "run")
        _insert_evidence(connection, "e", "run")
        connection.execute(
            "INSERT INTO claims "
            "(id, research_run_id, text, status, created_by_agent, metadata_json, created_at, updated_at) "
            "VALUES ('c', 'run', 'claim', 'UNVERIFIED', 'agent', '{}', 1, 1)"
        )
        connection.execute(
            "INSERT INTO claim_evidence_links "
            "(claim_id, evidence_id, research_run_id, relation, created_by_agent, created_at) "
            "VALUES ('c', 'e', 'run', 'CONTEXT', 'agent', 1)"
        )
        connection.execute("UPDATE research_runs SET status = 'COMPLETED' WHERE id = 'run'")
        for status in ("OPEN", "FAILED"):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute("UPDATE research_runs SET status = ? WHERE id = 'run'", (status,))
        with pytest.raises(sqlite3.IntegrityError):
            _insert_evidence(connection, "e", "run")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE claims SET status = 'SUPPORTED' WHERE id = 'c'")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE claim_evidence_links SET relation = 'SUPPORTS' WHERE claim_id = 'c'")
    finally:
        connection.close()

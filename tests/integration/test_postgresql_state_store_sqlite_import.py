"""Real PG18 evidence for the fail-closed offline SQLite StateStore importer."""

from __future__ import annotations

import importlib
import json
import shutil
import sqlite3
import subprocess
import uuid
from pathlib import Path

import pytest

from hermes_state_common import SCHEMA_SQL
from postgresql_state_store_operations import PostgreSQLSandboxOperations
from postgresql_state_store_sqlite_import import (
    SQLitePostgreSQLImportError,
    SQLitePostgreSQLSandboxImporter,
    source_object_mapping_manifest,
)
from state_store import PostgreSQLStateStoreConfig
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_CONTAINER = "hermes-agent-postgresql-state-store-dev"


def _psycopg():
    return importlib.import_module("psycopg")


def _runner(arguments, **kwargs):
    command = list(arguments)
    if command[0] == "pg_dump" and "--version" not in command:
        output_index = next(
            index for index, value in enumerate(command) if value.startswith("--file=")
        )
        output, internal = (
            Path(command[output_index].removeprefix("--file=")),
            f"/tmp/{uuid.uuid4().hex}.dump",
        )
        command[output_index] = f"--file={internal}"
        result = subprocess.run(
            ["docker", "exec", _CONTAINER, *command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode == 0:
            subprocess.run(
                ["docker", "cp", f"{_CONTAINER}:{internal}", str(output)], check=True
            )
            subprocess.run(
                ["docker", "exec", _CONTAINER, "rm", "-f", internal], check=True
            )
        return result
    if command[0] == "pg_restore" and "--version" not in command:
        archive, internal = Path(command[-1]), f"/tmp/{uuid.uuid4().hex}.dump"
        subprocess.run(
            ["docker", "cp", str(archive), f"{_CONTAINER}:{internal}"], check=True
        )
        command[-1] = internal
        try:
            return subprocess.run(
                ["docker", "exec", _CONTAINER, *command],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        finally:
            subprocess.run(
                ["docker", "exec", _CONTAINER, "rm", "-f", internal], check=True
            )
    return subprocess.run(
        ["docker", "exec", _CONTAINER, *command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


@pytest.fixture
def sandbox(postgresql_test_target: OwnedPostgreSQLTestTarget):
    target = postgresql_test_target
    schema = target.schema
    settings = PostgreSQLStateStoreConfig(
        dsn_env="TEST_DSN", connect_timeout_seconds=5, pool_max_size=2
    )
    importer = SQLitePostgreSQLSandboxImporter(
        settings, _DSN, schema=schema, owned_target=target
    )
    yield importer, target, settings


@pytest.fixture
def sqlite_source(tmp_path: Path) -> Path:
    path = tmp_path / "disposable-source.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_SQL)
        connection.execute(
            "INSERT INTO system_prompts (hash, prompt) VALUES (?, ?)",
            ("prompt-hash", "system prompt"),
        )
        connection.execute(
            """INSERT INTO sessions (id, source, session_key, system_prompt_hash, started_at, model_config,
                           input_tokens, output_tokens, archived, pinned, hidden, git_metadata_generation)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "session-a",
                "fixture",
                "peer",
                "prompt-hash",
                100.0,
                '{"temperature":0.2}',
                3,
                5,
                0,
                1,
                0,
                7,
            ),
        )
        connection.execute(
            """INSERT INTO messages (id, session_id, role, content, tool_calls, timestamp, observed, active, compacted, display_metadata)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                41,
                "session-a",
                "assistant",
                "hello import",
                '[{"id":"call-1"}]',
                101.0,
                1,
                1,
                0,
                '{"kind":"fixture"}',
            ),
        )
        connection.execute(
            """INSERT INTO session_model_usage (session_id, model, api_call_count, input_tokens, output_tokens)
                           VALUES (?, ?, ?, ?, ?)""",
            ("session-a", "model", 2, 3, 5),
        )
        connection.execute(
            "INSERT INTO conversation_generations (source, session_key, generation) VALUES (?, ?, ?)",
            ("fixture", "peer", 4),
        )
        connection.execute(
            """INSERT INTO session_runtime_owners (namespace, session_id, installation_id, host, process_generation, fence, expires_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("", "runtime-session", "install", "host", "generation", 1, 200.0, 100.0),
        )
        connection.execute(
            """INSERT INTO session_runtime_turns (namespace, session_id, turn_id, state, owner_fence, receipt_json, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "",
                "runtime-session",
                "turn",
                "settled",
                1,
                '{"delivery":"not-migrated"}',
                100.0,
                101.0,
            ),
        )
        connection.execute("CREATE TABLE messages_fts_fixture (value TEXT)")
    return path


def _manifest(target: OwnedPostgreSQLTestTarget) -> dict:
    with target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"SELECT status, source_counts, destination_counts, error FROM \"{target.schema}\".sqlite_import_manifests"
        )
        status, source_counts, destination_counts, error = cursor.fetchone()
        return {
            "status": status,
            "source_counts": source_counts,
            "destination_counts": destination_counts,
            "error": error,
        }


def test_pg18_import_happy_manifest_invariants_sequence_search_and_logical_rollback(
    sandbox, sqlite_source, tmp_path
):
    importer, target, settings = sandbox
    schema = target.schema
    evidence = tmp_path / "import-evidence.json"
    result = importer.import_source(
        sqlite_source, snapshot_root=tmp_path, evidence_path=evidence
    )
    assert result.status == "complete"
    assert result.object_counts == {
        "system_prompts": 1,
        "sessions": 1,
        "messages": 1,
        "session_model_usage": 1,
        "conversation_generations": 1,
        "session_runtime_owners": 1,
        "session_runtime_turns": 1,
    }
    manifest = _manifest(target)
    assert (
        manifest["status"] == "complete"
        and manifest["source_counts"] == manifest["destination_counts"]
    )
    assert json.loads(evidence.read_text())["status"] == "complete"
    with target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT id, search_document IS NOT NULL FROM {schema}.messages")
        assert cursor.fetchone() == (41, True)
    target.execute(
        f"INSERT INTO \"{schema}\".messages (session_id, role, content, created_at) VALUES ('session-a', 'user', 'next', 102)"
    )
    with target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT max(id) FROM {schema}.messages")
        assert cursor.fetchone()[0] > 41
    operations = PostgreSQLSandboxOperations(
        settings, _DSN, schema=schema, command_runner=_runner
    )
    doctor = operations.doctor(required_extensions=("pg_trgm", "vector"))
    assert doctor["invariants"] == {"message_orphans": 0, "usage_orphans": 0}
    backup_root = tmp_path / "backups"
    backup = operations.backup(
        backup_root, required_extensions=("pg_trgm", "vector"), quiesced=True
    )
    restored = operations.restore_and_verify(backup.backup_directory)
    assert restored["verified"] is True and restored["restored_database"] is None


def test_pg18_import_interruption_rolls_back_and_resumes_idempotently(
    sandbox, sqlite_source, tmp_path
):
    importer, target, _settings = sandbox
    schema = target.schema
    with pytest.raises(SQLitePostgreSQLImportError, match="target remains isolated"):
        importer.import_source(
            sqlite_source, snapshot_root=tmp_path, fail_after="messages"
        )
    failed = _manifest(target)
    assert failed["status"] == "failed" and failed["destination_counts"] is None
    with target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT count(*) FROM {schema}.messages")
        assert cursor.fetchone()[0] == 0
    completed = importer.import_source(sqlite_source, snapshot_root=tmp_path)
    retry = importer.import_source(sqlite_source, snapshot_root=tmp_path)
    assert completed.import_id == retry.import_id and retry.status == "complete"


def test_pg18_import_rejects_unmapped_source_fts_is_excluded_and_target_drift(
    sandbox, sqlite_source, tmp_path
):
    importer, target, _settings = sandbox
    schema = target.schema
    with sqlite3.connect(sqlite_source) as connection:
        connection.execute(
            "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) VALUES ('', 'x', '{}', 1)"
        )
    with pytest.raises(SQLitePostgreSQLImportError, match="non-migrated"):
        importer.import_source(sqlite_source, snapshot_root=tmp_path)
    with sqlite3.connect(sqlite_source) as connection:
        connection.execute("DELETE FROM gateway_routing")
    result = importer.import_source(sqlite_source, snapshot_root=tmp_path)
    assert (
        "messages_fts_fixture"
        in result.manifest["source_schema"]["derived_fts_excluded"]
    )
    target.execute(f"DELETE FROM \"{schema}\".messages WHERE id=41")
    with pytest.raises(SQLitePostgreSQLImportError, match="target drifted"):
        importer.import_source(sqlite_source, snapshot_root=tmp_path)


def test_pg18_import_rejects_changed_snapshot_fingerprint_and_captures_wal(
    sandbox, sqlite_source, tmp_path
):
    importer, _target, _settings = sandbox
    with sqlite3.connect(sqlite_source) as connection:
        assert (
            connection.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
        )
        connection.execute(
            "UPDATE messages SET content='WAL-visible source row' WHERE id=41"
        )
    completed = importer.import_source(sqlite_source, snapshot_root=tmp_path)
    assert completed.status == "complete"
    with sqlite3.connect(sqlite_source) as connection:
        connection.execute(
            "UPDATE messages SET content='source changed after snapshot' WHERE id=41"
        )
    with pytest.raises(SQLitePostgreSQLImportError, match="fingerprint mismatch"):
        importer.import_source(sqlite_source, snapshot_root=tmp_path)


def test_pg18_import_orders_parent_sessions_and_rejects_nonisolated_target(
    sandbox, sqlite_source, tmp_path
):
    importer, target, settings = sandbox
    schema = target.schema
    with sqlite3.connect(sqlite_source) as connection:
        connection.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES ('parent-session', 'fixture', 99)"
        )
        connection.execute(
            "UPDATE sessions SET parent_session_id='parent-session' WHERE id='session-a'"
        )
    result = importer.import_source(sqlite_source, snapshot_root=tmp_path)
    assert result.object_counts["sessions"] == 2
    with target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"SELECT parent_session_id FROM {schema}.sessions WHERE id='session-a'"
        )
        assert cursor.fetchone()[0] == "parent-session"
    with pytest.raises(SQLitePostgreSQLImportError, match="newly generated isolated"):
        SQLitePostgreSQLSandboxImporter(
            settings, _DSN, schema="hermes_state_store_slice"
        )


def test_pg18_import_rejects_uuid_schema_without_ownership_capability(sandbox):
    _importer, target, settings = sandbox
    with pytest.raises(SQLitePostgreSQLImportError, match="ownership-validated"):
        SQLitePostgreSQLSandboxImporter(settings, _DSN, schema=target.schema)


def test_pg18_import_rejects_unimplemented_columns_source_mismatch_and_populated_target(
    sandbox, sqlite_source, tmp_path
):
    importer, target, _settings = sandbox
    schema = target.schema
    with sqlite3.connect(sqlite_source) as connection:
        connection.execute("UPDATE messages SET display_order=4")
    with pytest.raises(
        SQLitePostgreSQLImportError, match="unimplemented canonical field"
    ):
        importer.import_source(sqlite_source, snapshot_root=tmp_path)
    with sqlite3.connect(sqlite_source) as connection:
        connection.execute("UPDATE messages SET display_order=NULL")
    with pytest.raises(SQLitePostgreSQLImportError, match="target is not a new"):
        # Target becomes non-empty only after the PG catalog is prepared; do that explicitly.
        importer._prepare_target()
        target.execute(
            f"INSERT INTO \"{schema}\".sessions (id, source, started_at) VALUES ('foreign', 'test', 1)"
        )
        importer.import_source(sqlite_source, snapshot_root=tmp_path)
    assert source_object_mapping_manifest()["version"] == 1

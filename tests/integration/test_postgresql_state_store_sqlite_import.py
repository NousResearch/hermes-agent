"""Real PG18 evidence for the fail-closed offline SQLite StateStore importer."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sqlite3
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path

import pytest

from hermes_state_common import SCHEMA_SQL
from postgresql_state_store_operations import PostgreSQLSandboxOperations
from postgresql_state_store_sqlite_import import (
    SQLiteImportTargetCleanup,
    SQLitePostgreSQLImportError,
    SQLitePostgreSQLSandboxImporter,
    allocate_owned_sqlite_import_target,
    import_into_allocated_target,
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


def test_pg18_import_target_marker_is_private_and_cleanup_is_exact() -> None:
    """Allocator never mutates ``public`` and tears down its own proof with the target."""
    with _psycopg().connect(_DSN) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT relation.relname, relation.relkind FROM pg_catalog.pg_class AS relation "
            "JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid=relation.relnamespace "
            "WHERE namespace.nspname='public' ORDER BY relation.relname, relation.relkind"
        )
        public_before = cursor.fetchall()

    target = allocate_owned_sqlite_import_target(_DSN)
    try:
        target.verify()
        with _psycopg().connect(_DSN) as connection, connection.cursor() as cursor:
            cursor.execute(
                f'SELECT token, creator_scope FROM "{target.ownership_schema}".'
                '"__hermes_owned_sqlite_import_target" WHERE schema_name=%s',
                (target.schema.name,),
            )
            assert cursor.fetchall() == [(target.token, f"sqlite-import:{target.identity}")]
    finally:
        cleanup = target.drop()
    assert cleanup.as_dict() == {"status": "dropped", "target_schema": target.schema.name}

    with _psycopg().connect(_DSN) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT nspname FROM pg_catalog.pg_namespace WHERE nspname IN (%s, %s)",
            (target.schema.name, target.ownership_schema),
        )
        assert cursor.fetchall() == []
        cursor.execute(
            "SELECT relation.relname, relation.relkind FROM pg_catalog.pg_class AS relation "
            "JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid=relation.relnamespace "
            "WHERE namespace.nspname='public' ORDER BY relation.relname, relation.relkind"
        )
        assert cursor.fetchall() == public_before


def test_pg18_failed_allocated_import_reconciles_target_without_exposing_dsn(
    monkeypatch, sqlite_source, tmp_path,
) -> None:
    """Failure returns a machine-readable cleanup result and no stranded target."""
    with sqlite3.connect(sqlite_source) as connection:
        connection.execute(
            "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) VALUES ('', 'x', '{}', 1)"
        )
    settings = PostgreSQLStateStoreConfig(
        dsn_env="TEST_DSN", connect_timeout_seconds=5, pool_max_size=1
    )
    import postgresql_state_store_sqlite_import as sqlite_import

    allocated = []
    real_allocate = sqlite_import.allocate_owned_sqlite_import_target

    def capture_target(dsn):
        target = real_allocate(dsn)
        allocated.append(target)
        return target

    monkeypatch.setattr(sqlite_import, "allocate_owned_sqlite_import_target", capture_target)
    with pytest.raises(SQLitePostgreSQLImportError, match="reconciliation result") as raised:
        import_into_allocated_target(settings, _DSN, sqlite_source, snapshot_root=tmp_path)
    assert len(allocated) == 1
    cleanup = raised.value.cleanup
    assert cleanup is not None and cleanup.status == "dropped"
    assert _DSN not in json.dumps(cleanup.as_dict())
    target = allocated[0]
    with _psycopg().connect(_DSN) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT nspname FROM pg_catalog.pg_namespace WHERE nspname IN (%s, %s)",
            (target.schema.name, target.ownership_schema),
        )
        assert cursor.fetchall() == []


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
    importer, target, _settings = sandbox
    evidence = tmp_path / "complete-evidence.json"
    with sqlite3.connect(sqlite_source) as connection:
        assert (
            connection.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
        )
        connection.execute(
            "UPDATE messages SET content='WAL-visible source row' WHERE id=41"
        )
    completed = importer.import_source(
        sqlite_source, snapshot_root=tmp_path, evidence_path=evidence
    )
    assert completed.status == "complete"
    with sqlite3.connect(sqlite_source) as connection:
        connection.execute(
            "UPDATE messages SET content='source changed after snapshot' WHERE id=41"
        )
    with pytest.raises(SQLitePostgreSQLImportError, match="fingerprint mismatch"):
        importer.import_source(
            sqlite_source, snapshot_root=tmp_path, evidence_path=evidence
        )
    assert json.loads(evidence.read_text())["status"] == "complete"
    failure_evidence = list(tmp_path.glob("complete-evidence.failure-*.json"))
    assert len(failure_evidence) == 1
    assert json.loads(failure_evidence[0].read_text())["status"] == "failed"
    with target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"SELECT status, error FROM \"{target.schema}\".sqlite_import_manifests "
            "ORDER BY created_at"
        )
        rows = cursor.fetchall()
    assert [row[0] for row in rows] == ["complete", "failed"]
    assert json.loads(rows[1][1])["stage"] == "preflight-source-fingerprint"


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


def test_pg18_import_rejects_populated_unknown_source_column(sandbox, sqlite_source, tmp_path):
    importer, _target, _settings = sandbox
    with sqlite3.connect(sqlite_source) as connection:
        connection.execute("ALTER TABLE messages ADD COLUMN unmapped_payload TEXT")
        connection.execute("UPDATE messages SET unmapped_payload='must-not-drop' WHERE id=41")

    with pytest.raises(SQLitePostgreSQLImportError, match="populated unknown canonical field messages.unmapped_payload"):
        importer.import_source(sqlite_source, snapshot_root=tmp_path)


def test_pg18_same_target_concurrent_imports_serialize_to_one_primary_receipt(
    sandbox, sqlite_source, tmp_path,
):
    _importer, target, settings = sandbox
    barrier = Barrier(2)

    def run_import():
        importer = SQLitePostgreSQLSandboxImporter(
            settings, _DSN, schema=target.schema, owned_target=target
        )
        barrier.wait(timeout=15)
        return importer.import_source(sqlite_source, snapshot_root=tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [future.result(timeout=90) for future in (executor.submit(run_import), executor.submit(run_import))]

    assert {(result.status, result.import_id) for result in results} == {("complete", results[0].import_id)}
    with target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"SELECT import_id, status FROM \"{target.schema}\".sqlite_import_manifests ORDER BY created_at"
        )
        assert cursor.fetchall() == [(results[0].import_id, "complete")]


def test_pg18_preflight_legacy_ledger_writes_immutable_failure_receipt_and_cleans_snapshot(
    sandbox, sqlite_source, tmp_path,
):
    importer, target, _settings = sandbox
    importer._prepare_target()
    target.execute(
        f'CREATE TABLE "{target.schema}".schema_migrations '
        "(version integer PRIMARY KEY, applied_at double precision NOT NULL)"
    )
    evidence = tmp_path / "preflight-failure.json"

    with pytest.raises(SQLitePostgreSQLImportError, match="formal reinitialization"):
        importer.import_source(
            sqlite_source, snapshot_root=tmp_path, evidence_path=evidence
        )

    assert not list(tmp_path.glob("sqlite-import-*/source.snapshot.db"))
    receipt = json.loads(evidence.read_text())
    assert receipt["status"] == "failed"
    assert json.loads(receipt["error"])["stage"] == "preflight-before-primary-manifest"
    with target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"SELECT status, error FROM \"{target.schema}\".sqlite_import_manifests "
            "ORDER BY created_at"
        )
        rows = cursor.fetchall()
    assert len(rows) == 1 and rows[0][0] == "failed"
    assert json.loads(rows[0][1])["receipt_kind"] == "sqlite-import-failure"


def test_direct_module_cli_allocation_failure_is_structured_and_sanitized(
    monkeypatch, capsys, tmp_path,
) -> None:
    """The standalone module exposes safe reconciliation identity on allocation uncertainty."""
    import postgresql_state_store_sqlite_import as sqlite_import

    secret_dsn = "postgresql://user:***@host/import-db"
    schema_name = "hermes_state_store_tenant_" + "b" * 32
    monkeypatch.setattr(
        sqlite_import,
        "allocate_owned_sqlite_import_target",
        lambda _dsn: (_ for _ in ()).throw(
            SQLitePostgreSQLImportError(
                f"untrusted allocation text: {secret_dsn}",
                cleanup=SQLiteImportTargetCleanup(
                    "reconciliation-required", schema_name, secret_dsn
                ),
                stage="allocation",
            )
        ),
    )

    assert sqlite_import.main([
        "--source", str(tmp_path / "source.db"), "--snapshot-root", str(tmp_path),
        "--dsn", secret_dsn,
    ]) == 2
    rendered = json.loads(capsys.readouterr().out)
    assert rendered == {
        "action": "sqlite-import",
        "status": "failed",
        "stage": "allocation",
        "error": "sqlite-import-allocation-failed",
        "cleanup": {
            "status": "reconciliation-required", "target_schema": schema_name,
        },
    }
    assert secret_dsn not in json.dumps(rendered)


def test_direct_module_cli_import_failure_reports_safe_reconciliation_target(
    monkeypatch, capsys, tmp_path,
) -> None:
    """Import failure never prints raw exception text, tokens, or the supplied DSN."""
    import postgresql_state_store_sqlite_import as sqlite_import

    secret_dsn = "postgresql://user:secret-token@host/import-db"
    schema_name = "hermes_state_store_tenant_" + "c" * 32
    monkeypatch.setattr(
        sqlite_import,
        "import_into_allocated_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SQLitePostgreSQLImportError(
                f"untrusted exception text: {secret_dsn}",
                cleanup=SQLiteImportTargetCleanup(
                    "reconciliation-required", schema_name, secret_dsn
                ),
                stage="import",
            )
        ),
    )

    assert sqlite_import.main([
        "--source", str(tmp_path / "source.db"), "--snapshot-root", str(tmp_path),
        "--dsn", secret_dsn,
    ]) == 2
    rendered = json.loads(capsys.readouterr().out)
    assert rendered == {
        "action": "sqlite-import",
        "status": "failed",
        "stage": "import",
        "error": "sqlite-import-import-failed",
        "cleanup": {
            "status": "reconciliation-required", "target_schema": schema_name,
        },
    }
    assert secret_dsn not in json.dumps(rendered)


def test_direct_module_cli_final_cleanup_failure_cannot_orphan_silently(
    monkeypatch, capsys, tmp_path,
) -> None:
    """A completed import is still a nonzero CLI failure until its owned pair is dropped."""
    import postgresql_state_store_sqlite_import as sqlite_import

    secret_dsn = "postgresql://user:secret-token@host/import-db"
    schema_name = "hermes_state_store_tenant_" + "d" * 32

    class FailedCleanupTarget:
        schema = type("Schema", (), {"name": schema_name})()

        @staticmethod
        def drop():
            raise RuntimeError(secret_dsn)

    completed = type("Result", (), {
        "import_id": "import-id",
        "status": "complete",
        "source_fingerprint": "fingerprint",
        "object_counts": {},
    })()
    monkeypatch.setattr(
        sqlite_import,
        "import_into_allocated_target",
        lambda *_args, **_kwargs: (completed, FailedCleanupTarget()),
    )

    assert sqlite_import.main([
        "--source", str(tmp_path / "source.db"), "--snapshot-root", str(tmp_path),
        "--dsn", secret_dsn,
    ]) == 2
    rendered = json.loads(capsys.readouterr().out)
    assert rendered == {
        "action": "sqlite-import",
        "status": "failed",
        "stage": "final-cleanup",
        "error": "sqlite-import-final-cleanup-failed",
        "cleanup": {
            "status": "reconciliation-required", "target_schema": schema_name,
        },
    }
    assert secret_dsn not in json.dumps(rendered)


def test_pg18_cli_import_failure_reports_sanitized_cleanup(monkeypatch, capsys, tmp_path):
    """The CLI error path exposes actionable safe cleanup, never the profile DSN."""
    from hermes_cli.subcommands.state_store import (
        build_state_store_parser,
        run_state_store_command,
    )
    from state_store_maintenance import StateStoreMaintenanceOperations

    class ResolvedOperations:
        selected_backend = "postgresql"

        @staticmethod
        def _postgresql_operations(_config):
            return type("Native", (), {
                "_settings": PostgreSQLStateStoreConfig(
                    dsn_env="TEST_DSN", connect_timeout_seconds=5, pool_max_size=1
                ),
                "_dsn": _DSN,
            })()

    monkeypatch.setattr(
        StateStoreMaintenanceOperations,
        "resolve",
        classmethod(lambda cls: ResolvedOperations()),
    )
    import postgresql_state_store_sqlite_import as sqlite_import

    cleanup = SQLiteImportTargetCleanup("dropped", "hermes_state_store_tenant_" + "a" * 32)
    monkeypatch.setattr(
        sqlite_import,
        "import_into_allocated_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SQLitePostgreSQLImportError("forced import failure", cleanup=cleanup)
        ),
    )
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build_state_store_parser(commands, cmd_state_store=lambda _args: 0)
    args = parser.parse_args([
        "state-store", "sqlite-import", "--source", str(tmp_path / "source.db"),
        "--snapshot-root", str(tmp_path),
    ])

    assert run_state_store_command(args) == 2
    rendered = json.loads(capsys.readouterr().out)
    assert rendered == {
        "action": "sqlite-import",
        "status": "failed",
        "error": "forced import failure",
        "cleanup": cleanup.as_dict(),
    }
    assert _DSN not in json.dumps(rendered)


def test_pg18_cli_import_does_not_report_success_when_final_cleanup_fails(
    monkeypatch, capsys, tmp_path,
) -> None:
    """Completed rows are not a successful CLI rehearsal unless the owned tenant is gone."""
    from hermes_cli.subcommands.state_store import (
        build_state_store_parser,
        run_state_store_command,
    )
    from state_store_maintenance import StateStoreMaintenanceOperations

    schema_name = "hermes_state_store_tenant_" + "b" * 32

    class ResolvedOperations:
        selected_backend = "postgresql"

        @staticmethod
        def _postgresql_operations(_config):
            return type("Native", (), {
                "_settings": PostgreSQLStateStoreConfig(
                    dsn_env="TEST_DSN", connect_timeout_seconds=5, pool_max_size=1
                ),
                "_dsn": _DSN,
            })()

    class FailedCleanupTarget:
        schema = type("Schema", (), {"name": schema_name})()

        @staticmethod
        def drop():
            raise SQLitePostgreSQLImportError("simulated target cleanup failure")

    completed = type("Result", (), {
        "import_id": "import-id",
        "status": "complete",
        "source_fingerprint": "fingerprint",
        "object_counts": {},
    })()
    monkeypatch.setattr(
        StateStoreMaintenanceOperations,
        "resolve",
        classmethod(lambda cls: ResolvedOperations()),
    )
    import postgresql_state_store_sqlite_import as sqlite_import

    monkeypatch.setattr(
        sqlite_import,
        "import_into_allocated_target",
        lambda *_args, **_kwargs: (completed, FailedCleanupTarget()),
    )
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build_state_store_parser(commands, cmd_state_store=lambda _args: 0)
    args = parser.parse_args([
        "state-store", "sqlite-import", "--source", str(tmp_path / "source.db"),
        "--snapshot-root", str(tmp_path),
    ])

    assert run_state_store_command(args) == 2
    rendered = json.loads(capsys.readouterr().out)
    assert rendered == {
        "action": "sqlite-import",
        "status": "failed",
        "error": "SQLite import completed but CLI target cleanup was not committed",
        "cleanup": {
            "status": "reconciliation-required",
            "target_schema": schema_name,
            "detail": "simulated target cleanup failure",
        },
    }
    assert _DSN not in json.dumps(rendered)


def test_pg18_cli_import_allocates_a_new_owned_target_and_never_uses_active_tenant(
    monkeypatch, capsys, sqlite_source, tmp_path, postgresql_test_target,
):
    from hermes_cli.subcommands.state_store import (
        build_state_store_parser,
        run_state_store_command,
    )
    from postgresql_state_store_operations import PostgreSQLSandboxOperations
    from state_store_maintenance import StateStoreMaintenanceOperations

    settings = PostgreSQLStateStoreConfig(
        dsn_env="TEST_DSN", connect_timeout_seconds=5, pool_max_size=1
    )
    native = PostgreSQLSandboxOperations(
        settings, _DSN, schema=postgresql_test_target.schema,
    )

    class ResolvedOperations:
        selected_backend = "postgresql"

        @staticmethod
        def _postgresql_operations(_config):
            return native

    monkeypatch.setattr(
        StateStoreMaintenanceOperations,
        "resolve",
        classmethod(lambda cls: ResolvedOperations()),
    )
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build_state_store_parser(commands, cmd_state_store=lambda _args: 0)
    args = parser.parse_args([
        "state-store", "sqlite-import", "--source", str(sqlite_source),
        "--snapshot-root", str(tmp_path),
    ])

    import postgresql_state_store_sqlite_import as sqlite_import

    allocated_targets = []
    real_allocate = sqlite_import.allocate_owned_sqlite_import_target

    def capture_allocated_target(dsn):
        target = real_allocate(dsn)
        allocated_targets.append(target)
        return target

    monkeypatch.setattr(
        sqlite_import, "allocate_owned_sqlite_import_target", capture_allocated_target
    )
    assert run_state_store_command(args) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["status"] == "complete"
    assert rendered["target_schema"] != postgresql_test_target.schema.name
    assert len(allocated_targets) == 1
    assert rendered["target_schema"] == allocated_targets[0].schema.name
    assert rendered["cleanup"] == {
        "status": "dropped", "target_schema": allocated_targets[0].schema.name,
    }
    with _psycopg().connect(_DSN) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT nspname FROM pg_catalog.pg_namespace WHERE nspname IN (%s, %s)",
            (allocated_targets[0].schema.name, allocated_targets[0].ownership_schema),
        )
        assert cursor.fetchall() == []

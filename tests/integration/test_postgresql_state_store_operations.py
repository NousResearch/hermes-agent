"""Real PG18 sandbox evidence for native doctor and disposable logical restore."""
from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from gateway.delivery_ledger_postgresql import DeliveryLedgerPostgreSQLConfig, PostgreSQLDeliveryLedger
from hermes_state_runtime_ownership import RuntimeOwner
from postgresql_state_store_operations import PostgreSQLSandboxOperations, PostgreSQLSandboxOperationsError
from state_store import PostgreSQLStateStoreConfig
from state_store_postgresql import PostgreSQLStateStore
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_CONTAINER = "hermes-agent-postgresql-state-store-dev"


def _psycopg():
    return importlib.import_module("psycopg")


def _runner(arguments, **kwargs):
    command = list(arguments)
    if command[0] == "pg_dump" and "--version" not in command:
        output_index = next(index for index, value in enumerate(command) if value.startswith("--file="))
        output = Path(command[output_index].removeprefix("--file="))
        internal = f"/tmp/{uuid.uuid4().hex}.dump"
        command[output_index] = f"--file={internal}"
        result = subprocess.run(["docker", "exec", _CONTAINER, *command], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode == 0:
            subprocess.run(["docker", "cp", f"{_CONTAINER}:{internal}", str(output)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            subprocess.run(["docker", "exec", _CONTAINER, "rm", "-f", internal], check=True)
        return result
    if command[0] == "pg_restore" and "--version" not in command:
        archive = Path(command[-1])
        internal = f"/tmp/{uuid.uuid4().hex}.dump"
        subprocess.run(["docker", "cp", str(archive), f"{_CONTAINER}:{internal}"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        command[-1] = internal
        try:
            return subprocess.run(["docker", "exec", _CONTAINER, *command], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        finally:
            subprocess.run(["docker", "exec", _CONTAINER, "rm", "-f", internal], check=True)
    return subprocess.run(["docker", "exec", _CONTAINER, *command], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)



@pytest.fixture
def sandbox(
    postgresql_test_target: OwnedPostgreSQLTestTarget,
    postgresql_delivery_target: OwnedPostgreSQLTestTarget,
):
    state_target, delivery_target = postgresql_test_target, postgresql_delivery_target
    state_schema, delivery_schema = state_target.schema, delivery_target.schema
    settings = PostgreSQLStateStoreConfig(dsn_env="TEST_DSN", connect_timeout_seconds=5, pool_max_size=2)
    store = PostgreSQLStateStore(settings, _DSN, schema=state_schema)
    delivery = PostgreSQLDeliveryLedger(_DSN, schema=delivery_schema, settings=DeliveryLedgerPostgreSQLConfig(connect_timeout_seconds=5))
    operations = PostgreSQLSandboxOperations(
        settings, _DSN, schema=state_schema, delivery_schema=delivery_schema, command_runner=_runner,
    )
    try:
        yield operations, store, delivery, state_target, delivery_target
    finally:
        store.close()
        delivery.close()
        state_target.verify()
        delivery_target.verify()


@pytest.fixture
def backup_root():
    root = Path("task-output") / "postgresql-state-store-operations" / uuid.uuid4().hex
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_pg18_doctor_backup_restore_isolated_and_reversible(sandbox, backup_root: Path):
    operations, store, delivery, _state_target, _delivery_target = sandbox
    session_id = f"operations-{uuid.uuid4()}"
    store.ensure_session(session_id, source="operations")
    store.append_message(session_id, role="user", content="logical backup round trip")
    receipt = store.acquire_session_runtime_ownership(
        session_id, RuntimeOwner("installation", "host", "generation"), ttl_seconds=30,
    )
    assert receipt is not None
    delivery.record_obligation(
        obligation_id=delivery.compute_obligation_id("session", "message-ref", "delivery payload"),
        session_key="session", platform="platform", chat_id="chat", thread_id=None, content="delivery payload",
    )

    status = operations.doctor(required_extensions=("pg_trgm", "vector"))
    assert status["search"]["available"] is True
    assert status["ownership"]["active_leases"] == 1
    assert status["delivery_ledger"] == {
        "schema": delivery._schema, "present": True, "migration_versions": [1], "obligation_count": 1,
    }
    backup = operations.backup(backup_root, required_extensions=("pg_trgm", "vector"), quiesced=True)
    assert backup.manifest["archive"]["sha256"]
    manifest = json.loads(backup.manifest_path.read_text())
    assert manifest["backend"] == "postgresql"
    assert manifest["created_at"].endswith("+00:00")
    assert manifest["archive"]["sha256"] == backup.manifest["archive"]["sha256"]
    assert manifest["tenant"]["table_counts"] == status["table_counts"]

    restore_target = f"hermes_state_restore_{uuid.uuid4().hex}"
    result = operations.restore_and_verify(backup.backup_directory, target_database=restore_target)
    assert result["verified"] is True and result["restored_database"] is None
    default_cleanup_target = f"hermes_state_restore_{uuid.uuid4().hex}"
    default_cleanup = operations.restore_and_verify(backup.backup_directory, target_database=default_cleanup_target)
    assert default_cleanup["verified"] is True and default_cleanup["restored_database"] is None
    # Parallel UUID-schema suites may be restoring their own disposable
    # databases.  Assert cleanup of this test's owned targets only.
    restored = set(_restored_databases())
    assert restore_target not in restored
    assert default_cleanup_target not in restored


def test_pg18_operations_fail_closed_for_missing_extension_bad_manifest_and_existing_target(sandbox, backup_root: Path):
    operations, store, _delivery, _state_target, _delivery_target = sandbox
    store.ensure_session(f"operations-{uuid.uuid4()}", source="operations")
    with pytest.raises(PostgreSQLSandboxOperationsError, match="invariant"):
        operations.doctor(required_extensions=("missing_extension",))
    with pytest.raises(PostgreSQLSandboxOperationsError, match="quiesced"):
        operations.backup(backup_root)
    backup = operations.backup(backup_root, quiesced=True)
    with pytest.raises(PostgreSQLSandboxOperationsError, match="generated isolated"):
        operations.restore_and_verify(backup.backup_directory, target_database="not-an-owned-generated-target")
    manifest = json.loads(backup.manifest_path.read_text())
    manifest["archive"]["sha256"] = "0" * 64
    backup.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    restore_target = f"hermes_state_restore_{uuid.uuid4().hex}"
    with pytest.raises(PostgreSQLSandboxOperationsError, match="manifest"):
        operations.restore_and_verify(backup.backup_directory, target_database=restore_target)
    # Invalid manifests are rejected before the owned target is created;
    # unrelated parallel tests must not influence this assertion.
    assert restore_target not in set(_restored_databases())


def test_pg18_backup_rejects_missing_explicit_quiescence(sandbox, backup_root: Path):
    operations, store, _delivery, _state_target, _delivery_target = sandbox
    store.ensure_session(f"operations-{uuid.uuid4()}", source="operations")
    with pytest.raises(PostgreSQLSandboxOperationsError, match="--quiesced"):
        operations.backup(backup_root)


def test_pg18_doctor_rejects_catalog_drift_without_migrating(sandbox):
    operations, store, _delivery, _state_target, _delivery_target = sandbox
    _state_target.execute(
        f"DELETE FROM \"{store._schema}\".schema_migrations WHERE version=18"
    )
    with pytest.raises(PostgreSQLSandboxOperationsError, match="migration catalog"):
        operations.doctor()
    with _psycopg().connect(_DSN) as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT count(*) FROM {store._schema}.schema_migrations WHERE version=18")
        assert cursor.fetchone()[0] == 0


def _restored_databases() -> list[str]:
    with _psycopg().connect(_DSN, dbname="postgres") as connection, connection.cursor() as cursor:
        cursor.execute("SELECT datname FROM pg_database WHERE datname LIKE 'hermes_state_restore_%'")
        return [str(row[0]) for row in cursor.fetchall()]

"""Unit tests for PostgreSQL migration target SQL and recovery protocol."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
from typing import Any

import pytest

from hermes_cli.state_postgres_migration import TableSpec
from state_store.postgres.migration_adapter import PostgresMigrationTargetAdapter


class Cursor:
    def __init__(self, *, one: Any = None, many: list[Any] | None = None, rowcount: int = 1):
        self._one = one
        self._many = many or []
        self.rowcount = rowcount

    def fetchone(self):
        return self._one

    def fetchmany(self, size: int):
        return self._many[:size]


class Connection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.marker: Any = None
        self.rows: list[Any] = []
        self.target_occupied = False
        self.success_rowcount = 1
        self.namespaces: set[str] = set()
        self.namespace_policy: Any = (True, True, True)
        self.namespace_objects: dict[str, Any] = {}

    def execute(self, statement: str, parameters: Any = None):
        self.calls.append((" ".join(statement.split()), parameters))
        normalized = " ".join(statement.split()).lower()
        if normalized.startswith("create schema "):
            match = re.search(r'create schema "([^"]+)"', statement, re.IGNORECASE)
            assert match is not None
            self.namespaces.add(match.group(1))
            return Cursor()
        if "select status, verified_source" in normalized:
            return Cursor(one=self.marker)
        if (
            "from pg_catalog.pg_namespace as namespace" in normalized
            and "select 1" in normalized
        ):
            schema = parameters[0]
            exists = schema in self.namespaces or (
                schema == "hermes" and self.target_occupied
            )
            return Cursor(one=(1,) if exists else None)
        if "owned_by_current_user" in normalized:
            return Cursor(one=self.namespace_policy)
        if "from pg_catalog.pg_proc" in normalized:
            return Cursor(one=self.namespace_objects.get(parameters[0]))
        if "select run_id, status, verified_source, report" in normalized:
            return Cursor(one=self.marker)
        if normalized.startswith("update"):
            return Cursor(rowcount=self.success_rowcount)
        if normalized.startswith("select") and "from \"hermes_stage" in normalized:
            return Cursor(many=self.rows)
        if normalized.startswith("select") and "from sessions" in normalized:
            return Cursor(many=self.rows)
        return Cursor()


class Lease:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        self.exited = False

    def __enter__(self):
        return self.connection

    def __exit__(self, *args):
        self.exited = True


class Pool:
    def __init__(self, connection: Connection) -> None:
        self.connection_value = connection
        self.leases: list[Lease] = []

    def connection(self):
        lease = Lease(self.connection_value)
        self.leases.append(lease)
        return lease


class Store:
    def __init__(self, connection: Connection) -> None:
        self._pool = Pool(connection)
        self.connection = connection
        self.opened = False

    def open(self):
        self.opened = True

    @contextmanager
    def transaction(self, **kwargs):
        assert kwargs == {"read_only": False, "configure_search_path": False}
        yield self.connection


def _adapter(connection: Connection) -> PostgresMigrationTargetAdapter:
    return PostgresMigrationTargetAdapter(
        dsn_env="HERMES_STATE_POSTGRES_DSN",
        schema="hermes",
        home=Path("/tmp/hermes"),
        store=Store(connection),  # type: ignore[arg-type]
    )


def _statements(connection: Connection) -> list[str]:
    return [statement for statement, _ in connection.calls]


def _lock(adapter: PostgresMigrationTargetAdapter, run_id: str) -> None:
    adapter.acquire_advisory_lock(run_id)


def test_staging_uses_authored_schema_and_two_phase_session_parent_fk():
    connection = Connection()
    adapter = _adapter(connection)
    manifest = (
        TableSpec("sessions", ("id",)),
        TableSpec("messages", ("id",)),
    )

    _lock(adapter, "run-1")
    staging = adapter.create_or_resume_staging("run-1", manifest)
    adapter.begin_session_parent_link_copy(staging)
    adapter.finalize_session_parent_links(staging)

    statements = _statements(connection)
    assert staging.startswith("hermes_stage_")
    assert any("CREATE TABLE IF NOT EXISTS sessions" in statement for statement in statements)
    assert any("DROP CONSTRAINT IF EXISTS sessions_parent_session_id_fkey" in statement for statement in statements)
    assert any("FOREIGN KEY (parent_session_id) REFERENCES sessions(id) NOT VALID" in statement for statement in statements)
    assert any("VALIDATE CONSTRAINT sessions_parent_session_id_fkey" in statement for statement in statements)


def test_copy_is_bounded_row_at_a_time_and_identity_is_reset():
    connection = Connection()
    adapter = _adapter(connection)
    table = TableSpec("messages", ("id",))
    _lock(adapter, "run-copy")
    staging = adapter.create_or_resume_staging("run-copy", (table, TableSpec("sessions", ("id",))))
    rows = [
        {
            "id": 5,
            "session_id": "session-1",
            "role": "user",
            "content": "one",
            "tool_call_id": None,
            "tool_calls": None,
            "tool_name": None,
            "effect_disposition": None,
            "timestamp": 1.0,
            "token_count": None,
            "finish_reason": None,
            "reasoning": None,
            "reasoning_content": None,
            "reasoning_details": None,
            "codex_reasoning_items": None,
            "codex_message_items": None,
            "platform_message_id": None,
            "observed": 0,
            "active": 1,
            "compacted": 0,
        },
        {
            "id": 6,
            "session_id": "session-1",
            "role": "assistant",
            "content": "two",
            "tool_call_id": None,
            "tool_calls": None,
            "tool_name": None,
            "effect_disposition": None,
            "timestamp": 2.0,
            "token_count": None,
            "finish_reason": None,
            "reasoning": None,
            "reasoning_content": None,
            "reasoning_details": None,
            "codex_reasoning_items": None,
            "codex_message_items": None,
            "platform_message_id": None,
            "observed": 0,
            "active": 1,
            "compacted": 0,
        },
    ]

    adapter.copy_rows(staging, table, rows)
    adapter.reset_identity(staging, table, "id")

    inserts = [
        statement
        for statement in _statements(connection)
        if '"messages"' in statement and statement.startswith("INSERT INTO")
    ]
    assert len(inserts) == 2
    assert all("ON CONFLICT (\"id\") DO UPDATE" in statement for statement in inserts)
    assert any("pg_get_serial_sequence" in statement for statement in _statements(connection))


def test_target_keyset_reads_use_fetchmany_and_no_fetchall():
    connection = Connection()
    adapter = _adapter(connection)
    table = TableSpec("sessions", ("id",))
    _lock(adapter, "run-read")
    staging = adapter.create_or_resume_staging("run-read", (table,))
    connection.rows = [{"id": "next", "source": "cli"}]

    rows = adapter.fetchmany_keyset(
        staging,
        table,
        ("id", "source"),
        ("previous",),
        10,
    )

    assert rows == [{"id": "next", "source": "cli"}]
    statement, parameters = connection.calls[-1]
    assert '"id" > %s' in statement
    assert parameters == ("previous", 10)
    assert not any("fetchall" in statement.lower() for statement in _statements(connection))


def test_advisory_lock_lifecycle_and_empty_target_refusal():
    connection = Connection()
    adapter = _adapter(connection)

    adapter.acquire_advisory_lock("run-lock")
    adapter.release_advisory_lock("run-lock")
    assert any("pg_advisory_lock" in statement for statement in _statements(connection))
    assert any("pg_advisory_unlock" in statement for statement in _statements(connection))

    connection.target_occupied = True
    assert not adapter.publish_schema_is_empty()


def test_namespace_admission_requires_lock_and_rejects_unsafe_reuse():
    connection = Connection()
    adapter = _adapter(connection)
    manifest = (TableSpec("sessions", ("id",)),)

    with pytest.raises(RuntimeError, match="requires its advisory lock"):
        adapter.create_or_resume_staging("run-namespace", manifest)

    _lock(adapter, "run-namespace")
    connection.namespace_policy = (False, True, True)
    with pytest.raises(RuntimeError, match="owned by the current database user"):
        adapter.create_or_resume_staging("run-namespace", manifest)

    connection.namespace_policy = (True, False, True)
    with pytest.raises(RuntimeError, match="grant CREATE"):
        adapter.create_or_resume_staging("run-namespace", manifest)

    connection.namespace_policy = (True, True, False)
    with pytest.raises(RuntimeError, match="explicit ACLs"):
        adapter.create_or_resume_staging("run-namespace", manifest)

    connection.namespace_policy = (True, True, True)
    staging_schema = next(
        schema for schema in connection.namespaces if schema.startswith("hermes_stage_")
    )
    connection.namespace_objects[staging_schema] = ("function or procedure",)
    with pytest.raises(RuntimeError, match="function or procedure"):
        adapter.create_or_resume_staging("run-namespace", manifest)


def test_publish_requires_a_fresh_target_schema_under_the_advisory_lock():
    connection = Connection()
    adapter = _adapter(connection)
    manifest = (TableSpec("sessions", ("id",)),)
    _lock(adapter, "run-publish-existing")
    staging = adapter.create_or_resume_staging("run-publish-existing", manifest)
    connection.target_occupied = True

    with pytest.raises(RuntimeError, match="fresh namespace is required"):
        adapter.atomic_publish(staging, "run-publish-existing", {"run_id": "run-publish-existing"})

    assert not any(" SET SCHEMA " in statement for statement in _statements(connection))


def test_published_marker_cannot_bypass_namespace_validation():
    connection = Connection()
    adapter = _adapter(connection)
    connection.namespaces.add("hermes")
    connection.marker = {
        "run_id": "published-safety",
        "status": "published",
        "verified_source": {"run_id": "published-safety"},
        "report": None,
    }
    connection.namespace_objects["hermes"] = ("domain",)

    with pytest.raises(RuntimeError, match="domain"):
        adapter.published_run_report("published-safety")


def test_atomic_publish_binds_evidence_to_marker_and_is_idempotent():
    connection = Connection()
    adapter = _adapter(connection)
    manifest = (TableSpec("sessions", ("id",)),)
    _lock(adapter, "run-publish")
    staging = adapter.create_or_resume_staging("run-publish", manifest)
    evidence = {
        "run_id": "run-publish",
        "manifest": ["sessions"],
        "tables": {
            "sessions": {
                "source_count": 1,
                "source_digest": "a" * 64,
                "target_count": 1,
                "target_digest": "a" * 64,
            }
        },
    }

    adapter.atomic_publish(staging, "run-publish", evidence)
    statements = _statements(connection)
    assert any('ALTER TABLE "' + staging + '"."schema_version" SET SCHEMA "hermes"' in statement for statement in statements)
    assert any("verified_source" in statement and "ON CONFLICT" in statement for statement in statements)

    before = len(connection.calls)
    connection.marker = {"status": "published", "verified_source": evidence}
    adapter.atomic_publish(staging, "run-publish", evidence)
    after = _statements(connection)[before:]
    assert not any("ALTER TABLE" in statement for statement in after)

    with pytest.raises(RuntimeError, match="conflicting verified evidence"):
        adapter.atomic_publish(staging, "run-publish", {**evidence, "manifest": []})


def test_failure_does_not_demote_published_state_and_success_requires_marker():
    connection = Connection()
    adapter = _adapter(connection)
    adapter.record_failure({"run_id": "run-failure", "failure": "safe"})
    assert any(
        "last_failure = EXCLUDED.last_failure" in statement
        for statement in _statements(connection)
    )

    connection.success_rowcount = 0
    with pytest.raises(RuntimeError, match="unpublished"):
        adapter.record_success({"run_id": "run-failure"})

"""Focused safety tests for SQLite-to-PostgreSQL migration orchestration."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import pytest

from hermes_cli.state_postgres_migration import (
    DEFAULT_DURABLE_TABLES,
    MigrationPhase,
    MigrationRequest,
    StatePostgresMigration,
    TableSpec,
    _redact_text,
)


CORE_COLUMNS = {
    "sessions": ("id", "source", "parent_session_id", "handoff_state"),
    "messages": ("id", "session_id", "role", "content", "timestamp"),
    "session_model_usage": (
        "session_id",
        "model",
        "billing_provider",
        "billing_base_url",
        "billing_mode",
        "task",
        "api_call_count",
    ),
    "state_meta": ("key", "value"),
    "gateway_routing": ("scope", "session_key", "entry_json", "updated_at"),
    "compression_locks": ("session_id", "holder", "acquired_at", "expires_at"),
    "async_delegations": ("delegation_id", "origin_session", "state", "updated_at"),
}


def _key(table: TableSpec, row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[column] for column in table.key_columns)


class FakeSnapshot:
    """A strict keyset-only snapshot; ``fetchall`` is intentionally forbidden."""

    def __init__(self, rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self._rows = {
            name: [dict(row) for row in rows] for name, rows in rows_by_table.items()
        }
        self.closed = False
        self.fetchmany_calls: list[tuple[str, Optional[tuple[Any, ...]], int]] = []
        self.fetchall_calls = 0

    def available_tables(self):
        return self._rows.keys()

    def columns(self, table: str):
        return CORE_COLUMNS[table]

    def fetchmany_keyset(self, table, key_columns, after_key, batch_size):
        self.fetchmany_calls.append((table, after_key, batch_size))
        rows = sorted(
            self._rows[table],
            key=lambda row: tuple(row[column] for column in key_columns),
        )
        if after_key is not None:
            rows = [
                row
                for row in rows
                if tuple(row[column] for column in key_columns) > after_key
            ]
        return rows[:batch_size]

    def fetchall(self, *args, **kwargs):
        del args, kwargs
        self.fetchall_calls += 1
        raise AssertionError("migration must not call source fetchall")

    def close(self):
        self.closed = True


class FakeWriterFence:
    def __init__(self, source):
        self._source = source

    def __enter__(self):
        self._source.events.append("fence_enter")
        self._source.fence_depth += 1

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        self._source.fence_depth -= 1
        self._source.events.append("fence_exit")


class FakeSource:
    def __init__(
        self, rows_by_table: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None
    ) -> None:
        self.rows_by_table = {
            table.name: [] for table in DEFAULT_DURABLE_TABLES if not table.optional
        }
        if rows_by_table is not None:
            self.rows_by_table.update(rows_by_table)
        self.integrity_errors: list[str] = []
        self.foreign_key_violations: list[str] = []
        self.writer_reasons: list[str] = []
        self.compression_leases: list[str] = []
        self.delegations: list[str] = []
        self.handoffs: list[str] = []
        self.snapshot_calls: list[str] = []
        self.snapshots: list[FakeSnapshot] = []
        self.snapshot: Optional[FakeSnapshot] = None
        self.events: list[str] = []
        self.fence_depth = 0
        self.writer_checks = 0
        self.writer_reason_after_preflight: Optional[str] = None

    def writer_fence(self, run_id):
        del run_id
        return FakeWriterFence(self)

    def available_tables(self):
        return self.rows_by_table.keys()

    def sqlite_integrity_errors(self):
        return self.integrity_errors

    def sqlite_foreign_key_violations(self):
        return self.foreign_key_violations

    def active_writer_reasons(self):
        self.writer_checks += 1
        if (
            self.writer_reason_after_preflight is not None
            and self.writer_checks >= 2
        ):
            return [self.writer_reason_after_preflight]
        return self.writer_reasons

    def active_compression_leases(self, now):
        del now
        return self.compression_leases

    def active_delegations(self):
        return self.delegations

    def active_handoffs(self):
        return self.handoffs

    def snapshot_via_sqlite_backup(self, run_id):
        self.snapshot_calls.append(run_id)
        self.events.append("snapshot")
        assert self.fence_depth == 1
        self.snapshot = FakeSnapshot(self.rows_by_table)
        self.snapshots.append(self.snapshot)
        return self.snapshot


class FakeTarget:
    def __init__(self) -> None:
        self.redacted_identity = "postgresql://***@state.example/hermes"
        self.empty = True
        self.events: list[str] = []
        self.staging: dict[
            str, dict[str, dict[tuple[Any, ...], dict[str, Any]]]
        ] = {}
        self.failure_reports: list[dict[str, Any]] = []
        self.success_reports: list[dict[str, Any]] = []
        self.copy_failure: Optional[Exception] = None
        self.tamper_messages = False
        self.identity_resets: list[tuple[str, str, str]] = []
        self.fetchmany_calls: list[tuple[str, Optional[tuple[Any, ...]], int]] = []
        self.fetchall_calls = 0
        self.foreign_key_violations: list[str] = []
        self.lineage_violations: list[str] = []
        self.published_runs: dict[str, dict[str, Any]] = {}
        self.publish_count = 0
        self.session_copy_started: set[str] = set()
        self.session_copy_keys: list[tuple[Any, ...]] = []
        self.record_success_failure: Optional[Exception] = None
        self.unlock_failure: Optional[Exception] = None

    def publish_schema_is_empty(self):
        self.events.append("target_empty")
        return self.empty

    def acquire_advisory_lock(self, run_id):
        self.events.append(f"lock:{run_id}")

    def release_advisory_lock(self, run_id):
        self.events.append(f"unlock:{run_id}")
        if self.unlock_failure is not None:
            raise self.unlock_failure

    def create_or_resume_staging(self, run_id, manifest):
        self.events.append("stage")
        schema = f"hermes_state_stage_{run_id}"
        self.staging.setdefault(schema, {table.name: {} for table in manifest})
        return schema

    def begin_session_parent_link_copy(self, staging_schema):
        self.events.append("begin_session_parent_links")
        self.session_copy_started.add(staging_schema)

    def copy_rows(self, staging_schema, table, rows):
        self.events.append(f"copy:{table.name}")
        if self.copy_failure is not None:
            raise self.copy_failure
        if table.name == "sessions":
            assert staging_schema in self.session_copy_started
        destination = self.staging[staging_schema][table.name]
        for row in rows:
            if table.name == "sessions":
                self.session_copy_keys.append(_key(table, row))
            destination[_key(table, row)] = dict(row)
        if self.tamper_messages and table.name == "messages" and destination:
            first_key = next(iter(destination))
            destination[first_key]["content"] = "tampered"

    def finalize_session_parent_links(self, staging_schema):
        self.events.append("finalize_session_parent_links")
        sessions = self.staging[staging_schema]["sessions"]
        for row in sessions.values():
            parent = row.get("parent_session_id")
            if parent is not None and (parent,) not in sessions:
                raise AssertionError(f"missing parent {parent}")

    def reset_identity(self, staging_schema, table, identity_column):
        assert table.name == "messages"
        assert identity_column == "id"
        self.events.append("reset_identity")
        self.identity_resets.append((staging_schema, table.name, identity_column))

    def fetchmany_keyset(self, staging_schema, table, columns, after_key, batch_size):
        self.events.append(f"target_stream:{table.name}")
        self.fetchmany_calls.append((table.name, after_key, batch_size))
        rows = sorted(
            self.staging[staging_schema][table.name].values(),
            key=lambda row: _key(table, row),
        )
        if after_key is not None:
            rows = [row for row in rows if _key(table, row) > after_key]
        return [
            {column: row.get(column) for column in columns}
            for row in rows[:batch_size]
        ]

    def fetchall(self, *args, **kwargs):
        del args, kwargs
        self.fetchall_calls += 1
        raise AssertionError("migration must not call target fetchall")

    def read_row(self, staging_schema, table, key, columns):
        self.events.append(f"probe:{table.name}")
        row = self.staging[staging_schema][table.name].get(key)
        if row is None:
            return None
        return {column: row.get(column) for column in columns}

    def validate_foreign_keys(self, staging_schema):
        del staging_schema
        self.events.append("verify_fk")
        return self.foreign_key_violations

    def validate_session_lineage(self, staging_schema):
        del staging_schema
        self.events.append("verify_lineage")
        return self.lineage_violations

    def atomic_publish(self, staging_schema, run_id, verified_source):
        self.events.append("publish")
        self.publish_count += 1
        self.empty = False
        self.published_runs[run_id] = {
            "run_id": run_id,
            "verified_source": {
                "run_id": verified_source["run_id"],
                "manifest": list(verified_source["manifest"]),
                "tables": {
                    name: dict(evidence)
                    for name, evidence in verified_source["tables"].items()
                },
            },
        }

    def published_run_report(self, run_id):
        self.events.append(f"published_marker:{run_id}")
        return self.published_runs.get(run_id)

    def record_success(self, report):
        self.events.append("success")
        if self.record_success_failure is not None:
            raise self.record_success_failure
        self.success_reports.append(dict(report))

    def record_failure(self, report):
        self.events.append("failure_report")
        self.failure_reports.append(dict(report))


def _durable_rows():
    return {
        "sessions": [
            {
                "id": "s1",
                "source": "cli",
                "parent_session_id": None,
                "handoff_state": None,
            },
            {
                "id": "s2",
                "source": "cli",
                "parent_session_id": "s1",
                "handoff_state": None,
            },
        ],
        "messages": [
            {
                "id": 10,
                "session_id": "s1",
                "role": "user",
                "content": "one",
                "timestamp": 1.0,
            },
            {
                "id": 20,
                "session_id": "s2",
                "role": "assistant",
                "content": "two",
                "timestamp": 2.0,
            },
        ],
        "session_model_usage": [
            {
                "session_id": "s1",
                "model": "model",
                "billing_provider": "",
                "billing_base_url": "",
                "billing_mode": "",
                "task": "",
                "api_call_count": 2,
            }
        ],
        "state_meta": [{"key": "scheduler", "value": "ok"}],
        "gateway_routing": [
            {
                "scope": "",
                "session_key": "telegram:1",
                "entry_json": "{}",
                "updated_at": 3.0,
            }
        ],
        # This stale lease must never be copied or included in verification.
        "compression_locks": [
            {
                "session_id": "s1",
                "holder": "dead",
                "acquired_at": 1.0,
                "expires_at": 99.0,
            }
        ],
        "async_delegations": [],
    }


def _engine(source, target, cutover=None):
    source.events = target.events
    if cutover is None:
        cutover = lambda report: target.events.append(f"cutover:{report.run_id}")
    return StatePostgresMigration(
        source,
        target,
        cutover=cutover,
        clock=lambda: 100.0,
    )


def test_apply_uses_bounded_keysets_verifies_and_publishes_before_cutover():
    source = FakeSource(_durable_rows())
    target = FakeTarget()

    report = _engine(source, target).run(
        MigrationRequest(apply=True, run_id="run-1", batch_size=1)
    )

    assert report.phase is MigrationPhase.COMPLETE
    assert report.succeeded
    assert source.snapshot_calls == ["run-1"]
    assert source.snapshot is not None
    assert source.snapshot.closed
    assert source.snapshot.fetchall_calls == 0
    assert target.fetchall_calls == 0
    assert all(size == 1 for _, _, size in source.snapshot.fetchmany_calls)
    assert all(size == 1 for _, _, size in target.fetchmany_calls)
    assert report.tables["messages"].source_count == 2
    assert (
        report.tables["messages"].source_digest
        == report.tables["messages"].target_digest
    )
    assert report.tables["compression_locks"].source_count == 0
    assert target.identity_resets == [
        ("hermes_state_stage_run-1", "messages", "id")
    ]
    assert target.events.index("fence_enter") < target.events.index("lock:run-1")
    assert target.events.index("lock:run-1") < target.events.index("snapshot")
    assert target.events.index("begin_session_parent_links") < target.events.index(
        "copy:sessions"
    )
    assert target.events.index("copy:sessions") < target.events.index(
        "finalize_session_parent_links"
    )
    assert target.events.index("verify_fk") < target.events.index("publish")
    assert target.events.index("verify_lineage") < target.events.index("publish")
    assert target.events.index("publish") < target.events.index("cutover:run-1")
    assert target.events.index("cutover:run-1") < target.events.index("fence_exit")
    assert report.published and report.cutover_complete


def test_dry_run_checks_source_and_target_without_snapshot_or_mutation():
    source = FakeSource(_durable_rows())
    target = FakeTarget()

    report = _engine(source, target).run(MigrationRequest(run_id="dry-run"))

    assert report.phase is MigrationPhase.DRY_RUN
    assert not report.apply
    assert source.snapshot_calls == []
    assert target.events == [
        "fence_enter",
        "published_marker:dry-run",
        "target_empty",
        "fence_exit",
    ]
    assert report.manifest == [table.name for table in DEFAULT_DURABLE_TABLES if not table.optional]


def test_preflight_refuses_integrity_writers_leases_delegations_and_handoffs():
    source = FakeSource(_durable_rows())
    source.integrity_errors = ["malformed page"]
    source.foreign_key_violations = ["messages row 20"]
    source.writer_reasons = ["gateway pid 123"]
    source.compression_leases = ["s1 held by worker"]
    source.delegations = ["delegation-1 running"]
    source.handoffs = ["s2 pending"]
    target = FakeTarget()

    report = _engine(source, target).run(MigrationRequest(run_id="blocked"))

    assert report.phase is MigrationPhase.FAILED
    assert "SQLite integrity: malformed page" in report.failure
    assert "SQLite foreign key: messages row 20" in report.failure
    assert "active writer: gateway pid 123" in report.failure
    assert "active compression lease: s1 held by worker" in report.failure
    assert "active delegation: delegation-1 running" in report.failure
    assert "active handoff: s2 pending" in report.failure
    assert source.snapshot_calls == []
    assert "stage" not in target.events
    assert target.failure_reports == []
    assert target.events == ["fence_enter", "fence_exit"]


def test_nonempty_target_refuses_without_snapshot_or_staging():
    source = FakeSource(_durable_rows())
    target = FakeTarget()
    target.empty = False

    report = _engine(source, target).run(MigrationRequest(apply=True, run_id="occupied"))

    assert report.phase is MigrationPhase.FAILED
    assert "target publish schema is not empty" in report.failure
    assert source.snapshot_calls == []
    assert "stage" not in target.events
    assert "snapshot" not in target.events


def test_copy_failure_records_redacted_resumable_report_without_cutover():
    source = FakeSource(_durable_rows())
    target = FakeTarget()
    target.copy_failure = RuntimeError(
        "connection postgresql://alice:super-secret@state.example/hermes refused"
    )

    report = _engine(source, target).run(
        MigrationRequest(apply=True, run_id="resume-7")
    )

    assert report.phase is MigrationPhase.FAILED
    assert report.resume_run_id == "resume-7"
    assert "super-secret" not in report.failure
    assert "postgresql://***@state.example/hermes" in report.failure
    assert not report.published
    assert not report.cutover_complete
    assert "publish" not in target.events
    assert not any(event.startswith("cutover:") for event in target.events)
    assert target.failure_reports[0]["run_id"] == "resume-7"
    assert target.failure_reports[0]["target_identity"] == target.redacted_identity
    assert target.events[-2:] == ["unlock:resume-7", "fence_exit"]


def test_digest_mismatch_blocks_publish_and_config_cutover():
    source = FakeSource(_durable_rows())
    target = FakeTarget()
    target.tamper_messages = True

    report = _engine(source, target).run(
        MigrationRequest(apply=True, run_id="bad-digest")
    )

    assert report.phase is MigrationPhase.FAILED
    assert "messages digest mismatch" in report.failure
    assert report.tables["messages"].source_count == 2
    assert report.tables["messages"].target_count == 2
    assert "publish" not in target.events
    assert not any(event.startswith("cutover:") for event in target.events)
    assert target.failure_reports


def test_target_fk_or_lineage_failure_blocks_publish_and_cutover():
    source = FakeSource(_durable_rows())
    target = FakeTarget()
    target.foreign_key_violations = ["messages.session_id=s9"]

    report = _engine(source, target).run(
        MigrationRequest(apply=True, run_id="bad-fk")
    )

    assert report.phase is MigrationPhase.FAILED
    assert "target foreign keys: messages.session_id=s9" in report.failure
    assert "publish" not in target.events
    assert not any(event.startswith("cutover:") for event in target.events)

    target = FakeTarget()
    target.lineage_violations = ["sessions cycle s1 -> s2 -> s1"]
    report = _engine(FakeSource(_durable_rows()), target).run(
        MigrationRequest(apply=True, run_id="bad-lineage")
    )
    assert report.phase is MigrationPhase.FAILED
    assert "target session lineage: sessions cycle" in report.failure
    assert "publish" not in target.events


def test_replace_is_rejected_without_contacting_source_or_target():
    source = FakeSource(_durable_rows())
    target = FakeTarget()

    with pytest.raises(ValueError, match="destructive --replace"):
        _engine(source, target).run(MigrationRequest(apply=True, replace=True))

    assert source.snapshot_calls == []
    assert target.events == []


def test_raw_adapter_identity_is_redacted_before_reports_are_persisted():
    source = FakeSource(_durable_rows())
    target = FakeTarget()
    target.redacted_identity = "postgresql://alice:super-secret@state.example/hermes"
    target.copy_failure = RuntimeError("copy failed")

    report = _engine(source, target).run(
        MigrationRequest(apply=True, run_id="identity-redaction")
    )

    assert report.target_identity == "postgresql://***@state.example/hermes"
    assert "super-secret" not in target.failure_reports[0]["target_identity"]


def test_writer_admission_is_rechecked_after_preflight_under_the_fence():
    source = FakeSource(_durable_rows())
    source.writer_reason_after_preflight = "gateway pid 987 started after preflight"
    target = FakeTarget()

    report = _engine(source, target).run(
        MigrationRequest(apply=True, run_id="writer-race")
    )

    assert report.phase is MigrationPhase.FAILED
    assert "active writer: gateway pid 987" in report.failure
    assert source.writer_checks == 2
    assert source.snapshot_calls == []
    assert "stage" not in target.events
    assert target.events.index("fence_enter") < target.events.index("lock:writer-race")
    assert target.events.index("lock:writer-race") < target.events.index(
        "unlock:writer-race"
    )
    assert target.events[-1] == "fence_exit"


def test_cutover_failure_retries_from_publish_marker_without_recopy_or_republish():
    source = FakeSource(_durable_rows())
    target = FakeTarget()
    cutover_attempts: list[str] = []

    def cutover(report):
        cutover_attempts.append(report.run_id)
        target.events.append(f"cutover:{report.run_id}")
        if len(cutover_attempts) == 1:
            raise RuntimeError("temporary config write failure")

    engine = _engine(source, target, cutover=cutover)
    first = engine.run(MigrationRequest(apply=True, run_id="cutover-retry"))

    assert first.phase is MigrationPhase.FAILED
    assert first.published
    assert not first.cutover_complete
    assert target.publish_count == 1
    assert source.snapshot_calls == ["cutover-retry"]
    assert target.failure_reports
    assert (
        target.published_runs["cutover-retry"]["verified_source"]["tables"]["messages"][
            "source_digest"
        ]
        == first.tables["messages"].source_digest
    )

    second = engine.run(MigrationRequest(apply=True, run_id="cutover-retry"))

    assert second.phase is MigrationPhase.COMPLETE
    assert second.published and second.cutover_complete
    assert second.recovered_published_run
    assert target.publish_count == 1
    assert source.snapshot_calls == ["cutover-retry", "cutover-retry"]
    assert cutover_attempts == ["cutover-retry", "cutover-retry"]
    assert target.success_reports[-1]["recovered_published_run"]


def test_record_success_failure_is_recoverable_and_unlock_failure_is_reported():
    source = FakeSource(_durable_rows())
    target = FakeTarget()
    target.record_success_failure = RuntimeError("status token = 'success secret'")

    report = _engine(source, target).run(
        MigrationRequest(apply=True, run_id="record-success-failure")
    )

    assert report.phase is MigrationPhase.FAILED
    assert report.published and report.cutover_complete
    assert "success secret" not in report.failure
    assert target.failure_reports
    assert "unlock:record-success-failure" in target.events

    source = FakeSource(_durable_rows())
    target = FakeTarget()
    target.unlock_failure = RuntimeError('password = "unlock multi word secret"')
    report = _engine(source, target).run(
        MigrationRequest(apply=True, run_id="unlock-failure")
    )

    assert report.phase is MigrationPhase.FAILED
    assert report.published and report.cutover_complete
    assert "unlock multi word secret" not in report.failure
    assert target.failure_reports
    assert target.events[-1] == "fence_exit"


def test_secret_scrubbing_handles_url_userinfo_and_quoted_multiword_values():
    text = (
        'postgresql://alice:"url multi word"@db.example/hermes '
        'password = "password multi word" host=db '
        "api_key='api multi word'; access_key: access multi word; "
        "SECRET_TOKEN = token multi word; "
        'PGPASSWORD="pg password multi word"; PGUSER=router-user; '
        "client_secret: client secret multi word; PRIVATE_KEY='private key value'\n"
        "Authorization: Bearer bearer token multi word\n"
        "proxy-authorization: Basic basic credential value"
    )

    redacted = _redact_text(text)

    for secret in (
        "url multi word",
        "password multi word",
        "api multi word",
        "access multi word",
        "token multi word",
        "pg password multi word",
        "router-user",
        "client secret multi word",
        "private key value",
        "bearer token multi word",
        "basic credential value",
    ):
        assert secret not in redacted
    assert "postgresql://***@db.example/hermes" in redacted
    assert "***" in redacted


def test_recovery_refuses_cutover_when_fenced_source_changed_after_publish():
    source = FakeSource(_durable_rows())
    target = FakeTarget()
    cutover_attempts: list[str] = []

    def cutover(report):
        cutover_attempts.append(report.run_id)
        if len(cutover_attempts) == 1:
            raise RuntimeError("config temporarily unavailable")

    engine = _engine(source, target, cutover=cutover)
    first = engine.run(MigrationRequest(apply=True, run_id="changed-source"))
    assert first.phase is MigrationPhase.FAILED
    assert first.published

    source.rows_by_table["messages"][0]["content"] = "changed after publish"
    second = engine.run(MigrationRequest(apply=True, run_id="changed-source"))

    assert second.phase is MigrationPhase.FAILED
    assert "current source differs from verified publish evidence for messages" in second.failure
    assert "require a new migration" in second.failure
    assert target.publish_count == 1
    assert source.snapshot_calls == ["changed-source", "changed-source"]
    assert cutover_attempts == ["changed-source"]


def test_recovery_refuses_publish_marker_without_durable_verified_evidence():
    source = FakeSource(_durable_rows())
    target = FakeTarget()
    target.empty = False
    target.published_runs["missing-evidence"] = {"run_id": "missing-evidence"}

    report = _engine(source, target).run(
        MigrationRequest(apply=True, run_id="missing-evidence")
    )

    assert report.phase is MigrationPhase.FAILED
    assert "lacks durable verified source evidence" in report.failure
    assert source.snapshot_calls == []
    assert target.publish_count == 0


def test_child_before_parent_session_batches_use_explicit_two_phase_parent_links():
    rows = _durable_rows()
    rows["sessions"] = [
        {
            "id": "a-child",
            "source": "cli",
            "parent_session_id": "z-parent",
            "handoff_state": None,
        },
        {
            "id": "z-parent",
            "source": "cli",
            "parent_session_id": None,
            "handoff_state": None,
        },
    ]
    rows["messages"] = [
        {
            "id": 10,
            "session_id": "a-child",
            "role": "user",
            "content": "child first",
            "timestamp": 1.0,
        }
    ]
    source = FakeSource(rows)
    target = FakeTarget()

    report = _engine(source, target).run(
        MigrationRequest(apply=True, run_id="child-before-parent", batch_size=1)
    )

    assert report.phase is MigrationPhase.COMPLETE
    assert target.session_copy_keys[:2] == [("a-child",), ("z-parent",)]
    assert target.events.index("begin_session_parent_links") < target.events.index(
        "copy:sessions"
    )
    assert target.events.index("copy:sessions") < target.events.index(
        "finalize_session_parent_links"
    )

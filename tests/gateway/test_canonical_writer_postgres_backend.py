from __future__ import annotations

import datetime as dt
import json

import pytest

from gateway.canonical_writer_db import QueryResult
from gateway.canonical_writer_handlers import (
    CapabilityGrantRequest,
    CanonicalWriterError,
    EventAppendRequest,
    ProjectorReadRequest,
    RouteBackTerminalRequest,
    RuntimeContext,
)
from gateway.canonical_writer_postgres_backend import (
    EXPECTED_HELPER_ROUTINE_SIGNATURES,
    EXPECTED_ROUTINE_SIGNATURES,
    POSTGRES_ROUTINE_BY_OPERATION,
    PRODUCTION_CATALOG_SHA256,
    PRODUCTION_STATEMENT_CATALOG,
    PostgresCanonicalWriterBackend,
)
from gateway.canonical_writer_protocol import CanonicalWriterOperation


class _FixedDatabase:
    statement_names = PRODUCTION_STATEMENT_CATALOG.names
    statement_catalog_sha256 = PRODUCTION_CATALOG_SHA256

    def __init__(self):
        self.calls = []
        self.response = {"ok": True, "result": {"stored": True}}

    def query_fixed(self, statement_name, parameters):
        self.calls.append((statement_name, parameters))
        return QueryResult(
            ("response",),
            ((json.dumps(self.response),),),
            "SELECT 1",
        )


def _runtime():
    return RuntimeContext(
        request_id="request-1",
        platform="discord",
        session_key_sha256="a" * 64,
        capability_epoch_sha256="b" * 64,
        user_id="owner-1",
        thread_id="thread-1",
    )


def _event(event_type):
    return EventAppendRequest(
        event_type=event_type,
        case_id="case-1",
        summary="summary",
        source_refs={"thread_id": "thread-1"},
        actors={},
        body={},
        safety={},
        idempotency_key="event-1",
    )


def test_catalog_covers_every_enum_once_with_one_fixed_routine():
    assert set(POSTGRES_ROUTINE_BY_OPERATION) == set(CanonicalWriterOperation)
    assert len(set(POSTGRES_ROUTINE_BY_OPERATION.values())) == len(
        CanonicalWriterOperation
    )
    assert len(PRODUCTION_STATEMENT_CATALOG.names) == len(CanonicalWriterOperation)
    assert len(EXPECTED_ROUTINE_SIGNATURES) == len(CanonicalWriterOperation) + 1
    assert (
        "canonical_brain.writer_canary_scope_preapproval_retire(jsonb, jsonb)"
        in EXPECTED_ROUTINE_SIGNATURES
    )
    assert "writer_canary_scope_preapproval_retire" not in (
        POSTGRES_ROUTINE_BY_OPERATION.values()
    )
    assert len(EXPECTED_HELPER_ROUTINE_SIGNATURES) == 12
    assert not set(EXPECTED_HELPER_ROUTINE_SIGNATURES).intersection(
        EXPECTED_ROUTINE_SIGNATURES
    )
    assert len(PRODUCTION_CATALOG_SHA256) == 64


def test_backend_rejects_catalog_name_or_digest_drift():
    database = _FixedDatabase()
    database.statement_names = ("wrong",)
    with pytest.raises(ValueError, match="names"):
        PostgresCanonicalWriterBackend(database)

    database = _FixedDatabase()
    database.statement_catalog_sha256 = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        PostgresCanonicalWriterBackend(database)


def test_each_protocol_operation_selects_only_its_catalog_statement():
    database = _FixedDatabase()
    backend = PostgresCanonicalWriterBackend(database)

    for operation in CanonicalWriterOperation:
        assert backend._invoke(operation, {"operation": operation.value}, _runtime()) == {
            "stored": True
        }

    assert [call[0] for call in database.calls] == [
        "op_" + operation.value.replace(".", "_")
        for operation in CanonicalWriterOperation
    ]
    assert all(set(call[1]) == {"request", "runtime"} for call in database.calls)


@pytest.mark.parametrize(
    "event_type,statement",
    [
        ("case.note.recorded", "op_event_append_model"),
        ("task.plan.updated", "op_plan_transition"),
        ("task.verification.recorded", "op_verification_append"),
    ],
)
def test_typed_event_interface_preserves_protocol_routine(event_type, statement):
    database = _FixedDatabase()
    backend = PostgresCanonicalWriterBackend(database)

    backend.event_append(_event(event_type), _runtime())

    assert database.calls[-1][0] == statement


@pytest.mark.parametrize(
    "outcome,statement",
    [
        ("sent", "op_routeback_finalize_sent"),
        ("blocked", "op_routeback_finalize_blocked"),
    ],
)
def test_routeback_terminal_has_distinct_fixed_routines(outcome, statement):
    database = _FixedDatabase()
    backend = PostgresCanonicalWriterBackend(database)
    request = RouteBackTerminalRequest(
        authorization_id="routeauth-1",
        outcome=outcome,
        receipt={"message_id": "message-1"} if outcome == "sent" else {},
        blocker_reason="blocked" if outcome == "blocked" else "",
    )

    backend.routeback_terminal(request, _runtime())

    assert database.calls[-1][0] == statement


def test_routeback_terminal_rejects_unknown_outcome_before_database_call():
    database = _FixedDatabase()
    backend = PostgresCanonicalWriterBackend(database)

    with pytest.raises(CanonicalWriterError, match="outcome"):
        backend.routeback_terminal(
            RouteBackTerminalRequest("routeauth-1", "other", {}, ""),
            _runtime(),
        )
    assert database.calls == []


def test_preclaim_blocked_uses_existing_fixed_routine_without_authorization():
    database = _FixedDatabase()
    backend = PostgresCanonicalWriterBackend(database)
    request = RouteBackTerminalRequest(
        authorization_id="",
        outcome="blocked",
        receipt={},
        blocker_reason="target unresolved",
        preclaim=True,
        case_id="case:1",
        target_ref={"id": "blocked-target:1"},
        message_summary="could not resolve target",
        source_refs={"thread_id": "thread-1"},
        idempotency_key="preclaim:1",
    )

    backend.routeback_terminal(request, _runtime())

    statement, parameters = database.calls[-1]
    assert statement == "op_routeback_finalize_blocked"
    assert parameters["request"] == {
        "preclaim": True,
        "case_id": "case:1",
        "target_ref": {"id": "blocked-target:1"},
        "message_summary": "could not resolve target",
        "source_refs": {"thread_id": "thread-1"},
        "idempotency_key": "preclaim:1",
        "outcome": "blocked",
        "receipt": {},
        "blocker_reason": "target unresolved",
    }
    assert "authorization_id" not in parameters["request"]


def test_capability_grant_serializes_exact_plan_revision_and_expiry():
    database = _FixedDatabase()
    backend = PostgresCanonicalWriterBackend(database)
    expires_at = dt.datetime(2026, 7, 12, 10, 0, tzinfo=dt.timezone.utc)

    backend.capability_grant(
        CapabilityGrantRequest(
            approval_id="approval:1",
            case_id="case:1",
            plan_id="plan:1",
            plan_revision=7,
            approval_source_sha256="c" * 64,
            command_hashes=("d" * 64,),
            expires_at=expires_at,
            max_uses=2,
        ),
        _runtime(),
    )

    request = database.calls[-1][1]["request"]
    assert request["plan_revision"] == 7
    assert request["expires_at"] == "2026-07-12T10:00:00+00:00"


def test_routine_error_envelope_becomes_bounded_semantic_error():
    database = _FixedDatabase()
    database.response = {
        "ok": False,
        "error": {"code": "idempotency_conflict", "message": "conflict"},
    }
    backend = PostgresCanonicalWriterBackend(database)

    with pytest.raises(CanonicalWriterError) as captured:
        backend.ping(_runtime())

    assert captured.value.code == "idempotency_conflict"


def test_datetime_payload_is_canonical_utc_json():
    database = _FixedDatabase()
    backend = PostgresCanonicalWriterBackend(database)
    aware = dt.datetime(2026, 7, 12, 12, 0, tzinfo=dt.timezone(dt.timedelta(hours=3)))

    backend._invoke(CanonicalWriterOperation.PING, {"at": aware}, _runtime())

    assert database.calls[-1][1]["request"]["at"] == "2026-07-12T09:00:00+00:00"


class _SnapshotTransaction:
    def __init__(self, database):
        self.database = database
        self.calls = []

    def query(self, parameters):
        self.calls.append(parameters)
        response = self.database.snapshot_responses.pop(0)
        return QueryResult(
            ("response",),
            ((json.dumps({"ok": True, "result": response}),),),
            "SELECT 1",
        )


class _SnapshotContext:
    def __init__(self, database):
        self.database = database
        self.transaction = _SnapshotTransaction(database)

    def __enter__(self):
        self.database.snapshot_entries += 1
        return self.transaction

    def __exit__(self, exc_type, _exc, _traceback):
        self.database.snapshot_exits.append(exc_type)
        return False


class _SnapshotDatabase(_FixedDatabase):
    def __init__(self):
        super().__init__()
        self.snapshot_entries = 0
        self.snapshot_exits = []
        self.snapshot_responses = []
        self.snapshot_contexts = []

    def projection_read_transaction(self):
        context = _SnapshotContext(self)
        self.snapshot_contexts.append(context)
        return context


def test_projection_export_scope_uses_projection_only_transaction_once():
    database = _SnapshotDatabase()
    database.snapshot_responses = [
        {"events": [{"event_id": "event-1"}], "has_more": True},
        {"events": [{"event_id": "event-2"}], "has_more": False},
    ]
    backend = PostgresCanonicalWriterBackend(database)

    with backend.projection_export_scope() as projection:
        first = projection.projector_read(
            ProjectorReadRequest(case_id="", after_event_id="", limit=10),
            _runtime(),
        )
        second = projection.projector_read(
            ProjectorReadRequest(
                case_id="",
                after_event_id="event-1",
                limit=10,
            ),
            _runtime(),
        )

    assert first["events"][0]["event_id"] == "event-1"
    assert second["events"][0]["event_id"] == "event-2"
    assert database.snapshot_entries == 1
    assert database.snapshot_exits == [None]
    assert database.calls == []
    transaction = database.snapshot_contexts[0].transaction
    assert len(transaction.calls) == 2
    assert all(set(call) == {"request", "runtime"} for call in transaction.calls)


def test_projection_export_scope_fails_closed_without_snapshot_transaction():
    backend = PostgresCanonicalWriterBackend(_FixedDatabase())

    with pytest.raises(CanonicalWriterError, match="consistent projection snapshot"):
        with backend.projection_export_scope():
            pytest.fail("scope must not be exposed")

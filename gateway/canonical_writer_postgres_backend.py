"""Production Canonical Writer backend backed only by fixed PostgreSQL routines.

Each wire operation maps one-to-one to an immutable, schema-qualified routine.
The two mechanical exceptions preserve the already-typed handler interface:
typed plan/verification events select their corresponding protocol operation,
and route-back terminal outcomes select sent versus blocked.  No caller can
provide a routine name or SQL string.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import json
import math
import re
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Protocol

from gateway.canonical_writer_db import (
    CanonicalWriterDB,
    FixedReadOnlyTransaction,
    FixedStatement,
    ParameterKind,
    ParameterSpec,
    QueryResult,
    StatementCatalog,
)
from gateway.canonical_writer_handlers import (
    CapabilityConsumeRequest,
    CapabilityGrantRequest,
    CapabilityRevokeRequest,
    CanonicalWriterError,
    EventAppendRequest,
    PlanActiveMatchRequest,
    ProjectorReadRequest,
    QueryRequest,
    RouteBackAuthorizeRequest,
    RouteBackContextRequest,
    RouteBackRecoveryRequest,
    RouteBackTerminalRequest,
    RuntimeContext,
)
from gateway.canonical_writer_protocol import CanonicalWriterOperation


CANONICAL_WRITER_SCHEMA = "canonical_brain"
CANONICAL_WRITER_ROLE = "canonical_brain_writer"
CANONICAL_WRITER_MIGRATION_OWNER = "canonical_brain_migration_owner"
_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

POSTGRES_ROUTINE_BY_OPERATION: Mapping[CanonicalWriterOperation, str] = (
    MappingProxyType(
        {
            CanonicalWriterOperation.PING: "writer_ping",
            CanonicalWriterOperation.CASE_QUERY: "writer_case_query",
            CanonicalWriterOperation.ROUTEBACK_CONTEXT: "writer_routeback_context",
            CanonicalWriterOperation.PLAN_ACTIVE_MATCH: "writer_plan_active_match",
            CanonicalWriterOperation.EVENT_APPEND_MODEL: "writer_event_append_model",
            CanonicalWriterOperation.PLAN_TRANSITION: "writer_plan_transition",
            CanonicalWriterOperation.VERIFICATION_APPEND: "writer_verification_append",
            CanonicalWriterOperation.ROUTEBACK_CLAIM: "writer_routeback_claim",
            CanonicalWriterOperation.ROUTEBACK_RECOVER: "writer_routeback_recover",
            CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT: (
                "writer_routeback_finalize_sent"
            ),
            CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED: (
                "writer_routeback_finalize_blocked"
            ),
            CanonicalWriterOperation.LEASE_SHADOW_RECORD: "writer_lease_shadow_record",
            CanonicalWriterOperation.CAPABILITY_GRANT: "writer_capability_grant",
            CanonicalWriterOperation.CAPABILITY_CONSUME: "writer_capability_consume",
            CanonicalWriterOperation.CAPABILITY_REVOKE: "writer_capability_revoke",
            CanonicalWriterOperation.CAPABILITY_REVOKE_SESSION: (
                "writer_capability_revoke_session"
            ),
            CanonicalWriterOperation.PROJECTION_READ_EVENTS: (
                "writer_projection_read_events"
            ),
        }
    )
)

if set(POSTGRES_ROUTINE_BY_OPERATION) != set(CanonicalWriterOperation):
    raise RuntimeError("PostgreSQL writer catalog does not cover the protocol enum")
if len(set(POSTGRES_ROUTINE_BY_OPERATION.values())) != len(
    POSTGRES_ROUTINE_BY_OPERATION
):
    raise RuntimeError("each writer operation must use a distinct routine")


def _statement_name(operation: CanonicalWriterOperation) -> str:
    return "op_" + operation.value.replace(".", "_")


def _build_production_statements() -> tuple[FixedStatement, ...]:
    request = ParameterSpec(
        "request",
        ParameterKind.JSON,
        maximum_bytes=128 * 1024,
    )
    runtime = ParameterSpec(
        "runtime",
        ParameterKind.JSON,
        maximum_bytes=32 * 1024,
    )
    return tuple(
        FixedStatement(
            name=_statement_name(operation),
            sql_template=(
                "SELECT * FROM "
                f"{CANONICAL_WRITER_SCHEMA}.{routine}({{{{request}}}}, {{{{runtime}}}})"
            ),
            parameters=(request, runtime),
            returns_rows=True,
            command_prefixes=("SELECT",),
            maximum_rows=1,
        )
        for operation, routine in POSTGRES_ROUTINE_BY_OPERATION.items()
    )


PRODUCTION_STATEMENT_CATALOG = StatementCatalog(_build_production_statements())
PRODUCTION_CATALOG_SHA256 = PRODUCTION_STATEMENT_CATALOG.sha256
EXPECTED_ROUTINE_SIGNATURES = tuple(
    sorted(
        f"{CANONICAL_WRITER_SCHEMA}.{routine}(jsonb, jsonb)"
        for routine in POSTGRES_ROUTINE_BY_OPERATION.values()
    )
)
EXPECTED_HELPER_ROUTINE_SIGNATURES = tuple(
    sorted(
        {
            "canonical_brain._ok(jsonb)",
            "canonical_brain._event_envelope(public.canonical_event_log)",
            (
                "canonical_brain._append_event(text, text, text, jsonb, jsonb, "
                "jsonb, jsonb, text, text, jsonb)"
            ),
            "canonical_brain._plan_head(text)",
            "canonical_brain._fail(text, text)",
            "canonical_brain._sha256_text(text)",
            "canonical_brain._sha256_json(jsonb)",
            "canonical_brain._deterministic_uuid(text)",
            "canonical_brain._keys_valid(jsonb, text[], text[])",
            "canonical_brain._runtime_valid(jsonb)",
            "canonical_brain._contains_forbidden_dm_ref(jsonb)",
            "canonical_brain._case_scope_authorized(text, jsonb, boolean)",
        }
    )
)


class _FixedDatabase(Protocol):
    @property
    def statement_names(self) -> tuple[str, ...]: ...

    @property
    def statement_catalog_sha256(self) -> str: ...

    def query_fixed(
        self,
        statement_name: str,
        parameters: Mapping[str, Any],
    ) -> QueryResult: ...

    def projection_read_transaction(
        self,
    ) -> contextlib.AbstractContextManager[FixedReadOnlyTransaction]: ...


class _ProjectionExportScope:
    """Projection-only facade bound to one attested database snapshot."""

    def __init__(self, transaction: FixedReadOnlyTransaction) -> None:
        self._transaction = transaction

    def projector_read(
        self,
        request: ProjectorReadRequest,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        result = self._transaction.query(
            {
                "request": _json_safe(request),
                "runtime": _json_safe(runtime),
            }
        )
        return _decode_routine_response(result)


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            raise CanonicalWriterError(
                "invalid_backend_request",
                "database routine payload contains a naive datetime",
            )
        return value.astimezone(dt.timezone.utc).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(nested) for nested in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise CanonicalWriterError(
        "invalid_backend_request",
        "database routine payload is not canonical JSON",
    )


def _decode_routine_response(result: QueryResult) -> Mapping[str, Any]:
    if len(result.rows) != 1 or len(result.rows[0]) != 1 or result.rows[0][0] is None:
        raise CanonicalWriterError(
            "invalid_database_response",
            "writer routine must return exactly one JSON object",
        )
    try:
        value = json.loads(result.rows[0][0])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CanonicalWriterError(
            "invalid_database_response",
            "writer routine returned invalid JSON",
        ) from exc
    if not isinstance(value, dict) or type(value.get("ok")) is not bool:
        raise CanonicalWriterError(
            "invalid_database_response",
            "writer routine response envelope is invalid",
        )
    if value["ok"]:
        if set(value) != {"ok", "result"} or not isinstance(
            value.get("result"), Mapping
        ):
            raise CanonicalWriterError(
                "invalid_database_response",
                "writer routine success envelope is invalid",
            )
        return dict(value["result"])
    if set(value) != {"ok", "error"} or not isinstance(value.get("error"), Mapping):
        raise CanonicalWriterError(
            "invalid_database_response",
            "writer routine error envelope is invalid",
        )
    error = value["error"]
    code = error.get("code")
    message = error.get("message")
    if (
        not isinstance(code, str)
        or not _ERROR_CODE.fullmatch(code)
        or not isinstance(message, str)
        or not message
    ):
        raise CanonicalWriterError(
            "invalid_database_response",
            "writer routine error fields are invalid",
        )
    raise CanonicalWriterError(code, message)


class PostgresCanonicalWriterBackend:
    """Semantic backend adapter over the immutable fixed-routine catalog."""

    def __init__(self, database: CanonicalWriterDB | _FixedDatabase) -> None:
        if database.statement_names != PRODUCTION_STATEMENT_CATALOG.names:
            raise ValueError("database statement names do not match production catalog")
        if database.statement_catalog_sha256 != PRODUCTION_CATALOG_SHA256:
            raise ValueError("database statement catalog digest does not match production")
        self._database = database

    def _invoke(
        self,
        operation: CanonicalWriterOperation,
        request: Any,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        result = self._database.query_fixed(
            _statement_name(operation),
            {
                "request": _json_safe(request),
                "runtime": _json_safe(runtime),
            },
        )
        return _decode_routine_response(result)

    def ping(self, runtime: RuntimeContext) -> Mapping[str, Any]:
        return self._invoke(CanonicalWriterOperation.PING, {}, runtime)

    def event_append(
        self,
        request: EventAppendRequest,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        operation = CanonicalWriterOperation.EVENT_APPEND_MODEL
        if request.event_type == "task.plan.updated":
            operation = CanonicalWriterOperation.PLAN_TRANSITION
        elif request.event_type == "task.verification.recorded":
            operation = CanonicalWriterOperation.VERIFICATION_APPEND
        return self._invoke(operation, request, runtime)

    def query(
        self,
        request: QueryRequest,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        return self._invoke(CanonicalWriterOperation.CASE_QUERY, request, runtime)

    def plan_active_match(
        self,
        request: PlanActiveMatchRequest,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        return self._invoke(CanonicalWriterOperation.PLAN_ACTIVE_MATCH, request, runtime)

    def routeback_context(
        self,
        request: RouteBackContextRequest,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        return self._invoke(CanonicalWriterOperation.ROUTEBACK_CONTEXT, request, runtime)

    def routeback_authorize(
        self,
        request: RouteBackAuthorizeRequest,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        return self._invoke(CanonicalWriterOperation.ROUTEBACK_CLAIM, request, runtime)

    def routeback_recover(
        self,
        request: RouteBackRecoveryRequest,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        return self._invoke(CanonicalWriterOperation.ROUTEBACK_RECOVER, request, runtime)

    def routeback_terminal(
        self,
        request: RouteBackTerminalRequest,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        if request.outcome == "sent":
            operation = CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT
        elif request.outcome == "blocked":
            operation = CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED
        else:
            raise CanonicalWriterError(
                "invalid_backend_request",
                "route-back terminal outcome is invalid",
            )
        if request.preclaim:
            if request.outcome != "blocked" or request.authorization_id:
                raise CanonicalWriterError(
                    "invalid_backend_request",
                    "preclaim blocked request cannot carry an authorization",
                )
            payload: Mapping[str, Any] = {
                "preclaim": True,
                "case_id": request.case_id,
                "target_ref": request.target_ref,
                "message_summary": request.message_summary,
                "source_refs": request.source_refs,
                "idempotency_key": request.idempotency_key,
                "outcome": "blocked",
                "receipt": request.receipt,
                "blocker_reason": request.blocker_reason,
            }
        else:
            payload = {
                "authorization_id": request.authorization_id,
                "outcome": request.outcome,
                "receipt": request.receipt,
                "blocker_reason": request.blocker_reason,
            }
        return self._invoke(operation, payload, runtime)

    def lease_shadow_record(
        self,
        payload: Mapping[str, Any],
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        return self._invoke(
            CanonicalWriterOperation.LEASE_SHADOW_RECORD,
            payload,
            runtime,
        )

    def capability_grant(
        self,
        request: CapabilityGrantRequest,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        return self._invoke(CanonicalWriterOperation.CAPABILITY_GRANT, request, runtime)

    def capability_consume(
        self,
        request: CapabilityConsumeRequest,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        return self._invoke(
            CanonicalWriterOperation.CAPABILITY_CONSUME,
            request,
            runtime,
        )

    def capability_revoke(
        self,
        request: CapabilityRevokeRequest,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        return self._invoke(CanonicalWriterOperation.CAPABILITY_REVOKE, request, runtime)

    def capability_revoke_session(
        self,
        session_key_sha256: str,
        reason: str,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        return self._invoke(
            CanonicalWriterOperation.CAPABILITY_REVOKE_SESSION,
            {"session_key_sha256": session_key_sha256, "reason": reason},
            runtime,
        )

    def projector_read(
        self,
        request: ProjectorReadRequest,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        return self._invoke(
            CanonicalWriterOperation.PROJECTION_READ_EVENTS,
            request,
            runtime,
        )

    @contextlib.contextmanager
    def projection_export_scope(self) -> Iterator[_ProjectionExportScope]:
        """Pin every page of one privileged export to one DB snapshot."""

        transaction_factory = getattr(
            self._database,
            "projection_read_transaction",
            None,
        )
        if not callable(transaction_factory):
            raise CanonicalWriterError(
                "projection_snapshot_unavailable",
                "database backend cannot provide a consistent projection snapshot",
            )
        with transaction_factory() as transaction:
            yield _ProjectionExportScope(transaction)


__all__ = [
    "CANONICAL_WRITER_SCHEMA",
    "CANONICAL_WRITER_ROLE",
    "CANONICAL_WRITER_MIGRATION_OWNER",
    "EXPECTED_HELPER_ROUTINE_SIGNATURES",
    "EXPECTED_ROUTINE_SIGNATURES",
    "POSTGRES_ROUTINE_BY_OPERATION",
    "PRODUCTION_CATALOG_SHA256",
    "PRODUCTION_STATEMENT_CATALOG",
    "PostgresCanonicalWriterBackend",
]

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
import time
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Protocol

from gateway.canonical_canary_bootstrap import (
    CanaryScopeBootstrapRequest,
    CanaryScopePreclaimRetirementRequest,
)
from gateway.canonical_writer_db import (
    CanonicalWriterDB,
    ManagedCloudSQLAdminHBAReceipt,
    FixedReadOnlyTransaction,
    FixedStatement,
    ParameterKind,
    ParameterSpec,
    QueryResult,
    StatementCatalog,
    WriterDBConfig,
    _RETRYABLE_TRANSACTION_ERRORS,
    _open_postgres_session,
    _render_fixed,
    _require_command,
    _rollback_quietly,
    _validate_active_managed_hba_receipt,
    collect_managed_cloudsqladmin_hba_receipt,
)
from gateway.canonical_writer_handlers import (
    CanaryScopeClaimRequest,
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
CANONICAL_CANARY_BOOTSTRAP_ROLE = "canonical_brain_canary_bootstrap"
CANONICAL_CANARY_BOOTSTRAP_LOGIN = "canonical_brain_canary_bootstrap_login"
_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_CANARY_BOOTSTRAP_ATTEMPTS = 3

POSTGRES_ROUTINE_BY_OPERATION: Mapping[CanonicalWriterOperation, str] = (
    MappingProxyType(
        {
            CanonicalWriterOperation.PING: "writer_ping",
            CanonicalWriterOperation.CANARY_SCOPE_CLAIM: "writer_canary_scope_claim",
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
CANARY_BOOTSTRAP_STATEMENT = FixedStatement(
    name="bootstrap_canary_scope_preapprove",
    sql_template=(
        "SELECT * FROM canonical_brain.writer_canary_scope_preapprove("
        "{{request}}, {{runtime}})"
    ),
    parameters=(
        ParameterSpec("request", ParameterKind.JSON, maximum_bytes=128 * 1024),
        ParameterSpec("runtime", ParameterKind.JSON, maximum_bytes=32 * 1024),
    ),
    returns_rows=True,
    command_prefixes=("SELECT",),
    maximum_rows=1,
)
CANARY_PRECLAIM_RETIREMENT_STATEMENT = FixedStatement(
    name="private_canary_scope_preclaim_retire",
    sql_template=(
        "SELECT * FROM canonical_brain."
        "writer_canary_scope_preapproval_retire({{request}}, {{runtime}})"
    ),
    parameters=(
        ParameterSpec("request", ParameterKind.JSON, maximum_bytes=128 * 1024),
        ParameterSpec("runtime", ParameterKind.JSON, maximum_bytes=32 * 1024),
    ),
    returns_rows=True,
    command_prefixes=("SELECT",),
    maximum_rows=1,
)
EXPECTED_ROUTINE_SIGNATURES = tuple(
    sorted(
        {
            *(
                f"{CANONICAL_WRITER_SCHEMA}.{routine}(jsonb, jsonb)"
                for routine in POSTGRES_ROUTINE_BY_OPERATION.values()
            ),
            (
                "canonical_brain."
                "writer_canary_scope_preapproval_retire(jsonb, jsonb)"
            ),
        }
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


class PostgresCanaryScopeBootstrapBackend:
    """Private fixed-call client for the separately provisioned one-shot ACL.

    The SQL routine performs the authority attestation and consumes its own
    schema/function ACL in the same transaction as the durable receipt.  This
    client intentionally has no statement selector and is never attached to
    the Canonical Writer socket dispatcher.
    """

    def __init__(
        self,
        *,
        config: WriterDBConfig,
        expected_hba_receipt: ManagedCloudSQLAdminHBAReceipt | None = None,
        expected_hba_receipt_sha256: str = "",
        _session_factory=None,
        _managed_hba_probe=None,
    ) -> None:
        if config.user != CANONICAL_CANARY_BOOTSTRAP_LOGIN:
            raise ValueError("canary bootstrap database login is not pinned")
        if (expected_hba_receipt is None) != (expected_hba_receipt_sha256 == ""):
            raise ValueError("canary bootstrap HBA receipt and digest must be paired")
        if expected_hba_receipt is not None:
            if (
                expected_hba_receipt.user != config.user
                or expected_hba_receipt.host != config.host
                or expected_hba_receipt.tls_server_name != config.tls_server_name
                or expected_hba_receipt.port != config.port
                or expected_hba_receipt.sha256 != expected_hba_receipt_sha256
            ):
                raise ValueError("canary bootstrap HBA receipt binding is invalid")
        self._config = config
        self._expected_hba_receipt = expected_hba_receipt
        self._session_factory = _session_factory or _open_postgres_session
        self._managed_hba_probe = (
            _managed_hba_probe or collect_managed_cloudsqladmin_hba_receipt
        )

    def preapprove(
        self,
        request: CanaryScopeBootstrapRequest,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        if runtime.platform != "writer_service" or runtime.service_internal is not True:
            raise CanonicalWriterError(
                "service_internal_required",
                "canary preapproval requires the in-process writer bootstrap",
            )
        sql = _render_fixed(
            CANARY_BOOTSTRAP_STATEMENT,
            {"request": _json_safe(request), "runtime": _json_safe(runtime)},
        )
        for attempt in range(_CANARY_BOOTSTRAP_ATTEMPTS):
            expected = self._expected_hba_receipt
            if expected is not None:
                observed = self._managed_hba_probe(self._config)
                _validate_active_managed_hba_receipt(
                    expected,
                    observed,
                    config=self._config,
                    now_unix=int(time.time()),
                    require_expected_fresh=False,
                )
            session = self._session_factory(self._config)
            try:
                _require_command(
                    session,
                    "BEGIN ISOLATION LEVEL SERIALIZABLE",
                    "BEGIN",
                )
                lock = session.query(
                    "SELECT pg_catalog.pg_advisory_xact_lock_shared("
                    "4841739663211427921)",
                    maximum_rows=1,
                )
                if not lock.command_tag.upper().startswith("SELECT") or len(lock.rows) != 1:
                    raise RuntimeError("canary bootstrap deployment lock failed")
                result = session.query(sql, maximum_rows=1)
                if not result.command_tag.upper().startswith("SELECT") or len(result.rows) > 1:
                    raise RuntimeError("canary bootstrap response shape is invalid")
                _require_command(session, "COMMIT", "COMMIT")
                return _decode_routine_response(result)
            except BaseException as exc:
                _rollback_quietly(session)
                if (
                    str(exc) in _RETRYABLE_TRANSACTION_ERRORS
                    and attempt + 1 < _CANARY_BOOTSTRAP_ATTEMPTS
                ):
                    continue
                raise
            finally:
                session.close()
        raise AssertionError("canary bootstrap retry bound is unreachable")


class PostgresCanaryScopePreclaimRetirementBackend:
    """Writer-UID fixed-call boundary for crash-safe preclaim retirement.

    This surface is deliberately absent from the public operation enum and
    socket dispatcher.  It always opens a fresh writer database connection so
    normal shutdown, ExecStopPost, and next-start reconciliation share the
    same durable database state machine.
    """

    def __init__(
        self,
        *,
        config: WriterDBConfig,
        expected_hba_receipt: ManagedCloudSQLAdminHBAReceipt | None = None,
        expected_hba_receipt_sha256: str = "",
        _session_factory=None,
        _managed_hba_probe=None,
    ) -> None:
        if (expected_hba_receipt is None) != (expected_hba_receipt_sha256 == ""):
            raise ValueError("preclaim retirement HBA receipt and digest must be paired")
        if expected_hba_receipt is not None:
            if (
                expected_hba_receipt.user != config.user
                or expected_hba_receipt.host != config.host
                or expected_hba_receipt.tls_server_name != config.tls_server_name
                or expected_hba_receipt.port != config.port
                or expected_hba_receipt.sha256 != expected_hba_receipt_sha256
            ):
                raise ValueError("preclaim retirement HBA receipt binding is invalid")
        self._config = config
        self._expected_hba_receipt = expected_hba_receipt
        self._session_factory = _session_factory or _open_postgres_session
        self._managed_hba_probe = (
            _managed_hba_probe or collect_managed_cloudsqladmin_hba_receipt
        )

    def retire(
        self,
        request: CanaryScopePreclaimRetirementRequest,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        if runtime.platform != "writer_service" or runtime.service_internal is not True:
            raise CanonicalWriterError(
                "service_internal_required",
                "preclaim retirement requires the in-process writer boundary",
            )
        sql = _render_fixed(
            CANARY_PRECLAIM_RETIREMENT_STATEMENT,
            {"request": _json_safe(request), "runtime": _json_safe(runtime)},
        )
        for attempt in range(_CANARY_BOOTSTRAP_ATTEMPTS):
            expected = self._expected_hba_receipt
            if expected is not None:
                observed = self._managed_hba_probe(self._config)
                _validate_active_managed_hba_receipt(
                    expected,
                    observed,
                    config=self._config,
                    now_unix=int(time.time()),
                    require_expected_fresh=False,
                )
            session = self._session_factory(self._config)
            try:
                _require_command(
                    session,
                    "BEGIN ISOLATION LEVEL SERIALIZABLE",
                    "BEGIN",
                )
                lock = session.query(
                    "SELECT pg_catalog.pg_advisory_xact_lock_shared("
                    "4841739663211427921)",
                    maximum_rows=1,
                )
                if not lock.command_tag.upper().startswith("SELECT") or len(lock.rows) != 1:
                    raise RuntimeError("preclaim retirement deployment lock failed")
                result = session.query(sql, maximum_rows=1)
                if not result.command_tag.upper().startswith("SELECT") or len(result.rows) > 1:
                    raise RuntimeError("preclaim retirement response shape is invalid")
                _require_command(session, "COMMIT", "COMMIT")
                return _decode_routine_response(result)
            except BaseException as exc:
                _rollback_quietly(session)
                if (
                    str(exc) in _RETRYABLE_TRANSACTION_ERRORS
                    and attempt + 1 < _CANARY_BOOTSTRAP_ATTEMPTS
                ):
                    continue
                raise
            finally:
                session.close()
        raise AssertionError("preclaim retirement retry bound is unreachable")


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

    def canary_scope_claim(
        self,
        request: CanaryScopeClaimRequest,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        return self._invoke(
            CanonicalWriterOperation.CANARY_SCOPE_CLAIM,
            request,
            runtime,
        )

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
    "CANONICAL_CANARY_BOOTSTRAP_ROLE",
    "CANONICAL_CANARY_BOOTSTRAP_LOGIN",
    "CANARY_BOOTSTRAP_STATEMENT",
    "CANARY_PRECLAIM_RETIREMENT_STATEMENT",
    "EXPECTED_HELPER_ROUTINE_SIGNATURES",
    "EXPECTED_ROUTINE_SIGNATURES",
    "POSTGRES_ROUTINE_BY_OPERATION",
    "PRODUCTION_CATALOG_SHA256",
    "PRODUCTION_STATEMENT_CATALOG",
    "PostgresCanonicalWriterBackend",
    "PostgresCanaryScopeBootstrapBackend",
    "PostgresCanaryScopePreclaimRetirementBackend",
]

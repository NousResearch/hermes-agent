"""SessionDB-owned execution-route attempts and immutable event projection."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent.execution_router import (
    CONTRACT_VERSION,
    MAX_EVENT_PAGE_RECORDS,
    ExecutionKind,
    ExecutionRouteDecisionKind,
    ExecutionRouteDecisionV1,
    ExecutionRouteEventType,
    ExecutionRouteEventV1,
    ExecutionRouteIdentityV1,
    ExecutionRouteRequestV1,
    canonical_json_bytes,
)
from hermes_cli.execution_router_runtime import (
    ExecutionRouteResolution,
    ExecutionRouteResolutionState,
    ExecutionRouterRegistration,
)

if TYPE_CHECKING:  # pragma: no cover
    from sqlite3 import Connection, Row

_PAGE_TOKEN_VERSION = 1
_PAGE_TOKEN_TTL_MS = 300_000
_PAGE_TOKEN_SECRET = os.urandom(32)
_PAGE_TOKEN_FIELDS = frozenset({
    "v", "contract_version", "scope", "session_id", "filters", "cursor", "expires_utc_ms",
})


@dataclass(frozen=True)
class ExecutionRouteEventPageV1:
    events: tuple[ExecutionRouteEventV1, ...]
    next_page_token: str | None


def _token_encode(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(_PAGE_TOKEN_SECRET, body, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(body + signature).decode("ascii").rstrip("=")


def _token_decode(token: str) -> dict[str, Any] | None:
    try:
        if type(token) is not str or re.fullmatch(r"[A-Za-z0-9_-]+", token) is None:
            return None
        padding = "=" * (-len(token) % 4)
        raw = base64.b64decode(
            (token + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
        if base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=") != token:
            return None
        body, signature = raw[:-32], raw[-32:]
        if len(signature) != 32 or not hmac.compare_digest(
            signature, hmac.new(_PAGE_TOKEN_SECRET, body, hashlib.sha256).digest()
        ):
            return None
        payload = json.loads(body)
        return payload if type(payload) is dict else None
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _identity_from_dict(raw: Any) -> ExecutionRouteIdentityV1 | None:
    if raw is None:
        return None
    return ExecutionRouteIdentityV1(
        candidate_id=raw["candidate_id"],
        provider=raw["provider"],
        model=raw["model"],
        reasoning=raw["reasoning"],
    )


def _event_from_json(raw: str) -> ExecutionRouteEventV1:
    data = json.loads(raw)
    data["execution_kind"] = ExecutionKind(data["execution_kind"])
    data["event_type"] = ExecutionRouteEventType(data["event_type"])
    if data["decision_state"] is not None:
        data["decision_state"] = ExecutionRouteDecisionKind(data["decision_state"])
    data["accepted_route"] = _identity_from_dict(data["accepted_route"])
    data["actual_route"] = _identity_from_dict(data["actual_route"])
    return ExecutionRouteEventV1(**data)


def _decision_from_json(raw: str | None) -> ExecutionRouteDecisionV1 | None:
    if raw is None:
        return None
    data = json.loads(raw)
    data["kind"] = ExecutionRouteDecisionKind(data["kind"])
    return ExecutionRouteDecisionV1(**data)


def _resolution_from_row(row: "Row") -> ExecutionRouteResolution | None:
    if row["resolution_state"] is None:
        return None
    accepted = _identity_from_dict(json.loads(row["accepted_route_json"])) if row["accepted_route_json"] else None
    return ExecutionRouteResolution(
        state=ExecutionRouteResolutionState(row["resolution_state"]),
        decision=_decision_from_json(row["decision_json"]),
        accepted_route=accepted,
        reason_code=row["reason_code"],
        reason_text=row["reason_text"],
    )


@dataclass(frozen=True)
class SessionExecutionRouteLifecycle:
    db: Any
    session_id: str

    def get_execution_route_resolution(
        self, request_id: str, attempt_id: str
    ) -> ExecutionRouteResolution | None:
        with self.db._read_ctx() as conn:
            row = conn.execute(
                "SELECT * FROM execution_route_attempts WHERE session_id = ? "
                "AND request_id = ? AND attempt_id = ?",
                (self.session_id, request_id, attempt_id),
            ).fetchone()
        return _resolution_from_row(row) if row is not None else None

    def _append_event(
        self,
        conn: "Connection",
        row: "Row",
        event_type: ExecutionRouteEventType,
        *,
        decision_state: ExecutionRouteDecisionKind | None = None,
        requested_candidate_id: str | None = None,
        accepted_route: ExecutionRouteIdentityV1 | None = None,
        actual_route: ExecutionRouteIdentityV1 | None = None,
        reason_code: str | None = None,
        reason_text: str | None = None,
        terminal_state: str | None = None,
    ) -> ExecutionRouteEventV1:
        existing = conn.execute(
            "SELECT event_json FROM execution_route_events WHERE request_id = ? "
            "AND attempt_id = ? AND event_type = ?",
            (row["request_id"], row["attempt_id"], event_type.value),
        ).fetchone()
        if existing is not None:
            event = _event_from_json(existing["event_json"])
            if (
                event.decision_state != decision_state
                or event.requested_candidate_id != requested_candidate_id
                or event.accepted_route != accepted_route
                or event.actual_route != actual_route
                or event.reason_code != reason_code
                or event.reason_text != reason_text
                or event.terminal_state != terminal_state
            ):
                raise ValueError(f"conflicting duplicate {event_type.value} transition")
            return event
        sequence = conn.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM execution_route_events "
            "WHERE request_id = ? AND attempt_id = ?",
            (row["request_id"], row["attempt_id"]),
        ).fetchone()[0]
        event = ExecutionRouteEventV1(
            contract_version=CONTRACT_VERSION,
            event_id=str(uuid.uuid4()),
            root_id=row["root_id"],
            task_id=row["task_id"],
            execution_id=row["execution_id"],
            attempt_id=row["attempt_id"],
            request_id=row["request_id"],
            previous_attempt_id=row["previous_attempt_id"],
            sequence=sequence,
            timestamp_utc_ms=int(time.time() * 1000),
            execution_kind=ExecutionKind(row["execution_kind"]),
            surface_class=row["surface_class"],
            router_plugin_id=row["router_plugin_id"],
            router_provider_id=row["router_provider_id"],
            router_contract_version=row["router_contract_version"],
            event_type=event_type,
            decision_state=decision_state,
            requested_candidate_id=requested_candidate_id,
            accepted_route=accepted_route,
            actual_route=actual_route,
            reason_code=reason_code,
            reason_text=reason_text,
            terminal_state=terminal_state,
            request_digest=row["request_digest"] if event_type is ExecutionRouteEventType.REQUESTED else None,
            instruction_digest=row["instruction_digest"] if event_type is ExecutionRouteEventType.REQUESTED else None,
            eligibility_revision=row["eligibility_revision"] if event_type is ExecutionRouteEventType.REQUESTED else None,
        )
        conn.execute(
            "INSERT INTO execution_route_events "
            "(event_id, session_id, request_id, attempt_id, sequence, event_type, event_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                self.session_id,
                event.request_id,
                event.attempt_id,
                event.sequence,
                event.event_type.value,
                canonical_json_bytes(event).decode("utf-8"),
            ),
        )
        return event

    def record_execution_route_requested(
        self, request: ExecutionRouteRequestV1, registration: ExecutionRouterRegistration
    ) -> bool:
        descriptor = registration.descriptor

        def write(conn: "Connection") -> bool:
            if conn.execute("SELECT 1 FROM sessions WHERE id = ?", (self.session_id,)).fetchone() is None:
                raise ValueError("owning session does not exist in this profile state.db")
            row = conn.execute(
                "SELECT * FROM execution_route_attempts WHERE request_id = ? AND attempt_id = ?",
                (request.request_id, request.attempt_id),
            ).fetchone()
            created = row is None
            if created:
                conn.execute(
                    "INSERT INTO execution_route_attempts "
                    "(session_id, request_id, attempt_id, root_id, task_id, execution_id, "
                    "previous_attempt_id, execution_kind, surface_class, router_plugin_id, "
                    "router_provider_id, router_contract_version, router_generation, request_digest, "
                    "instruction_digest, eligibility_revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        self.session_id, request.request_id, request.attempt_id, request.root_id,
                        request.task_id, request.execution_id,
                        request.previous_attempt.attempt_id if request.previous_attempt else None,
                        request.execution_kind.value, request.surface_class, descriptor.plugin_id,
                        descriptor.provider_id, descriptor.contract_version, registration.generation,
                        request.request_digest, request.instruction.digest, request.eligibility_revision,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM execution_route_attempts WHERE request_id = ? AND attempt_id = ?",
                    (request.request_id, request.attempt_id),
                ).fetchone()
            elif (
                row["session_id"] != self.session_id
                or row["request_digest"] != request.request_digest
                or row["instruction_digest"] != request.instruction.digest
                or row["eligibility_revision"] != request.eligibility_revision
            ):
                raise ValueError("execution-route request identity is already bound differently")
            self._append_event(conn, row, ExecutionRouteEventType.REQUESTED)
            return created

        return self.db._execute_write(write)

    def record_execution_route_result(
        self,
        request: ExecutionRouteRequestV1,
        result: ExecutionRouteResolution,
        registration: ExecutionRouterRegistration,
    ) -> ExecutionRouteResolution:
        def write(conn: "Connection") -> ExecutionRouteResolution:
            row = conn.execute(
                "SELECT * FROM execution_route_attempts WHERE session_id = ? "
                "AND request_id = ? AND attempt_id = ?",
                (self.session_id, request.request_id, request.attempt_id),
            ).fetchone()
            if row is None:
                raise ValueError("route result has no requested lifecycle row")
            existing = _resolution_from_row(row)
            if existing is not None:
                if existing != result:
                    raise ValueError("execution-route resolution is immutable")
                return existing
            decision_json = (
                canonical_json_bytes(result.decision).decode("utf-8") if result.decision is not None else None
            )
            accepted_json = (
                canonical_json_bytes(result.accepted_route).decode("utf-8")
                if result.accepted_route is not None else None
            )
            conn.execute(
                "UPDATE execution_route_attempts SET resolution_state = ?, decision_json = ?, "
                "accepted_route_json = ?, reason_code = ?, reason_text = ? "
                "WHERE request_id = ? AND attempt_id = ?",
                (
                    result.state.value, decision_json, accepted_json, result.reason_code,
                    result.reason_text, request.request_id, request.attempt_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM execution_route_attempts WHERE request_id = ? AND attempt_id = ?",
                (request.request_id, request.attempt_id),
            ).fetchone()
            decision_kind = (
                result.decision.kind if result.decision is not None
                else ExecutionRouteDecisionKind.ROUTER_ERROR
            )
            if result.state is ExecutionRouteResolutionState.ROUTE:
                self._append_event(
                    conn,
                    row,
                    ExecutionRouteEventType.ACCEPTED,
                    decision_state=ExecutionRouteDecisionKind.ROUTE,
                    requested_candidate_id=result.decision.candidate_id,
                    accepted_route=result.accepted_route,
                    reason_code=result.reason_code,
                    reason_text=result.reason_text,
                )
            elif result.state in {ExecutionRouteResolutionState.STOP, ExecutionRouteResolutionState.ROUTER_ERROR}:
                self._append_event(
                    conn,
                    row,
                    ExecutionRouteEventType.NOT_STARTED,
                    decision_state=decision_kind,
                    reason_code=result.reason_code,
                    reason_text=result.reason_text,
                )
            return result

        return self.db._execute_write(write)

    def _attempt_row(
        self, conn: "Connection", request_id: str, attempt_id: str
    ) -> "Row":
        row = conn.execute(
            "SELECT * FROM execution_route_attempts WHERE session_id = ? "
            "AND request_id = ? AND attempt_id = ?",
            (self.session_id, request_id, attempt_id),
        ).fetchone()
        if row is None:
            raise ValueError("execution-route attempt does not belong to session")
        return row

    def record_started(
        self,
        actual_route: ExecutionRouteIdentityV1,
        *,
        request_id: str,
        attempt_id: str,
    ) -> ExecutionRouteEventV1:
        if type(actual_route) is not ExecutionRouteIdentityV1:
            raise TypeError("actual_route must be exact ExecutionRouteIdentityV1")

        def write(conn: "Connection") -> ExecutionRouteEventV1:
            row = self._attempt_row(conn, request_id, attempt_id)
            if row["resolution_state"] not in {
                ExecutionRouteResolutionState.ROUTE.value,
                ExecutionRouteResolutionState.PASS_THROUGH.value,
            }:
                raise ValueError("route_started requires route or pass-through resolution")
            decision = _decision_from_json(row["decision_json"])
            accepted = _identity_from_dict(json.loads(row["accepted_route_json"])) if row["accepted_route_json"] else None
            if accepted is not None and actual_route != accepted:
                raise ValueError("actual route does not equal the immutable accepted route")
            return self._append_event(
                conn,
                row,
                ExecutionRouteEventType.STARTED,
                decision_state=decision.kind if decision else ExecutionRouteDecisionKind.PASS_THROUGH,
                requested_candidate_id=decision.candidate_id if decision else None,
                accepted_route=accepted,
                actual_route=actual_route,
                reason_code=row["reason_code"],
                reason_text=row["reason_text"],
            )

        return self.db._execute_write(write)

    def record_not_started(
        self,
        reason_code: str,
        reason_text: str | None = None,
        *,
        request_id: str,
        attempt_id: str,
    ) -> ExecutionRouteEventV1:
        def write(conn: "Connection") -> ExecutionRouteEventV1:
            row = self._attempt_row(conn, request_id, attempt_id)
            if conn.execute(
                "SELECT 1 FROM execution_route_events WHERE request_id = ? AND attempt_id = ? "
                "AND event_type = ?",
                (row["request_id"], row["attempt_id"], ExecutionRouteEventType.STARTED.value),
            ).fetchone():
                raise ValueError("started attempt cannot become not-started")
            decision = _decision_from_json(row["decision_json"])
            accepted = (
                _identity_from_dict(json.loads(row["accepted_route_json"]))
                if row["accepted_route_json"] else None
            )
            return self._append_event(
                conn,
                row,
                ExecutionRouteEventType.NOT_STARTED,
                decision_state=decision.kind if decision else ExecutionRouteDecisionKind.ROUTER_ERROR,
                requested_candidate_id=decision.candidate_id if decision else None,
                accepted_route=accepted,
                reason_code=reason_code,
                reason_text=reason_text,
            )

        return self.db._execute_write(write)

    def record_finished(
        self,
        terminal_state: str,
        *,
        request_id: str,
        attempt_id: str,
    ) -> ExecutionRouteEventV1:
        def write(conn: "Connection") -> ExecutionRouteEventV1:
            row = self._attempt_row(conn, request_id, attempt_id)
            started = conn.execute(
                "SELECT event_json FROM execution_route_events WHERE request_id = ? AND attempt_id = ? "
                "AND event_type = ?",
                (row["request_id"], row["attempt_id"], ExecutionRouteEventType.STARTED.value),
            ).fetchone()
            if started is None:
                raise ValueError("route_finished requires route_started")
            start_event = _event_from_json(started["event_json"])
            return self._append_event(
                conn,
                row,
                ExecutionRouteEventType.FINISHED,
                decision_state=start_event.decision_state,
                requested_candidate_id=start_event.requested_candidate_id,
                accepted_route=start_event.accepted_route,
                actual_route=start_event.actual_route,
                reason_code=start_event.reason_code,
                reason_text=start_event.reason_text,
                terminal_state=terminal_state,
            )

        return self.db._execute_write(write)

    def read_events(
        self,
        *,
        attempt_id: str | None = None,
        request_id: str | None = None,
        execution_kind: ExecutionKind | None = None,
        event_type: ExecutionRouteEventType | None = None,
        after_sequence: int | None = None,
        before_sequence: int | None = None,
        limit: int = MAX_EVENT_PAGE_RECORDS,
    ) -> tuple[ExecutionRouteEventV1, ...]:
        if type(limit) is not int or not 1 <= limit <= MAX_EVENT_PAGE_RECORDS:
            raise ValueError(f"limit must be in 1..{MAX_EVENT_PAGE_RECORDS}")
        for name, value in (("after_sequence", after_sequence), ("before_sequence", before_sequence)):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if (after_sequence is not None or before_sequence is not None) and (
            request_id is None or attempt_id is None
        ):
            raise ValueError("sequence bounds require exact request_id and attempt_id")
        if request_id is not None and attempt_id is not None:
            with self.db._read_ctx() as conn:
                authoritative = conn.execute(
                    "SELECT 1 FROM execution_route_attempts WHERE session_id = ? AND request_id = ? AND attempt_id = ?",
                    (self.session_id, request_id, attempt_id),
                ).fetchone()
            if authoritative is None:
                raise ValueError("request_id and attempt_id do not identify an authoritative attempt")
        clauses = ["e.session_id = ?"]
        params: list[Any] = [self.session_id]
        for column, value in (("e.attempt_id", attempt_id), ("e.request_id", request_id)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if execution_kind is not None:
            clauses.append("a.execution_kind = ?")
            params.append(execution_kind.value)
        if event_type is not None:
            clauses.append("e.event_type = ?")
            params.append(event_type.value)
        if after_sequence is not None:
            clauses.append("e.sequence > ?")
            params.append(after_sequence)
        if before_sequence is not None:
            clauses.append("e.sequence < ?")
            params.append(before_sequence)
        params.append(limit)
        sql = (
            "SELECT e.event_json FROM execution_route_events e "
            "JOIN execution_route_attempts a ON a.request_id = e.request_id AND a.attempt_id = e.attempt_id "
            f"WHERE {' AND '.join(clauses)} ORDER BY "
            + ("e.sequence, e.event_id" if request_id is not None and attempt_id is not None
               else "CAST(json_extract(e.event_json, '$.timestamp_utc_ms') AS INTEGER), e.event_id")
            + " LIMIT ?"
        )
        with self.db._read_ctx() as conn:
            rows = conn.execute(sql, params).fetchall()
        return tuple(_event_from_json(row["event_json"]) for row in rows)

    def read_event_page(
        self,
        *,
        execution_kind: ExecutionKind | None = None,
        event_type: ExecutionRouteEventType | None = None,
        limit: int = MAX_EVENT_PAGE_RECORDS,
        page_token: str | None = None,
        after_sequence: int | None = None,
    ) -> ExecutionRouteEventPageV1:
        if after_sequence is not None:
            raise ValueError("page-token mode is incompatible with sequence bounds")
        if type(limit) is not int or not 1 <= limit <= MAX_EVENT_PAGE_RECORDS:
            raise ValueError(f"limit must be in 1..{MAX_EVENT_PAGE_RECORDS}")
        filters: dict[str, str | None] = {
            "execution_kind": execution_kind.value if execution_kind is not None else None,
            "event_type": event_type.value if event_type is not None else None,
        }
        scope = hashlib.sha256(
            os.fspath(self.db.db_path.resolve()).encode("utf-8")
        ).hexdigest()
        cursor: tuple[int, str] | None = None
        if page_token is not None:
            if execution_kind is not None or event_type is not None:
                return ExecutionRouteEventPageV1((), None)
            payload = _token_decode(page_token)
            now_ms = int(time.time() * 1000)
            token_filters = payload.get("filters") if payload is not None else None
            if (
                payload is None
                or frozenset(payload) != _PAGE_TOKEN_FIELDS
                or payload.get("v") != _PAGE_TOKEN_VERSION
                or payload.get("contract_version") != CONTRACT_VERSION
                or payload.get("scope") != scope
                or payload.get("session_id") != self.session_id
                or type(token_filters) is not dict
                or frozenset(token_filters) != {"execution_kind", "event_type"}
                or token_filters["execution_kind"] not in (
                    None, *(kind.value for kind in ExecutionKind)
                )
                or token_filters["event_type"] not in (
                    None, *(kind.value for kind in ExecutionRouteEventType)
                )
                or type(payload.get("expires_utc_ms")) is not int
                or payload["expires_utc_ms"] < now_ms
                or type(payload.get("cursor")) is not list
                or len(payload["cursor"]) != 2
                or type(payload["cursor"][0]) is not int
                or type(payload["cursor"][1]) is not str
            ):
                return ExecutionRouteEventPageV1((), None)
            filters = token_filters
            cursor = (payload["cursor"][0], payload["cursor"][1])
        clauses = ["e.session_id = ?"]
        params: list[Any] = [self.session_id]
        if filters["execution_kind"] is not None:
            clauses.append("a.execution_kind = ?")
            params.append(filters["execution_kind"])
        if filters["event_type"] is not None:
            clauses.append("e.event_type = ?")
            params.append(filters["event_type"])
        timestamp_sql = "CAST(json_extract(e.event_json, '$.timestamp_utc_ms') AS INTEGER)"
        if cursor is not None:
            clauses.append(f"({timestamp_sql} > ? OR ({timestamp_sql} = ? AND e.event_id > ?))")
            params.extend((cursor[0], cursor[0], cursor[1]))
        params.append(limit + 1)
        sql = (
            "SELECT e.event_json FROM execution_route_events e "
            "JOIN execution_route_attempts a ON a.request_id = e.request_id AND a.attempt_id = e.attempt_id "
            f"WHERE {' AND '.join(clauses)} ORDER BY {timestamp_sql}, e.event_id LIMIT ?"
        )
        with self.db._read_ctx() as conn:
            rows = conn.execute(sql, params).fetchall()
        events = tuple(_event_from_json(row["event_json"]) for row in rows[:limit])
        next_token = None
        if len(rows) > limit and events:
            last = events[-1]
            next_token = _token_encode({
                "v": _PAGE_TOKEN_VERSION,
                "contract_version": CONTRACT_VERSION,
                "scope": scope,
                "session_id": self.session_id,
                "filters": filters,
                "cursor": [last.timestamp_utc_ms, last.event_id],
                "expires_utc_ms": int(time.time() * 1000) + _PAGE_TOKEN_TTL_MS,
            })
        return ExecutionRouteEventPageV1(events, next_token)


class SessionExecutionRouterMixin:
    def execution_route_lifecycle(self, session_id: str) -> SessionExecutionRouteLifecycle:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id is required")
        return SessionExecutionRouteLifecycle(self, session_id)

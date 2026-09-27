"""Narrow durable gateway route authority.

This module deliberately owns only key-to-current-session routing.  It does not
persist gateway transcripts, delivery, or lifecycle side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SessionRoute:
    tenant_namespace: str
    session_key: str
    session_id: str
    generation: int
    flags: dict[str, bool]
    metadata: dict[str, Any]
    created_at: float
    updated_at: float


@runtime_checkable
class SessionRouteStore(Protocol):
    def get_or_create_route(self, *, session_key: str, session_id: str, metadata: dict[str, Any]) -> SessionRoute: ...
    def lookup_by_key(self, session_key: str) -> SessionRoute | None: ...
    def lookup_by_session_id(self, session_id: str) -> SessionRoute | None: ...
    def switch_route(self, *, session_key: str, session_id: str, expected_session_id: str, expected_generation: int, metadata: dict[str, Any], flags: dict[str, bool] | None = None) -> SessionRoute | None: ...
    def delete_or_repair_route(self, *, session_key: str, expected_session_id: str, expected_generation: int, replacement_session_id: str | None = None, metadata: dict[str, Any] | None = None) -> SessionRoute | bool | None: ...


class PostgreSQLSessionRouteStore:
    """Adapter over the tenant-bound PostgreSQL state store route primitives."""

    def __init__(self, state_store: Any, *, tenant_namespace: str) -> None:
        self._state_store = state_store
        self._tenant_namespace = tenant_namespace

    def _route(self, row: dict[str, Any] | None) -> SessionRoute | None:
        if row is None:
            return None
        return SessionRoute(
            tenant_namespace=str(row["tenant_namespace"]), session_key=str(row["session_key"]),
            session_id=str(row["session_id"]), generation=int(row["generation"]),
            flags=dict(row.get("flags") or {}), metadata=dict(row.get("metadata") or {}),
            created_at=float(row["created_at"]), updated_at=float(row["updated_at"]),
        )

    def get_or_create_route(self, *, session_key: str, session_id: str, metadata: dict[str, Any]) -> SessionRoute:
        return self._route(self._state_store.get_or_create_gateway_session_route(
            self._tenant_namespace, session_key, session_id, metadata))  # type: ignore[return-value]

    def lookup_by_key(self, session_key: str) -> SessionRoute | None:
        return self._route(self._state_store.get_gateway_session_route_by_key(self._tenant_namespace, session_key))

    def lookup_by_session_id(self, session_id: str) -> SessionRoute | None:
        return self._route(self._state_store.get_gateway_session_route_by_session_id(self._tenant_namespace, session_id))

    def switch_route(self, *, session_key: str, session_id: str, expected_session_id: str, expected_generation: int, metadata: dict[str, Any], flags: dict[str, bool] | None = None) -> SessionRoute | None:
        return self._route(self._state_store.switch_gateway_session_route(
            self._tenant_namespace, session_key, session_id, expected_session_id, expected_generation,
            metadata, flags or {}))

    def delete_or_repair_route(self, *, session_key: str, expected_session_id: str, expected_generation: int, replacement_session_id: str | None = None, metadata: dict[str, Any] | None = None) -> SessionRoute | bool | None:
        row = self._state_store.delete_or_repair_gateway_session_route(
            self._tenant_namespace, session_key, expected_session_id, expected_generation,
            replacement_session_id, metadata or {})
        return self._route(row) if isinstance(row, dict) else row

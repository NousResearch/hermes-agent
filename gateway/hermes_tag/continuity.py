"""Cross-surface continuity with optimistic writes and replay fences."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import IdentityConflict, ReplayDetected, StaleWriteError, StorageError
from .ledger import HermesTagLedger
from .model import (
    ContinuityEnvelope,
    ContinuityMode,
    Principal,
    SurfaceRef,
    canonical_json,
    new_id,
    utc_text,
)


@dataclass(frozen=True, slots=True)
class ContinuityRecord:
    continuity_id: str
    principal_id: str
    mode: ContinuityMode
    project_id: str | None
    version: int
    objective: str
    state: Mapping[str, Any]
    created_at: str
    updated_at: str


class ContinuityStore:
    def __init__(self, ledger: HermesTagLedger) -> None:
        self.ledger = ledger

    @staticmethod
    def _record(row: sqlite3.Row) -> ContinuityRecord:
        return ContinuityRecord(
            continuity_id=row["continuity_id"],
            principal_id=row["principal_id"],
            mode=ContinuityMode(row["mode"]),
            project_id=row["project_id"],
            version=int(row["version"]),
            objective=row["objective"],
            state=json.loads(row["state_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get(self, continuity_id: str) -> ContinuityRecord:
        connection = self.ledger.connection()
        try:
            row = connection.execute(
                "SELECT * FROM hermes_tag_continuities WHERE continuity_id=?",
                (continuity_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise StorageError("unknown continuity")
        return self._record(row)

    def create(
        self,
        principal: Principal,
        *,
        mode: ContinuityMode,
        project_id: str | None = None,
        objective: str = "",
        state: Mapping[str, Any] | None = None,
        continuity_id: str | None = None,
    ) -> ContinuityRecord:
        mode = ContinuityMode(mode)
        if mode == ContinuityMode.PROJECT and not project_id:
            raise ValueError("project continuity requires project_id")
        if mode != ContinuityMode.PROJECT and project_id and mode != ContinuityMode.EXPLICIT:
            raise ValueError("project_id is only valid for project or explicit continuity")
        identifier = continuity_id or new_id("continuity")
        timestamp = utc_text()
        payload = state or {}
        payload_json = canonical_json(payload)
        with self.ledger.transaction() as connection:
            connection.execute(
                """
                INSERT INTO hermes_tag_continuities(
                    continuity_id, principal_id, mode, project_id, version,
                    objective, state_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    principal.principal_id,
                    mode.value,
                    project_id,
                    objective,
                    payload_json,
                    timestamp,
                    timestamp,
                ),
            )
        self.ledger.append_receipt(
            event_id=new_id("event"),
            kind="continuity.created",
            payload={
                "continuity_id": identifier,
                "principal_id": principal.principal_id,
                "mode": mode.value,
                "project_id": project_id,
            },
        )
        return self.get(identifier)

    def binding(self, surface: SurfaceRef) -> ContinuityRecord | None:
        connection = self.ledger.connection()
        try:
            row = connection.execute(
                """
                SELECT c.*
                FROM hermes_tag_surface_bindings b
                JOIN hermes_tag_continuities c
                  ON c.continuity_id = b.continuity_id
                WHERE b.surface_key=?
                """,
                (surface.key,),
            ).fetchone()
        finally:
            connection.close()
        return self._record(row) if row is not None else None

    def bind_surface(
        self,
        surface: SurfaceRef,
        continuity_id: str,
        *,
        principal_id: str,
        allow_rebind: bool = False,
    ) -> ContinuityRecord:
        continuity = self.get(continuity_id)
        if continuity.principal_id != principal_id:
            raise IdentityConflict("surface and continuity principal differ")
        timestamp = utc_text()
        with self.ledger.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM hermes_tag_surface_bindings WHERE surface_key=?",
                (surface.key,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["continuity_id"] == continuity_id
                    and existing["principal_id"] == principal_id
                ):
                    return continuity
                if not allow_rebind:
                    raise IdentityConflict("surface is already bound to another continuity")
                if existing["principal_id"] != principal_id:
                    raise IdentityConflict("surface rebind cannot cross principals")
                connection.execute(
                    "DELETE FROM hermes_tag_surface_bindings WHERE surface_key=?",
                    (surface.key,),
                )
            connection.execute(
                """
                INSERT INTO hermes_tag_surface_bindings(
                    surface_key, continuity_id, principal_id, platform, profile,
                    scope_id, chat_id, thread_id, bound_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    surface.key,
                    continuity_id,
                    principal_id,
                    surface.platform,
                    surface.profile,
                    surface.scope_id,
                    surface.chat_id,
                    surface.thread_id,
                    timestamp,
                ),
            )
        self.ledger.append_receipt(
            event_id=new_id("event"),
            kind="continuity.surface_bound",
            payload={
                "surface_key": surface.key,
                "continuity_id": continuity_id,
                "principal_id": principal_id,
            },
        )
        return continuity

    def resolve_or_create(
        self,
        principal: Principal,
        surface: SurfaceRef,
        *,
        mode: ContinuityMode,
        project_id: str | None = None,
        explicit_id: str | None = None,
    ) -> ContinuityRecord:
        existing = self.binding(surface)
        if existing is not None:
            if existing.principal_id != principal.principal_id:
                raise IdentityConflict("surface binding belongs to another principal")
            return existing

        mode = ContinuityMode(mode)
        if mode == ContinuityMode.EXPLICIT:
            if not explicit_id:
                raise ValueError("explicit continuity mode requires explicit_id")
            continuity = self.get(explicit_id)
            if continuity.principal_id != principal.principal_id:
                raise IdentityConflict("explicit continuity belongs to another principal")
        else:
            continuity = self._find_reusable(
                principal.principal_id,
                surface,
                mode=mode,
                project_id=project_id,
            )
            if continuity is None:
                continuity = self.create(
                    principal,
                    mode=mode,
                    project_id=project_id,
                )
        try:
            return self.bind_surface(
                surface,
                continuity.continuity_id,
                principal_id=principal.principal_id,
            )
        except sqlite3.IntegrityError:
            bound = self.binding(surface)
            if bound is None:
                raise
            return bound

    def _find_reusable(
        self,
        principal_id: str,
        surface: SurfaceRef,
        *,
        mode: ContinuityMode,
        project_id: str | None,
    ) -> ContinuityRecord | None:
        if mode == ContinuityMode.ISOLATED:
            return None
        connection = self.ledger.connection()
        try:
            if mode == ContinuityMode.PRINCIPAL:
                row = connection.execute(
                    """
                    SELECT * FROM hermes_tag_continuities
                    WHERE principal_id=? AND mode=? AND project_id IS NULL
                    ORDER BY created_at LIMIT 1
                    """,
                    (principal_id, mode.value),
                ).fetchone()
            elif mode == ContinuityMode.PROJECT:
                if not project_id:
                    raise ValueError("project continuity requires project_id")
                row = connection.execute(
                    """
                    SELECT * FROM hermes_tag_continuities
                    WHERE principal_id=? AND mode=? AND project_id=?
                    ORDER BY created_at LIMIT 1
                    """,
                    (principal_id, mode.value, project_id),
                ).fetchone()
            elif mode == ContinuityMode.WORKSPACE:
                row = connection.execute(
                    """
                    SELECT c.*
                    FROM hermes_tag_continuities c
                    JOIN hermes_tag_surface_bindings b
                      ON b.continuity_id = c.continuity_id
                    WHERE c.principal_id=? AND c.mode=?
                      AND b.platform=? AND b.profile=? AND b.scope_id=?
                    ORDER BY c.created_at LIMIT 1
                    """,
                    (
                        principal_id,
                        mode.value,
                        surface.platform,
                        surface.profile,
                        surface.scope_id,
                    ),
                ).fetchone()
            else:
                row = None
        finally:
            connection.close()
        return self._record(row) if row is not None else None

    def update_checkpoint(
        self,
        continuity_id: str,
        *,
        expected_version: int,
        payload: Mapping[str, Any],
        objective: str | None = None,
    ) -> ContinuityRecord:
        payload_json = canonical_json(payload)
        timestamp = utc_text()
        checkpoint_id = new_id("checkpoint")
        with self.ledger.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM hermes_tag_continuities WHERE continuity_id=?",
                (continuity_id,),
            ).fetchone()
            if row is None:
                raise StorageError("unknown continuity")
            current_version = int(row["version"])
            if current_version != expected_version:
                raise StaleWriteError(
                    f"expected continuity version {expected_version}, found {current_version}"
                )
            new_version = current_version + 1
            connection.execute(
                """
                UPDATE hermes_tag_continuities
                SET version=?, objective=?, state_json=?, updated_at=?
                WHERE continuity_id=? AND version=?
                """,
                (
                    new_version,
                    row["objective"] if objective is None else objective,
                    payload_json,
                    timestamp,
                    continuity_id,
                    expected_version,
                ),
            )
            connection.execute(
                """
                INSERT INTO hermes_tag_checkpoints(
                    checkpoint_id, continuity_id, version, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (checkpoint_id, continuity_id, new_version, payload_json, timestamp),
            )
        self.ledger.append_receipt(
            event_id=new_id("event"),
            kind="continuity.checkpoint_updated",
            payload={
                "continuity_id": continuity_id,
                "version": new_version,
                "checkpoint_id": checkpoint_id,
            },
        )
        return self.get(continuity_id)

    def accept_envelope(
        self,
        envelope: ContinuityEnvelope,
        *,
        max_hops: int,
    ) -> None:
        continuity = self.get(envelope.continuity_id)
        origin_binding = self.binding(envelope.origin)
        if (
            origin_binding is None
            or origin_binding.continuity_id != continuity.continuity_id
            or origin_binding.principal_id != continuity.principal_id
        ):
            raise IdentityConflict(
                "continuity envelope origin is not bound to this continuity"
            )
        if envelope.hop_count > max_hops:
            raise ReplayDetected("continuity propagation hop limit exceeded")
        if envelope.origin.platform in envelope.propagation_path:
            raise ReplayDetected("continuity envelope would cycle to its origin")
        self.ledger.register_replay_fingerprint(
            fingerprint=envelope.fingerprint,
            event_id=envelope.event_id,
            continuity_id=continuity.continuity_id,
        )
        self.ledger.append_receipt(
            event_id=envelope.event_id,
            kind="continuity.envelope_admitted",
            payload={
                "continuity_id": envelope.continuity_id,
                "fingerprint": envelope.fingerprint,
                "hop_count": envelope.hop_count,
            },
        )

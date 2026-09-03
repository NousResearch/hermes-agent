"""SQLite WAL store for Agent Computers.

auth_blob is persisted but never mapped onto public dataclasses or APIs.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import (
    AgentComputer,
    AuditEvent,
    BrowserIdentity,
    Checkpoint,
    CheckpointStatus,
    ControlAuthority,
    ControlLease,
    Controller,
    LeaseStatus,
    Lifecycle,
    TakeoverToken,
)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS computers (
    id TEXT PRIMARY KEY,
    agent_profile_id TEXT NOT NULL UNIQUE,
    backend TEXT NOT NULL,
    persistence_ref TEXT NOT NULL,
    lifecycle TEXT NOT NULL,
    control_authority TEXT NOT NULL,
    fencing_epoch INTEGER NOT NULL DEFAULT 0,
    resume_observe_required INTEGER NOT NULL DEFAULT 0,
    active_browser_identity_id TEXT,
    workspace_url TEXT NOT NULL DEFAULT '',
    workspace_title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS browser_identities (
    id TEXT PRIMARY KEY,
    profile_ref TEXT NOT NULL,
    ownership_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0,
    lock_computer_id TEXT,
    lock_holder TEXT,
    auth_blob TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    computer_id TEXT NOT NULL,
    controller TEXT NOT NULL,
    fencing_epoch INTEGER NOT NULL,
    acquired_at TEXT NOT NULL,
    expires_at TEXT,
    status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_leases_computer ON leases(computer_id, status);

CREATE TABLE IF NOT EXISTS takeover_tokens (
    token_id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL UNIQUE,
    computer_id TEXT NOT NULL,
    owner_principal TEXT NOT NULL,
    fencing_epoch INTEGER NOT NULL,
    expires_at TEXT NOT NULL,
    consumed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    computer_id TEXT NOT NULL,
    action_class TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    computer_id TEXT,
    actor TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _row_computer(row: sqlite3.Row) -> AgentComputer:
    return AgentComputer(
        id=row["id"],
        agent_profile_id=row["agent_profile_id"],
        backend=row["backend"],
        persistence_ref=row["persistence_ref"],
        lifecycle=Lifecycle(row["lifecycle"]),
        control_authority=ControlAuthority(row["control_authority"]),
        fencing_epoch=int(row["fencing_epoch"]),
        resume_observe_required=bool(row["resume_observe_required"]),
        active_browser_identity_id=row["active_browser_identity_id"],
        workspace_url=row["workspace_url"] or "",
        workspace_title=row["workspace_title"] or "",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_identity(row: sqlite3.Row) -> BrowserIdentity:
    return BrowserIdentity(
        id=row["id"],
        profile_ref=row["profile_ref"],
        ownership=list(json.loads(row["ownership_json"] or "[]")),
        metadata=dict(json.loads(row["metadata_json"] or "{}")),
        created_at=row["created_at"],
        revoked=bool(row["revoked"]),
        lock_computer_id=row["lock_computer_id"],
        lock_holder=row["lock_holder"],
    )


def _row_lease(row: sqlite3.Row) -> ControlLease:
    return ControlLease(
        lease_id=row["lease_id"],
        computer_id=row["computer_id"],
        controller=Controller(row["controller"]),
        fencing_epoch=int(row["fencing_epoch"]),
        acquired_at=row["acquired_at"],
        expires_at=row["expires_at"],
        status=LeaseStatus(row["status"]),
    )


class AgentComputerStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert_computer(self, computer: AgentComputer) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO computers (
                    id, agent_profile_id, backend, persistence_ref, lifecycle,
                    control_authority, fencing_epoch, resume_observe_required,
                    active_browser_identity_id, workspace_url, workspace_title,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    agent_profile_id=excluded.agent_profile_id,
                    backend=excluded.backend,
                    persistence_ref=excluded.persistence_ref,
                    lifecycle=excluded.lifecycle,
                    control_authority=excluded.control_authority,
                    fencing_epoch=excluded.fencing_epoch,
                    resume_observe_required=excluded.resume_observe_required,
                    active_browser_identity_id=excluded.active_browser_identity_id,
                    workspace_url=excluded.workspace_url,
                    workspace_title=excluded.workspace_title,
                    updated_at=excluded.updated_at
                """,
                (
                    computer.id,
                    computer.agent_profile_id,
                    computer.backend,
                    computer.persistence_ref,
                    computer.lifecycle.value,
                    computer.control_authority.value,
                    computer.fencing_epoch,
                    int(computer.resume_observe_required),
                    computer.active_browser_identity_id,
                    computer.workspace_url,
                    computer.workspace_title,
                    computer.created_at,
                    computer.updated_at,
                ),
            )
            self._conn.commit()

    def get_computer(self, computer_id: str) -> AgentComputer | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM computers WHERE id = ?", (computer_id,)
            ).fetchone()
            return _row_computer(row) if row else None

    def get_computer_by_profile(self, profile_id: str) -> AgentComputer | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM computers WHERE agent_profile_id = ?", (profile_id,)
            ).fetchone()
            return _row_computer(row) if row else None

    def list_computers(self) -> list[AgentComputer]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM computers ORDER BY created_at"
            ).fetchall()
            return [_row_computer(r) for r in rows]

    def upsert_identity(self, identity: BrowserIdentity) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO browser_identities (
                    id, profile_ref, ownership_json, metadata_json, created_at,
                    revoked, lock_computer_id, lock_holder, auth_blob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(
                    (SELECT auth_blob FROM browser_identities WHERE id = ?), ''
                ))
                ON CONFLICT(id) DO UPDATE SET
                    profile_ref=excluded.profile_ref,
                    ownership_json=excluded.ownership_json,
                    metadata_json=excluded.metadata_json,
                    revoked=excluded.revoked,
                    lock_computer_id=excluded.lock_computer_id,
                    lock_holder=excluded.lock_holder
                """,
                (
                    identity.id,
                    identity.profile_ref,
                    json.dumps(list(identity.ownership)),
                    json.dumps(dict(identity.metadata)),
                    identity.created_at,
                    int(identity.revoked),
                    identity.lock_computer_id,
                    identity.lock_holder,
                    identity.id,
                ),
            )
            self._conn.commit()

    def try_lock_identity(
        self, identity_id: str, computer_id: str, holder: str
    ) -> BrowserIdentity | None:
        """Atomic exclusive mount. Succeeds only if unlocked or already ours."""
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE browser_identities
                SET lock_computer_id = ?, lock_holder = ?
                WHERE id = ? AND revoked = 0
                  AND (lock_computer_id IS NULL OR lock_computer_id = ?)
                """,
                (computer_id, holder, identity_id, computer_id),
            )
            self._conn.commit()
            if cur.rowcount != 1:
                return None
            return self.get_identity(identity_id)

    def consume_checkpoint(self, checkpoint_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE checkpoints SET status = ? WHERE id = ?",
                (CheckpointStatus.CONSUMED.value, checkpoint_id),
            )
            self._conn.commit()

    def get_identity(self, identity_id: str) -> BrowserIdentity | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, profile_ref, ownership_json, metadata_json, created_at, "
                "revoked, lock_computer_id, lock_holder FROM browser_identities WHERE id = ?",
                (identity_id,),
            ).fetchone()
            return _row_identity(row) if row else None

    def list_identities(self) -> list[BrowserIdentity]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, profile_ref, ownership_json, metadata_json, created_at, "
                "revoked, lock_computer_id, lock_holder FROM browser_identities "
                "ORDER BY created_at"
            ).fetchall()
            return [_row_identity(r) for r in rows]

    def upsert_lease(self, lease: ControlLease) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO leases (
                    lease_id, computer_id, controller, fencing_epoch,
                    acquired_at, expires_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(lease_id) DO UPDATE SET
                    status=excluded.status,
                    expires_at=excluded.expires_at,
                    fencing_epoch=excluded.fencing_epoch
                """,
                (
                    lease.lease_id,
                    lease.computer_id,
                    lease.controller.value,
                    lease.fencing_epoch,
                    lease.acquired_at,
                    lease.expires_at,
                    lease.status.value,
                ),
            )
            self._conn.commit()

    def get_lease(self, lease_id: str) -> ControlLease | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM leases WHERE lease_id = ?", (lease_id,)
            ).fetchone()
            return _row_lease(row) if row else None

    def active_lease_for_computer(self, computer_id: str) -> ControlLease | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM leases WHERE computer_id = ? AND status = ? "
                "ORDER BY fencing_epoch DESC LIMIT 1",
                (computer_id, LeaseStatus.ACTIVE.value),
            ).fetchone()
            return _row_lease(row) if row else None

    def revoke_leases(self, computer_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE leases SET status = ? WHERE computer_id = ? AND status = ?",
                (LeaseStatus.REVOKED.value, computer_id, LeaseStatus.ACTIVE.value),
            )
            self._conn.commit()

    def insert_token(self, token: TakeoverToken) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO takeover_tokens (
                    token_id, token_hash, computer_id, owner_principal,
                    fencing_epoch, expires_at, consumed
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token.token_id,
                    token.token_hash,
                    token.computer_id,
                    token.owner_principal,
                    token.fencing_epoch,
                    token.expires_at,
                    int(token.consumed),
                ),
            )
            self._conn.commit()

    def get_token_by_hash(self, token_hash: str) -> TakeoverToken | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM takeover_tokens WHERE token_hash = ?", (token_hash,)
            ).fetchone()
            if not row:
                return None
            return TakeoverToken(
                token_id=row["token_id"],
                token_hash=row["token_hash"],
                computer_id=row["computer_id"],
                owner_principal=row["owner_principal"],
                fencing_epoch=int(row["fencing_epoch"]),
                expires_at=row["expires_at"],
                consumed=bool(row["consumed"]),
            )

    def mark_token_consumed(self, token_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE takeover_tokens SET consumed = 1 WHERE token_id = ?",
                (token_id,),
            )
            self._conn.commit()

    def expire_tokens_for_computer(self, computer_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE takeover_tokens SET consumed = 1 WHERE computer_id = ?",
                (computer_id,),
            )
            self._conn.commit()

    def upsert_checkpoint(self, checkpoint: Checkpoint) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO checkpoints (id, computer_id, action_class, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET status=excluded.status
                """,
                (
                    checkpoint.id,
                    checkpoint.computer_id,
                    checkpoint.action_class,
                    checkpoint.status.value,
                    checkpoint.created_at,
                ),
            )
            self._conn.commit()

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,)
            ).fetchone()
            if not row:
                return None
            return Checkpoint(
                id=row["id"],
                computer_id=row["computer_id"],
                action_class=row["action_class"],
                status=CheckpointStatus(row["status"]),
                created_at=row["created_at"],
            )

    def open_checkpoint(self, computer_id: str, action_class: str) -> Checkpoint | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoints WHERE computer_id = ? AND action_class = ? "
                "AND status = ? ORDER BY created_at DESC LIMIT 1",
                (computer_id, action_class, CheckpointStatus.APPROVED.value),
            ).fetchone()
            if not row:
                return None
            return Checkpoint(
                id=row["id"],
                computer_id=row["computer_id"],
                action_class=row["action_class"],
                status=CheckpointStatus(row["status"]),
                created_at=row["created_at"],
            )

    def append_audit(self, event: AuditEvent) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO audit (event_type, computer_id, actor, detail_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.event_type,
                    event.computer_id,
                    event.actor,
                    json.dumps(event.detail),
                    event.created_at,
                ),
            )
            self._conn.commit()

    def list_audit(self, computer_id: str | None = None, limit: int = 200) -> list[AuditEvent]:
        with self._lock:
            if computer_id:
                rows = self._conn.execute(
                    "SELECT * FROM audit WHERE computer_id = ? ORDER BY id DESC LIMIT ?",
                    (computer_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            return [
                AuditEvent(
                    id=row["id"],
                    event_type=row["event_type"],
                    computer_id=row["computer_id"],
                    actor=row["actor"],
                    detail=json.loads(row["detail_json"] or "{}"),
                    created_at=row["created_at"],
                )
                for row in rows
            ]

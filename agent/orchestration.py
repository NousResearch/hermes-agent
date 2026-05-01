"""Durable control plane for orchestrating Hermes profiles as peer agents."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from acp_adapter.client import HermesACPClient
from hermes_constants import get_default_hermes_root, get_hermes_home
from hermes_cli.profiles import validate_profile_name


DEFAULT_LEASE_SECONDS = 900.0


def _now() -> float:
    return time.time()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def _normalize_profile(profile: str | None) -> str:
    normalized = (profile or "default").strip() or "default"
    validate_profile_name(normalized)
    return normalized


def _normalize_cwd(cwd: str | None) -> str:
    normalized = os.path.abspath(os.path.expanduser(cwd or os.getcwd()))
    if not os.path.isdir(normalized):
        raise FileNotFoundError(f"Working directory does not exist: {normalized}")
    return normalized


class AgentControlStore:
    """SQLite-backed handles, runs, and cross-process leases."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else get_hermes_home() / "agent-control.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_handles (
                    id TEXT PRIMARY KEY,
                    profile TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'idle',
                    idempotency_key TEXT UNIQUE,
                    lease_owner TEXT,
                    lease_until REAL,
                    last_run_id TEXT,
                    last_response TEXT,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_agent_handles_profile
                    ON agent_handles(profile);
                CREATE INDEX IF NOT EXISTS idx_agent_handles_status
                    ON agent_handles(status);

                CREATE TABLE IF NOT EXISTS agent_session_leases (
                    profile TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    handle_id TEXT,
                    lease_owner TEXT NOT NULL,
                    lease_until REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (profile, session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_agent_session_leases_until
                    ON agent_session_leases(lease_until);

                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY,
                    handle_id TEXT NOT NULL REFERENCES agent_handles(id) ON DELETE CASCADE,
                    profile TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stop_reason TEXT,
                    response TEXT,
                    error TEXT,
                    usage_json TEXT,
                    started_at REAL NOT NULL,
                    finished_at REAL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_agent_runs_handle
                    ON agent_runs(handle_id, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_agent_runs_status
                    ON agent_runs(status);
                """
            )

    def refresh_expired_leases(self) -> int:
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            expired = conn.execute(
                """
                SELECT id, last_run_id
                FROM agent_handles
                WHERE lease_until IS NOT NULL
                  AND lease_until < ?
                  AND status = 'running'
                """,
                (now,),
            ).fetchall()
            cur = conn.execute(
                """
                UPDATE agent_handles
                SET lease_owner = NULL, lease_until = NULL, status = 'error',
                    last_error = 'agent_control lease expired', updated_at = ?
                WHERE lease_until IS NOT NULL
                  AND lease_until < ?
                  AND status = 'running'
                """,
                (now, now),
            )
            for row in expired:
                run_id = row["last_run_id"]
                if not run_id:
                    continue
                conn.execute(
                    """
                    UPDATE agent_runs
                    SET status = 'error', error = 'agent_control lease expired',
                        finished_at = ?, updated_at = ?
                    WHERE id = ? AND status = 'running'
                    """,
                    (now, now, run_id),
                )
            conn.execute("DELETE FROM agent_session_leases WHERE lease_until < ?", (now,))
            return int(cur.rowcount or 0)

    def create_handle(
        self,
        *,
        profile: str,
        session_id: str,
        cwd: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        handle_id = f"agent-{uuid.uuid4().hex[:12]}"
        now = _now()
        with self._connect() as conn:
            if idempotency_key:
                existing = conn.execute(
                    "SELECT * FROM agent_handles WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing is not None:
                    return dict(existing)
            conn.execute(
                """
                INSERT INTO agent_handles (
                    id, profile, session_id, cwd, status, idempotency_key,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'idle', ?, ?, ?)
                """,
                (handle_id, profile, session_id, cwd, idempotency_key, now, now),
            )
            row = conn.execute(
                "SELECT * FROM agent_handles WHERE id = ?",
                (handle_id,),
            ).fetchone()
            return dict(row)

    def get_handle(self, handle_id: str) -> dict[str, Any] | None:
        self.refresh_expired_leases()
        with self._connect() as conn:
            return _row_to_dict(
                conn.execute("SELECT * FROM agent_handles WHERE id = ?", (handle_id,)).fetchone()
            )

    def get_handle_by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        if not key:
            return None
        with self._connect() as conn:
            return _row_to_dict(
                conn.execute(
                    "SELECT * FROM agent_handles WHERE idempotency_key = ?",
                    (key,),
                ).fetchone()
            )

    def list_handles(self, profile: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        self.refresh_expired_leases()
        limit = max(1, min(int(limit or 50), 200))
        with self._connect() as conn:
            if profile:
                rows = conn.execute(
                    """
                    SELECT * FROM agent_handles
                    WHERE profile = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (profile, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM agent_handles
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    def update_handle(
        self,
        handle_id: str,
        *,
        status: str | None = None,
        session_id: str | None = None,
        cwd: str | None = None,
        last_run_id: str | None = None,
        last_response: str | None = None,
        last_error: str | None = None,
    ) -> None:
        fields: list[str] = ["updated_at = ?"]
        values: list[Any] = [_now()]
        for name, value in (
            ("status", status),
            ("session_id", session_id),
            ("cwd", cwd),
            ("last_run_id", last_run_id),
            ("last_response", last_response),
            ("last_error", last_error),
        ):
            if value is not None:
                fields.append(f"{name} = ?")
                values.append(value)
        values.append(handle_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE agent_handles SET {', '.join(fields)} WHERE id = ?",
                tuple(values),
            )

    def acquire_handle_lease(
        self,
        handle_id: str,
        *,
        owner: str,
        ttl_seconds: float = DEFAULT_LEASE_SECONDS,
        wait_seconds: float = 0.0,
    ) -> bool:
        deadline = _now() + max(0.0, wait_seconds)
        while True:
            now = _now()
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    UPDATE agent_handles
                    SET lease_owner = ?, lease_until = ?, status = 'running',
                        updated_at = ?
                    WHERE id = ?
                      AND (lease_until IS NULL OR lease_until < ? OR lease_owner = ?)
                    """,
                    (owner, now + ttl_seconds, now, handle_id, now, owner),
                )
                if cur.rowcount == 1:
                    return True
            if _now() >= deadline:
                return False
            time.sleep(0.1)

    def release_handle_lease(self, handle_id: str, *, owner: str, status: str = "idle") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE agent_handles
                SET lease_owner = NULL, lease_until = NULL, status = ?,
                    updated_at = ?
                WHERE id = ? AND lease_owner = ?
                """,
                (status, _now(), handle_id, owner),
            )

    def acquire_session_lease(
        self,
        *,
        profile: str,
        session_id: str,
        handle_id: str,
        owner: str,
        ttl_seconds: float = DEFAULT_LEASE_SECONDS,
        wait_seconds: float = 0.0,
    ) -> bool:
        profile = _normalize_profile(profile)
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("session_id is required for agent session lease")

        deadline = _now() + max(0.0, wait_seconds)
        while True:
            now = _now()
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute(
                    """
                    UPDATE agent_session_leases
                    SET handle_id = ?, lease_owner = ?, lease_until = ?, updated_at = ?
                    WHERE profile = ?
                      AND session_id = ?
                      AND (lease_until < ? OR lease_owner = ?)
                    """,
                    (
                        handle_id,
                        owner,
                        now + ttl_seconds,
                        now,
                        profile,
                        session_id,
                        now,
                        owner,
                    ),
                )
                if cur.rowcount == 1:
                    return True
                try:
                    conn.execute(
                        """
                        INSERT INTO agent_session_leases (
                            profile, session_id, handle_id, lease_owner,
                            lease_until, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            profile,
                            session_id,
                            handle_id,
                            owner,
                            now + ttl_seconds,
                            now,
                            now,
                        ),
                    )
                    return True
                except sqlite3.IntegrityError:
                    pass
            if _now() >= deadline:
                return False
            time.sleep(0.1)

    def release_session_lease(self, *, profile: str, session_id: str, owner: str) -> None:
        profile = _normalize_profile(profile)
        session_id = str(session_id or "").strip()
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM agent_session_leases
                WHERE profile = ? AND session_id = ? AND lease_owner = ?
                """,
                (profile, session_id, owner),
            )

    @contextmanager
    def leased_handle(
        self,
        handle_id: str,
        *,
        owner: str,
        ttl_seconds: float = DEFAULT_LEASE_SECONDS,
        wait_seconds: float = 0.0,
    ):
        acquired = self.acquire_handle_lease(
            handle_id,
            owner=owner,
            ttl_seconds=ttl_seconds,
            wait_seconds=wait_seconds,
        )
        if not acquired:
            raise TimeoutError(f"agent handle {handle_id} is busy")
        status = "idle"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            self.release_handle_lease(handle_id, owner=owner, status=status)

    @contextmanager
    def leased_session(
        self,
        handle_id: str,
        *,
        profile: str,
        session_id: str,
        owner: str,
        ttl_seconds: float = DEFAULT_LEASE_SECONDS,
        wait_seconds: float = 0.0,
    ):
        acquired_session = self.acquire_session_lease(
            profile=profile,
            session_id=session_id,
            handle_id=handle_id,
            owner=owner,
            ttl_seconds=ttl_seconds,
            wait_seconds=wait_seconds,
        )
        if not acquired_session:
            raise TimeoutError(f"agent session {profile}/{session_id} is busy")

        acquired_handle = self.acquire_handle_lease(
            handle_id,
            owner=owner,
            ttl_seconds=ttl_seconds,
            wait_seconds=0.0,
        )
        if not acquired_handle:
            self.release_session_lease(profile=profile, session_id=session_id, owner=owner)
            raise TimeoutError(f"agent handle {handle_id} is busy")

        status = "idle"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            try:
                self.release_handle_lease(handle_id, owner=owner, status=status)
            finally:
                self.release_session_lease(profile=profile, session_id=session_id, owner=owner)

    def create_run(
        self,
        *,
        handle_id: str,
        profile: str,
        session_id: str,
        prompt: str,
    ) -> dict[str, Any]:
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_runs (
                    id, handle_id, profile, session_id, prompt, status,
                    started_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?)
                """,
                (run_id, handle_id, profile, session_id, prompt, now, now),
            )
            conn.execute(
                """
                UPDATE agent_handles
                SET last_run_id = ?, status = 'running', updated_at = ?
                WHERE id = ?
                """,
                (run_id, now, handle_id),
            )
        return self.get_run(run_id) or {"id": run_id, "status": "running"}

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        response: str | None = None,
        stop_reason: str | None = None,
        error: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE agent_runs
                SET status = ?, response = ?, stop_reason = ?, error = ?,
                    usage_json = ?, finished_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    response,
                    stop_reason,
                    error,
                    _json_dumps(usage or {}),
                    now,
                    now,
                    run_id,
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM agent_runs WHERE id = ?", (run_id,)).fetchone()
        out = _row_to_dict(row)
        if out and out.get("usage_json"):
            try:
                out["usage"] = json.loads(out["usage_json"])
            except json.JSONDecodeError:
                out["usage"] = {}
        return out

    def last_run_for_handle(self, handle_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM agent_runs
                WHERE handle_id = ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (handle_id,),
            ).fetchone()
        out = _row_to_dict(row)
        if out and out.get("usage_json"):
            try:
                out["usage"] = json.loads(out["usage_json"])
            except json.JSONDecodeError:
                out["usage"] = {}
        return out


class AgentController:
    """High-level operations for controlling peer Hermes profiles."""

    def __init__(
        self,
        *,
        store: AgentControlStore | None = None,
        client_factory: Callable[..., HermesACPClient] = HermesACPClient,
    ):
        self.store = store or AgentControlStore()
        self.client_factory = client_factory

    def _profile_exists(self, profile: str) -> bool:
        profile = _normalize_profile(profile)
        if profile == "default":
            return True
        return (get_default_hermes_root() / "profiles" / profile).is_dir()

    def _new_client(
        self,
        *,
        profile: str,
        cwd: str,
        approval_policy: str = "deny",
    ) -> HermesACPClient:
        return self.client_factory(
            profile=profile,
            cwd=cwd,
            approval_policy=approval_policy,
        )

    def start_agent(
        self,
        *,
        profile: str,
        cwd: str | None = None,
        session_id: str | None = None,
        idempotency_key: str | None = None,
        approval_policy: str = "deny",
    ) -> dict[str, Any]:
        try:
            profile = _normalize_profile(profile)
            cwd = _normalize_cwd(cwd)
        except (FileNotFoundError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        if not self._profile_exists(profile):
            return {
                "ok": False,
                "error": f"Profile '{profile}' does not exist.",
            }

        if idempotency_key:
            existing = self.store.get_handle_by_idempotency_key(idempotency_key)
            if existing:
                return {"ok": True, "reused": True, "agent": existing}

        client = self._new_client(profile=profile, cwd=cwd, approval_policy=approval_policy)
        try:
            client.connect()
            actual_session_id = (
                client.load_session(session_id, cwd)
                if session_id
                else client.new_session(cwd)
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            client.close()

        handle = self.store.create_handle(
            profile=profile,
            session_id=actual_session_id,
            cwd=cwd,
            idempotency_key=idempotency_key,
        )
        return {"ok": True, "reused": False, "agent": handle}

    def list_agents(self, *, profile: str | None = None, limit: int = 50) -> dict[str, Any]:
        try:
            normalized_profile = _normalize_profile(profile) if profile else None
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "agents": self.store.list_handles(profile=normalized_profile, limit=limit)}

    def status(self, *, agent_id: str) -> dict[str, Any]:
        handle = self.store.get_handle(agent_id)
        if handle is None:
            return {"ok": False, "error": f"Unknown agent_id: {agent_id}"}
        last_run = self.store.last_run_for_handle(agent_id)
        return {"ok": True, "agent": handle, "last_run": last_run}

    def fork_agent(
        self,
        *,
        agent_id: str,
        cwd: str | None = None,
        idempotency_key: str | None = None,
        approval_policy: str = "deny",
        lease_wait_seconds: float = 5.0,
    ) -> dict[str, Any]:
        handle = self.store.get_handle(agent_id)
        if handle is None:
            return {"ok": False, "error": f"Unknown agent_id: {agent_id}"}
        if idempotency_key:
            existing = self.store.get_handle_by_idempotency_key(idempotency_key)
            if existing:
                return {"ok": True, "agent": existing, "forked_from": agent_id, "reused": True}
        profile = str(handle["profile"])
        try:
            target_cwd = _normalize_cwd(cwd or handle["cwd"])
        except FileNotFoundError as exc:
            return {"ok": False, "error": str(exc)}
        client = self._new_client(profile=profile, cwd=target_cwd, approval_policy=approval_policy)
        owner = f"{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex[:8]}"
        session_id = str(handle["session_id"])
        try:
            with self.store.leased_session(
                agent_id,
                profile=profile,
                session_id=session_id,
                owner=owner,
                ttl_seconds=DEFAULT_LEASE_SECONDS,
                wait_seconds=lease_wait_seconds,
            ):
                try:
                    client.connect()
                    new_session_id = client.fork_session(session_id, target_cwd)
                except Exception as exc:
                    return {"ok": False, "error": str(exc)}
                finally:
                    client.close()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        new_handle = self.store.create_handle(
            profile=profile,
            session_id=new_session_id,
            cwd=target_cwd,
            idempotency_key=idempotency_key,
        )
        return {"ok": True, "agent": new_handle, "forked_from": agent_id}

    def prompt_agent(
        self,
        *,
        agent_id: str,
        prompt: str,
        timeout_seconds: float = 600.0,
        lease_wait_seconds: float = 5.0,
        approval_policy: str = "deny",
    ) -> dict[str, Any]:
        handle = self.store.get_handle(agent_id)
        if handle is None:
            return {"ok": False, "error": f"Unknown agent_id: {agent_id}"}
        if not prompt or not str(prompt).strip():
            return {"ok": False, "error": "prompt is required."}
        try:
            target_cwd = _normalize_cwd(str(handle["cwd"]))
        except FileNotFoundError as exc:
            return {"ok": False, "agent_id": agent_id, "error": str(exc)}

        owner = f"{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex[:8]}"
        run_id: str | None = None

        try:
            with self.store.leased_session(
                agent_id,
                profile=str(handle["profile"]),
                session_id=str(handle["session_id"]),
                owner=owner,
                ttl_seconds=max(DEFAULT_LEASE_SECONDS, float(timeout_seconds) + 60.0),
                wait_seconds=lease_wait_seconds,
            ):
                run = self.store.create_run(
                    handle_id=agent_id,
                    profile=str(handle["profile"]),
                    session_id=str(handle["session_id"]),
                    prompt=str(prompt),
                )
                run_id = str(run["id"])
                client = self._new_client(
                    profile=str(handle["profile"]),
                    cwd=target_cwd,
                    approval_policy=approval_policy,
                )
                try:
                    client.connect()
                    session_id = client.load_session(str(handle["session_id"]), target_cwd)
                    result = client.prompt(session_id, str(prompt), timeout=float(timeout_seconds))
                finally:
                    client.close()

                response = str(result.get("text") or "")
                stop_reason = result.get("stop_reason")
                usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
                self.store.finish_run(
                    run_id,
                    status="completed",
                    response=response,
                    stop_reason=str(stop_reason or ""),
                    usage=usage,
                )
                self.store.update_handle(
                    agent_id,
                    status="idle",
                    session_id=session_id,
                    last_run_id=run_id,
                    last_response=response,
                    last_error="",
                )
                return {
                    "ok": True,
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "profile": handle["profile"],
                    "session_id": session_id,
                    "stop_reason": stop_reason,
                    "response": response,
                    "usage": usage,
                }
        except Exception as exc:
            error = str(exc)
            if run_id:
                self.store.finish_run(run_id, status="error", error=error)
                self.store.update_handle(
                    agent_id,
                    status="error",
                    last_run_id=run_id,
                    last_error=error,
                )
            return {"ok": False, "run_id": run_id, "agent_id": agent_id, "error": error}

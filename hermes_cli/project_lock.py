"""Project-scoped Production delivery leases for Kanban workers."""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import secrets
import shlex
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
from urllib.parse import urlparse

from hermes_cli import kanban_db as kb

_VALID_OPERATION = {"deploy", "migration"}
_PROJECT_PART = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def production_delivery_guard(command: str) -> Optional[str]:
    """Reject unleased Production mutations from dispatcher workers."""
    if not os.environ.get("HERMES_KANBAN_TASK"):
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        tokens = command.split()
    lowered = [token.lower() for token in tokens]
    wrapped = (
        len(lowered) >= 4
        and os.path.basename(lowered[0]) == "hermes"
        and lowered[1:4] == ["kanban", "lock", "run"]
        and "--" in lowered
    )
    if wrapped and "\n" not in command and not any(
        token and set(token) <= {";", "&", "|"} for token in lowered
    ):
        return None

    segments = []
    segment = []
    for token in lowered:
        if token and set(token) <= {";", "&", "|"}:
            if segment:
                segments.append(" ".join(segment))
                segment = []
        else:
            segment.append(token)
    if segment:
        segments.append(" ".join(segment))
    segments = [
        part
        for raw in segments
        for part in re.split(r"\s*(?:&&|\|\||[;|])\s*", raw)
        if part
    ]

    production_mutation = False
    for raw in segments:
        local_or_preview = bool(re.search(
            r"(?:^|\s)--local(?:\s|$)|--target(?:=|\s+)preview\b",
            raw,
        ))
        prod_flag = bool(re.search(
            r"--prod(?:uction)?\b|--(?:target|environment)(?:=|\s+)production\b",
            raw,
        ))
        production_mutation = (
            (bool(re.search(r"\bvercel\b", raw)) and prod_flag)
            or (bool(re.search(r"\bsupabase\s+db\s+push\b", raw)) and not local_or_preview)
            or (bool(re.search(r"\bsupabase\s+migration\s+up\b", raw)) and not local_or_preview)
            or bool(re.search(r"\bprisma\s+migrate\s+deploy\b", raw))
            or bool(re.search(r"\bgh\b.*\bpr\b.*\bmerge\b", raw))
            or bool(re.search(r"\bgit\b.*\bpush\b.*\bmain\b", raw))
            or (prod_flag and any(part in raw for part in ("deploy", "migrat")))
        )
        if production_mutation:
            break
    if not production_mutation:
        return None
    return (
        "Production deploys and migrations from Kanban workers must run through "
        "`hermes kanban lock run --project <owner/repo> "
        "--operation <deploy|migration> -- <command>`."
    )


@dataclass(frozen=True)
class LeaseOwner:
    task_id: str
    run_id: int
    claim_lock: str
    instance_id: str
    host: str = ""
    pid: int = 0

    @classmethod
    def from_env(cls) -> "LeaseOwner":
        task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
        run_id = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
        claim_lock = os.environ.get("HERMES_KANBAN_CLAIM_LOCK", "").strip()
        if not task_id or not run_id or not claim_lock:
            raise RuntimeError(
                "Production delivery locks require an active Kanban worker claim"
            )
        try:
            parsed_run_id = int(run_id)
        except ValueError as exc:
            raise RuntimeError("HERMES_KANBAN_RUN_ID must be an integer") from exc
        return cls(
            task_id=task_id,
            run_id=parsed_run_id,
            claim_lock=claim_lock,
            instance_id=f"{claim_lock}:{os.getpid()}:{secrets.token_hex(8)}",
            host=socket.gethostname() or "unknown",
            pid=os.getpid(),
        )


@dataclass(frozen=True)
class ProjectLease:
    key: str
    project: str
    operation: str
    owner: LeaseOwner
    token: str
    fence: int
    expires_at: float


def _canonical_project(project: str) -> str:
    value = project.strip().rstrip("/")
    if value.startswith("git@") and ":" in value:
        host, path = value[4:].split(":", 1)
        value = f"{host}/{path}"
    elif "://" in value:
        parsed = urlparse(value)
        value = f"{parsed.hostname or ''}/{parsed.path.lstrip('/')}"
    value = value.removesuffix(".git").lower()
    parts = value.split("/")
    if len(parts) == 2:
        parts.insert(0, "github.com")
    if len(parts) != 3 or any(not _PROJECT_PART.fullmatch(part) for part in parts):
        raise ValueError(
            "project must be owner/repo or a canonical repository URL"
        )
    return "/".join(parts)


def project_lock_key(project: str, *, target: str = "production") -> str:
    if target.strip().lower() != "production":
        raise ValueError("Production-only lock; preview and local work do not use it")
    return f"delivery:{_canonical_project(project)}:production"


def project_lock_db_path():
    """Return the one shared lock store for every Kanban board."""
    return kb.kanban_home() / "kanban" / "project-delivery-locks.db"


@contextlib.contextmanager
def connect_project_locks():
    with kb.connect_closing(project_lock_db_path()) as conn:
        yield conn


def _owner_valid(conn, owner: LeaseOwner) -> bool:
    row = conn.execute(
        "SELECT 1 FROM tasks WHERE id=? AND status='running' "
        "AND current_run_id=? AND claim_lock=?",
        (owner.task_id, owner.run_id, owner.claim_lock),
    ).fetchone()
    return row is not None


def _validate_owner(conn, owner: LeaseOwner) -> None:
    if not _owner_valid(conn, owner):
        raise RuntimeError("Kanban run no longer owns its task claim")


@contextlib.contextmanager
def _lock_write_txn(conn):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise
    else:
        try:
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise


def acquire_project_lock(
    conn,
    *,
    owner_conn=None,
    project: str,
    operation: str,
    owner: LeaseOwner,
    lease_seconds: float,
    target: str = "production",
    now: Optional[float] = None,
) -> Optional[ProjectLease]:
    if operation not in _VALID_OPERATION:
        raise ValueError("operation must be deploy or migration")
    if not math.isfinite(lease_seconds) or lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    key = project_lock_key(project, target=target)
    canonical_project = _canonical_project(project)
    timestamp = time.time() if now is None else float(now)
    token = secrets.token_urlsafe(32)
    expires_at = timestamp + float(lease_seconds)

    _validate_owner(owner_conn or conn, owner)
    with _lock_write_txn(conn):
        row = conn.execute(
            "SELECT owner_token, expires_at, fence, owner_host, owner_pid, command_pid "
            "FROM project_delivery_locks WHERE resource_key=?",
            (key,),
        ).fetchone()
        if row is not None and row["owner_token"] is not None and row["expires_at"] > timestamp:
            return None
        local_host = socket.gethostname() or "unknown"
        if (
            row is not None
            and row["owner_token"] is not None
            and row["owner_host"] == local_host
            and any(
                pid and kb._pid_alive(int(pid))
                for pid in (row["owner_pid"], row["command_pid"])
            )
        ):
            return None
        fence = (int(row["fence"]) + 1) if row is not None else 1
        owner_host = owner.host or local_host
        owner_pid = owner.pid or os.getpid()
        if row is None:
            conn.execute(
                "INSERT INTO project_delivery_locks ("
                "resource_key, project, target, operation, owner_task_id, "
                "owner_run_id, owner_claim_lock, owner_instance, owner_host, "
                "owner_pid, command_pid, owner_token, "
                "fence, acquired_at, renewed_at, expires_at"
                ") VALUES (?, ?, 'production', ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)",
                (
                    key, canonical_project, operation, owner.task_id, owner.run_id,
                    owner.claim_lock, owner.instance_id, owner_host, owner_pid,
                    token, fence,
                    timestamp, timestamp, expires_at,
                ),
            )
        else:
            cur = conn.execute(
                "UPDATE project_delivery_locks SET operation=?, owner_task_id=?, "
                "owner_run_id=?, owner_claim_lock=?, owner_instance=?, owner_host=?, "
                "owner_pid=?, command_pid=NULL, owner_token=?, fence=?, acquired_at=?, "
                "renewed_at=?, expires_at=?, released_at=NULL "
                "WHERE resource_key=? AND (owner_token IS NULL OR expires_at<=?)",
                (
                    operation, owner.task_id, owner.run_id, owner.claim_lock,
                    owner.instance_id, owner_host, owner_pid, token, fence, timestamp, timestamp,
                    expires_at, key, timestamp,
                ),
            )
            if cur.rowcount != 1:
                return None
    return ProjectLease(
        key=key,
        project=canonical_project,
        operation=operation,
        owner=owner,
        token=token,
        fence=fence,
        expires_at=expires_at,
    )


def renew_project_lock(
    conn,
    lease: ProjectLease,
    *,
    owner_conn=None,
    lease_seconds: float,
    now: Optional[float] = None,
) -> bool:
    timestamp = time.time() if now is None else float(now)
    expires_at = timestamp + float(lease_seconds)
    if not _owner_valid(owner_conn or conn, lease.owner):
        return False
    with _lock_write_txn(conn):
        cur = conn.execute(
            "UPDATE project_delivery_locks SET renewed_at=?, expires_at=? "
            "WHERE resource_key=? AND owner_token=? AND fence=? AND expires_at>? "
            "AND owner_task_id=? AND owner_run_id=? AND owner_claim_lock=?",
            (
                timestamp, expires_at, lease.key, lease.token, lease.fence, timestamp,
                lease.owner.task_id, lease.owner.run_id, lease.owner.claim_lock,
            ),
        )
        return cur.rowcount == 1


def attach_project_lock_process(conn, lease: ProjectLease, pid: int) -> bool:
    with _lock_write_txn(conn):
        cur = conn.execute(
            "UPDATE project_delivery_locks SET command_pid=? WHERE resource_key=? "
            "AND owner_token=? AND fence=?",
            (int(pid), lease.key, lease.token, lease.fence),
        )
        return cur.rowcount == 1


def release_project_lock(
    conn,
    lease: ProjectLease,
    *,
    owner_conn=None,
    now: Optional[float] = None,
) -> bool:
    timestamp = time.time() if now is None else float(now)
    if not _owner_valid(owner_conn or conn, lease.owner):
        return False
    with _lock_write_txn(conn):
        cur = conn.execute(
            "UPDATE project_delivery_locks SET owner_task_id=NULL, owner_run_id=NULL, "
            "owner_claim_lock=NULL, owner_instance=NULL, owner_host=NULL, "
            "owner_pid=NULL, command_pid=NULL, owner_token=NULL, "
            "expires_at=NULL, released_at=? WHERE resource_key=? "
            "AND owner_token=? AND fence=?",
            (
                timestamp, lease.key, lease.token, lease.fence,
            ),
        )
        return cur.rowcount == 1


def project_lock_status(conn, project: str) -> Optional[dict]:
    key = project_lock_key(project)
    row = conn.execute(
        "SELECT resource_key, project, target, operation, owner_task_id, "
        "owner_run_id, owner_instance, owner_host, owner_pid, command_pid, "
        "fence, acquired_at, renewed_at, "
        "expires_at, released_at FROM project_delivery_locks WHERE resource_key=?",
        (key,),
    ).fetchone()
    return dict(row) if row is not None else None


def acquire_with_wait(
    conn,
    *,
    owner_conn=None,
    project: str,
    operation: str,
    owner: LeaseOwner,
    lease_seconds: float,
    wait_seconds: float,
    poll_seconds: float = 1.0,
    on_wait: Optional[Callable[[Optional[dict]], None]] = None,
) -> Optional[ProjectLease]:
    if not math.isfinite(wait_seconds) or wait_seconds < 0:
        raise ValueError("wait_seconds must be a finite non-negative number")
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    deadline = time.monotonic() + max(0.0, wait_seconds)
    previous_busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    conn.execute("PRAGMA busy_timeout=0")
    try:
        while True:
            busy = False
            try:
                lease = acquire_project_lock(
                    conn,
                    owner_conn=owner_conn,
                    project=project,
                    operation=operation,
                    owner=owner,
                    lease_seconds=lease_seconds,
                )
            except sqlite3.OperationalError as exc:
                if not kb._is_busy_error(exc):
                    raise
                lease = None
                busy = True
            if lease is not None:
                return lease
            if time.monotonic() >= deadline:
                return None
            if on_wait is not None:
                on_wait(None if busy else project_lock_status(conn, project))
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    finally:
        conn.execute(f"PRAGMA busy_timeout={int(previous_busy_timeout)}")


def _emit(state: str, **fields) -> None:
    print(json.dumps({"state": state, "timestamp": time.time(), **fields}, sort_keys=True), file=sys.stderr, flush=True)


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _supervised_command(command: Sequence[str]) -> list[str]:
    """Run the command under the existing parent-death process supervisor."""
    if os.name != "posix":
        raise RuntimeError("Production delivery command supervision requires POSIX")
    try:
        import psutil

        parent_create_time = psutil.Process(os.getpid()).create_time()
    except Exception as exc:
        raise RuntimeError("Cannot establish crash-safe command supervision") from exc
    watchdog = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tools",
        "mcp_stdio_watchdog.py",
    )
    return [
        sys.executable,
        watchdog,
        "--ppid",
        str(os.getpid()),
        "--create-time",
        repr(parent_create_time),
        "--",
        *command,
    ]


def run_locked_command(
    command: Sequence[str],
    *,
    project: str,
    operation: str,
    lease_seconds: float,
    wait_seconds: float,
) -> int:
    if not command:
        raise ValueError("lock run requires a command after --")
    if lease_seconds < 2:
        raise ValueError("--lease must be at least 2 seconds")
    owner = LeaseOwner.from_env()
    with kb.connect_closing() as owner_conn, connect_project_locks() as conn:
        lease = acquire_with_wait(
            conn,
            owner_conn=owner_conn,
            project=project,
            operation=operation,
            owner=owner,
            lease_seconds=lease_seconds,
            wait_seconds=wait_seconds,
            on_wait=lambda holder: _emit("waiting", holder=holder),
        )
        if lease is None:
            _emit("timeout", project=_canonical_project(project), waited_seconds=wait_seconds)
            return 73
        _emit(
            "acquired",
            project=lease.project,
            operation=operation,
            owner_task_id=owner.task_id,
            owner_run_id=owner.run_id,
            owner_instance=owner.instance_id,
            fence=lease.fence,
            expires_at=lease.expires_at,
        )
        try:
            proc = subprocess.Popen(_supervised_command(command))  # noqa: S603 -- explicit argv, no shell
        except Exception:
            release_project_lock(conn, lease, owner_conn=owner_conn)
            raise
        if not attach_project_lock_process(conn, lease, proc.pid):
            _stop_process(proc)
            release_project_lock(conn, lease, owner_conn=owner_conn)
            raise RuntimeError("lost project lease before command process registration")
        interrupted = False
        old_handlers = {}

        def stop(_signum, _frame):
            nonlocal interrupted
            interrupted = True

        if __import__("threading").current_thread() is __import__("threading").main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.signal(sig, stop)
        next_renewal = time.monotonic() + lease_seconds / 3
        lost = False
        try:
            while proc.poll() is None:
                if interrupted:
                    _stop_process(proc)
                    break
                if time.monotonic() >= next_renewal:
                    if not renew_project_lock(
                        conn, lease, owner_conn=owner_conn, lease_seconds=lease_seconds,
                    ):
                        lost = True
                        _emit("lost", fence=lease.fence)
                        _stop_process(proc)
                        break
                    next_renewal = time.monotonic() + lease_seconds / 3
                time.sleep(min(0.2, max(0.01, next_renewal - time.monotonic())))
            return_code = proc.wait()
        except BaseException:
            _stop_process(proc)
            raise
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            released = release_project_lock(conn, lease, owner_conn=owner_conn)
            _emit("released" if released else "stale_release_rejected", fence=lease.fence)
        if lost or not released:
            return 74
        if interrupted:
            return 130
        return int(return_code)

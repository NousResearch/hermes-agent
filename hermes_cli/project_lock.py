"""Project-scoped Production delivery leases for Kanban workers."""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence
from urllib.parse import urlparse

from hermes_cli import kanban_db as kb

_VALID_OPERATION = {"deploy", "migration"}
_PROJECT_PART = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_LOCK_SCHEMA = """
CREATE TABLE IF NOT EXISTS project_delivery_locks (
    resource_key TEXT PRIMARY KEY, project TEXT NOT NULL, target TEXT NOT NULL,
    operation TEXT NOT NULL, owner_task_id TEXT, owner_run_id INTEGER,
    owner_claim_lock TEXT, owner_instance TEXT, owner_host TEXT,
    owner_pid INTEGER, owner_started_at REAL, command_pid INTEGER,
    command_started_at REAL, owner_token TEXT, fence INTEGER NOT NULL DEFAULT 0,
    acquired_at REAL, renewed_at REAL, expires_at REAL, released_at REAL
)
"""


class ProjectLockTimeout(RuntimeError):
    pass


def _trusted_hermes_executable(token: str) -> bool:
    resolved = shutil.which(token) if os.sep not in token else token
    if not resolved:
        return False
    try:
        return (
            Path(resolved).resolve()
            == Path(sys.executable).with_name("hermes").resolve()
        )
    except OSError:
        return False


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
        and _trusted_hermes_executable(tokens[0])
        and lowered[1:4] == ["kanban", "lock", "run"]
        and "--" in lowered
    )
    has_shell_control = any(
        marker in command for marker in ("\n", "$(", "`", "<(", ">(")
    ) or any(
        token and set(token) <= {";", "&", "|"} for token in lowered
    )
    if wrapped and not has_shell_control:
        try:
            project_index = lowered.index("--project") + 1
            requested = _canonical_project(tokens[project_index])
            actual = _workspace_project(
                os.environ.get("HERMES_KANBAN_WORKSPACE", "")
            )
        except (IndexError, RuntimeError, ValueError):
            return (
                "Production lock wrapper must match the claimed task's "
                "Git workspace origin."
            )
        if requested != actual:
            return (
                "Production lock project does not match the claimed task's "
                "Git workspace origin."
            )
        return None
    if wrapped or any(
        lowered[index:index + 3] == ["kanban", "lock", "run"]
        for index in range(len(lowered) - 2)
    ):
        return "Production lock wrapper must be one trusted Hermes command."

    segments = []
    segment = []
    for token in tokens:
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
        raw_lower = raw.lower()
        local_or_preview = bool(re.search(
            r"(?:^|\s)--local(?:\s|$)|--target(?:=|\s+)preview\b",
            raw_lower,
        ))
        prod_flag = bool(re.search(
            r"--prod(?:uction)?\b|--(?:target|environment)(?:=|\s+)production\b",
            raw_lower,
        ))
        try:
            words = shlex.split(raw)
        except ValueError:
            words = raw.split()
        executable = os.path.basename(words[0]).lower() if words else ""
        script = executable
        if executable in {"bash", "python", "python3", "sh", "zsh"}:
            options_with_value = (
                {"-W", "-X", "--check-hash-based-pycs"}
                if executable in {"python", "python3"}
                else {"-O", "-o"}
            )
            index = 1
            while index < len(words):
                word = words[index]
                if word == "--":
                    index += 1
                    break
                if word in {"-c", "-m"}:
                    index += 1
                    break
                if word in options_with_value:
                    index += 2
                    continue
                if word.startswith("-"):
                    index += 1
                    continue
                break
            script = os.path.basename(words[index]) if index < len(words) else ""
        opaque_delivery_script = bool(
            re.search(r"deploy|migrat|release", script.lower())
        )
        production_mutation = (
            (bool(re.search(r"\bvercel\b", raw_lower)) and prod_flag)
            or (bool(re.search(r"\bsupabase\s+db\s+push\b", raw_lower)) and not local_or_preview)
            or (bool(re.search(r"\bsupabase\s+migration\s+up\b", raw_lower)) and not local_or_preview)
            or bool(re.search(r"\bprisma\s+migrate\s+deploy\b", raw_lower))
            or bool(re.search(r"\bgh\b.*\bpr\b.*\bmerge\b", raw_lower))
            or bool(re.search(r"\bgit\b.*\bpush\b.*\bmain\b", raw_lower))
            or (prod_flag and any(part in raw_lower for part in ("deploy", "migrat")))
            or (opaque_delivery_script and not local_or_preview)
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
    started_at: float = 0.0
    workspace: str = ""
    project: str = ""

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
        workspace = os.environ.get("HERMES_KANBAN_WORKSPACE", "").strip()
        try:
            import psutil

            started_at = psutil.Process(os.getpid()).create_time()
        except Exception as exc:
            raise RuntimeError(
                "Cannot establish Kanban worker process identity"
            ) from exc
        return cls(
            task_id=task_id,
            run_id=parsed_run_id,
            claim_lock=claim_lock,
            instance_id=f"{claim_lock}:{os.getpid()}:{secrets.token_hex(8)}",
            host=socket.gethostname() or "unknown",
            pid=os.getpid(),
            started_at=started_at,
            workspace=str(Path(workspace).resolve()) if workspace else "",
            project=_workspace_project(workspace),
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


def _workspace_project(workspace: str) -> str:
    if not workspace:
        raise RuntimeError(
            "Production delivery locks require a Kanban Git workspace"
        )
    try:
        result = subprocess.run(
            ["git", "-C", workspace, "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Cannot resolve the Kanban workspace origin") from exc
    return _canonical_project(result.stdout.strip())


def project_lock_key(project: str, *, target: str = "production") -> str:
    if target.strip().lower() != "production":
        raise ValueError("Production-only lock; preview and local work do not use it")
    return f"delivery:{_canonical_project(project)}:production"


def project_lock_db_path():
    """Return the one shared lock store for every Kanban board."""
    return kb.kanban_home() / "kanban" / "project-delivery-locks.db"


def _ensure_lock_schema(conn) -> None:
    conn.execute(_LOCK_SCHEMA)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(project_delivery_locks)")
    }
    for column in ("owner_started_at", "command_started_at"):
        if column not in columns:
            conn.execute(
                f"ALTER TABLE project_delivery_locks ADD COLUMN {column} REAL"
            )


@contextlib.contextmanager
def connect_project_locks(
    *,
    owner_db_path: Optional[Path] = None,
    deadline: Optional[float] = None,
    on_wait: Optional[Callable[[Optional[dict]], None]] = None,
):
    path = project_lock_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=0")
    try:
        while True:
            try:
                _ensure_lock_schema(conn)
                break
            except sqlite3.OperationalError as exc:
                if not kb._is_busy_error(exc):
                    raise
                if deadline is None or time.monotonic() >= deadline:
                    raise ProjectLockTimeout("project lock store is busy") from exc
                if on_wait is not None:
                    on_wait(None)
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        if owner_db_path is not None:
            conn.execute("ATTACH DATABASE ? AS ownerdb", (str(owner_db_path),))
        yield conn
    finally:
        conn.close()


def _owner_row(conn, owner: LeaseOwner, *, schema: str = "main"):
    if schema not in {"main", "ownerdb"}:
        raise ValueError("invalid owner schema")
    return conn.execute(
        f"SELECT workspace_path FROM {schema}.tasks "
        "WHERE id=? AND status='running' AND current_run_id=? AND claim_lock=?",
        (owner.task_id, owner.run_id, owner.claim_lock),
    ).fetchone()


def _owner_valid(conn, owner: LeaseOwner, *, schema: str = "main") -> bool:
    return _owner_row(conn, owner, schema=schema) is not None


def _validate_owner(conn, owner: LeaseOwner, *, schema: str = "main") -> None:
    row = _owner_row(conn, owner, schema=schema)
    if row is None:
        raise RuntimeError("Kanban run no longer owns its task claim")
    if owner.workspace:
        recorded = row["workspace_path"] or ""
        try:
            same_workspace = (
                Path(recorded).resolve() == Path(owner.workspace).resolve()
            )
        except OSError:
            same_workspace = False
        if not same_workspace:
            raise RuntimeError("Kanban task workspace no longer matches this worker")


def _process_alive(pid: Optional[int], started_at: Optional[float]) -> bool:
    if not pid or not started_at:
        return False
    try:
        import psutil

        return psutil.pid_exists(int(pid)) and abs(
            psutil.Process(int(pid)).create_time() - float(started_at)
        ) < 0.01
    except Exception:
        return False


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
    owner_schema: str = "main",
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
    canonical_project = _canonical_project(project)
    if owner.project and canonical_project != owner.project:
        raise ValueError(
            "project does not match the claimed task's Git workspace origin"
        )
    key = project_lock_key(canonical_project, target=target)
    timestamp = time.time() if now is None else float(now)
    token = secrets.token_urlsafe(32)
    expires_at = timestamp + float(lease_seconds)

    with _lock_write_txn(conn):
        _validate_owner(owner_conn or conn, owner, schema=owner_schema)
        row = conn.execute(
            "SELECT owner_token, expires_at, fence, owner_host, owner_pid, "
            "owner_started_at, command_pid, command_started_at "
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
                _process_alive(pid, started_at)
                for pid, started_at in (
                    (row["owner_pid"], row["owner_started_at"]),
                    (row["command_pid"], row["command_started_at"]),
                )
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
                "owner_pid, owner_started_at, command_pid, command_started_at, "
                "owner_token, "
                "fence, acquired_at, renewed_at, expires_at"
                ") VALUES (?, ?, 'production', ?, ?, ?, ?, ?, ?, ?, ?, "
                "NULL, NULL, ?, ?, ?, ?, ?)",
                (
                    key, canonical_project, operation, owner.task_id, owner.run_id,
                    owner.claim_lock, owner.instance_id, owner_host, owner_pid,
                    owner.started_at, token, fence,
                    timestamp, timestamp, expires_at,
                ),
            )
        else:
            cur = conn.execute(
                "UPDATE project_delivery_locks SET operation=?, owner_task_id=?, "
                "owner_run_id=?, owner_claim_lock=?, owner_instance=?, owner_host=?, "
                "owner_pid=?, owner_started_at=?, command_pid=NULL, "
                "command_started_at=NULL, owner_token=?, fence=?, acquired_at=?, "
                "renewed_at=?, expires_at=?, released_at=NULL "
                "WHERE resource_key=? AND (owner_token IS NULL OR expires_at<=?)",
                (
                    operation, owner.task_id, owner.run_id, owner.claim_lock,
                    owner.instance_id, owner_host, owner_pid, owner.started_at,
                    token, fence, timestamp, timestamp,
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
    owner_schema: str = "main",
    lease_seconds: float,
    now: Optional[float] = None,
) -> bool:
    timestamp = time.time() if now is None else float(now)
    expires_at = timestamp + float(lease_seconds)
    with _lock_write_txn(conn):
        if not _owner_valid(
            owner_conn or conn, lease.owner, schema=owner_schema
        ):
            return False
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


def attach_project_lock_process(
    conn,
    lease: ProjectLease,
    pid: int,
    *,
    owner_conn=None,
    owner_schema: str = "main",
) -> bool:
    try:
        import psutil

        started_at = psutil.Process(int(pid)).create_time()
    except Exception:
        return False
    with _lock_write_txn(conn):
        if not _owner_valid(
            owner_conn or conn, lease.owner, schema=owner_schema
        ):
            return False
        cur = conn.execute(
            "UPDATE project_delivery_locks SET command_pid=?, command_started_at=? "
            "WHERE resource_key=? "
            "AND owner_token=? AND fence=?",
            (int(pid), started_at, lease.key, lease.token, lease.fence),
        )
        return cur.rowcount == 1


def release_project_lock(
    conn,
    lease: ProjectLease,
    *,
    owner_conn=None,
    owner_schema: str = "main",
    now: Optional[float] = None,
) -> bool:
    timestamp = time.time() if now is None else float(now)
    with _lock_write_txn(conn):
        if not _owner_valid(
            owner_conn or conn, lease.owner, schema=owner_schema
        ):
            return False
        cur = conn.execute(
            "UPDATE project_delivery_locks SET owner_task_id=NULL, owner_run_id=NULL, "
            "owner_claim_lock=NULL, owner_instance=NULL, owner_host=NULL, "
            "owner_pid=NULL, owner_started_at=NULL, command_pid=NULL, "
            "command_started_at=NULL, owner_token=NULL, "
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
    owner_schema: str = "main",
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
                    owner_schema=owner_schema,
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


def _supervised_command(
    command: Sequence[str], *, start_gate_fd: Optional[int] = None
) -> list[str]:
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
    argv = [
        sys.executable,
        watchdog,
        "--ppid",
        str(os.getpid()),
        "--create-time",
        repr(parent_create_time),
    ]
    if start_gate_fd is not None:
        argv.extend(["--start-gate-fd", str(start_gate_fd)])
    return [*argv, "--", *command]


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
    deadline = time.monotonic() + max(0.0, wait_seconds)
    owner = LeaseOwner.from_env()
    try:
        with connect_project_locks(
            owner_db_path=kb.kanban_db_path(),
            deadline=deadline,
            on_wait=lambda holder: _emit("waiting", holder=holder),
        ) as conn:
            lease = acquire_with_wait(
                conn,
                owner_schema="ownerdb",
                project=project,
                operation=operation,
                owner=owner,
                lease_seconds=lease_seconds,
                wait_seconds=max(0.0, deadline - time.monotonic()),
                on_wait=lambda holder: _emit("waiting", holder=holder),
            )
            if lease is None:
                _emit(
                    "timeout",
                    project=_canonical_project(project),
                    waited_seconds=wait_seconds,
                )
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
            gate_read, gate_write = os.pipe()
            try:
                proc = subprocess.Popen(  # noqa: S603 -- explicit argv, no shell
                    _supervised_command(command, start_gate_fd=gate_read),
                    pass_fds=(gate_read,),
                )
            except Exception:
                os.close(gate_read)
                os.close(gate_write)
                release_project_lock(conn, lease, owner_schema="ownerdb")
                raise
            os.close(gate_read)
            if not attach_project_lock_process(
                conn, lease, proc.pid, owner_schema="ownerdb"
            ):
                os.close(gate_write)
                _stop_process(proc)
                release_project_lock(conn, lease, owner_schema="ownerdb")
                raise RuntimeError(
                    "lost project lease before command process registration"
                )
            os.write(gate_write, b"1")
            os.close(gate_write)
            interrupted = False
            old_handlers = {}

            def stop(_signum, _frame):
                nonlocal interrupted
                interrupted = True

            if (
                __import__("threading").current_thread()
                is __import__("threading").main_thread()
            ):
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
                            conn,
                            lease,
                            owner_schema="ownerdb",
                            lease_seconds=lease_seconds,
                        ):
                            lost = True
                            _emit("lost", fence=lease.fence)
                            _stop_process(proc)
                            break
                        next_renewal = time.monotonic() + lease_seconds / 3
                    time.sleep(
                        min(0.2, max(0.01, next_renewal - time.monotonic()))
                    )
                return_code = proc.wait()
            except BaseException:
                _stop_process(proc)
                raise
            finally:
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)
                released = release_project_lock(
                    conn, lease, owner_schema="ownerdb"
                )
                _emit(
                    "released" if released else "stale_release_rejected",
                    fence=lease.fence,
                )
            if lost or not released:
                return 74
            if interrupted:
                return 130
            return int(return_code)
    except ProjectLockTimeout:
        pass
    _emit(
        "timeout",
        project=_canonical_project(project),
        waited_seconds=wait_seconds,
    )
    return 73

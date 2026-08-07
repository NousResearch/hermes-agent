"""Structured provenance and visibility for bounded cross-profile executions.

This is a same-user policy boundary, not hostile-process isolation. A process
running as the same OS user can alter its environment or the ledger. The
module provides fail-closed Hermes entry-point enforcement and an auditable
record, but does not claim cryptographic authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import sqlite3
import time
from pathlib import Path
from typing import Any

import psutil

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None

_REQUIRED_DIRECT_FIELDS = {
    "authority_class",
    "authority_reference",
    "source",
    "target",
    "scope",
    "one_shot",
    "expires_at",
    "execution_id",
    "evidence",
    "terminal_condition",
}
_READ_ONLY_FORMS = {
    ("gateway", "status"),
    ("gateway", "health"),
    ("profile", "list"),
    ("kanban", "list"),
    ("kanban", "show"),
    ("kanban", "log"),
    ("kanban", "stats"),
    ("session", "list"),
    ("session", "search"),
}
_PROMPT_FLAGS = {"-q", "--query", "-z", "--oneshot", "--prompt"}
_AGENT_MODE_FLAGS = _PROMPT_FLAGS | {"-c", "--continue", "-r", "--resume", "--tui"}
_TOP_LEVEL_VALUE_FLAGS = {
    "-p",
    "--profile",
    "-m",
    "--model",
    "--provider",
    "--reasoning",
    "-t",
    "--toolsets",
    "-s",
    "--skills",
    "--usage-file",
}
_SENSITIVE_ARG_TERMS = {
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "passphrase",
    "passwd",
    "password",
    "secret",
    "token",
}
_SENSITIVE_KEY_PREFIXES = {
    "access",
    "api",
    "auth",
    "authentication",
    "client",
    "encryption",
    "private",
    "secret",
    "signing",
    "webhook",
}
_EXACT_SENSITIVE_OPTIONS = {
    "--authority",
    "--body",
    "--content",
    "--message",
    "--system-prompt",
}


class ExecutionAuthorityError(RuntimeError):
    """Raised when a cross-profile execution lacks valid bounded authority."""


def _ledger_path() -> Path:
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "execution-provenance.jsonl"


def _kanban_db_path() -> Path:
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "kanban.db"


def _command_and_args(argv: list[str]) -> tuple[str | None, list[str]]:
    """Return the real top-level command without treating values as commands."""
    args = list(argv[1:])
    index = 0
    while index < len(args):
        arg = args[index]
        option = arg.split("=", 1)[0]
        if option in _AGENT_MODE_FLAGS:
            return "__agent__", []
        if option in _TOP_LEVEL_VALUE_FLAGS:
            index += 1 if "=" in arg else 2
            continue
        if arg.startswith("-"):
            index += 1
            continue
        return arg, args[index + 1 :]
    return None, []


def is_read_only_invocation(argv: list[str]) -> bool:
    command, args = _command_and_args(argv)
    return command is not None and bool(args) and (command, args[0]) in _READ_ONLY_FORMS


def is_agent_invocation(argv: list[str]) -> bool:
    command, _args = _command_and_args(argv)
    return command in {None, "__agent__", "chat"}


def _is_noninteractive(argv: list[str]) -> bool:
    return any(
        arg in _PROMPT_FLAGS or arg.startswith(("--query=", "--oneshot=", "--prompt="))
        for arg in argv[1:]
    )


def _is_sensitive_option(arg: str) -> bool:
    """Return whether a long option name conventionally carries secret material."""
    if not arg.startswith("--"):
        return False
    option = arg.split("=", 1)[0]
    if option in _EXACT_SENSITIVE_OPTIONS:
        return True
    name = option[2:].lower().replace("_", "-")
    terms = [term for term in name.split("-") if term]
    if not terms:
        return False
    if terms[-1] in _SENSITIVE_ARG_TERMS:
        return True
    return terms[-1] == "key" and any(
        term in _SENSITIVE_KEY_PREFIXES for term in terms[:-1]
    )


def _redacted_execution_path(argv: list[str]) -> str:
    safe = list(argv)
    redact_next = False
    for index, arg in enumerate(safe):
        if redact_next:
            safe[index] = "[REDACTED]"
            redact_next = False
            continue
        option, separator, _value = arg.partition("=")
        if option in _PROMPT_FLAGS or _is_sensitive_option(option):
            if separator:
                safe[index] = option + "=[REDACTED]"
            elif index + 1 < len(safe):
                redact_next = True
    return shlex.join(safe)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    return bool(psutil.pid_exists(pid))


_STATUS_MAX_BYTES = 256 * 1024
_STATUS_MAX_ROWS = 50
_STATUS_MAX_LINE_BYTES = 16 * 1024


def _load_rows(
    path: Path,
    *,
    max_bytes: int = _STATUS_MAX_BYTES,
    max_rows: int = _STATUS_MAX_ROWS,
    max_line_bytes: int = _STATUS_MAX_LINE_BYTES,
) -> list[dict[str, Any]]:
    """Read a bounded tail of the ledger for synchronous status rendering."""
    try:
        size = path.stat().st_size
        start = max(0, size - max_bytes)
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(max_bytes)
    except OSError:
        return []
    if start:
        separator = data.find(b"\n")
        if separator < 0:
            return []
        data = data[separator + 1 :]

    rows: list[dict[str, Any]] = []
    for raw_line in data.splitlines()[-max_rows:]:
        if not raw_line or len(raw_line) > max_line_bytes:
            continue
        try:
            value = json.loads(raw_line.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte or fail; os.write is permitted to short-write."""
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise OSError("zero-length ledger write")
        offset += written


def _fsync_directory(path: Path) -> None:
    """Persist directory entries on POSIX after creating audit artifacts."""
    if (
        os.name == "nt"
    ):  # File fsync persists NTFS file metadata; directories cannot be os.open'ed.
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _consumption_marker_path(path: Path, execution_id: str) -> Path:
    digest = hashlib.sha256(execution_id.encode("utf-8")).hexdigest()
    return path.with_name(path.name + ".consumed") / digest


def _append_row_once(path: Path, row: dict[str, Any], *, reject_replay: bool) -> None:
    """Persist an audit row while atomically consuming one-shot IDs."""
    try:
        payload = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ExecutionAuthorityError("execution record is not finite JSON") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    ledger_existed = path.exists()
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    windows_lock_fd: int | None = None
    marker_fd: int | None = None
    locked = False
    original_size = 0
    append_started = False
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True
        elif msvcrt is not None:  # pragma: no cover - Windows
            lock_path = path.with_name(path.name + ".lock")
            windows_lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            if os.fstat(windows_lock_fd).st_size == 0:
                _write_all(windows_lock_fd, b"\0")
            os.lseek(windows_lock_fd, 0, os.SEEK_SET)
            getattr(msvcrt, "locking")(windows_lock_fd, getattr(msvcrt, "LK_LOCK"), 1)
            locked = True
        else:  # pragma: no cover - unsupported Python platform
            raise ExecutionAuthorityError("cross-process ledger locking is unavailable")

        original_size = os.fstat(fd).st_size
        execution_id = str(row.get("execution_id"))
        marker_path: Path | None = None
        if reject_replay:
            marker_path = _consumption_marker_path(path, execution_id)
            if marker_path.exists():
                raise ExecutionAuthorityError("one-shot execution ID already used")

            # Preserve compatibility with ledgers written before durable marker
            # files existed and fail closed on ambiguous/corrupt audit history.
            with os.fdopen(os.dup(fd), "rb") as prior_handle:
                prior_handle.seek(0)
                while prior_handle.tell() < original_size:
                    raw_line = prior_handle.readline(_STATUS_MAX_LINE_BYTES + 1)
                    if len(raw_line) > _STATUS_MAX_LINE_BYTES or not raw_line.endswith(
                        b"\n"
                    ):
                        raise ExecutionAuthorityError(
                            "execution ledger integrity check failed"
                        )
                    if not raw_line.strip():
                        continue
                    try:
                        value = json.loads(raw_line.decode("utf-8", errors="strict"))
                    except (json.JSONDecodeError, UnicodeError) as exc:
                        raise ExecutionAuthorityError(
                            "execution ledger integrity check failed"
                        ) from exc
                    if (
                        isinstance(value, dict)
                        and str(value.get("execution_id")) == execution_id
                    ):
                        raise ExecutionAuthorityError(
                            "one-shot execution ID already used"
                        )

            marker_directory = marker_path.parent
            marker_directory_existed = marker_directory.exists()
            marker_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not marker_directory_existed:
                _fsync_directory(marker_directory.parent)
            try:
                marker_fd = os.open(
                    marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
            except FileExistsError as exc:
                raise ExecutionAuthorityError(
                    "one-shot execution ID already used"
                ) from exc
            _write_all(marker_fd, (execution_id + "\n").encode("utf-8"))
            os.fsync(marker_fd)
            os.close(marker_fd)
            marker_fd = None
            _fsync_directory(marker_directory)

        os.lseek(fd, 0, os.SEEK_END)
        append_started = True
        _write_all(fd, payload)
        os.fsync(fd)
        if not ledger_existed:
            _fsync_directory(path.parent)
    except OSError as exc:
        if append_started:
            try:
                os.ftruncate(fd, original_size)
                os.fsync(fd)
            except OSError:
                pass
        raise ExecutionAuthorityError("execution record persistence failed") from exc
    finally:
        if marker_fd is not None:
            os.close(marker_fd)
        if locked and fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        elif (
            locked and msvcrt is not None and windows_lock_fd is not None
        ):  # pragma: no cover
            os.lseek(windows_lock_fd, 0, os.SEEK_SET)
            getattr(msvcrt, "locking")(windows_lock_fd, getattr(msvcrt, "LK_UNLCK"), 1)
        if windows_lock_fd is not None:
            os.close(windows_lock_fd)
        os.close(fd)


def _is_finite_future(value: Any, now: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(numeric) and numeric > now


def _validate_kanban_custody(
    *,
    source: str,
    target: str,
    task_id: str,
    run_id: str,
    claim_lock: str,
    db_path: Path,
    pid: int,
) -> None:
    del source  # Source labels are audit context; live custody is authoritative.
    if not db_path.is_file():
        raise ExecutionAuthorityError("Kanban custody database is unavailable")
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        task = conn.execute(
            "SELECT assignee, status, claim_lock, claim_expires, current_run_id, worker_pid "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        run = conn.execute(
            "SELECT task_id, profile, status, claim_lock, claim_expires, worker_pid "
            "FROM task_runs WHERE id = ?",
            (int(run_id),),
        ).fetchone()
        conn.close()
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        raise ExecutionAuthorityError("Kanban custody lookup failed closed") from exc
    now = time.time()
    valid = (
        task is not None
        and run is not None
        and task["assignee"] == target
        and run["profile"] == target
        and task["status"] == "running"
        and run["status"] == "running"
        and str(task["current_run_id"]) == str(run_id)
        and run["task_id"] == task_id
        and task["claim_lock"] == claim_lock
        and run["claim_lock"] == claim_lock
        and _is_finite_future(task["claim_expires"], now)
        and _is_finite_future(run["claim_expires"], now)
        and all(
            value in (None, pid) for value in (task["worker_pid"], run["worker_pid"])
        )
    )
    if not valid:
        raise ExecutionAuthorityError(
            "Kanban custody does not match the active assigned run"
        )


def authorize_profile_execution(
    *,
    source: str,
    target: str,
    argv: list[str],
    authority_json: str | None,
    interactive: bool = False,
    ledger_path: Path | None = None,
    kanban_task: str | None = None,
    kanban_run: str | None = None,
    kanban_claim_lock: str | None = None,
    kanban_db_path: Path | None = None,
    pid: int | None = None,
) -> dict[str, Any] | None:
    """Authorize and record a cross-profile agent execution."""
    source = str(source or "").strip()
    target = str(target or "").strip()
    if (
        not source
        or not target
        or source == target
        or not is_agent_invocation(argv)
        or is_read_only_invocation(argv)
    ):
        return None
    if interactive and not _is_noninteractive(argv):
        return None

    path = ledger_path or _ledger_path()
    now = time.time()
    if kanban_task and kanban_run and kanban_claim_lock:
        db_path = kanban_db_path or Path(
            os.environ.get("HERMES_KANBAN_DB") or _kanban_db_path()
        )
        _validate_kanban_custody(
            source=source,
            target=target,
            task_id=kanban_task,
            run_id=kanban_run,
            claim_lock=kanban_claim_lock,
            db_path=db_path,
            pid=int(pid or os.getpid()),
        )
        record: dict[str, Any] = {
            "authority_class": "kanban_dispatch",
            "authority_reference": f"kanban:{kanban_task}:run:{kanban_run}",
            "source": source,
            "target": target,
            "scope": f"assigned Kanban task {kanban_task}",
            "one_shot": True,
            "expires_at": None,
            "execution_id": f"kanban-{kanban_task}-run-{kanban_run}",
            "evidence": f"kanban:{kanban_task}",
            "terminal_condition": "Kanban terminal action or worker exit",
        }
    else:
        if not authority_json:
            raise ExecutionAuthorityError(
                f"cross-profile execution {source}->{target} requires structured authority"
            )
        try:
            parsed = json.loads(authority_json)
        except json.JSONDecodeError as exc:
            raise ExecutionAuthorityError(
                "structured authority is not valid JSON"
            ) from exc
        if not isinstance(parsed, dict):
            raise ExecutionAuthorityError("structured authority must be a JSON object")
        missing = sorted(_REQUIRED_DIRECT_FIELDS - parsed.keys())
        if missing:
            raise ExecutionAuthorityError(
                f"structured authority missing: {', '.join(missing)}"
            )
        record = dict(parsed)
        for field in _REQUIRED_DIRECT_FIELDS - {"one_shot", "expires_at"}:
            if not str(record[field]).strip():
                raise ExecutionAuthorityError(
                    f"authority field {field} must not be empty"
                )
        if record["authority_class"] != "direct_one_shot":
            raise ExecutionAuthorityError("authority class must be direct_one_shot")
        if record["source"] != source:
            raise ExecutionAuthorityError("authority source mismatch")
        if record["target"] != target:
            raise ExecutionAuthorityError("authority target mismatch")
        if record["one_shot"] is not True:
            raise ExecutionAuthorityError("direct exception must be one-shot")
        try:
            expires_at = float(record["expires_at"])
        except (TypeError, ValueError) as exc:
            raise ExecutionAuthorityError(
                "authority expiry must be a Unix timestamp"
            ) from exc
        if not math.isfinite(expires_at):
            raise ExecutionAuthorityError("authority expiry must be finite")
        if expires_at <= now:
            raise ExecutionAuthorityError("structured authority expired")

    record.update({
        "execution_path": _redacted_execution_path(argv),
        "kanban_tracked": bool(kanban_task and kanban_run),
        "pid": int(pid or os.getpid()),
        "started_at": now,
        "state": "running",
    })
    _append_row_once(path, record, reject_replay=True)
    return record


def _coerce_number(value: Any, *, integer: bool = False) -> int | float:
    try:
        numeric = int(value) if integer else float(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not integer and not math.isfinite(numeric):
        return 0
    return numeric


def list_execution_status(*, ledger_path: Path | None = None) -> list[dict[str, Any]]:
    rows = _load_rows(ledger_path or _ledger_path())
    for row in rows:
        if row.get("state") == "running" and not _pid_alive(
            int(_coerce_number(row.get("pid"), integer=True))
        ):
            row["state"] = "terminal"
    return sorted(
        rows,
        key=lambda row: float(_coerce_number(row.get("started_at"))),
        reverse=True,
    )


def _bounded_status_value(value: Any, *, max_chars: int = 32) -> str:
    """Render one ledger field as bounded, single-line, printable status text."""
    text = " ".join(str(value).split())
    text = "".join(char for char in text if char.isprintable()) or "unknown"
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


def format_execution_status(
    *, ledger_path: Path | None = None, limit: int = 10, max_total_chars: int = 1200
) -> list[str]:
    """Return bounded status lines safe to append to gateway status responses."""
    lines: list[str] = []
    total_chars = 0
    for row in list_execution_status(ledger_path=ledger_path)[: max(0, min(limit, 10))]:
        tracked = "yes" if row.get("kanban_tracked") else "no"
        line = (
            "Execution "
            f"{_bounded_status_value(row.get('execution_id', 'unknown'))}: "
            f"source={_bounded_status_value(row.get('source', 'unknown'))} "
            f"authority={_bounded_status_value(row.get('authority_class', 'unknown'))}/"
            f"{_bounded_status_value(row.get('authority_reference', 'unknown'))} "
            f"target={_bounded_status_value(row.get('target', 'unknown'))} "
            f"path={_bounded_status_value(row.get('execution_path', 'unknown'), max_chars=64)} "
            f"kanban={tracked} scope={_bounded_status_value(row.get('scope', 'unknown'))} "
            f"one_shot={_bounded_status_value(row.get('one_shot', 'unknown'))} "
            f"expires={_bounded_status_value(row.get('expires_at', 'none'))} "
            f"evidence={_bounded_status_value(row.get('evidence', 'unknown'))} "
            f"terminal={_bounded_status_value(row.get('terminal_condition', 'unknown'))} "
            f"state={_bounded_status_value(row.get('state', 'unknown'))}"
        )
        line = line[:600]
        added_chars = len(line) + (1 if lines else 0)
        if total_chars + added_chars > max_total_chars:
            break
        lines.append(line)
        total_chars += added_chars
    return lines

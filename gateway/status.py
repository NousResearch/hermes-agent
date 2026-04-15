"""
Gateway runtime status helpers.

Provides PID-file based detection of whether the gateway daemon is running,
used by send_message's check_fn to gate availability in the CLI.

The PID file lives at ``{HERMES_HOME}/gateway.pid``.  HERMES_HOME defaults to
``~/.hermes`` but can be overridden via the environment variable.  This means
separate HERMES_HOME directories naturally get separate PID files — a property
that will be useful when we add named profiles (multiple agents running
concurrently under distinct configurations).
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Any, Optional

_GATEWAY_KIND = "hermes-gateway"
_RUNTIME_STATUS_FILE = "gateway_state.json"
_LOCKS_DIRNAME = "gateway-locks"
_IS_WINDOWS = sys.platform == "win32"
_UNSET = object()
_VALID_STARTUP_CHECK_STATES = {"pending", "ready", "failed"}


def _get_pid_path() -> Path:
    """Return the path to the gateway PID file, respecting HERMES_HOME."""
    home = get_hermes_home()
    return home / "gateway.pid"


def _get_runtime_status_path() -> Path:
    """Return the persisted runtime health/status file path."""
    return _get_pid_path().with_name(_RUNTIME_STATUS_FILE)


def _get_lock_dir() -> Path:
    """Return the machine-local directory for token-scoped gateway locks."""
    override = os.getenv("HERMES_GATEWAY_LOCK_DIR")
    if override:
        return Path(override)
    state_home = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state_home / "hermes" / _LOCKS_DIRNAME


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def terminate_pid(pid: int, *, force: bool = False) -> None:
    """Terminate a PID with platform-appropriate force semantics.

    POSIX uses SIGTERM/SIGKILL. Windows uses taskkill /T /F for true force-kill
    because os.kill(..., SIGTERM) is not equivalent to a tree-killing hard stop.
    """
    if force and _IS_WINDOWS:
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            os.kill(pid, signal.SIGTERM)
            return

        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise OSError(details or f"taskkill failed for PID {pid}")
        return

    sig = signal.SIGTERM if not force else getattr(signal, "SIGKILL", signal.SIGTERM)
    os.kill(pid, sig)


def _scope_hash(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _get_scope_lock_path(scope: str, identity: str) -> Path:
    return _get_lock_dir() / f"{scope}-{_scope_hash(identity)}.lock"


def _get_process_start_time(pid: int) -> Optional[int]:
    """Return the kernel start time for a process when available."""
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        # Field 22 in /proc/<pid>/stat is process start time (clock ticks).
        return int(stat_path.read_text().split()[21])
    except (FileNotFoundError, IndexError, PermissionError, ValueError, OSError):
        return None


def _read_process_cmdline(pid: int) -> Optional[str]:
    """Return the process command line as a space-separated string."""
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        raw = cmdline_path.read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return None

    if not raw:
        return None
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()


def _looks_like_gateway_process(pid: int) -> bool:
    """Return True when the live PID still looks like the Hermes gateway."""
    cmdline = _read_process_cmdline(pid)
    if not cmdline:
        return False

    patterns = (
        "hermes_cli.main gateway",
        "hermes_cli/main.py gateway",
        "hermes gateway",
        "gateway/run.py",
    )
    return any(pattern in cmdline for pattern in patterns)


def _record_looks_like_gateway(record: dict[str, Any]) -> bool:
    """Validate gateway identity from PID-file metadata when cmdline is unavailable."""
    if record.get("kind") != _GATEWAY_KIND:
        return False

    argv = record.get("argv")
    if not isinstance(argv, list) or not argv:
        return False

    cmdline = " ".join(str(part) for part in argv)
    patterns = (
        "hermes_cli.main gateway",
        "hermes_cli/main.py gateway",
        "hermes gateway",
        "gateway/run.py",
    )
    return any(pattern in cmdline for pattern in patterns)


def _build_pid_record() -> dict:
    return {
        "pid": os.getpid(),
        "kind": _GATEWAY_KIND,
        "argv": list(sys.argv),
        "start_time": _get_process_start_time(os.getpid()),
    }


def _build_runtime_status_record() -> dict[str, Any]:
    payload = _build_pid_record()
    payload.update({
        "gateway_state": "starting",
        "exit_reason": None,
        "restart_requested": False,
        "active_agents": 0,
        "platforms": {},
        "startup_checks": {},
        "updated_at": _utc_now_iso(),
    })
    return payload


def _normalize_startup_check_entries(
    startup_checks: Optional[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Normalize persisted startup readiness entries."""
    if not isinstance(startup_checks, dict):
        return {}

    now = _utc_now_iso()
    normalized: dict[str, dict[str, Any]] = {}
    for raw_id, raw_payload in startup_checks.items():
        check_id = str(raw_id).strip()
        if not check_id:
            continue
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        state = str(payload.get("state", "pending")).strip().lower()
        if state not in _VALID_STARTUP_CHECK_STATES:
            state = "pending"
        normalized[check_id] = {
            "state": state,
            "required": bool(payload.get("required", True)),
            "source": payload.get("source"),
            "detail": payload.get("detail"),
            "updated_at": payload.get("updated_at") or now,
        }
    return normalized


def _read_json_file(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _read_pid_record() -> Optional[dict]:
    pid_path = _get_pid_path()
    if not pid_path.exists():
        return None

    raw = pid_path.read_text().strip()
    if not raw:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        try:
            return {"pid": int(raw)}
        except ValueError:
            return None

    if isinstance(payload, int):
        return {"pid": payload}
    if isinstance(payload, dict):
        return payload
    return None


def write_pid_file() -> None:
    """Write the current process PID and metadata to the gateway PID file."""
    _write_json_file(_get_pid_path(), _build_pid_record())


def write_runtime_status(
    *,
    gateway_state: Any = _UNSET,
    exit_reason: Any = _UNSET,
    restart_requested: Any = _UNSET,
    active_agents: Any = _UNSET,
    startup_checks: Any = _UNSET,
    platform: Any = _UNSET,
    platform_state: Any = _UNSET,
    error_code: Any = _UNSET,
    error_message: Any = _UNSET,
) -> None:
    """Persist gateway runtime health information for diagnostics/status."""
    path = _get_runtime_status_path()
    payload = _read_json_file(path) or _build_runtime_status_record()
    payload.setdefault("platforms", {})
    payload.setdefault("kind", _GATEWAY_KIND)
    payload["pid"] = os.getpid()
    payload["start_time"] = _get_process_start_time(os.getpid())
    payload["updated_at"] = _utc_now_iso()

    if gateway_state is not _UNSET:
        payload["gateway_state"] = gateway_state
    if exit_reason is not _UNSET:
        payload["exit_reason"] = exit_reason
    if restart_requested is not _UNSET:
        payload["restart_requested"] = bool(restart_requested)
    if active_agents is not _UNSET:
        payload["active_agents"] = max(0, int(active_agents))
    if startup_checks is not _UNSET:
        payload["startup_checks"] = _normalize_startup_check_entries(startup_checks)

    if platform is not _UNSET:
        platform_payload = payload["platforms"].get(platform, {})
        if platform_state is not _UNSET:
            platform_payload["state"] = platform_state
        if error_code is not _UNSET:
            platform_payload["error_code"] = error_code
        if error_message is not _UNSET:
            platform_payload["error_message"] = error_message
        platform_payload["updated_at"] = _utc_now_iso()
        payload["platforms"][platform] = platform_payload

    _write_json_file(path, payload)


def read_runtime_status() -> Optional[dict[str, Any]]:
    """Read the persisted gateway runtime health/status information."""
    payload = _read_json_file(_get_runtime_status_path())
    if payload is None:
        return None
    payload.setdefault("platforms", {})
    payload["startup_checks"] = _normalize_startup_check_entries(payload.get("startup_checks"))
    return payload


def reset_startup_checks(checks: Optional[list[dict[str, Any]]] = None) -> dict[str, dict[str, Any]]:
    """Replace persisted startup readiness checks for the current run."""
    normalized: dict[str, dict[str, Any]] = {}
    now = _utc_now_iso()

    for hook in checks or []:
        if not isinstance(hook, dict):
            continue
        readiness = hook.get("startup_readiness")
        if not isinstance(readiness, dict):
            continue
        check_id = str(readiness.get("id", "")).strip()
        if not check_id:
            continue
        normalized[check_id] = {
            "state": "pending",
            "required": bool(readiness.get("required", True)),
            "source": hook.get("name"),
            "detail": None,
            "updated_at": now,
        }

    write_runtime_status(startup_checks=normalized)
    return normalized


def update_startup_check(
    check_id: str,
    state: str,
    *,
    detail: Any = _UNSET,
    required: Any = _UNSET,
    source: Any = _UNSET,
) -> dict[str, Any]:
    """Update a single startup readiness check in the runtime status file."""
    normalized_id = str(check_id).strip()
    if not normalized_id:
        raise ValueError("startup readiness check id is required")

    normalized_state = str(state).strip().lower()
    if normalized_state not in _VALID_STARTUP_CHECK_STATES:
        raise ValueError(f"invalid startup readiness state: {state}")

    path = _get_runtime_status_path()
    payload = _read_json_file(path) or _build_runtime_status_record()
    checks = _normalize_startup_check_entries(payload.get("startup_checks"))
    existing = checks.get(normalized_id, {})
    now = _utc_now_iso()

    checks[normalized_id] = {
        "state": normalized_state,
        "required": bool(existing.get("required", True) if required is _UNSET else required),
        "source": existing.get("source") if source is _UNSET else source,
        "detail": existing.get("detail") if detail is _UNSET else detail,
        "updated_at": now,
    }

    payload["startup_checks"] = checks
    payload.setdefault("platforms", {})
    payload.setdefault("kind", _GATEWAY_KIND)
    payload["pid"] = os.getpid()
    payload["start_time"] = _get_process_start_time(os.getpid())
    payload["updated_at"] = now
    _write_json_file(path, payload)
    return checks[normalized_id]


def mark_startup_check_pending(
    check_id: str,
    *,
    detail: Any = _UNSET,
    required: Any = _UNSET,
    source: Any = _UNSET,
) -> dict[str, Any]:
    return update_startup_check(check_id, "pending", detail=detail, required=required, source=source)


def mark_startup_check_ready(
    check_id: str,
    *,
    detail: Any = _UNSET,
    required: Any = _UNSET,
    source: Any = _UNSET,
) -> dict[str, Any]:
    return update_startup_check(check_id, "ready", detail=detail, required=required, source=source)


def mark_startup_check_failed(
    check_id: str,
    *,
    detail: Any = _UNSET,
    required: Any = _UNSET,
    source: Any = _UNSET,
) -> dict[str, Any]:
    return update_startup_check(check_id, "failed", detail=detail, required=required, source=source)


def remove_pid_file() -> None:
    """Remove the gateway PID file, but only if it belongs to this process.

    During --replace handoffs, the old process's atexit handler can fire AFTER
    the new process has written its own PID file.  Blindly removing the file
    would delete the new process's record, leaving the gateway running with no
    PID file (invisible to ``get_running_pid()``).
    """
    try:
        path = _get_pid_path()
        record = _read_json_file(path)
        if record is not None:
            try:
                file_pid = int(record["pid"])
            except (KeyError, TypeError, ValueError):
                file_pid = None
            if file_pid is not None and file_pid != os.getpid():
                # PID file belongs to a different process — leave it alone.
                return
        path.unlink(missing_ok=True)
    except Exception:
        pass


def acquire_scoped_lock(scope: str, identity: str, metadata: Optional[dict[str, Any]] = None) -> tuple[bool, Optional[dict[str, Any]]]:
    """Acquire a machine-local lock keyed by scope + identity.

    Used to prevent multiple local gateways from using the same external identity
    at once (e.g. the same Telegram bot token across different HERMES_HOME dirs).
    """
    lock_path = _get_scope_lock_path(scope, identity)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **_build_pid_record(),
        "scope": scope,
        "identity_hash": _scope_hash(identity),
        "metadata": metadata or {},
        "updated_at": _utc_now_iso(),
    }

    existing = _read_json_file(lock_path)
    if existing is None and lock_path.exists():
        # Lock file exists but is empty or contains invalid JSON — treat as
        # stale.  This happens when a previous process was killed between
        # O_CREAT|O_EXCL and the subsequent json.dump() (e.g. DNS failure
        # during rapid Slack reconnect retries).
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass
    if existing:
        try:
            existing_pid = int(existing["pid"])
        except (KeyError, TypeError, ValueError):
            existing_pid = None

        if existing_pid == os.getpid() and existing.get("start_time") == record.get("start_time"):
            _write_json_file(lock_path, record)
            return True, existing

        stale = existing_pid is None
        if not stale:
            try:
                os.kill(existing_pid, 0)
            except (ProcessLookupError, PermissionError):
                stale = True
            else:
                current_start = _get_process_start_time(existing_pid)
                if (
                    existing.get("start_time") is not None
                    and current_start is not None
                    and current_start != existing.get("start_time")
                ):
                    stale = True
                # Check if process is stopped (Ctrl+Z / SIGTSTP) — stopped
                # processes still respond to os.kill(pid, 0) but are not
                # actually running. Treat them as stale so --replace works.
                if not stale:
                    try:
                        _proc_status = Path(f"/proc/{existing_pid}/status")
                        if _proc_status.exists():
                            for _line in _proc_status.read_text().splitlines():
                                if _line.startswith("State:"):
                                    _state = _line.split()[1]
                                    if _state in ("T", "t"):  # stopped or tracing stop
                                        stale = True
                                    break
                    except (OSError, PermissionError):
                        pass
        if stale:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
        else:
            return False, existing

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False, _read_json_file(lock_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
    except Exception:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return True, None


def release_scoped_lock(scope: str, identity: str) -> None:
    """Release a previously-acquired scope lock when owned by this process."""
    lock_path = _get_scope_lock_path(scope, identity)
    existing = _read_json_file(lock_path)
    if not existing:
        return
    if existing.get("pid") != os.getpid():
        return
    if existing.get("start_time") != _get_process_start_time(os.getpid()):
        return
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


def release_all_scoped_locks() -> int:
    """Remove all scoped lock files in the lock directory.

    Called during --replace to clean up stale locks left by stopped/killed
    gateway processes that did not release their locks gracefully.
    Returns the number of lock files removed.
    """
    lock_dir = _get_lock_dir()
    removed = 0
    if lock_dir.exists():
        for lock_file in lock_dir.glob("*.lock"):
            try:
                lock_file.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
    return removed


def get_running_pid() -> Optional[int]:
    """Return the PID of a running gateway instance, or ``None``.

    Checks the PID file and verifies the process is actually alive.
    Cleans up stale PID files automatically.
    """
    record = _read_pid_record()
    if not record:
        remove_pid_file()
        return None

    try:
        pid = int(record["pid"])
    except (KeyError, TypeError, ValueError):
        remove_pid_file()
        return None

    try:
        os.kill(pid, 0)  # signal 0 = existence check, no actual signal sent
    except (ProcessLookupError, PermissionError):
        remove_pid_file()
        return None

    recorded_start = record.get("start_time")
    current_start = _get_process_start_time(pid)
    if recorded_start is not None and current_start is not None and current_start != recorded_start:
        remove_pid_file()
        return None

    if not _looks_like_gateway_process(pid):
        if not _record_looks_like_gateway(record):
            remove_pid_file()
            return None

    return pid


def is_gateway_running() -> bool:
    """Check if the gateway daemon is currently running."""
    return get_running_pid() is not None

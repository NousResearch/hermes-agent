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

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

_GATEWAY_KIND = "hermes-gateway"
_RUNTIME_STATUS_FILE = "gateway_state.json"
_LOCKS_DIRNAME = "gateway-locks"
_IS_WINDOWS = sys.platform == "win32"
_UNSET = object()
_GATEWAY_LOCK_FILENAME = "gateway.lock"
_gateway_lock_handle = None


def _get_pid_path() -> Path:
    """Return the path to the gateway PID file, respecting HERMES_HOME."""
    home = get_hermes_home()
    return home / "gateway.pid"


def _get_gateway_lock_path(pid_path: Optional[Path] = None) -> Path:
    """Return the path to the runtime gateway lock file."""
    if pid_path is not None:
        return pid_path.with_name(_GATEWAY_LOCK_FILENAME)
    home = get_hermes_home()
    return home / _GATEWAY_LOCK_FILENAME


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
        "hermes-gateway",
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
        "updated_at": _utc_now_iso(),
    })
    return payload


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


def _read_pid_record(pid_path: Optional[Path] = None) -> Optional[dict]:
    pid_path = pid_path or _get_pid_path()
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


def _read_gateway_lock_record(lock_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    return _read_pid_record(lock_path or _get_gateway_lock_path())


def _pid_from_record(record: Optional[dict[str, Any]]) -> Optional[int]:
    if not record:
        return None
    try:
        return int(record["pid"])
    except (KeyError, TypeError, ValueError):
        return None


def _cleanup_invalid_pid_path(pid_path: Path, *, cleanup_stale: bool) -> None:
    """Delete a stale gateway PID file (and its sibling lock metadata).

    Called from ``get_running_pid()`` after the runtime lock has already been
    confirmed inactive, so the on-disk metadata is known to belong to a dead
    process.  Unlike ``remove_pid_file()`` (which defensively refuses to delete
    a PID file whose ``pid`` field differs from ``os.getpid()`` to protect
    ``--replace`` handoffs), this path force-unlinks both files so the next
    startup sees a clean slate.
    """
    if not cleanup_stale:
        return
    try:
        pid_path.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        _get_gateway_lock_path(pid_path).unlink(missing_ok=True)
    except Exception:
        pass


def _write_gateway_lock_record(handle) -> None:
    handle.seek(0)
    handle.truncate()
    json.dump(_build_pid_record(), handle)
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError:
        pass


def _try_acquire_file_lock(handle) -> bool:
    try:
        if _IS_WINDOWS:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("\n")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _release_file_lock(handle) -> None:
    try:
        if _IS_WINDOWS:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def acquire_gateway_runtime_lock() -> bool:
    """Claim the cross-process runtime lock for the gateway.

    Unlike the PID file, the lock is owned by the live process itself. If the
    process dies abruptly, the OS releases the lock automatically.
    """
    global _gateway_lock_handle
    if _gateway_lock_handle is not None:
        return True

    path = _get_gateway_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+", encoding="utf-8")
    if not _try_acquire_file_lock(handle):
        handle.close()
        return False
    _write_gateway_lock_record(handle)
    _gateway_lock_handle = handle
    return True


def release_gateway_runtime_lock() -> None:
    """Release the gateway runtime lock when owned by this process."""
    global _gateway_lock_handle
    handle = _gateway_lock_handle
    if handle is None:
        return
    _gateway_lock_handle = None
    _release_file_lock(handle)
    try:
        handle.close()
    except OSError:
        pass


def is_gateway_runtime_lock_active(lock_path: Optional[Path] = None) -> bool:
    """Return True when some process currently owns the gateway runtime lock."""
    global _gateway_lock_handle
    resolved_lock_path = lock_path or _get_gateway_lock_path()
    if _gateway_lock_handle is not None and resolved_lock_path == _get_gateway_lock_path():
        return True

    if not resolved_lock_path.exists():
        return False

    handle = open(resolved_lock_path, "a+", encoding="utf-8")
    try:
        if _try_acquire_file_lock(handle):
            _release_file_lock(handle)
            return False
        return True
    finally:
        try:
            handle.close()
        except OSError:
            pass


def write_pid_file() -> None:
    """Write the current process PID and metadata to the gateway PID file.

    Uses atomic O_CREAT | O_EXCL creation so that concurrent --replace
    invocations race: exactly one process wins and the rest get
    FileExistsError.
    """
    path = _get_pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps(_build_pid_record())
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise  # Let caller decide: another gateway is racing us
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(record)
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_runtime_status(
    *,
    gateway_state: Any = _UNSET,
    exit_reason: Any = _UNSET,
    restart_requested: Any = _UNSET,
    active_agents: Any = _UNSET,
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
    return _read_json_file(_get_runtime_status_path())


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
            except (ProcessLookupError, PermissionError, OSError):
                # Windows raises OSError with WinError 87 for invalid pid check
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


# ── --replace takeover marker ─────────────────────────────────────────
#
# When a new gateway starts with ``--replace``, it SIGTERMs the existing
# gateway so it can take over the bot token. PR #5646 made SIGTERM exit
# the gateway with code 1 so ``Restart=on-failure`` can revive it after
# unexpected kills — but that also means a --replace takeover target
# exits 1, which tricks systemd into reviving it 30 seconds later,
# starting a flap loop against the replacer when both services are
# enabled in the user's systemd (e.g. ``hermes.service`` + ``hermes-
# gateway.service``).
#
# The takeover marker breaks the loop: the replacer writes a short-lived
# file naming the target PID + start_time BEFORE sending SIGTERM.
# The target's shutdown handler reads the marker and, if it names
# this process, treats the SIGTERM as a planned takeover and exits 0.
# The marker is unlinked after the target has consumed it, so a stale
# marker left by a crashed replacer can grief at most one future
# shutdown on the same PID — and only within _TAKEOVER_MARKER_TTL_S.

_TAKEOVER_MARKER_FILENAME = ".gateway-takeover.json"
_TAKEOVER_MARKER_TTL_S = 60  # Marker older than this is treated as stale


def _get_takeover_marker_path() -> Path:
    """Return the path to the --replace takeover marker file."""
    home = get_hermes_home()
    return home / _TAKEOVER_MARKER_FILENAME


def write_takeover_marker(target_pid: int) -> bool:
    """Record that ``target_pid`` is being replaced by the current process.

    Captures the target's ``start_time`` so that PID reuse after the
    target exits cannot later match the marker. Also records the
    replacer's PID and a UTC timestamp for TTL-based staleness checks.

    Returns True on successful write, False on any failure. The caller
    should proceed with the SIGTERM even if the write fails (the marker
    is a best-effort signal, not a correctness requirement).
    """
    try:
        target_start_time = _get_process_start_time(target_pid)
        record = {
            "target_pid": target_pid,
            "target_start_time": target_start_time,
            "replacer_pid": os.getpid(),
            "written_at": _utc_now_iso(),
        }
        _write_json_file(_get_takeover_marker_path(), record)
        return True
    except (OSError, PermissionError):
        return False


def consume_takeover_marker_for_self() -> bool:
    """Check & unlink the takeover marker if it names the current process.

    Returns True only when a valid (non-stale) marker names this PID +
    start_time. A returning True indicates the current SIGTERM is a
    planned --replace takeover; the caller should exit 0 instead of
    signalling ``_signal_initiated_shutdown``.

    Always unlinks the marker on match (and on detected staleness) so
    subsequent unrelated signals don't re-trigger.
    """
    path = _get_takeover_marker_path()
    record = _read_json_file(path)
    if not record:
        return False

    # Any malformed or stale marker → drop it and return False
    try:
        target_pid = int(record["target_pid"])
        target_start_time = record.get("target_start_time")
        written_at = record.get("written_at") or ""
    except (KeyError, TypeError, ValueError):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return False

    # TTL guard: a stale marker older than _TAKEOVER_MARKER_TTL_S is ignored.
    stale = False
    try:
        written_dt = datetime.fromisoformat(written_at)
        age = (datetime.now(timezone.utc) - written_dt).total_seconds()
        if age > _TAKEOVER_MARKER_TTL_S:
            stale = True
    except (TypeError, ValueError):
        stale = True  # Unparseable timestamp — treat as stale

    if stale:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return False

    # Does the marker name THIS process?
    our_pid = os.getpid()
    our_start_time = _get_process_start_time(our_pid)
    matches = (
        target_pid == our_pid
        and target_start_time is not None
        and our_start_time is not None
        and target_start_time == our_start_time
    )

    # Consume the marker whether it matched or not — a marker that doesn't
    # match our identity is stale-for-us anyway.
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass

    return matches


def clear_takeover_marker() -> None:
    """Remove the takeover marker unconditionally. Safe to call repeatedly."""
    try:
        _get_takeover_marker_path().unlink(missing_ok=True)
    except OSError:
        pass


def get_running_pid(
    pid_path: Optional[Path] = None,
    *,
    cleanup_stale: bool = True,
) -> Optional[int]:
    """Return the PID of a running gateway instance, or ``None``.

    Checks the PID file and verifies the process is actually alive.
    Cleans up stale PID files automatically.
    """
    resolved_pid_path = pid_path or _get_pid_path()
    resolved_lock_path = _get_gateway_lock_path(resolved_pid_path)
    lock_active = is_gateway_runtime_lock_active(resolved_lock_path)
    if not lock_active:
        _cleanup_invalid_pid_path(resolved_pid_path, cleanup_stale=cleanup_stale)
        return None

    primary_record = _read_pid_record(resolved_pid_path)
    fallback_record = _read_gateway_lock_record(resolved_lock_path)

    for record in (primary_record, fallback_record):
        pid = _pid_from_record(record)
        if pid is None:
            continue

        try:
            os.kill(pid, 0)  # signal 0 = existence check, no actual signal sent
        except ProcessLookupError:
            continue
        except PermissionError:
            # The process exists but belongs to another user/service scope.
            # With the runtime lock still held, prefer keeping it visible
            # rather than deleting the PID file as "stale".
            if _record_looks_like_gateway(record):
                return pid
            continue
        except OSError:
            # Windows raises OSError with WinError 87 for an invalid pid
            # (process is definitely gone). Treat as "process doesn't exist".
            continue

        recorded_start = record.get("start_time")
        current_start = _get_process_start_time(pid)
        if recorded_start is not None and current_start is not None and current_start != recorded_start:
            continue

        if _looks_like_gateway_process(pid) or _record_looks_like_gateway(record):
            return pid

    _cleanup_invalid_pid_path(resolved_pid_path, cleanup_stale=cleanup_stale)
    return None


def is_gateway_running(
    pid_path: Optional[Path] = None,
    *,
    cleanup_stale: bool = True,
) -> bool:
    """Check if the gateway daemon is currently running."""
    return get_running_pid(pid_path, cleanup_stale=cleanup_stale) is not None

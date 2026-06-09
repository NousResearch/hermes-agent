"""Gateway restart state management — intent, locks, status, and JSONL logging.

Provides the durable state layer for the Windows transactional restart
coordinator.  Each restart transaction gets its own request-scoped directory
under ``{HERMES_HOME}/run/gateway-restart/{profile}/{request_id}/``.

Design invariants:
- Profile-level ``active.lock`` prevents concurrent coordinators.
- Per-request directories eliminate TOCTOU on intent/status cleanup.
- Lease files use ``O_EXCL`` for atomic one-time-only worker claim.
- Intent state transitions require request_id + nonce verification.
- JSONL log is append-only, one record per state change.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re as _re
import secrets
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_SCHEMA_VERSION = 1
_INTENT_MAX_BYTES = 4096
_DEFAULT_TTL_S = 300  # 5 minutes
_LOCK_TTL_S = 120     # 2 minutes

# UUID v4 format — strict match for request_id
_UUID_RE = _re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
)

# Profile allowlist: alphanumeric start, then alphanumeric/hyphen/underscore,
# max 64 chars.  Rejects anything that could escape a directory or inject
# into a command.
_PROFILE_RE = _re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$')


def _validate_profile(profile: str) -> str:
    """Validate profile name.  Raises ValueError on illegal input."""
    if not isinstance(profile, str) or not profile:
        raise ValueError("profile must be a non-empty string")
    if not _PROFILE_RE.match(profile):
        raise ValueError(
            f"profile {profile!r} contains disallowed characters — "
            "must match ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$"
        )
    return profile


def _validate_request_id(request_id: str) -> str:
    """Validate request_id is a UUID.  Raises ValueError on illegal input."""
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be a non-empty string")
    if not _UUID_RE.match(request_id):
        raise ValueError(
            f"request_id {request_id!r} is not a valid UUID — "
            "must match ^[0-9a-f]{{8}}-[0-9a-f]{{4}}-...$"
        )
    return request_id


def _safe_restart_path(profile: str, request_id: str = "") -> Path:
    """Build and validate a restart path, ensuring containment.

    Returns the resolved path.  Raises ValueError if the resolved path
    escapes the restart base (traversal, absolute path, UNC, etc.).
    """
    _validate_profile(profile)
    base = _get_restart_base().resolve()
    profile_dir = (base / profile).resolve()
    # Containment check: profile_dir must start with base
    if not str(profile_dir).startswith(str(base)):
        raise ValueError(
            f"profile path escapes restart base: {profile_dir} not under {base}"
        )
    if request_id:
        _validate_request_id(request_id)
        req_dir = (profile_dir / request_id).resolve()
        if not str(req_dir).startswith(str(profile_dir)):
            raise ValueError(
                f"request_id path escapes profile dir: {req_dir} not under {profile_dir}"
            )
        return req_dir
    return profile_dir


# ---------------------------------------------------------------------------
# Windows named mutex for profile-level serialization
# ---------------------------------------------------------------------------


class _ProfileMutex:
    """Windows named mutex for profile-level serialization.

    Uses CreateMutexW/WaitForSingleObject/ReleaseMutex/CloseHandle
    with explicit argtypes/restype and a finite acquisition timeout.
    On non-Windows platforms, this is a no-op context manager.

    Mutex naming: ``Global\\hermes-restart-{sanitized_profile}-{hash}``
    where *hash* = ``sha256(restart_root + "|" + profile)[:16]``.

    ``Global\\`` rationale: ensures visibility across all Windows sessions
    including Session 0 where Task Scheduler runs.  Without ``Global\\``,
    different user sessions could hold independent mutexes for the same
    gateway, defeating serialization.

    The hash suffix prevents name collisions between different Hermes
    installations that share the same profile name (e.g. dev vs prod).
    """

    _WAIT_OBJECT_0 = 0x000
    _WAIT_ABANDONED = 0x080
    _WAIT_TIMEOUT = 0x102  # 258
    _MUTEX_TIMEOUT_MS = 30_000  # 30 s — all protected ops are sub-second

    def __init__(self, profile: str):
        self._handle = None
        self._closed = False
        self._profile_name = profile
        if sys.platform != "win32":
            return
        import ctypes

        # -- compute stable name from installation path + profile --
        try:
            restart_root = str(_get_restart_base())
        except Exception:
            restart_root = profile   # degraded fallback
        sanitized = _re.sub(r'[^a-zA-Z0-9\-]', '-', profile)
        canonical = f"{restart_root}|{profile}"
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        mutex_name = f"Global\\hermes-restart-{sanitized}-{digest}"

        try:
            k32 = ctypes.windll.kernel32
        except AttributeError:
            raise OSError(
                f"kernel32 not available — cannot create restart mutex "
                f"(profile={profile})"
            )

        # explicit argtypes / restype for all four Win32 functions
        try:
            k32.CreateMutexW.argtypes = [
                ctypes.c_void_p,    # lpMutexAttributes
                ctypes.c_bool,      # bInitialOwner
                ctypes.c_wchar_p,   # lpName
            ]
            k32.CreateMutexW.restype = ctypes.c_void_p
            k32.WaitForSingleObject.argtypes = [
                ctypes.c_void_p,    # hHandle
                ctypes.c_uint,      # dwMilliseconds
            ]
            k32.WaitForSingleObject.restype = ctypes.c_uint
            k32.ReleaseMutex.argtypes = [ctypes.c_void_p]
            k32.ReleaseMutex.restype = ctypes.c_bool
            k32.CloseHandle.argtypes = [ctypes.c_void_p]
            k32.CloseHandle.restype = ctypes.c_bool
        except AttributeError as exc:
            raise OSError(
                f"Failed to configure kernel32 function signatures — "
                f"cannot create restart mutex (profile={profile}): {exc}"
            )

        try:
            handle = k32.CreateMutexW(None, False, mutex_name)
        except OSError as exc:
            raise OSError(
                f"CreateMutexW failed for restart mutex "
                f"(profile={profile}, name={mutex_name}): {exc}"
            )

        # NULL/0/INVALID_HANDLE_VALUE = failure
        if not handle:
            err = k32.GetLastError() if hasattr(k32, "GetLastError") else -1
            raise OSError(
                f"CreateMutexW returned NULL for restart mutex "
                f"(profile={profile}, name={mutex_name}, "
                f"GetLastError={err})"
            )

        self._handle = handle

    # -- context-manager protocol: __enter__ acquires, __exit__ releases --

    def __enter__(self):
        if self._handle is None:
            return self
        import ctypes
        k32 = ctypes.windll.kernel32
        result = k32.WaitForSingleObject(self._handle, self._MUTEX_TIMEOUT_MS)

        if result == self._WAIT_OBJECT_0:
            return self                      # normal acquisition

        if result == self._WAIT_ABANDONED:
            # Previous holder terminated without calling ReleaseMutex.
            # We now own the mutex.  The protected operations are all
            # self-verifying (read → compare → write), so the logical
            # state is still correct even though the previous holder
            # didn't clean up.
            logging.getLogger("gateway.restart").warning(
                "Mutex acquired after WAIT_ABANDONED (profile=%s) — "
                "previous holder exited without ReleaseMutex; "
                "protected ops will self-verify",
                self._profile_name,
            )
            return self

        if result == self._WAIT_TIMEOUT:
            raise TimeoutError(
                f"Failed to acquire restart mutex after "
                f"{self._MUTEX_TIMEOUT_MS} ms — another restart "
                f"operation may be stuck (profile={self._profile_name})"
            )

        # WAIT_FAILED or any other unexpected return value
        err = k32.GetLastError() if hasattr(k32, "GetLastError") else -1
        raise OSError(
            f"WaitForSingleObject returned {result:#x}, "
            f"GetLastError={err} (profile={self._profile_name})"
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is None:
            return
        import ctypes
        ok = ctypes.windll.kernel32.ReleaseMutex(self._handle)
        if not ok:
            err = ctypes.windll.kernel32.GetLastError()
            logging.getLogger("gateway.restart").error(
                "ReleaseMutex failed (profile=%s, handle=%s, "
                "GetLastError=%d) — mutex may be abandoned",
                self._profile_name, self._handle, err,
            )
            raise OSError(
                f"ReleaseMutex failed for restart mutex "
                f"(profile={self._profile_name}, GetLastError={err})"
            )

    # -- handle lifecycle: exactly one CloseHandle per handle --

    def close(self) -> None:
        """Release the OS handle.  Idempotent — safe to call more than once."""
        if self._handle is not None and not self._closed:
            self._closed = True
            import ctypes
            ctypes.windll.kernel32.CloseHandle(self._handle)
            self._handle = None

    def __del__(self) -> None:
        """Safety-net destructor — calls close() if the caller forgot."""
        try:
            self.close()
        except Exception:
            pass  # Never raise from __del__

    @property
    def mutex_name(self) -> str:
        """Diagnostic accessor for the Win32 mutex name."""
        return getattr(self, "_profile_name", "")


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
# run/gateway-restart/
#   {profile}/
#     active.lock                    ← profile-level lock (prevents concurrent coordinators)
#     {request_id}/
#       intent.json                  ← restart intent (signed with nonce)
#       status.json                  ← current state
#       claim.lock                   ← O_EXCL race exclusion marker
#       lease.json                   ← atomic lease publication (Coordinator → Worker handoff)
# ---------------------------------------------------------------------------

def _get_restart_base() -> Path:
    """Return ``{HERMES_HOME}/run/gateway-restart/``, creating if needed."""
    from hermes_cli.config import get_hermes_home
    base = Path(get_hermes_home()) / "run" / "gateway-restart"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _get_profile_dir(profile: str = "default") -> Path:
    """Return ``{HERMES_HOME}/run/gateway-restart/{profile}/``."""
    _validate_profile(profile)
    d = _get_restart_base() / profile
    d_resolved = d.resolve()
    base_resolved = _get_restart_base().resolve()
    if not str(d_resolved).startswith(str(base_resolved)):
        raise ValueError(f"profile path escapes restart base: {profile!r}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_request_dir(profile: str, request_id: str) -> Path:
    """Return ``{HERMES_HOME}/run/gateway-restart/{profile}/{request_id}/``."""
    _validate_profile(profile)
    _validate_request_id(request_id)
    d = _get_profile_dir(profile) / request_id
    d_resolved = d.resolve()
    profile_resolved = _get_profile_dir(profile).resolve()
    if not str(d_resolved).startswith(str(profile_resolved)):
        raise ValueError(
            f"request_id path escapes profile dir: {request_id!r}"
        )
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_logs_dir() -> Path:
    from hermes_cli.config import get_hermes_home
    logs_dir = Path(get_hermes_home()) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def intent_path(profile: str = "default", request_id: str = "") -> Path:
    """Path to intent file.  If request_id given, uses per-request dir."""
    if request_id:
        return _get_request_dir(profile, request_id) / "intent.json"
    return _get_profile_dir(profile) / "active-intent.json"


def lock_path(profile: str = "default") -> Path:
    """Path to profile-level active lock."""
    return _get_profile_dir(profile) / "active.lock"


def lease_path(profile: str, request_id: str) -> Path:
    """Path to request-scoped lease file (O_EXCL atomic claim)."""
    return _get_request_dir(profile, request_id) / "lease.lock"


def claim_lock_path(profile: str, request_id: str) -> Path:
    """Path to claim lock file (O_EXCL race marker)."""
    return _get_request_dir(profile, request_id) / "claim.lock"


def lease_json_path(profile: str, request_id: str) -> Path:
    """Path to lease JSON (atomic publish after intent state update)."""
    return _get_request_dir(profile, request_id) / "lease.json"


def status_path(profile: str = "default", request_id: str = "") -> Path:
    """Path to status file.  If request_id given, uses per-request dir."""
    if request_id:
        return _get_request_dir(profile, request_id) / "status.json"
    return _get_profile_dir(profile) / "active-status.json"


def request_dir_path(profile: str, request_id: str) -> Path:
    """Public accessor for the request directory."""
    return _get_request_dir(profile, request_id)


def jsonl_log_path() -> Path:
    return _get_logs_dir() / "gateway-restart.jsonl"


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------

# Strict allowlist for Scheduled Task names interpolated into PowerShell
# commands.  Rejects quotes, backticks, semicolons, $(), pipes, etc.
_TASK_NAME_RE = _re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9 _\-.]{0,126}$')


def validate_intent(intent: dict[str, Any]) -> tuple[bool, str]:
    """Validate an intent dict against the schema.

    Checks:
    - expires_at: must be int/float and finite
    - target_pid: must be int and >= 0
    - profile: must be non-empty string, no path traversal
    - request_id: must be non-empty string, no path traversal
    - hermes_home: must be non-empty string
    - task_name: strict allowlist — prevents PowerShell injection when
      task_name is interpolated into ``schtasks`` or ``powershell -Command``

    Returns (valid, error_message).
    """
    # expires_at: must be int/float and finite
    expires_at = intent.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        return False, "expires_at must be int or float"
    if not math.isfinite(expires_at):
        return False, "expires_at must be finite"

    # target_pid: must be int and >= 0
    target_pid = intent.get("target_pid")
    if not isinstance(target_pid, int):
        return False, "target_pid must be int"
    if target_pid < 0:
        return False, "target_pid must be >= 0"

    # profile: must be non-empty string, no path traversal
    profile = intent.get("profile")
    if not isinstance(profile, str) or not profile:
        return False, "profile must be a non-empty string"
    if "/" in profile or "\\" in profile or ".." in profile:
        return False, "profile must not contain path separators or '..'"

    # request_id: must be non-empty string, no path traversal, no absolute path
    request_id = intent.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        return False, "request_id must be a non-empty string"
    if "/" in request_id or "\\" in request_id or ".." in request_id:
        return False, "request_id must not contain path separators or '..'"
    # Absolute-path check (Windows drive letter or UNC)
    if len(request_id) >= 2 and request_id[1] == ":":
        return False, "request_id must not be an absolute path"
    if request_id.startswith("\\\\"):
        return False, "request_id must not be a UNC path"

    # hermes_home: must be non-empty string
    hermes_home = intent.get("hermes_home")
    if not isinstance(hermes_home, str) or not hermes_home:
        return False, "hermes_home must be a non-empty string"

    # task_name: strict allowlist for Scheduled Task names.
    # Permitted: alphanumeric start, then [a-zA-Z0-9 _-.], max 127 chars.
    # Rejected: quotes, backticks, semicolons, $(), pipes, etc.
    # This prevents PowerShell injection when task_name is embedded in
    # f-strings for ``powershell -Command`` or ``schtasks /TN``.
    task_name = intent.get("task_name", "")
    if task_name:
        if any(ord(c) < 0x20 for c in task_name):
            return False, "task_name must not contain control characters"
        if not _TASK_NAME_RE.match(task_name):
            return False, (
                "task_name contains disallowed characters — "
                "must match ^[a-zA-Z0-9][a-zA-Z0-9 _\\-.]{0,126}$"
            )

    return True, ""


def create_intent(
    *,
    request_id: str | None = None,
    profile: str = "default",
    hermes_home: str | None = None,
    target_pid: int = 0,
    task_name: str = "",
    origin: str = "external-cli",
    ttl_s: int = _DEFAULT_TTL_S,
) -> dict[str, Any]:
    """Create and atomically write a restart intent.  Returns the intent dict.

    Intent is written to the per-request directory.
    """
    from hermes_cli.config import get_hermes_home
    rid = request_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    intent = {
        "schema_version": _SCHEMA_VERSION,
        "request_id": rid,
        "nonce": secrets.token_urlsafe(32),
        "profile": profile,
        "hermes_home": hermes_home or str(Path(get_hermes_home()).resolve()),
        "target_pid": target_pid,
        "task_name": task_name,
        "origin": origin,
        "created_at": now.isoformat(),
        "expires_at": now.timestamp() + ttl_s,
        "state": "scheduled",
    }
    valid, error = validate_intent(intent)
    if not valid:
        raise ValueError(f"Invalid intent: {error}")
    _atomic_write_json(intent_path(profile, rid), intent)
    return intent


def read_intent(profile: str = "default", request_id: str = "") -> Optional[dict[str, Any]]:
    """Read and validate the intent file.  Returns None if missing/expired/malformed."""
    path = intent_path(profile, request_id)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if len(raw.encode("utf-8")) > _INTENT_MAX_BYTES:
            return None
        data = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != _SCHEMA_VERSION:
        return None
    expires = data.get("expires_at", 0)
    if isinstance(expires, (int, float)) and time.time() > expires:
        return None
    for key in ("request_id", "nonce", "profile", "hermes_home", "target_pid"):
        if key not in data:
            return None
    # Full schema validation — rejects path traversal, invalid task_name, etc.
    valid, _error = validate_intent(data)
    if not valid:
        return None
    return data


def read_intent_by_profile(profile: str = "default") -> Optional[dict[str, Any]]:
    """Read intent from the profile-level fallback path (backward compat)."""
    return read_intent(profile, request_id="")


def validate_intent_nonce(intent: dict[str, Any], nonce: str) -> bool:
    """Constant-time nonce comparison."""
    expected = intent.get("nonce", "")
    if not expected or not nonce:
        return False
    return secrets.compare_digest(expected, nonce)


def update_intent_state(profile: str, request_id: str, state: str,
                        expected_state: str = "") -> bool:
    """Update intent state with optional expected_state guard.

    P0-3: Only updates if current state matches expected_state (when provided).
    Returns True on success.
    """
    path = intent_path(profile, request_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        if expected_state and data.get("state") != expected_state:
            return False
        data["state"] = state
        _atomic_write_json(path, data)
        return True
    except (OSError, json.JSONDecodeError):
        return False


def release_lease(profile: str, request_id: str,
                  *, owner_token: str, worker_pid: int) -> bool:
    """Release lease only if owner_token and worker_pid match."""
    if not owner_token or worker_pid <= 0:
        return False
    if not request_id:
        return False
    lp = lease_json_path(profile, request_id)
    if not lp.exists():
        return False
    try:
        data = json.loads(lp.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        if data.get("owner_token") != owner_token:
            return False
        if data.get("worker_pid") != worker_pid:
            return False
        lp.unlink(missing_ok=True)
        # Also remove claim.lock
        clp = claim_lock_path(profile, request_id)
        clp.unlink(missing_ok=True)
        return True
    except (OSError, json.JSONDecodeError):
        return False


def sanitize_intent(profile: str, request_id: str,
                    *, expected_nonce: str, owner_token: str,
                    worker_pid: int) -> bool:
    """Clear nonce only if owner matches lease."""
    if not owner_token or worker_pid <= 0:
        return False
    if not request_id:
        return False
    # Verify ownership via lease
    lp = lease_json_path(profile, request_id)
    if not lp.exists():
        return False
    try:
        lease_data = json.loads(lp.read_text(encoding="utf-8"))
        if not isinstance(lease_data, dict):
            return False
        if lease_data.get("owner_token") != owner_token:
            return False
        if lease_data.get("worker_pid") != worker_pid:
            return False
    except (OSError, json.JSONDecodeError):
        return False
    # Clear nonce
    ip = intent_path(profile, request_id)
    try:
        data = json.loads(ip.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        if expected_nonce and not secrets.compare_digest(
                data.get("nonce", ""), expected_nonce):
            return False
        data["nonce"] = ""
        _atomic_write_json(ip, data)
        return True
    except (OSError, json.JSONDecodeError):
        return False


def gc_expired_request_dirs(profile: str = "default",
                            max_age_s: int = 3600,
                            active_request_id: str = "") -> int:
    """Garbage-collect expired request directories.

    P1-3: Skips directories that are still running or have active leases.
    Never deletes the active_request_id directory.
    """
    profile_dir = _get_profile_dir(profile)
    now = time.time()
    removed = 0
    _TERMINAL_STATES = frozenset({"completed", "failed"})
    try:
        for d in profile_dir.iterdir():
            if not d.is_dir():
                continue
            # Never delete the active request
            if d.name == active_request_id:
                continue
            # Check for orphan lease.json and claim.lock
            ljp = d / "lease.json"
            clp = d / "claim.lock"
            # Track whether we determined this dir is an orphan
            _orphan_lease = False
            _orphan_claim = False

            # P1-2: Check lease.json — if old + dead worker → orphan
            if ljp.exists():
                try:
                    import json as _lj
                    lj_data = _lj.loads(ljp.read_text(encoding="utf-8"))
                    if isinstance(lj_data, dict):
                        lj_age = now - lj_data.get("claimed_at", ljp.stat().st_mtime)
                        lj_pid = lj_data.get("worker_pid", 0)
                        if lj_age < max_age_s:
                            continue  # Lease too recent — skip
                        if lj_pid > 0 and _pid_exists(lj_pid):
                            continue  # Worker still alive — skip
                        _orphan_lease = True
                    else:
                        # Malformed lease.json — check file age
                        if now - ljp.stat().st_mtime < max_age_s:
                            continue
                        _orphan_lease = True
                except (OSError, ValueError, json.JSONDecodeError):
                    try:
                        if now - ljp.stat().st_mtime < max_age_s:
                            continue
                        _orphan_lease = True
                    except OSError:
                        continue

            # P1-1: Check claim.lock — if old + dead worker → orphan
            # (only reached if lease.json absent or already determined orphan)
            if clp.exists() and not _orphan_lease:
                try:
                    import json as _cl
                    cl_data = _cl.loads(clp.read_text(encoding="utf-8"))
                    if isinstance(cl_data, dict):
                        cl_age = now - cl_data.get("created_at", clp.stat().st_mtime)
                        cl_pid = cl_data.get("worker_pid", 0)
                        if cl_age < max_age_s:
                            continue  # Claim too recent — skip
                        if cl_pid > 0 and _pid_exists(cl_pid):
                            continue  # Worker still alive — skip
                        _orphan_claim = True
                    else:
                        if now - clp.stat().st_mtime < max_age_s:
                            continue
                        _orphan_claim = True
                except (OSError, ValueError, json.JSONDecodeError):
                    try:
                        if now - clp.stat().st_mtime < max_age_s:
                            continue
                        _orphan_claim = True
                    except OSError:
                        continue

            # If we found an orphan (lease or claim), delete the directory
            if _orphan_lease or _orphan_claim:
                import shutil
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
                continue

            # No lease/claim — check status.json for terminal state
            sp = d / "status.json"
            if not sp.exists():
                try:
                    age = now - d.stat().st_mtime
                    if age > max_age_s:
                        import shutil
                        shutil.rmtree(d, ignore_errors=True)
                        removed += 1
                except OSError:
                    continue
                continue
            try:
                data = json.loads(sp.read_text(encoding="utf-8"))
                state = data.get("state", "")
                if state not in _TERMINAL_STATES:
                    continue
                ts_str = data.get("updated_at", "")
                if ts_str:
                    from datetime import datetime as _dt
                    ts = _dt.fromisoformat(ts_str).timestamp()
                    if now - ts > max_age_s:
                        import shutil
                        shutil.rmtree(d, ignore_errors=True)
                        removed += 1
            except (OSError, json.JSONDecodeError, ValueError):
                continue
    except OSError:
        pass
    return removed


def gc_stale_lock_tmp(profile: str = "default") -> int:
    """Garbage-collect stale .lock-*.tmp files in the profile directory.

    Only removes files matching the pattern that are older than _LOCK_TTL_S.
    Never touches active.lock or any other file.
    Returns the number of files removed.
    """
    profile_dir = _get_profile_dir(profile)
    now = time.time()
    removed = 0
    try:
        for f in profile_dir.iterdir():
            if not f.is_file():
                continue
            if not f.name.startswith(".lock-") or not f.name.endswith(".tmp"):
                continue
            try:
                age = now - f.stat().st_mtime
                if age > _LOCK_TTL_S:
                    f.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue
    except OSError:
        pass
    return removed


def cleanup_intent(profile: str = "default", request_id: str = "") -> None:
    """Remove the entire request directory (intent + status + lease).

    DEPRECATED for Worker use — Workers should use release_lease() +
    sanitize_intent() to preserve terminal status for the Coordinator.

    Only use this for:
    - Coordinator crash recovery (invalid request cleanup)
    - _fail_closed() on validation failures
    """
    if not request_id:
        return
    try:
        d = _get_request_dir(profile, request_id)
        if d.exists():
            import shutil
            shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Profile lock (prevents concurrent coordinators)
# ---------------------------------------------------------------------------

class RestartLock:
    """Profile-scoped restart lock with ownership, TTL, and stale recovery.

    Lock is created with ``O_EXCL`` on the final path.  Stores owner_token,
    owner_pid, worker_pid, and claim_deadline for stale recovery.

    P1-1: If the lock records a worker_pid and claim_deadline, and both
    conditions are met (deadline expired + worker PID dead), the lock
    can be safely reclaimed even if the coordinator PID is still alive.

    P1-4: No coalesce — same request_id does NOT reuse an existing lock.
    """

    _LOCK_SCHEMA_VERSION = 1

    def __init__(self, profile: str = "default"):
        self._path = lock_path(profile)
        self._profile_name = profile
        self._owner_token: str = ""
        self._owner_request_id: str = ""
        self._mutex = _ProfileMutex(profile)

    @property
    def owner_token(self) -> str:
        """Expose owner_token for handoff verification."""
        return self._owner_token

    def try_acquire(self, request_id: str, ttl_s: int = _LOCK_TTL_S,
                    worker_pid: int = 0) -> bool:
        """Try to acquire the lock.  Returns True on success.

        No coalesce — each request_id gets a fresh lock.
        Stale recovery considers worker_pid + claim_deadline (P1-1).
        """
        with self._mutex:
            self._path.parent.mkdir(parents=True, exist_ok=True)

            # B2: GC stale tmp files from crashed processes (best-effort)
            try:
                gc_stale_lock_tmp(self._profile())
            except Exception:
                pass  # GC failure must not block lock acquisition

            existing = self._read_lock()
            if existing:
                age = time.time() - existing.get("created_at", 0)

                # P1-1: Check claim_deadline-based recovery
                phase = existing.get("phase", "")
                claim_deadline = existing.get("claim_deadline", 0)
                existing_worker_pid = existing.get("worker_pid", 0)

                if phase == "awaiting_claim" and claim_deadline > 0 and time.time() > claim_deadline:
                    # Claim deadline expired — check if worker is dead
                    if existing_worker_pid > 0 and not _pid_exists(existing_worker_pid):
                        # Worker dead and never claimed — safe to reclaim
                        self._force_release(expected=existing)
                    elif existing_worker_pid <= 0:
                        # No worker PID recorded — safe to reclaim
                        self._force_release(expected=existing)
                    else:
                        # Worker still alive — wait
                        return False
                elif age > ttl_s:
                    # Standard TTL expiry — verify owner is dead
                    owner_pid = existing.get("owner_pid", 0)
                    if owner_pid > 0 and _pid_exists(owner_pid):
                        return False
                    self._force_release(expected=existing)
                else:
                    # Active lock by another request
                    return False

            # Generate owner token for this acquisition
            self._owner_token = secrets.token_urlsafe(32)
            self._owner_request_id = request_id

            lock_data = {
                "schema_version": self._LOCK_SCHEMA_VERSION,
                "request_id": request_id,
                "owner_token": self._owner_token,
                "owner_pid": os.getpid(),
                "worker_pid": worker_pid,
                "claim_deadline": time.time() + 30 if worker_pid else 0,
                "phase": "awaiting_claim" if worker_pid else "acquired",
                "profile": self._profile(),
                "created_at": time.time(),
                "expires_at": time.time() + ttl_s,
            }
            # B1: Crash-safe atomic publish.
            #   1. Write complete JSON to a unique tmp file (same directory).
            #   2. flush + fsync to guarantee durability.
            #   3. os.link(tmp, lock_path) — atomic no-clobber.  If lock_path
            #      already exists, link fails with FileExistsError (never overwrites).
            #   4. Clean up tmp in finally block (covers all exit paths).
            #   If crash occurs between step 1 and 3, a stale tmp file remains
            #   but does not block future acquisitions (unique filename).
            tmp = self._path.with_name(
                f".lock-{uuid.uuid4().hex}.tmp"
            )
            try:
                fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(lock_data, f)
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as e:
                # tmp write failure (disk full, permissions, UUID collision, etc.)
                append_restart_log(
                    request_id=request_id, profile=self._profile(), old_pid=0,
                    origin="try_acquire", state="lock_publish_failed",
                    error=f"tmp write failed ({tmp}): {e}",
                )
                tmp.unlink(missing_ok=True)
                return False

            try:
                os.link(str(tmp), str(self._path))
            except FileExistsError:
                # Lock already held — do NOT overwrite. This is the normal
                # contention path, not an error worth logging.
                return False
            except OSError as e:
                # Non-conflict OSError (permissions, cross-device, etc.) — fail closed
                append_restart_log(
                    request_id=request_id, profile=self._profile(), old_pid=0,
                    origin="try_acquire", state="lock_publish_failed",
                    error=f"os.link({tmp} -> {self._path}) failed: {e}",
                )
                return False
            finally:
                tmp.unlink(missing_ok=True)

            return True
    def release(self) -> None:
        """Release the lock ONLY if we are the owner."""
        with self._mutex:
            existing = self._read_lock()
            if not existing:
                return
            if (existing.get("owner_token") != self._owner_token
                    or existing.get("request_id") != self._owner_request_id):
                return
            try:
                self._path.unlink(missing_ok=True)
            except OSError:
                pass

    def close(self) -> None:
        """Release the OS mutex handle.  Idempotent — safe to call more than once.

        Call this when the RestartLock is no longer needed (typically in a
        ``finally`` block).  Does NOT release the file lock — call
        ``release()`` first if the lock is still held.
        """
        self._mutex.close()

    def mark_phase(self, phase: str) -> None:
        """Update the lock's phase field (must be owner)."""
        with self._mutex:
            existing = self._read_lock()
            if not existing:
                return
            if existing.get("owner_token") != self._owner_token:
                return
            existing["phase"] = phase
            try:
                _atomic_write_json(self._path, existing)
            except OSError:
                pass

    def mark_worker_spawned(self, worker_pid: int,
                            claim_deadline: float) -> bool:
        """Record that a worker has been spawned.

        P0-3: Atomically updates phase, worker_pid, and claim_deadline.
        This enables claim-timeout stale recovery even if the coordinator
        process is still alive (the coordinator PID check in TTL recovery
        would block because the coordinator is alive, but the worker may
        have silently died).

        Returns True on success.
        """
        with self._mutex:
            existing = self._read_lock()
            if not existing:
                return False
            if existing.get("owner_token") != self._owner_token:
                return False
            existing["phase"] = "awaiting_claim"
            existing["worker_pid"] = worker_pid
            existing["claim_deadline"] = claim_deadline
            try:
                _atomic_write_json(self._path, existing)
                return True
            except OSError:
                return False
    def handoff_active_lock(
        self,
        request_id: str,
        coordinator_owner_token: str,
        worker_pid: int,
        lease_owner_token: str,
    ) -> bool:
        """Transfer active.lock ownership from Coordinator to Worker.

        P0-1 + P0-2: After the Worker claims the lease, the Coordinator
        hands off the active.lock so the Worker can release it when done.
        This prevents the Coordinator from releasing the lock while the
        Worker is still running drain/stop/start/verify.

        Validates:
        - active.lock.request_id matches
        - Coordinator's owner_token matches
        - Lease file exists with matching worker_pid and lease_owner_token

        On success, active.lock.owner_token = lease_owner_token,
        active.lock.owner_pid = worker_pid, phase = "running".

        Returns True on success.
        """
        with self._mutex:
            existing = self._read_lock()
            if not existing:
                return False
            if existing.get("request_id") != request_id:
                return False
            if existing.get("owner_token") != coordinator_owner_token:
                return False

            # Verify lease exists with matching owner_token and worker_pid
            lp = lease_json_path(self._profile(), request_id)
            if not lp.exists():
                return False
            try:
                lease_data = json.loads(lp.read_text(encoding="utf-8"))
                if not isinstance(lease_data, dict):
                    return False
                if lease_data.get("owner_token") != lease_owner_token:
                    return False
                if lease_data.get("worker_pid") != worker_pid:
                    return False
            except (OSError, json.JSONDecodeError):
                return False

            # Transfer ownership
            existing["owner_token"] = lease_owner_token
            existing["owner_pid"] = worker_pid
            existing["phase"] = "running"
            try:
                _atomic_write_json(self._path, existing)
                # Update our in-memory token so release() works
                self._owner_token = lease_owner_token
                return True
            except OSError:
                return False
    def claim_lease(self, request_id: str, nonce: str,
                    expected_state: str = "scheduled") -> bool:
        """Atomically claim the lease using two-phase publication.

        Phase 1: O_EXCL create claim.lock (race exclusion)
        Phase 2: Update intent state, then atomically publish lease.json
        """
        clp = claim_lock_path(self._profile(), request_id)
        ljp = lease_json_path(self._profile(), request_id)

        # Verify intent exists and is in expected state
        ip = intent_path(self._profile(), request_id)
        try:
            intent_data = json.loads(ip.read_text(encoding="utf-8"))
            if not isinstance(intent_data, dict):
                return False
            if intent_data.get("request_id") != request_id:
                return False
            if not validate_intent_nonce(intent_data, nonce):
                return False
            if expected_state and intent_data.get("state") != expected_state:
                return False
        except (OSError, json.JSONDecodeError):
            return False

        # Phase 1: O_EXCL claim.lock with metadata
        claim_data = {
            "request_id": request_id,
            "worker_pid": os.getpid(),
            "created_at": time.time(),
        }
        try:
            fd = os.open(str(clp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(claim_data, f)
                f.flush()
                os.fsync(f.fileno())
        except FileExistsError:
            return False
        except OSError:
            return False

        # Update intent state
        self._owner_token = secrets.token_urlsafe(32)
        self._owner_request_id = request_id
        if not update_intent_state(self._profile(), request_id, "claimed",
                                   expected_state=expected_state):
            try:
                clp.unlink(missing_ok=True)
            except OSError:
                pass
            self._owner_token = ""
            self._owner_request_id = ""
            return False

        # Phase 2: Atomic lease.json publication
        lease_data = {
            "request_id": request_id,
            "owner_token": self._owner_token,
            "worker_pid": os.getpid(),
            "claimed_at": time.time(),
        }
        try:
            _atomic_write_json(ljp, lease_data)
        except OSError:
            # Rollback: restore intent state to scheduled
            try:
                update_intent_state(self._profile(), request_id, "scheduled",
                                   expected_state="claimed")
            except Exception:
                pass
            try:
                clp.unlink(missing_ok=True)
            except OSError:
                pass
            self._owner_token = ""
            self._owner_request_id = ""
            return False
        return True

    def _profile(self) -> str:
        """Extract profile name from lock path."""
        return self._path.parent.name

    def _read_lock(self) -> Optional[dict[str, Any]]:
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            # A1: Corrupt / truncated / unreadable lock — fail-closed.
            # Do NOT move, delete, or modify the file.  The caller sees
            # None, interprets it as "locked by someone", and retries or
            # gives up.  Human operator must inspect and clean up.
            abs_path = str(self._path.resolve())
            append_restart_log(
                request_id="", profile=self._profile(), old_pid=0,
                origin="read_lock", state="corrupt_lock_detected",
                error=(
                    f"active.lock at {abs_path} is corrupt or unreadable; "
                    "manual cleanup required — refusing automatic recovery. "
                    "Before removing the file, confirm no restart worker is "
                    "running (check for pythonw.exe processes with "
                    "gateway_windows_restart_worker in cmdline). "
                    f"To clean up: del \"{abs_path}\""
                ),
            )
            return None

    def _force_release(self, expected: dict[str, Any] | None = None) -> None:
        """Force-release an expired lock whose owner is confirmed dead.

        Re-reads and compares before deleting (TOCTOU protection).
        """
        with self._mutex:
            if expected:
                current = self._read_lock()
                if not current:
                    return
                if (current.get("request_id") != expected.get("request_id")
                        or current.get("created_at") != expected.get("created_at")
                        or current.get("owner_token") != expected.get("owner_token")):
                    return
            try:
                self._path.unlink(missing_ok=True)
            except OSError:
                pass
# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

_VALID_STATES = frozenset({
    "scheduled",
    "claimed",
    "preflight_ok",
    "draining",
    "stopping",
    "waiting_pid_exit",
    "waiting_port_release",
    "waiting_task_ready",
    "starting_task",
    "starting_direct_fallback",
    "verifying",
    "completed",
    "failed",
})


def write_status(profile: str, state: str, request_id: str = "",
                 **extra: Any) -> None:
    """Write the status file to the per-request directory."""
    if state not in _VALID_STATES:
        raise ValueError(f"Invalid state: {state!r}")
    payload = {
        "state": state,
        "request_id": request_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    _atomic_write_json(status_path(profile, request_id), payload)


def read_status(profile: str = "default", request_id: str = "") -> Optional[dict[str, Any]]:
    path = status_path(profile, request_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_latest_status(profile: str = "default") -> Optional[dict[str, Any]]:
    """Read the most recent status for this profile.

    Scans per-request directories for the latest status.json.
    """
    profile_dir = _get_profile_dir(profile)
    best = None
    best_time = ""
    try:
        for d in profile_dir.iterdir():
            if not d.is_dir():
                continue
            sp = d / "status.json"
            if not sp.exists():
                continue
            try:
                data = json.loads(sp.read_text(encoding="utf-8"))
                ts = data.get("updated_at", "")
                if ts > best_time:
                    best_time = ts
                    best = data
            except (OSError, json.JSONDecodeError):
                continue
    except OSError:
        pass
    return best


# ---------------------------------------------------------------------------
# JSONL log
# ---------------------------------------------------------------------------

def append_restart_log(
    *,
    request_id: str = "",
    profile: str = "default",
    old_pid: int = 0,
    new_pid: int = 0,
    origin: str = "",
    state: str = "",
    launcher: str = "",
    reason: str = "",
    error: str = "",
    detail: str = "",
    listener_pid: int = 0,
    port: int = 0,
) -> None:
    """Append one record to the JSONL restart log."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "profile": profile,
        "old_pid": old_pid,
        "new_pid": new_pid,
        "origin": origin,
        "state": state,
        "launcher": launcher,
        "reason": reason,
        "error": error,
        "detail": detail,
        "listener_pid": listener_pid,
        "port": port,
    }
    path = jsonl_log_path()
    try:
        with open(str(path), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (tmpfile + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _pid_exists(pid: int) -> bool:
    """Cross-platform PID existence check (best-effort)."""
    if pid <= 0:
        return False
    try:
        import psutil
        return bool(psutil.pid_exists(int(pid)))
    except ImportError:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.OpenProcess.restype = ctypes.c_void_p
            k32.WaitForSingleObject.restype = ctypes.c_uint
            k32.GetLastError.restype = ctypes.c_uint
            h = k32.OpenProcess(0x1000 | 0x100000, False, int(pid))
            if not h:
                return k32.GetLastError() != 87
            try:
                return k32.WaitForSingleObject(h, 0) == 0x102
            finally:
                k32.CloseHandle(h)
        except (OSError, AttributeError):
            return False
    else:
        try:
            os.kill(int(pid), 0)
            return True
        except (ProcessLookupError, OSError):
            return False
        except PermissionError:
            return True

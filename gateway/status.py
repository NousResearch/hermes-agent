"""Gateway runtime status helpers: PID/lock/marker files under ``{HERMES_HOME}`` (one set per
home/profile) that tell whether the gateway daemon is running."""

import contextlib
import copy
import hashlib
import json
import logging
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

from hermes_constants import _get_platform_default_hermes_home, get_hermes_home
from utils import atomic_json_write

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
# Windows byte-range locks are mandatory for other readers: lock a byte well past
# the JSON payload so status/PID readers can read while another process holds it.
_WINDOWS_LOCK_OFFSET = 1024 * 1024
_GATEWAY_RUNNING_PID_CACHE_TTL_SECONDS = 1.0
_gateway_running_pid_cache_lock = threading.Lock()
# key: (pid_path, cleanup_stale, include_runtime_status) -> (cached_at, file signature, pid)
_gateway_running_pid_cache: dict[tuple[str, bool, bool], tuple[float, tuple, Optional[int]]] = {}

logger = logging.getLogger(__name__)


class StormInfo(NamedTuple):
    """Respawn-storm check result: start count, window, and backoff to sleep."""

    count: int
    window_s: float
    backoff_s: float


def record_start_and_check_storm(
    max_starts: int = 5, window_s: float = 120.0, *, backoff_cap_s: float = 300.0
) -> Optional[StormInfo]:
    """Record this start; :class:`StormInfo` when > ``max_starts`` landed in ``window_s``.
    Best-effort: a broken ``gateway-starts.log`` ledger is logged and swallowed, never fatal."""
    try:
        path = get_hermes_home() / "gateway-starts.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).timestamp()
        existing: list[float] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                with contextlib.suppress(ValueError):
                    existing.append(float(line))
        existing.append(now)
        recent = [ts for ts in existing if now - ts <= window_s]
        # Ring-buffer the persisted file so it stays bounded.
        to_write = existing[-max(max_starts * 4, 40):]
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(repr(ts) for ts in to_write) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        if len(recent) <= max_starts:
            return None
        backoff = min(backoff_cap_s, 5.0 * (2 ** min(len(recent) - max_starts, 6)))
        return StormInfo(count=len(recent), window_s=window_s, backoff_s=backoff)
    except Exception as _e:
        logger.debug("respawn-storm breaker bookkeeping failed (non-fatal): %s", _e)
        return None


def _get_process_hermes_home() -> Path:
    """Launch-home HERMES_HOME for identity files (PID, lock, status, markers):
    ``get_hermes_home()`` honors the per-session ``_HERMES_HOME_OVERRIDE`` and would misroute
    them."""
    val = os.environ.get("HERMES_HOME", "").strip()
    return Path(val) if val else _get_platform_default_hermes_home()


def _canonical_hermes_home(path: Path | str) -> Path:
    """Stable absolute HERMES_HOME path for persisted identity data."""
    return Path(path).expanduser().resolve(strict=False)


def _same_hermes_home(left: Path | str, right: Path | str) -> bool:
    """Compare HERMES_HOME paths with the host platform's case semantics."""
    left_c = os.path.normcase(str(_canonical_hermes_home(left)))
    return left_c == os.path.normcase(str(_canonical_hermes_home(right)))


def recorded_gateway_home_conflicts(
    record: Optional[dict[str, Any]], *, expected_home: Optional[Path | str] = None
) -> bool:
    """True when a persisted gateway record names a DIFFERENT HERMES_HOME (cross-profile kill guard:
    profile B's stop must never SIGTERM profile A). ``expected_home`` overrides the comparison base.
    Legacy records without ``hermes_home`` prove nothing -> False; a comparison failure fails
    closed -> True."""
    recorded_home = record.get("hermes_home") if isinstance(record, dict) else None
    if not isinstance(recorded_home, str) or not recorded_home.strip():
        return False
    try:
        base = expected_home if expected_home is not None else _get_process_hermes_home()
        return not _same_hermes_home(recorded_home, base)
    except Exception:
        return True


# Mirrors hermes_cli.profiles._PROFILE_ID_RE -- duplicated so gateway identity code
# stays import-light (hermes_constants + stdlib only).
_PROFILE_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _profile_label_for_home(home: Path | str) -> Optional[str]:
    """Best-effort label: ``<root>/profiles/<name>`` -> name, root home -> "default", else None."""
    try:
        canonical = _canonical_hermes_home(home)
    except Exception:
        return None
    if canonical.parent.name == "profiles" and _PROFILE_LABEL_RE.match(canonical.name):
        return canonical.name
    import hermes_constants
    default_homes = (hermes_constants.get_default_hermes_root, _get_platform_default_hermes_home)
    for default_home in default_homes:
        with contextlib.suppress(Exception):
            if _same_hermes_home(canonical, default_home()):
                return "default"
    return None


def scoped_lock_owner_label(record: Optional[dict[str, Any]]) -> Optional[str]:
    """Profile label of a scoped-lock owner (None: PID-only wording): the validated ``profile``
    field stamped by :func:`acquire_scoped_lock`, else inferred from ``hermes_home`` (old locks)."""
    if not isinstance(record, dict):
        return None
    profile = record.get("profile")
    if isinstance(profile, str) and _PROFILE_LABEL_RE.match(profile.strip()):
        return profile.strip()
    home = record.get("hermes_home")
    return _profile_label_for_home(home) if isinstance(home, str) and home.strip() else None


def _get_pid_path() -> Path:
    return _get_process_hermes_home() / "gateway.pid"


def _get_gateway_lock_path(pid_path: Optional[Path] = None) -> Path:
    return (pid_path or _get_pid_path()).with_name(_GATEWAY_LOCK_FILENAME)


def _get_runtime_status_path() -> Path:
    return _get_process_hermes_home() / _RUNTIME_STATUS_FILE


def _get_lock_dir() -> Path:
    """Machine-local dir for token-scoped gateway locks; ``HERMES_GATEWAY_LOCK_DIR`` overrides."""
    override = os.getenv("HERMES_GATEWAY_LOCK_DIR")
    if override:
        return Path(override)
    state_home = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state_home / "hermes" / _LOCKS_DIRNAME


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Epochs before 2000-01-01 are corrupt/hand-edited state (e.g. an accidental 0).
_EPOCH_MIN_PLAUSIBLE = 946684800.0  # 2000-01-01T00:00:00Z


def normalize_updated_at(value: Any) -> Optional[str]:
    """Coerce a persisted ``updated_at`` (ISO string, legacy epoch, hand edit, garbage) to the
    RFC3339 ``string | null`` that ``/api/status`` promises. ``str``: iff fromisoformat parses
    (trailing ``Z`` tolerated; naive -> UTC). Epoch: before 2000-01-01, > 1 day ahead or
    non-finite -> None. ``bool``/other -> None."""
    if isinstance(value, str):
        raw = value.strip()
        # Python < 3.11 fromisoformat rejects a trailing 'Z'; tolerate it.
        if raw.endswith(("Z", "z")):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).isoformat()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = float(value)
        now = datetime.now(timezone.utc).timestamp()
        if not math.isfinite(seconds) or seconds < _EPOCH_MIN_PLAUSIBLE or seconds > now + 86400:
            return None
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    return None


def terminate_pid(
    pid: int, *, force: bool = False, expected_start_time: Optional[float] = None
) -> None:
    """Terminate a PID; POSIX SIGTERM/SIGKILL, Windows taskkill /T /F for force. Identity guard:
    Windows ``force`` REQUIRES a matching ``expected_start_time`` (taskkill on a recycled PID has
    killed svchost.exe); POSIX optional, but a provided mismatch refuses the kill everywhere.

    On POSIX an expectation is optional, but when the caller provides one and it no longer matches the live
    process, the kill is refused on every platform — a mismatched fingerprint always means the PID was
    recycled. See #89614.
    """
    if force and (_IS_WINDOWS or expected_start_time is not None):
        if expected_start_time is None:
            raise OSError(f"refusing to force-kill PID {pid} without a process start-time guard")
        current_start_time = _get_process_start_time(pid)
        if current_start_time is None:
            raise OSError(f"refusing to force-kill PID {pid}; process start time is unavailable")
        try:
            if not _start_times_agree(current_start_time, expected_start_time):
                raise OSError(f"refusing to force-kill PID {pid}; process identity changed")
        except (TypeError, ValueError) as exc:
            raise OSError(f"refusing to force-kill PID {pid}; malformed start time") from exc
    if not (force and _IS_WINDOWS):
        os.kill(pid, signal.SIGTERM if not force else getattr(signal, "SIGKILL", signal.SIGTERM))
        return
    # Hide flags: a bare taskkill spawn from windowless pythonw.exe would flash a conhost window.
    from hermes_cli._subprocess_compat import windows_hide_flags

    try:
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10, creationflags=windows_hide_flags(),
        )
    except FileNotFoundError:
        os.kill(pid, signal.SIGTERM)
        return
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise OSError(details or f"taskkill failed for PID {pid}")


def _start_times_agree(current: Any, *recorded: Any) -> bool:
    """Same process object: all fingerprints > 0 and within 1ms of ``current``; raises on junk."""
    cur = float(current)
    return cur > 0 and all(r > 0 and abs(r - cur) <= 0.001 for r in map(float, recorded))


def _scope_hash(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _get_scope_lock_path(scope: str, identity: str) -> Path:
    return _get_lock_dir() / f"{scope}-{_scope_hash(identity)}.lock"


def _get_process_start_time(pid: int) -> Optional[int]:
    """Per-process start-time fingerprint (PID-reuse guard), or None: ``/proc/<pid>/stat`` field 22
    on Linux, else psutil ``create_time()`` in centiseconds. Units differ per platform; the guard
    only compares same-host values."""
    with contextlib.suppress(IndexError, ValueError, OSError):
        return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])
    try:
        import psutil  # type: ignore
        return int(round(psutil.Process(pid).create_time() * 100))
    except Exception:
        return None


def get_process_start_time(pid: int) -> Optional[int]:
    """Public wrapper for retrieving a process start time when available."""
    return _get_process_start_time(pid)


def _read_process_cmdline(pid: int) -> Optional[str]:
    """Process command line as one string: /proc, then ``ps``, then psutil (Windows)."""
    with contextlib.suppress(OSError):
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        if raw:
            return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
    if not _IS_WINDOWS:
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
    with contextlib.suppress(Exception):
        import psutil  # type: ignore
        cmdline_parts = psutil.Process(pid).cmdline()
        if cmdline_parts:
            return " ".join(cmdline_parts)
    return None


def _gateway_command_subcommand(command: str | None) -> str | None:
    """Hermes gateway lifecycle subcommand from a command line, or None. No loose substring matches
    (``"gateway" in cmdline`` also matched ``gateway status`` / ``python -m tui_gateway``): needs a
    Hermes entrypoint plus the ``gateway`` subcommand, or a gateway-dedicated entrypoint. Tokenizes
    quote-aware (Windows paths with spaces); ``--profile``/``-p`` selectors are stripped anywhere in
    argv since ``_apply_profile_override`` removes them before argparse."""
    if not command:
        return None
    try:
        raw_tokens = shlex.split(command, posix=False)
    except ValueError:
        raw_tokens = command.split()
    # Strip surrounding quotes, normalize slashes + case per token.
    tokens = [t.strip("\"'").replace("\\", "/").lower() for t in raw_tokens]
    if not tokens:
        return None
    basenames = [t.rsplit("/", 1)[-1] for t in tokens]
    # Gateway-dedicated entrypoints carry no subcommand to inspect.
    if any(t == "gateway/run.py" or t.endswith("/gateway/run.py") for t in tokens):
        return "run"
    if any(b in ("hermes-gateway", "hermes-gateway.exe") for b in basenames):
        return "run"
    joined = " ".join(tokens)
    if "hermes_cli.main" not in joined and "hermes_cli/main.py" not in joined and not any(
        b in ("hermes", "hermes.exe") for b in basenames
    ):
        return None
    # Drop --profile X / -p X / --profile=X / -p=X (consumes a VALUE of "gateway" too).
    filtered: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
        elif token in ("--profile", "-p"):
            skip_next = True
        elif not token.startswith(("--profile=", "-p=")):
            filtered.append(token)
    for i, token in enumerate(filtered):
        if token == "gateway":
            # Bare `hermes gateway` defaults to `run`.
            return filtered[i + 1] if i + 1 < len(filtered) else "run"
    return None


def looks_like_gateway_command_line(command: str | None) -> bool:
    """True only for a real ``gateway run`` process command line."""
    return _gateway_command_subcommand(command) == "run"


def looks_like_gateway_runtime_command_line(command: str | None) -> bool:
    """True for command lines that can host the runtime (``run`` or ``restart``: without a service
    manager the manual restart fallback runs ``run_gateway()`` in-process). For validating
    Hermes-owned records / cleanup scans only; ``looks_like_gateway_command_line`` stays strict."""
    return _gateway_command_subcommand(command) in {"run", "restart"}


def _looks_like_gateway_process(pid: int) -> bool:
    """True when the live PID still looks like the Hermes gateway."""
    cmdline = _read_process_cmdline(pid)
    return bool(cmdline) and looks_like_gateway_command_line(cmdline)


def _record_looks_like_gateway(record: dict[str, Any]) -> bool:
    """Validate gateway identity from PID-file metadata when cmdline is unavailable."""
    argv = record.get("argv")
    if record.get("kind") != _GATEWAY_KIND or not isinstance(argv, list) or not argv:
        return False
    return looks_like_gateway_runtime_command_line(" ".join(str(part) for part in argv))


def _profile_name_for_home(profile_home: Path) -> Optional[str]:
    """Profile id for ``<root>/profiles/<name>``; None for the root/default home (bare gateway)."""
    return profile_home.name if profile_home.parent.name == "profiles" else None


def _command_line_belongs_to_profile(command: str, profile_home: Path) -> bool:
    """True when a gateway command line belongs to ``profile_home`` (mirrors
    ``hermes_cli.gateway._matches_current_profile``): a stale state file can record a PID recycled
    onto ANOTHER profile's live gateway. Named profiles carry ``-p``/``--profile <name>`` or
    ``HERMES_HOME=`` on argv; the default gateway runs bare. Separators normalized."""
    command_lc = command.lower().replace("\\", "/")
    profile_name = _profile_name_for_home(profile_home)
    home_lc = str(profile_home).lower().replace("\\", "/")
    if profile_name is not None and profile_name != "default":
        profile_lc = profile_name.lower()
        return any(needle in command_lc for needle in (
            f"--profile {profile_lc}", f"-p {profile_lc}", f"hermes_home={home_lc}"
        ))
    # Default profile: accept unless argv names another profile or a conflicting explicit
    # HERMES_HOME= (its absence is not disqualifying -- HERMES_HOME usually arrives via the env).
    if "--profile " in command_lc or " -p " in command_lc:
        return False
    return not ("hermes_home=" in command_lc and f"hermes_home={home_lc}" not in command_lc)


def _record_matches_live_gateway_pid(
    record: dict[str, Any], pid: int, *, expected_home: Optional[Path] = None
) -> bool:
    """True when a live PID still identifies as this gateway record. The live command line wins (a
    stale record's argv must not make a recycled PID count as a gateway; with ``expected_home`` it
    must also belong to that profile); unreadable cmdline (Windows/EACCES) -> persisted record."""
    live_cmdline = _read_process_cmdline(pid)
    if not live_cmdline:
        return _record_looks_like_gateway(record)
    if not looks_like_gateway_runtime_command_line(live_cmdline):
        return False
    return expected_home is None or _command_line_belongs_to_profile(live_cmdline, expected_home)


def _build_pid_record() -> dict:
    return {
        "pid": os.getpid(), "kind": _GATEWAY_KIND, "argv": list(sys.argv),
        "start_time": _get_process_start_time(os.getpid()),
        # Scoped locks are machine-global; the owner's home lets a cross-profile
        # --replace place its takeover marker where the target will read it.
        "hermes_home": str(_canonical_hermes_home(_get_process_hermes_home())),
    }


def _get_code_identity_fields() -> dict[str, Any]:
    """Code identity of THIS process for ``gateway_state.json`` (restart picked up new code?).
    Lazy import keeps ``gateway.status`` free of ``hermes_cli`` at import time. Never raises.

    A gateway keeps serving the module versions it imported at startup, so stamping the identity into
    ``gateway_state.json`` lets `hermes update` (and the dashboard) prove whether a running gateway actually
    picked up new code after the restart phase — instead of assuming it did (#88654, #69754). Never raises;
    degrades to absent fields.
    """
    try:
        from hermes_cli.build_info import get_code_identity
        identity = get_code_identity()
        return {"code_sha": identity.get("sha"), "code_version": identity.get("version")}
    except Exception:
        return {}


def _pid_record_belongs_to_current_profile(record: Optional[dict[str, Any]]) -> bool:
    """True when the record's ``hermes_home`` matches the current process (legacy records: True);
    another HERMES_HOME's record must be ignored or the default gateway assumes its identity."""
    if not isinstance(record, dict):
        return False
    record_home = record.get("hermes_home")
    return not record_home or _same_hermes_home(record_home, _get_process_hermes_home())


def _build_runtime_status_record() -> dict[str, Any]:
    return {
        **_build_pid_record(), "gateway_state": "starting", "exit_reason": None,
        "restart_requested": False, "active_agents": 0, "platforms": {},
        "session_store": {"status": "unknown"}, "updated_at": _utc_now_iso(),
        **_get_code_identity_fields(),
    }


def _read_json_file(path: Path, *, bare_pid_ok: bool = False) -> Optional[dict[str, Any]]:
    """JSON object at ``path``, or None when absent/empty/unreadable/invalid. ``bare_pid_ok`` also
    accepts legacy bare-integer PID files as ``{"pid": N}``."""
    try:
        raw = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    except (OSError, UnicodeDecodeError):  # vanished, EACCES, non-UTF-8 garbage
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
        if bare_pid_ok:
            with contextlib.suppress(ValueError):
                payload = int(raw)
    if bare_pid_ok and isinstance(payload, int):
        return {"pid": payload}
    return payload if isinstance(payload, dict) else None


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    atomic_json_write(path, payload, indent=None, separators=(",", ":"))


def _unlink_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _read_pid_record(pid_path: Optional[Path] = None) -> Optional[dict]:
    return _read_json_file(pid_path or _get_pid_path(), bare_pid_ok=True)


def _read_gateway_lock_record(lock_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    return _read_json_file(lock_path or _get_gateway_lock_path(), bare_pid_ok=True)


def _pid_from_record(record: Optional[dict[str, Any]], key: str = "pid") -> Optional[int]:
    try:
        return int(record[key])
    except (KeyError, TypeError, ValueError):
        return None


def _start_times_conflict(recorded_start: Any, current_start: Any) -> bool:
    """PID-reuse guard: True only when BOTH start times are known and differ."""
    return None not in (recorded_start, current_start) and current_start != recorded_start


def _live_pid_from_record(record: Optional[dict[str, Any]]) -> Optional[int]:
    """Record's PID when it is alive and passes the start-time PID-reuse guard, else None."""
    pid = _pid_from_record(record)
    if pid is None or not _pid_exists(pid):
        return None
    if _start_times_conflict(record.get("start_time"), _get_process_start_time(pid)):
        return None
    return pid


def _clear_running_pid_cache() -> None:
    with _gateway_running_pid_cache_lock:
        _gateway_running_pid_cache.clear()


def _file_cache_signature(path: Path) -> tuple[bool, Optional[int], Optional[int]]:
    try:
        st = path.stat()
    except OSError:
        return (False, None, None)
    return (True, st.st_mtime_ns, st.st_size)


def _cleanup_invalid_pid_path(pid_path: Path, *, cleanup_stale: bool) -> None:
    """Force-unlink a stale PID file + sibling lock (lock confirmed inactive, so no pid check)."""
    if not cleanup_stale:
        return
    _clear_running_pid_cache()
    for path in (pid_path, _get_gateway_lock_path(pid_path)):
        with contextlib.suppress(Exception):
            path.unlink(missing_ok=True)


def _try_acquire_file_lock(handle) -> bool:
    try:
        if _IS_WINDOWS:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("\n")
                handle.flush()
            handle.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _pid_exists(pid: int) -> bool:
    """Cross-platform "is this PID alive" check that does NOT kill the target. CRITICAL on Windows:
    ``os.kill(pid, 0)`` sends ``CTRL_C_EVENT`` to the whole console group (bpo-14484), so prefer
    psutil, then ctypes ``OpenProcess`` (Windows) / ``os.kill(pid, 0)`` (POSIX). Zombies report
    dead: treating one as alive makes --replace wait forever under systemd Restart=always."""
    pid = int(pid)
    try:
        import psutil  # type: ignore
        # Best-effort zombie check: status-read failures fall through to pid_exists().
        try:
            # A zombie (defunct) process is still in the process table, so ``psutil.pid_exists()`` returns
            # True for it — but it is already dead: SIGKILL has no effect and it cannot be a running
            # gateway. Treating a zombie as alive makes ``--replace`` wait for the old PID to die (it never
            # does, until its parent reaps it), then abort with exit 1 — a silent crash loop under systemd
            # ``Restart=always``, which respawns the gateway before reaping the previous process (issue
            # #42126). Report zombies as dead so the takeover proceeds. Best-effort: any failure to read
            # status (partial/stub psutil, access denied, transient race) falls through to the authoritative
            # ``pid_exists()`` below rather than raising.
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return False
        except getattr(psutil, "NoSuchProcess", ()):
            return False
        except Exception:
            pass
        return bool(psutil.pid_exists(pid))
    except ImportError:
        pass  # Fall through to stdlib fallback.
    if _IS_WINDOWS:
        return _pid_exists_win32_ctypes(pid)
    if _posix_is_zombie(pid):  # a zombie still answers os.kill(pid, 0)
        return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok — POSIX-only branch (the whole point of _pid_exists)
    except PermissionError:
        return True  # Exists but we can't signal it.
    except OSError:  # ProcessLookupError included
        return False
    return True


def _posix_is_zombie(pid: int) -> bool:
    """Zombie via ``/proc/<pid>/stat`` field 3, or ``ps -o state=`` without /proc (macOS/BSD)."""
    try:
        stat_fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return len(stat_fields) > 2 and stat_fields[2] == "Z"
    except FileNotFoundError:
        with contextlib.suppress(Exception):
            # --compile-bytecode: uv does NOT write __pycache__ by default (pip does), so without it the
            # first `import <backend>` in the foreground of a user request recompiles every module of the
            # backend *and* its transitive deps (#100461). This covers the whole install;
            # _warm_installed_bytecode below is the belt-and-braces pass for the spec's own roots on any
            # tier.
            # CREATE_NO_WINDOW on Windows — under the desktop GUI's windowless parent, this spawn otherwise
            # flashes a console (#56747).
            r = subprocess.run(
                ["ps", "-o", "state=", "-p", str(pid)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
            )
            return r.returncode == 0 and r.stdout.strip().startswith("Z")
    except (IndexError, PermissionError, OSError):
        pass
    return False


def _pid_exists_win32_ctypes(pid: int) -> bool:
    """psutil-free Windows liveness probe via OpenProcess/WaitForSingleObject."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # Pin restypes: default c_int mangles WAIT_* DWORDs into negatives.
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.restype = ctypes.c_uint
        kernel32.GetLastError.restype = ctypes.c_uint
        PROCESS_QUERY_LIMITED_INFORMATION, SYNCHRONIZE = 0x1000, 0x100000  # SYNCHRONIZE: for Wait*
        WAIT_TIMEOUT, ERROR_ACCESS_DENIED = 0x00000102, 5
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
        if not handle:
            # ERROR_INVALID_PARAMETER (87): PID definitely gone. ACCESS_DENIED: exists
            # but owned by another user/session. Any other error: conservative False.
            return kernel32.GetLastError() == ERROR_ACCESS_DENIED
        try:
            # WAIT_TIMEOUT = still running; anything else = gone.
            return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, AttributeError):
        return False


def _release_file_lock(handle) -> None:
    with contextlib.suppress(OSError):
        if _IS_WINDOWS:
            handle.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def acquire_gateway_runtime_lock() -> bool:
    """Claim the cross-process runtime lock; the OS releases it if the process dies."""
    global _gateway_lock_handle
    if _gateway_lock_handle is not None:
        return True
    path = _get_gateway_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = open(path, "a+", encoding="utf-8")
    except PermissionError:
        # Stale root-owned lock (launchd session that ran as root): the directory owner can
        # unlink it; retry once with a fresh file.
        try:
            path.unlink()
            handle = open(path, "a+", encoding="utf-8")
        except OSError:
            return False
    if not _try_acquire_file_lock(handle):
        handle.close()
        return False
    handle.seek(0)
    handle.truncate()
    json.dump(_build_pid_record(), handle)
    handle.flush()
    with contextlib.suppress(OSError):
        os.fsync(handle.fileno())
    _gateway_lock_handle = handle
    _clear_running_pid_cache()
    return True


def release_gateway_runtime_lock() -> None:
    """Release the gateway runtime lock when owned by this process."""
    global _gateway_lock_handle
    handle, _gateway_lock_handle = _gateway_lock_handle, None
    if handle is None:
        return
    _release_file_lock(handle)
    with contextlib.suppress(OSError):
        handle.close()
    _clear_running_pid_cache()


def owns_gateway_runtime_lock() -> bool:
    """True when THIS process holds the runtime lock. ``is_gateway_runtime_lock_active`` answers
    "does anyone?"; re-probing our own flock succeeds on POSIX, so only the handle discriminates."""
    return _gateway_lock_handle is not None


def _probe_lock_file(handle) -> bool:
    """True when another process holds the lock (a won probe is released); closes ``handle``."""
    try:
        held = not _try_acquire_file_lock(handle)
        if not held:
            _release_file_lock(handle)
        return held
    finally:
        with contextlib.suppress(OSError):
            handle.close()


def is_gateway_runtime_lock_active(lock_path: Optional[Path] = None) -> bool:
    """True when some process currently owns the gateway runtime lock."""
    resolved_lock_path = lock_path or _get_gateway_lock_path()
    if _gateway_lock_handle is not None and resolved_lock_path == _get_gateway_lock_path():
        return True
    if not resolved_lock_path.exists():
        return False
    try:
        handle = open(resolved_lock_path, "a+", encoding="utf-8")
    except PermissionError:
        # Stale root-owned lock (see acquire_gateway_runtime_lock): report inactive.
        _unlink_quietly(resolved_lock_path)
        return False
    return _probe_lock_file(handle)


def _strict_path_exists(path: Path, label: str) -> bool:
    """Like ``path.exists()`` but raises RuntimeError instead of False on EACCES-style errors."""
    try:
        path.stat()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(f"{label} metadata is not inspectable: {exc}") from exc


def _is_gateway_runtime_lock_active_strict(lock_path: Path) -> bool:
    """Probe ownership without treating access failures as absence."""
    try:
        handle = open(lock_path, "r+", encoding="utf-8")
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(f"gateway runtime lock is not inspectable: {exc}") from exc
    try:
        return _probe_lock_file(handle)
    except OSError as exc:
        raise RuntimeError(f"gateway runtime lock probe failed: {exc}") from exc


def write_pid_file() -> None:
    """Write this process's PID record via O_CREAT|O_EXCL; a racing gateway's FileExistsError
    propagates for the caller to decide."""
    path = _get_pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_excl(path, _build_pid_record())
    _clear_running_pid_cache()


def _write_json_excl(path: Path, record: dict[str, Any]) -> None:
    """Create ``path`` with O_CREAT|O_EXCL and dump ``record``; unlinks on a failed write."""
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
    except Exception:
        _unlink_quietly(path)
        raise


def _apply_set_fields(target: dict[str, Any], fields) -> None:
    """Assign each ``(key, value, coerce)`` whose value was explicitly passed (not ``_UNSET``)."""
    for key, value, coerce in fields:
        if value is not _UNSET:
            target[key] = coerce(value) if coerce is not None else value


def _coerce_session_store(session_store: Any) -> dict[str, str]:
    state = str(session_store.get("status") or "") if isinstance(session_store, dict) else ""
    return {"status": state if state in {"ok", "unavailable", "retrying"} else "unknown"}


def write_runtime_status(
    *, gateway_state: Any = _UNSET, exit_reason: Any = _UNSET, restart_requested: Any = _UNSET,
    active_agents: Any = _UNSET, platform: Any = _UNSET, platform_state: Any = _UNSET,
    error_code: Any = _UNSET, error_message: Any = _UNSET, needs_attention: Any = _UNSET,
    retrying_since: Any = _UNSET, served_profiles: Any = _UNSET, session_store: Any = _UNSET,
    clear_profile_platforms: bool = False,
) -> None:
    """Persist gateway runtime health information for diagnostics/status."""
    path = _get_runtime_status_path()
    payload = _read_json_file(path) or _build_runtime_status_record()
    previous_payload = copy.deepcopy(payload)
    current_record = _build_pid_record()
    payload.setdefault("platforms", {})
    if clear_profile_platforms:
        # Secondary-profile entries are keyed ``<profile>:<platform>``. A fresh process must not
        # inherit them or /api/status stays degraded until every old adapter re-emits.
        platforms = payload["platforms"] if isinstance(payload["platforms"], dict) else {}
        payload["platforms"] = {
            k: v for k, v in platforms.items() if not isinstance(k, str) or ":" not in k
        }
    # Re-stamp identity + code fields on every write: the file can outlive its creator and the
    # top-level record must describe the CURRENT writer.
    payload.update({key: current_record[key] for key in ("kind", "pid", "argv", "start_time")})
    payload["updated_at"] = _utc_now_iso()
    payload.update(_get_code_identity_fields())
    _apply_set_fields(payload, (
        ("gateway_state", gateway_state, None), ("exit_reason", exit_reason, None),
        ("restart_requested", restart_requested, bool),
        ("active_agents", active_agents, parse_active_agents),
        # Multiplexed profiles; absent/empty for a single-profile gateway.
        ("served_profiles", served_profiles, lambda v: list(v or [])),
        ("session_store", session_store, _coerce_session_store),
    ))
    if platform is not _UNSET:
        platform_payload = payload["platforms"].get(platform, {})
        _apply_set_fields(platform_payload, (
            ("state", platform_state, None), ("error_code", error_code, None),
            ("error_message", error_message, None),
            # Reconnect-loop escalation past the attention threshold: a signal for owners/fleet
            # monitoring, not a circuit breaker (retry never stops). Cleared on reconnect.
            ("needs_attention", needs_attention, bool),
            # ISO start of the current retry episode; None clears it.
            ("retrying_since", retrying_since, None),
        ))
        # Per-entry writer provenance: top-level pid/start_time only identify the most recent
        # writer; /api/status tells "live" from "preserved" by exact (pid, start_time) equality.
        platform_payload.update(updated_at=_utc_now_iso(), writer_pid=current_record["pid"],
                                writer_start_time=current_record["start_time"])
        payload["platforms"][platform] = platform_payload
    _write_json_file(path, payload)
    with contextlib.suppress(Exception):
        from agent.monitoring.gateway_health import emit_runtime_status_transition
        emit_runtime_status_transition(previous_payload, payload)


def read_runtime_status(path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Read ``gateway_state.json``; ``path`` lets callers inspect another profile's file."""
    return _read_json_file(path or _get_runtime_status_path())


# Max age of a ``gateway_state.json`` snapshot before its liveness claim is suspect:
# an older record outlived an ungracefully-killed writer (taskkill /F, OOM, power loss).
_RUNTIME_STATUS_STALE_TTL_S = 120


def runtime_status_is_stale(
    record: Optional[dict[str, Any]], ttl_s: int = _RUNTIME_STATUS_STALE_TTL_S
) -> bool:
    """True when the snapshot's ``updated_at`` is older than ``ttl_s`` (or missing/unparseable)."""
    return not isinstance(record, dict) or _marker_is_stale(record.get("updated_at") or "", ttl_s)


def runtime_status_pid_is_live(record: Optional[dict[str, Any]]) -> bool:
    """True when the snapshot's PID is alive and passes the start-time PID-reuse guard."""
    return _live_pid_from_record(record) is not None


def parse_active_agents(raw: Any) -> int:
    """Coerce ``active_agents`` to a non-negative int; shared by writer and both HTTP readers."""
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


# Only a live ``running`` gateway is a valid begin-drain target.
_DRAINABLE_GATEWAY_STATES = frozenset({"running"})


def derive_gateway_busy(*, gateway_running: bool, gateway_state: Any, active_agents: Any) -> bool:
    """Busy iff live, ``running``, and ``active_agents > 0`` -- the contract NAS gates on. Liveness
    keys off ``gateway_running``, NEVER ``updated_at`` (an idle gateway never advances it)."""
    if not derive_gateway_drainable(gateway_running=gateway_running, gateway_state=gateway_state):
        return False
    return parse_active_agents(active_agents) > 0


def derive_gateway_drainable(*, gateway_running: bool, gateway_state: Any) -> bool:
    """Drainable iff live and ``running``; independent of ``active_agents`` (idle drains finish)."""
    return bool(gateway_running) and gateway_state in _DRAINABLE_GATEWAY_STATES


@dataclass(frozen=True)
class GatewayLiveness:
    """Resolved gateway liveness for one dashboard surface. ``source``: which ladder rung answered
    (logging/tests only -- never branch product behavior on it). ``probe_error``: a rung raised;
    lets fail-open callers tell "down" from "unknown"."""

    running: bool
    pid: Optional[int]
    source: str
    health_body: Optional[dict[str, Any]] = None
    probe_error: bool = False


def resolve_gateway_liveness(
    *, profile_dir: Optional[Path] = None, runtime: Any = _UNSET,
    health_probe: Optional[Callable[[], tuple[bool, Optional[dict[str, Any]]]]] = None,
    use_cache: bool = True, pid_probe: Optional[Callable[..., Optional[int]]] = None,
    runtime_reader: Optional[Callable[..., Optional[dict[str, Any]]]] = None,
    runtime_pid_probe: Optional[Callable[..., Optional[int]]] = None,
) -> GatewayLiveness:
    """Single source of truth for "is the gateway up?" across dashboard surfaces. Ladder, most to
    least authoritative: (1) PID file + runtime lock (scoped to ``profile_dir``; cached by default
    so polling does not re-flock ``gateway.lock``); (2) caller-supplied HTTP health probe (gateway
    in another container); (3) LOCAL runtime status PID validated against the live process table
    with ``expected_home`` (a recycled PID of another profile never counts; pass ``runtime`` if
    already read). ``*_probe``/``runtime_reader`` are the dashboard's injection/test seam. A rung
    that raises degrades to the next (never 500 a status endpoint) and sets ``probe_error``.

    Before this existed, ``/api/status`` and ``/api/messaging/platforms`` each open-coded their own ladder
    and disagreed on the same page load — the sidebar read "running" while the Channels page rendered "The
    gateway is not running."  Three deployments hit it: a cross-container gateway (only ``/api/status`` ran
    the HTTP health probe), a profile-scoped dashboard (only ``/api/status`` passed the profile's paths, so
    messaging borrowed another profile's runtime state — issue #71211), and a launch-service-managed gateway
    with no PID file (only some callers used the runtime-status fallback).
    """
    _pid_probe = pid_probe or (get_running_pid_cached if use_cache else get_running_pid)
    _runtime_reader = runtime_reader or read_runtime_status
    _runtime_pid_probe = runtime_pid_probe or get_runtime_status_running_pid
    probe_error = False
    scoped = profile_dir is not None

    def guarded(fn, *args, fallback=None, **kwargs):
        nonlocal probe_error
        try:
            return fn(*args, **kwargs)
        except Exception:
            probe_error = True
            return fallback

    # Zero-arg call when unscoped: callers monkeypatch with zero-arg lambdas and
    # /api/status's cache signature is keyed on the call shape.
    pid = guarded(_pid_probe, profile_dir / "gateway.pid") if scoped else guarded(_pid_probe)
    if pid is not None:
        return GatewayLiveness(running=True, pid=pid, source="pid")
    health_body: Optional[dict[str, Any]] = None
    if health_probe is not None:
        alive, health_body = guarded(health_probe, fallback=(False, None))
        if alive:
            # Display-only PID: it belongs to the remote container.
            remote_pid = health_body.get("pid") if health_body else None
            return GatewayLiveness(
                running=True, pid=remote_pid, source="health", health_body=health_body
            )
    if runtime is _UNSET:
        reader_kwargs = {"path": profile_dir / "gateway_state.json"} if scoped else {}
        runtime = guarded(_runtime_reader, **reader_kwargs)
    probe_kwargs = {"expected_home": profile_dir} if scoped else {}
    runtime_pid = guarded(_runtime_pid_probe, runtime, **probe_kwargs)
    if runtime_pid is not None:
        return GatewayLiveness(
            running=True, pid=runtime_pid, source="runtime_status", health_body=health_body
        )
    return GatewayLiveness(
        running=False, pid=None, source="none", health_body=health_body, probe_error=probe_error
    )


def get_runtime_status_running_pid(
    runtime: Optional[dict[str, Any]] = None, *, expected_home: Optional[Path] = None
) -> Optional[int]:
    """Live gateway PID from the runtime status record, or None: the ``get_running_pid()`` fallback
    for launch-service-managed gateways with a fresh ``gateway_state.json`` but no ``gateway.pid``.
    ``expected_home`` scopes the OS-identity check to another profile's home so a PID recycled onto
    a different profile's gateway is not reported running for the dead one."""
    payload = runtime if runtime is not None else read_runtime_status()
    if not isinstance(payload, dict):
        return None
    if payload.get("gateway_state") in {None, "stopped", "startup_failed"}:
        return None
    pid = _live_pid_from_record(payload)
    if pid is None:
        return None
    # Active-profile context: the record's hermes_home must match this process so a stale record
    # cannot lend another profile's identity.
    if expected_home is None and not _pid_record_belongs_to_current_profile(payload):
        return None
    if not _record_matches_live_gateway_pid(payload, pid, expected_home=expected_home):
        return None
    return pid


def remove_pid_file() -> None:
    """Remove the PID file only if it belongs to this process: during --replace the old process's
    atexit can fire AFTER the new process wrote its own record."""
    with contextlib.suppress(Exception):
        path = _get_pid_path()
        file_pid = _pid_from_record(_read_json_file(path))
        if file_pid is not None and file_pid != os.getpid():
            return  # Belongs to a different process — leave it alone.
        path.unlink(missing_ok=True)
        _clear_running_pid_cache()


def _scoped_lock_record_is_stale(existing: dict[str, Any], existing_pid: Optional[int]) -> bool:
    """True when a foreign scoped-lock record no longer names a live gateway: PID missing/dead,
    start time changed (PID reuse), or the live process is not a gateway -- a readable cmdline says
    so (also catches boot-time PID+start_time collisions; systemd spawns deterministically);
    cmdline unreadable AND start_time unknown on either side => the lock record's own argv is the
    only signal left. Stopped (SIGTSTP) processes look alive to _pid_exists; stale so --replace
    works."""
    if existing_pid is None or not _pid_exists(existing_pid):
        return True
    recorded_start = existing.get("start_time")
    current_start = _get_process_start_time(existing_pid)
    if _start_times_conflict(recorded_start, current_start):
        return True
    if not _looks_like_gateway_process(existing_pid):
        if _read_process_cmdline(existing_pid) is not None:
            return True
        if None in (recorded_start, current_start) and not _record_looks_like_gateway(existing):
            return True
    return _process_is_stopped(existing_pid)


def _process_is_stopped(pid: int) -> bool:
    """True for a stopped / tracing-stop state (T/t) in ``/proc/<pid>/status``."""
    with contextlib.suppress(OSError):
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("State:"):
                return line.split()[1] in {"T", "t"}
    return False


def acquire_scoped_lock(
    scope: str, identity: str, metadata: Optional[dict[str, Any]] = None
) -> tuple[bool, Optional[dict[str, Any]]]:
    """Acquire a machine-local lock keyed by scope + identity (one Telegram token across homes)."""
    lock_path = _get_scope_lock_path(scope, identity)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **_build_pid_record(), "scope": scope, "identity_hash": _scope_hash(identity),
        "metadata": metadata or {}, "updated_at": _utc_now_iso(),
    }
    # Profile label for cross-profile conflict diagnostics ("token already in use (PID 559)" alone
    # does not say WHICH profile). Omitted when not inferable; readers fall back to hermes_home.
    profile = _profile_label_for_home(_get_process_hermes_home())
    if profile:
        record["profile"] = profile
    existing = _read_json_file(lock_path)
    if existing is None and lock_path.exists():
        # Empty/invalid JSON: previous process died between O_EXCL create and json.dump().
        _unlink_quietly(lock_path)
    if existing:
        existing_pid = _pid_from_record(existing)
        # Our own PID: always self-reacquire. start_time guards reuse of OTHER PIDs; requiring
        # equality here rejects reconnects when the on-disk record has start_time null.
        # Same live PID as this process: always self-reacquire. ``start_time`` is a PID-reuse guard for
        # *other* PIDs; it cannot distinguish two processes that share the caller's own PID (impossible
        # while we are alive). Requiring start_time equality here falsely rejects reconnects when the
        # on-disk record has ``start_time: null`` (older writers / psutil failure at first write) while the
        # freshly built record has a real value — the gateway then reports itself as the foreign squatter of
        # its own token (#81468).
        if existing_pid == os.getpid():
            _write_json_file(lock_path, record)
            return True, existing
        if not _scoped_lock_record_is_stale(existing, existing_pid):
            return False, existing
        # Rename to a tombstone instead of unlink(): with unlink()+O_EXCL two racing starters
        # could both win. os.replace() lets exactly one claim it; a failed replace means another
        # racer claimed it and O_EXCL below decides.
        with contextlib.suppress(OSError):
            tombstone = lock_path.with_name(lock_path.name + ".stale")
            os.replace(lock_path, tombstone)
            _unlink_quietly(tombstone)
    try:
        _write_json_excl(lock_path, record)
    except FileExistsError:
        return False, _read_json_file(lock_path)
    return True, None


def release_scoped_lock(scope: str, identity: str) -> None:
    """Release a scope lock owned by this PID. No start_time equality check: on-disk null vs a live
    fingerprint would wedge reconnects."""
    lock_path = _get_scope_lock_path(scope, identity)
    if (_read_json_file(lock_path) or {}).get("pid") == os.getpid():
        _unlink_quietly(lock_path)


def release_all_scoped_locks(
    *, owner_pid: Optional[int] = None, owner_start_time: Optional[int] = None
) -> int:
    """Remove scoped lock files (--replace cleanup); returns the count removed. With ``owner_pid``
    only that gateway's records go (``owner_start_time`` narrows against PID reuse)."""
    lock_dir = _get_lock_dir()
    if not lock_dir.exists():
        return 0
    removed = 0
    for lock_file in lock_dir.glob("*.lock"):
        if owner_pid is not None:
            record = _read_json_file(lock_file) or {}
            if _pid_from_record(record) != owner_pid or (
                owner_start_time is not None and record.get("start_time") != owner_start_time
            ):
                continue
        with contextlib.suppress(OSError):
            lock_file.unlink(missing_ok=True)
            removed += 1
    return removed


# ── --replace takeover marker ─────────────────────────────────────────
# SIGTERM exits the gateway with code 1 so Restart=on-failure revives it after unexpected
# kills -- which would also revive a --replace target (flap loop against the replacer). The
# replacer therefore writes a short-lived marker naming the target PID + start_time BEFORE
# SIGTERM; the target's shutdown handler treats a matching marker as a planned takeover and
# exits 0. Unlinked once consumed, so a stale one can grief at most one future shutdown on
# the same PID, within _TAKEOVER_MARKER_TTL_S.
# When a new gateway starts with ``--replace``, it SIGTERMs the existing gateway so it can take over the bot
# token. ``hermes.service`` + ``hermes- gateway.service``). See #5646.
_TAKEOVER_MARKER_FILENAME = ".gateway-takeover.json"
_TAKEOVER_MARKER_TTL_S = 60  # Marker older than this is treated as stale
_PLANNED_STOP_MARKER_FILENAME = ".gateway-planned-stop.json"
_PLANNED_STOP_MARKER_TTL_S = 60


def _get_takeover_marker_path(hermes_home: Optional[Path] = None) -> Path:
    """Takeover marker path; ``hermes_home`` is given only for a verified cross-home handoff."""
    home = _canonical_hermes_home(hermes_home or _get_process_hermes_home())
    return home / _TAKEOVER_MARKER_FILENAME


def _get_planned_stop_marker_path() -> Path:
    return _get_process_hermes_home() / _PLANNED_STOP_MARKER_FILENAME


def _marker_is_stale(written_at: str, ttl_s: int) -> bool:
    try:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(written_at)
        return age.total_seconds() > ttl_s
    except (TypeError, ValueError):
        return True


def _read_live_pid_marker(path: Path, ttl_s: int) -> Optional[tuple[dict[str, Any], int, Any]]:
    """``(record, target_pid, target_start_time)`` for a usable marker, else None. Malformed/expired
    markers can never match anyone, so they are unlinked here (must not wedge a new instance)."""
    record = _read_json_file(path)
    if not record:
        return None
    target_pid = _pid_from_record(record, "target_pid")
    if target_pid is None or _marker_is_stale(record.get("written_at") or "", ttl_s):
        _unlink_quietly(path)
        return None
    return record, target_pid, record.get("target_start_time")


def _pid_marker_names_self(target_pid: int, target_start_time: Any) -> bool:
    """PID match with an optional start-time PID-reuse guard (watcher probe + consume). Both start
    times known -> must match; either unknown -> PID equality decides (bounded by the marker TTL):
    ``_get_process_start_time`` is None without /proc (macOS, native Windows -- where the
    planned-stop watcher matters most) and requiring a match there would misclassify a legitimate
    ``hermes gateway stop`` as an unexpected exit revived by the service manager."""
    if target_pid != os.getpid():
        return False
    our_start_time = _get_process_start_time(target_pid)
    return None in (target_start_time, our_start_time) or target_start_time == our_start_time


def _consume_pid_marker_for_self(path: Path, *, ttl_s: int) -> bool:
    parsed = _read_live_pid_marker(path, ttl_s)
    if parsed is None:
        return False
    record, target_pid, target_start_time = parsed
    # Cross-profile guard: new markers name the verified TARGET home, which permits a deliberate
    # cross-HERMES_HOME --replace while ignoring a marker accidentally written into another
    # profile's directory. Legacy markers have no target field: keep the same-replacer-home rule.
    # See #29092.
    our_home = _get_process_hermes_home()
    target_home = record.get("target_hermes_home")
    if target_home is not None:
        if not isinstance(target_home, str) or not _same_hermes_home(target_home, our_home):
            return False
    else:
        replacer_home = record.get("replacer_hermes_home")
        if replacer_home is not None and not _same_hermes_home(replacer_home, our_home):
            return False
    matches = _pid_marker_names_self(target_pid, target_start_time)
    _unlink_quietly(path)
    return matches


def write_takeover_marker(
    target_pid: int, *, target_home: Optional[Path] = None, target_start_time: Any = _UNSET
) -> bool:
    """Record that ``target_pid`` is being replaced by this process; True on success. Captures the
    target's ``start_time`` (PID-reuse guard) + a timestamp for TTL. A verified cross-home handoff
    passes ``target_home`` + validated ``target_start_time`` so the marker lands in the target's
    home; such callers must fail closed on False (the target's supervisor could revive it)."""
    try:
        marker_home = _canonical_hermes_home(target_home or _get_process_hermes_home())
        if target_start_time is _UNSET:
            target_start_time = _get_process_start_time(target_pid)
        return _write_marker(_get_takeover_marker_path(marker_home), {
            "target_pid": target_pid, "target_start_time": target_start_time,
            "target_hermes_home": str(marker_home), "replacer_pid": os.getpid(),
            "replacer_hermes_home": str(_canonical_hermes_home(_get_process_hermes_home())),
            "written_at": _utc_now_iso(),
        })
    except OSError:
        return False


def _write_marker(path: Path, record: dict[str, Any]) -> bool:
    """Atomically write a marker record; False (never raise) on OS failure."""
    try:
        _write_json_file(path, record)
        return True
    except OSError:
        return False


def consume_takeover_marker_for_self() -> bool:
    """Consume the takeover marker; True => planned takeover (exit 0); unlinked on match/stale."""
    return _consume_pid_marker_for_self(_get_takeover_marker_path(), ttl_s=_TAKEOVER_MARKER_TTL_S)


def clear_takeover_marker(target_home: Optional[Path] = None) -> None:
    """Remove the takeover marker unconditionally. Safe to call repeatedly."""
    _unlink_quietly(_get_takeover_marker_path(target_home))


def _validated_scoped_lock_gateway_owner(record: dict[str, Any]) -> Optional[tuple[int, int, Path]]:
    """Resolve a live scoped-lock owner to a verified ``(pid, start_time, home)``. A lock file is
    only a claim: the record, the target home's PID record, and the live process must agree on
    PID, start-time, gateway identity, and home. Missing legacy metadata fails closed."""
    if not isinstance(record, dict) or not _record_looks_like_gateway(record):
        return None
    owner_pid = _pid_from_record(record)
    owner_start_time = record.get("start_time")
    raw_home = record.get("hermes_home")
    if (
        owner_pid is None or owner_pid <= 0 or owner_pid == os.getpid()
        or not isinstance(owner_start_time, int) or isinstance(owner_start_time, bool)
        or not isinstance(raw_home, str) or not raw_home.strip()
        or not Path(raw_home).expanduser().is_absolute()
    ):
        return None
    target_home = _canonical_hermes_home(raw_home)
    if _scoped_lock_owner_state(owner_pid, owner_start_time) != "same":
        return None
    live_cmdline = _read_process_cmdline(owner_pid)
    if live_cmdline is not None and not looks_like_gateway_runtime_command_line(live_cmdline):
        return None
    # The target home's own PID record must corroborate the claim.
    pid_record = _read_json_file(target_home / "gateway.pid") or {}
    pid_record_home = pid_record.get("hermes_home")
    if (
        not _record_looks_like_gateway(pid_record)
        or _pid_from_record(pid_record) != owner_pid
        or pid_record.get("start_time") != owner_start_time
        or not isinstance(pid_record_home, str)
        or not _same_hermes_home(pid_record_home, target_home)
    ):
        return None
    return owner_pid, owner_start_time, target_home


def _scoped_lock_owner_state(owner_pid: int, owner_start_time: int) -> str:
    """Return ``same``, ``exited``, or ``unknown`` for a validated owner."""
    if not _pid_exists(owner_pid):
        return "exited"
    live_start_time = _get_process_start_time(owner_pid)
    # A different start time means the PID was recycled; never signal the replacement.
    if live_start_time is None:
        return "unknown"
    return "same" if live_start_time == owner_start_time else "exited"


def _wait_for_scoped_lock_owner_exit(
    owner_pid: int, owner_start_time: int, *, attempts: int, delay: float
) -> tuple[bool, bool]:
    """Return ``(exited, safe_to_force)`` after bounded identity-aware waits."""
    for _ in range(max(0, attempts)):
        state = _scoped_lock_owner_state(owner_pid, owner_start_time)
        if state == "exited":
            return True, False
        if state == "unknown":
            return False, False
        time.sleep(max(0.0, delay))
    return False, _scoped_lock_owner_state(owner_pid, owner_start_time) == "same"


def _snapshot_gateway_children(pid: int) -> list:
    """Best-effort snapshot of ``pid``'s live descendants (POSIX only; never raises). Take it while
    the parent is alive -- once it exits the children are reparented and undiscoverable. ``[]`` on
    Windows (taskkill /T tree-kills)."""
    if _IS_WINDOWS:
        return []
    try:
        import psutil  # type: ignore
        return psutil.Process(int(pid)).children(recursive=True)
    except Exception:
        logger.debug("Could not snapshot children of gateway PID %d", pid, exc_info=True)
        return []


def reap_gateway_children(children: list, *, parent_pid: int, timeout: float = 5.0) -> int:
    """Best-effort reap of a dead gateway's orphaned descendants (POSIX; surviving adapter
    subprocesses keep holding token locks); returns count signalled. Call only AFTER the parent is
    confirmed dead, with a :func:`_snapshot_gateway_children` snapshot. ``is_running()`` is
    identity-aware so a recycled child PID is never signalled; a child whose ppid still equals
    ``parent_pid`` is skipped (parent alive => not an orphan). SIGTERM, bounded wait, SIGKILL
    survivors. Never raises."""
    if _IS_WINDOWS or not children:
        return 0
    reaped = 0
    try:
        import psutil  # type: ignore
        live = []
        for child in children:
            try:
                if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                    continue
                if child.ppid() == parent_pid:
                    logger.debug("Skipping child PID %d of old gateway %d: parent still appears "
                                 "alive", child.pid, parent_pid)
                    continue
                child.terminate()
                live.append(child)
            except psutil.NoSuchProcess:
                continue
            except Exception:
                logger.debug("Could not terminate child PID %s of old gateway %d",
                             getattr(child, "pid", "?"), parent_pid, exc_info=True)
        if not live:
            return 0
        gone, alive = psutil.wait_procs(live, timeout=max(0.0, timeout))
        reaped = len(gone)
        for child in alive:
            try:
                child.kill()
                reaped += 1
            except Exception:
                logger.debug("Could not force-kill child PID %s of old gateway %d",
                             getattr(child, "pid", "?"), parent_pid, exc_info=True)
        if reaped:
            logger.info("Reaped %d orphaned child process(es) of replaced gateway PID %d.",
                        reaped, parent_pid)
    except Exception:
        logger.debug("Child reap for replaced gateway PID %d failed", parent_pid, exc_info=True)
    return reaped


def take_over_scoped_lock_holder(
    record: dict[str, Any], *, graceful_attempts: int = 20, force_attempts: int = 20
) -> Optional[int]:
    """Terminate one verified scoped-lock holder for explicit ``--replace``. Returns the owner PID
    only after that exact PID/start-time identity exited; validation or marker-write failure returns
    None without signalling (a cross-home handoff must place a consumable marker in the target's
    home or its supervisor revives it: flap loop). On POSIX the snapshotted children are reaped."""
    owner = _validated_scoped_lock_gateway_owner(record)
    if owner is None:
        return None
    owner_pid, owner_start_time, target_home = owner
    # Snapshot while the owner is alive; afterwards children are reparented.
    owner_children = _snapshot_gateway_children(owner_pid)
    if not write_takeover_marker(
        owner_pid, target_home=target_home, target_start_time=owner_start_time
    ):
        return None
    try:
        replaced = _terminate_verified_owner(
            owner_pid, owner_start_time, graceful_attempts=graceful_attempts,
            force_attempts=force_attempts,
        )
    finally:
        # The target normally consumes the marker; clean up any remainder.
        clear_takeover_marker(target_home)
    if replaced is not None:
        reap_gateway_children(owner_children, parent_pid=owner_pid)
    return replaced


def _terminate_verified_owner(
    owner_pid: int, owner_start_time: int, *, graceful_attempts: int, force_attempts: int
) -> Optional[int]:
    """Bounded identity-aware SIGTERM-then-SIGKILL of a verified owner; the PID once it exited, else
    None. Per signal step: ``ProcessLookupError`` => already gone; other ``OSError`` => refuse."""
    state = _scoped_lock_owner_state(owner_pid, owner_start_time)
    if state == "exited":
        return owner_pid
    if state != "same":
        return None
    for attempts, delay, kwargs in (
        (graceful_attempts, 0.5, {"force": False}),
        (force_attempts, 0.25, {"force": True, "expected_start_time": owner_start_time}),
    ):
        try:
            terminate_pid(owner_pid, **kwargs)
        except ProcessLookupError:
            return owner_pid
        except OSError:
            return None
        exited, safe_to_force = _wait_for_scoped_lock_owner_exit(
            owner_pid, owner_start_time, attempts=attempts, delay=delay
        )
        if exited:
            return owner_pid
        if not safe_to_force:
            return None
    return None


def write_planned_stop_marker(target_pid: int) -> bool:
    """Record that ``target_pid`` is being stopped intentionally: unexpected SIGTERM exits non-zero
    so service managers revive the gateway; the CLI writes this first so a deliberate stop exits
    cleanly."""
    return _write_marker(_get_planned_stop_marker_path(), {
        "target_pid": target_pid, "target_start_time": _get_process_start_time(target_pid),
        "stopper_pid": os.getpid(), "written_at": _utc_now_iso(),
    })


def consume_planned_stop_marker_for_self() -> bool:
    """Return True when the current process is being intentionally stopped."""
    return _consume_pid_marker_for_self(
        _get_planned_stop_marker_path(), ttl_s=_PLANNED_STOP_MARKER_TTL_S
    )


def planned_stop_marker_targets_self() -> bool:
    """Non-destructive watcher probe: True when a live planned-stop marker names us. Never unlinks a
    matching marker (the shutdown handler does the authoritative consume); malformed/expired ones
    are still cleaned up; markers naming another PID are left alone."""
    parsed = _read_live_pid_marker(_get_planned_stop_marker_path(), _PLANNED_STOP_MARKER_TTL_S)
    return parsed is not None and _pid_marker_names_self(parsed[1], parsed[2])


def get_running_pid(
    pid_path: Optional[Path] = None, *, cleanup_stale: bool = True
) -> Optional[int]:
    """PID of a running gateway (lock + PID file verified against the live process), or None."""
    resolved_pid_path = pid_path or _get_pid_path()
    resolved_lock_path = _get_gateway_lock_path(resolved_pid_path)
    if is_gateway_runtime_lock_active(resolved_lock_path):
        records = (
            _read_pid_record(resolved_pid_path), _read_gateway_lock_record(resolved_lock_path),
        )
        for record in records:
            pid = _live_pid_from_record(record)
            if pid is None or not _pid_record_belongs_to_current_profile(record):
                continue
            if _record_matches_live_gateway_pid(record, pid):
                return pid
        _cleanup_invalid_pid_path(resolved_pid_path, cleanup_stale=cleanup_stale)
        return get_runtime_status_running_pid() if pid_path is None else None
    # Lock inactive: the runtime-status fallback runs BEFORE cleanup here.
    runtime_pid = get_runtime_status_running_pid() if pid_path is None else None
    if runtime_pid is None:
        _cleanup_invalid_pid_path(resolved_pid_path, cleanup_stale=cleanup_stale)
    return runtime_pid


def get_running_pid_identity_strict(pid_path: Path) -> Optional[tuple[int, float]]:
    """Return a verified process identity or fail on ambiguous runtime state."""
    resolved_pid_path = Path(pid_path)
    resolved_lock_path = _get_gateway_lock_path(resolved_pid_path)
    pid_exists = _strict_path_exists(resolved_pid_path, "gateway PID")
    # A stale PID file without a lock is not a live gateway; the lock probe is authoritative
    # for absence.
    if not _strict_path_exists(resolved_lock_path, "gateway lock"):
        return None
    if not _is_gateway_runtime_lock_active_strict(resolved_lock_path):
        return None
    if not pid_exists:
        raise RuntimeError("active gateway lock has no PID metadata")
    records = (_read_pid_record(resolved_pid_path), _read_gateway_lock_record(resolved_lock_path))
    if not all(records):
        raise RuntimeError("gateway PID or lock metadata is malformed")
    pid = _pid_from_record(records[0])
    if pid is None or pid <= 0 or _pid_from_record(records[1]) != pid:
        raise RuntimeError("gateway PID and lock identities disagree")
    if not _pid_exists(pid):
        raise RuntimeError("gateway identity is not live")
    current_start = _get_process_start_time(pid)
    starts = tuple(record.get("start_time") for record in records)
    if current_start is None or any(start is None for start in starts):
        raise RuntimeError("gateway creation time is unavailable")
    try:
        if not _start_times_agree(current_start, *starts):
            raise RuntimeError("gateway process identity changed")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("gateway creation time is malformed") from exc
    if not all(_record_matches_live_gateway_pid(record, pid) for record in records):
        raise RuntimeError("runtime metadata does not identify a live gateway")
    current = float(current_start)
    if not _IS_WINDOWS:
        return pid, current
    # Windows persists a centisecond fingerprint; SCM checks need the exact psutil epoch.
    # Re-read only after validation and prove it rounds to the same value.
    try:
        import psutil  # type: ignore

        exact_create_time = float(psutil.Process(pid).create_time())
    except Exception as exc:
        raise RuntimeError("exact gateway creation time is unavailable") from exc
    if int(round(exact_create_time * 100)) != int(current):
        raise RuntimeError("gateway process identity changed")
    return pid, exact_create_time


def get_running_pid_cached(
    pid_path: Optional[Path] = None, *, cleanup_stale: bool = True,
    ttl_seconds: float = _GATEWAY_RUNNING_PID_CACHE_TTL_SECONDS,
) -> Optional[int]:
    """Cached ``get_running_pid()`` for dashboard polling: short TTL, invalidated on PID/lock/
    runtime-status file changes, so status endpoints do not re-flock ``gateway.lock`` constantly."""
    if ttl_seconds <= 0:
        return get_running_pid(pid_path, cleanup_stale=cleanup_stale)
    resolved_pid_path = pid_path or _get_pid_path()
    include_runtime_status = pid_path is None
    # The signature covers the PID file, its sibling lock and (unscoped) the runtime status file.
    watched = [resolved_pid_path, _get_gateway_lock_path(resolved_pid_path)]
    if include_runtime_status:
        watched.append(_get_runtime_status_path())
    signature = tuple(_file_cache_signature(p) for p in watched)
    key = (str(resolved_pid_path), bool(cleanup_stale), include_runtime_status)
    now = time.monotonic()
    with _gateway_running_pid_cache_lock:
        cached = _gateway_running_pid_cache.get(key)
        if cached is not None and now - cached[0] <= ttl_seconds and cached[1] == signature:
            return cached[2]
    pid = get_running_pid(pid_path, cleanup_stale=cleanup_stale)
    refreshed_signature = tuple(_file_cache_signature(p) for p in watched)
    with _gateway_running_pid_cache_lock:
        _gateway_running_pid_cache[key] = (time.monotonic(), refreshed_signature, pid)
    return pid


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def clear_planned_stop_marker() -> None:
    """Remove the planned-stop marker unconditionally."""
    try:
        _get_planned_stop_marker_path().unlink(missing_ok=True)
    except OSError:
        pass

def is_gateway_running(
    pid_path: Optional[Path] = None,
    *,
    cleanup_stale: bool = True,
) -> bool:
    """Check if the gateway daemon is currently running."""
    return get_running_pid(pid_path, cleanup_stale=cleanup_stale) is not None
# ---- END PLUGIN-COMPAT ----

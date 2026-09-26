"""Machine-wide, read-only inventory for ``hermes monitor``.

Process discovery is the inventory source so older Hermes installs appear immediately. Structured
spawn-ledger and delegation manifests enrich rows when available; neither is required for a process
to be visible.
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class ProcessKey:
    pid: int
    create_time: float


@dataclass
class ProcessRow:
    pid: int
    create_time: float
    kind: str
    profile: str
    cpu_percent: float
    rss: int
    elapsed: float
    status: str
    tty: str
    tmux: str
    activity: str
    confidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MonitorSnapshot:
    observed_at: float
    processes: list[ProcessRow] = field(default_factory=list)
    delegations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_processes(self) -> int:
        return len(self.processes)

    @property
    def total_delegations(self) -> int:
        return len(self.delegations)

    @property
    def total_profiles(self) -> int:
        return len({row.profile for row in self.processes})

    @property
    def total_rss(self) -> int:
        return sum(max(0, row.rss) for row in self.processes)

    @property
    def total_cpu(self) -> float:
        return sum(max(0.0, row.cpu_percent) for row in self.processes)


def _default_process_iter():
    import psutil

    return psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time", "status"])


def _default_process_key_resolver(pid: int) -> ProcessKey | None:
    """Read a PID incarnation through a newly constructed psutil handle."""
    try:
        import psutil

        proc = psutil.Process(pid)
        return ProcessKey(int(proc.pid), float(proc.create_time()))
    except Exception:
        return None


def _default_ledger_reader() -> list[dict[str, Any]]:
    from hermes_cli.process_identity import _ledger_path, _read_ledger

    # Monitoring is observational: unlike ledger_entries(), this must not quarantine a corrupt
    # ledger. Process discovery below independently proves every matching PID incarnation.
    return _read_ledger(_ledger_path()) or []


def _default_home_resolver(pid: int) -> str | None:
    from hermes_cli.dashboard_procs import _hermes_home_for_pid

    return _hermes_home_for_pid(pid)


def _default_start_fingerprint_resolver(proc: Any, key: ProcessKey) -> int | None:
    from gateway.status import get_process_start_time

    if not _process_matches_key(proc, key):
        return None
    fingerprint = get_process_start_time(key.pid)
    return fingerprint if _process_matches_key(proc, key) else None


def _process_matches_key(proc: Any, key: ProcessKey) -> bool:
    """Cheap handle check; security-sensitive callers additionally use a fresh resolver."""
    try:
        return int(proc.pid) == key.pid and float(proc.create_time()) == key.create_time
    except Exception:
        return False


_PROFILE_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_SESSION_RE = re.compile(r"[0-9]{8}_[0-9]{6}_[0-9a-f]{6,32}")
_KANBAN_TASK_RE = re.compile(r"t_[0-9a-f]{8,32}")
_TMUX_PANE_RE = re.compile(r"%[0-9]{1,10}")
_TTY_RE = re.compile(r"(?:pts/[0-9]{1,10}|tty[0-9]{1,10}|console|con)", re.IGNORECASE)
_DELEGATION_ID_RE = re.compile(r"deleg_[0-9a-f]{8}")
_SUBAGENT_ID_RE = re.compile(r"sa-[0-9]{1,8}-[0-9a-f]{8}")
_PROCESS_STATUSES = frozenset({
    "running", "sleeping", "disk-sleep", "stopped", "tracing-stop", "zombie", "dead",
    "wake-kill", "waking", "parked", "idle", "locked", "waiting",
})
_DELEGATION_STATUSES = frozenset({"queued", "running"})


def _safe_match(value: Any, pattern: re.Pattern[str]) -> str | None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        return None
    try:
        from agent.redact import redact_sensitive_text

        if redact_sensitive_text(value, force=True) != value:
            return None
    except Exception:
        return None
    return value


def _profile_from_home(home: str | None) -> str:
    if not home:
        return "?"
    path = Path(home)
    if path.parent.name == "profiles":
        return _safe_match(path.name, _PROFILE_RE) or "?"
    return "default"


def _windows_sid_for_pid(pid: int) -> str | None:
    """Return the process token's account SID, failing closed on access/race errors."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.LocalFree.argtypes = (wintypes.HLOCAL,)
        kernel32.LocalFree.restype = wintypes.HLOCAL
        advapi32.OpenProcessToken.argtypes = (
            wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE),
        )
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = (
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        )
        advapi32.GetTokenInformation.restype = wintypes.BOOL
        advapi32.ConvertSidToStringSidW.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR),
        )
        advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL

        process = kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            return None
        token = wintypes.HANDLE()
        try:
            if not advapi32.OpenProcessToken(process, 0x0008, ctypes.byref(token)):
                return None
            size = wintypes.DWORD()
            advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
            if not size.value:
                return None
            buffer = ctypes.create_string_buffer(size.value)
            if not advapi32.GetTokenInformation(
                token, 1, buffer, size.value, ctypes.byref(size),
            ):
                return None

            class _SidAndAttributes(ctypes.Structure):
                _fields_ = [("sid", ctypes.c_void_p), ("attributes", wintypes.DWORD)]

            sid = ctypes.cast(buffer, ctypes.POINTER(_SidAndAttributes)).contents.sid
            rendered = wintypes.LPWSTR()
            if not advapi32.ConvertSidToStringSidW(sid, ctypes.byref(rendered)):
                return None
            try:
                return str(rendered.value) if rendered.value else None
            finally:
                kernel32.LocalFree(rendered)
        finally:
            if token:
                kernel32.CloseHandle(token)
            kernel32.CloseHandle(process)
    except Exception:
        return None


def _default_account_identity_resolver(proc: Any) -> object | None:
    if sys.platform == "win32":
        return _windows_sid_for_pid(int(proc.pid))
    try:
        return int(proc.uids().real)
    except Exception:
        return None


def _default_current_account_identity() -> object | None:
    if sys.platform == "win32":
        return _windows_sid_for_pid(os.getpid())
    try:
        return int(getattr(os, "getuid")())
    except (AttributeError, OSError):
        return None


_HERMES_EXECUTABLES = frozenset({
    "hermes", "hermes.exe", "hermes-agent", "hermes-acp", "hermes-gateway",
})
_HERMES_MODULES = frozenset({"hermes_cli.main", "agent.legacy_cli", "acp_adapter.entry", "gateway.run"})
_HERMES_SCRIPT_SUFFIXES = (
    "/hermes_cli/main.py", "/agent/legacy_cli.py", "/acp_adapter/entry.py", "/gateway/run.py",
)


def _argv_basename(token: str) -> str:
    return str(token).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].casefold()


def _launcher_tail_index(argv: list[str]) -> int | None:
    if not argv:
        return None
    executable = _argv_basename(argv[0])
    if executable in _HERMES_EXECUTABLES:
        return 1
    if not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", executable):
        return None
    if len(argv) >= 3 and argv[1] == "-m" and argv[2] in _HERMES_MODULES:
        return 3
    if len(argv) < 2:
        return None
    launcher = str(argv[1]).replace("\\", "/")
    launcher_basename = _argv_basename(launcher)
    if launcher_basename in _HERMES_EXECUTABLES:
        return 2
    lowered = launcher.casefold()
    if any(lowered.endswith(suffix) for suffix in _HERMES_SCRIPT_SUFFIXES):
        return 2
    if launcher_basename == "desktop-gateway.py":
        return 2
    return None


def _is_hermes_candidate(_name: str, argv: list[str]) -> bool:
    return _launcher_tail_index(argv) is not None


def _option_value(argv: list[str], *options: str) -> str | None:
    for i, token in enumerate(argv):
        if token in options and i + 1 < len(argv):
            return str(argv[i + 1])
        for option in options:
            prefix = f"{option}="
            if token.startswith(prefix):
                return token[len(prefix):]
    return None


def _command_tokens(argv: list[str]) -> list[str]:
    """Canonical Hermes command tail, excluding the launcher and top-level options."""
    start = _launcher_tail_index(argv)
    if start is None:
        return []
    try:
        from hermes_cli._parser import command_argv

        return [str(token) for token in command_argv(argv[start:])]
    except Exception:
        return []


def _classify(argv: list[str], env: dict[str, str], ledger: dict[str, Any] | None) -> tuple[str, str]:
    task = _safe_match(env.get("HERMES_KANBAN_TASK", "").strip(), _KANBAN_TASK_RE)
    if not task:
        for i, token in enumerate(argv[:-1]):
            if token == "task" and i >= 2 and argv[i - 2:i] == ["work", "kanban"]:
                task = _safe_match(str(argv[i + 1]), _KANBAN_TASK_RE)
                break
    if task:
        return "kanban", f"task {task[:80]}"

    purpose = str((ledger or {}).get("purpose") or "").strip().casefold()
    command = _command_tokens(argv)
    lowered = [token.casefold() for token in command]
    if purpose in {"gateway", "serve", "dashboard", "mcp-helper"}:
        kind = "backend" if purpose in {"serve", "dashboard"} else purpose
        return kind, purpose
    if lowered[:2] == ["gateway", "run"]:
        return "gateway", "gateway run"
    if lowered[:1] in (["serve"], ["dashboard"]):
        return "backend", "web backend"
    if lowered[:1] == ["monitor"]:
        return "monitor", "fleet dashboard"

    session = _option_value(command, "--resume", "-r") or _option_value(argv, "--resume", "-r")
    safe_session = _safe_match(session, _SESSION_RE)
    if safe_session:
        return "interactive", f"session {safe_session}"
    if lowered[:1] == ["chat"] or any(token in argv for token in ("--tui", "--cli")) or len(argv) > 1:
        return "interactive", "interactive"
    return "hermes", "running"


def _default_delegation_scanner(homes: Iterable[str]) -> list[dict[str, Any]]:
    from tools.delegation_live_log import scan_live_delegations

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for home in homes:
        try:
            key = os.path.normcase(str(Path(home).resolve()))
        except (OSError, RuntimeError, ValueError):
            continue
        if not key or key in seen:
            continue
        seen.add(key)
        root = Path(home) / "cache" / "delegation" / "live"
        try:
            trusted = scan_live_delegations(root)
        except Exception:
            trusted = []
        for row in trusted:
            projected = {
                key: row.get(key) for key in (
                    "owner_pid", "owner_started_at", "subagent_id", "delegation_id", "task_index", "status",
                    "updated_at",
                )
            }
            projected["confidence"] = "verified"
            rows.append(projected)
    return rows


def _safe_delegation(
    row: dict[str, Any], owners: dict[int, tuple[ProcessRow, ProcessKey, int]],
) -> dict[str, Any] | None:
    try:
        owner_pid = row["owner_pid"]
        owner_started_at = float(row["owner_started_at"])
        task_index = row["task_index"]
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        not isinstance(owner_pid, int) or isinstance(owner_pid, bool) or owner_pid <= 0
        or not math.isfinite(owner_started_at)
        or not isinstance(task_index, int) or isinstance(task_index, bool) or task_index < 0
    ):
        return None
    owner_entry = owners.get(owner_pid)
    delegation_id = _safe_match(row.get("delegation_id"), _DELEGATION_ID_RE)
    status = row.get("status")
    if owner_entry is None:
        return None
    owner, owner_key, owner_start_fingerprint = owner_entry
    if (
        owner.pid != owner_key.pid
        or owner.create_time != owner_key.create_time
        or owner_started_at != owner_start_fingerprint
        or delegation_id is None
        or status not in _DELEGATION_STATUSES
    ):
        return None
    try:
        updated_at = float(row["updated_at"])
    except (KeyError, TypeError, ValueError, OverflowError):
        updated_at = None
    if updated_at is not None and not math.isfinite(updated_at):
        updated_at = None
    confidence = row.get("confidence")
    return {
        "owner_pid": owner_pid,
        "subagent_id": _safe_match(row.get("subagent_id"), _SUBAGENT_ID_RE),
        "delegation_id": delegation_id,
        "task_index": task_index,
        "status": status,
        "profile": owner.profile,
        "updated_at": updated_at,
        "confidence": confidence if confidence == "verified" else None,
    }


class MonitorSampler:
    """Stateful process sampler; state is only prior CPU totals keyed by PID incarnation."""

    def __init__(
        self,
        *,
        process_iter: Callable[[], Iterable[Any]] | None = None,
        ledger_reader: Callable[[], list[dict[str, Any]]] | None = None,
        home_resolver: Callable[[int], str | None] | None = None,
        delegation_scanner: Callable[[Iterable[str]], list[dict[str, Any]]] | None = None,
        start_fingerprint_resolver: Callable[[Any, ProcessKey], int | None] | None = None,
        process_key_resolver: Callable[[int], ProcessKey | None] | None = None,
        account_identity_resolver: Callable[[Any], object | None] | None = None,
        current_account_identity: object | None = None,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._process_iter = process_iter or _default_process_iter
        self._ledger_reader = ledger_reader or _default_ledger_reader
        self._home_resolver = home_resolver or _default_home_resolver
        self._delegation_scanner = delegation_scanner or _default_delegation_scanner
        self._start_fingerprint_resolver = (
            start_fingerprint_resolver or _default_start_fingerprint_resolver
        )
        self._process_key_resolver = process_key_resolver or _default_process_key_resolver
        self._account_identity_resolver = (
            account_identity_resolver or _default_account_identity_resolver
        )
        self._current_account_identity = (
            _default_current_account_identity()
            if current_account_identity is None else current_account_identity
        )
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._cpu_samples: dict[ProcessKey, tuple[float, float]] = {}

    def _is_fresh(self, key: ProcessKey) -> bool:
        try:
            return self._process_key_resolver(key.pid) == key
        except Exception:
            return False

    def _read_if_fresh(self, key: ProcessKey, read: Callable[[], Any]) -> Any:
        if not self._is_fresh(key):
            raise RuntimeError("process incarnation changed")
        value = read()
        if not self._is_fresh(key):
            raise RuntimeError("process incarnation changed")
        return value

    def sample(self) -> MonitorSnapshot:
        observed = float(self._wall_clock())
        monotonic = float(self._monotonic_clock())
        try:
            ledgers = self._ledger_reader()
        except Exception:
            ledgers = []
        ledger_by_pid = {
            int(entry["pid"]): entry for entry in ledgers
            if isinstance(entry, dict) and isinstance(entry.get("pid"), int)
        }

        rows: list[ProcessRow] = []
        discovered_homes: set[str] = set()
        owner_fingerprints: dict[ProcessKey, tuple[int, Any]] = {}
        live_keys: set[ProcessKey] = set()
        discovered: dict[int, tuple[Any, dict[str, Any], ProcessKey, list[str], int, bool]] = {}
        try:
            processes = self._process_iter()
        except Exception:
            processes = []
        for proc in processes:
            try:
                info = getattr(proc, "info", {}) or {}
                pid = int(info.get("pid", getattr(proc, "pid")))
                key = ProcessKey(pid, float(info.get("create_time")))
                identity = self._read_if_fresh(
                    key, lambda: self._account_identity_resolver(proc),
                )
                if identity is None or identity != self._current_account_identity:
                    continue
                name, argv, ppid = self._read_if_fresh(key, lambda: (
                    str(info.get("name") or ""),
                    [str(token) for token in (info.get("cmdline") or [])],
                    int(info.get("ppid") or 0),
                ))
                discovered[pid] = (proc, info, key, argv, ppid, _is_hermes_candidate(name, argv))
            except Exception:
                continue

        included = {pid for pid, record in discovered.items() if record[-1]}
        while True:
            descendants = {
                pid for pid, record in discovered.items()
                if (
                    pid not in included
                    and record[4] in included
                    and discovered[record[4]][2].create_time <= record[2].create_time
                    and self._is_fresh(record[2])
                    and self._is_fresh(discovered[record[4]][2])
                )
            }
            if not descendants:
                break
            included.update(descendants)

        for pid in included:
            proc, info, key, argv, _ppid, canonical = discovered[pid]
            try:
                create_time = key.create_time
                ledger = ledger_by_pid.get(pid)
                if ledger is not None:
                    try:
                        ledger_create = float(ledger["create_time"])
                    except (KeyError, TypeError, ValueError):
                        ledger = None
                    else:
                        if ledger_create != create_time:
                            ledger = None
                start_fingerprint = None
                if canonical:
                    try:
                        start_fingerprint = self._read_if_fresh(
                            key, lambda: self._start_fingerprint_resolver(proc, key),
                        )
                    except Exception:
                        start_fingerprint = None
                cpu_times = self._read_if_fresh(key, proc.cpu_times)
                cpu_total = float(cpu_times.user) + float(cpu_times.system)
                previous = self._cpu_samples.get(key)
                cpu_percent = 0.0
                if previous is not None:
                    wall_delta = monotonic - previous[0]
                    if wall_delta > 0:
                        cpu_percent = max(0.0, 100.0 * (cpu_total - previous[1]) / wall_delta)
                if not math.isfinite(cpu_percent):
                    cpu_percent = 0.0
                env = self._read_if_fresh(key, proc.environ)
                safe_env = {
                    key: str(env.get(key) or "")
                    for key in ("HERMES_KANBAN_TASK", "TMUX_PANE")
                }
                kind, activity = _classify(argv, safe_env, ledger)
                if not canonical and not (ledger or {}).get("purpose"):
                    kind, activity = "helper", "helper process"
                ledger_home = (ledger or {}).get("hermes_home")
                home = self._read_if_fresh(key, lambda: (
                    ledger_home
                    if isinstance(ledger_home, str) and ledger_home
                    else self._home_resolver(pid)
                ))
                tty = str(self._read_if_fresh(key, proc.terminal) or "—")
                if tty.startswith("/dev/"):
                    tty = tty[5:]
                tty = _safe_match(tty, _TTY_RE) or "—"
                status = str(self._read_if_fresh(key, proc.status) or "?")
                if status not in _PROCESS_STATUSES:
                    status = "?"
                tmux = _safe_match(safe_env["TMUX_PANE"], _TMUX_PANE_RE) or "—"
                rss = int(self._read_if_fresh(key, proc.memory_info).rss)
                row = ProcessRow(
                    pid=pid,
                    create_time=create_time,
                    kind=kind,
                    profile=_profile_from_home(home),
                    cpu_percent=round(cpu_percent, 1),
                    rss=max(0, rss),
                    elapsed=max(0.0, observed - create_time),
                    status=status,
                    tty=tty,
                    tmux=tmux,
                    activity=activity,
                    confidence="verified" if ledger else "discovered",
                )
                if not self._is_fresh(key):
                    continue
                self._cpu_samples[key] = (monotonic, cpu_total)
                live_keys.add(key)
                rows.append(row)
                if (
                    isinstance(start_fingerprint, int)
                    and not isinstance(start_fingerprint, bool)
                    and start_fingerprint > 0
                ):
                    owner_fingerprints[key] = (start_fingerprint, proc)
                if home:
                    discovered_homes.add(str(home))
            except Exception:
                continue

        self._cpu_samples = {
            key: sample for key, sample in self._cpu_samples.items() if key in live_keys
        }
        rows.sort(key=lambda row: row.pid)
        try:
            delegations = self._delegation_scanner(sorted(discovered_homes))
        except Exception:
            delegations = []
        rows_by_key = {ProcessKey(row.pid, row.create_time): row for row in rows}
        owners = {}
        for key, (fingerprint, proc) in owner_fingerprints.items():
            row = rows_by_key.get(key)
            if row is not None and self._is_fresh(key):
                owners[key.pid] = (row, key, fingerprint)
        safe_delegations = []
        for delegation in delegations:
            if not isinstance(delegation, dict):
                continue
            owner_pid = delegation.get("owner_pid")
            owner_entry = owners.get(owner_pid) if isinstance(owner_pid, int) else None
            if owner_entry is None or not self._is_fresh(owner_entry[1]):
                continue
            safe = _safe_delegation(delegation, owners)
            if safe is not None and self._is_fresh(owner_entry[1]):
                safe_delegations.append(safe)
        return MonitorSnapshot(observed_at=observed, processes=rows, delegations=safe_delegations)

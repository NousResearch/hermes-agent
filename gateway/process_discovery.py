"""Gateway-owned process and supervisor discovery.

This module owns process-table discovery and service-PID exclusion used by
gateway lifecycle code. CLI surfaces may re-export these names for compatibility,
but Gateway backends must depend here rather than reaching through hermes_cli.gateway.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from gateway import service_identity as _service_identity
from gateway import systemd_runtime as _systemd_runtime
from gateway.launchd_service import (
    _locate_launchd_gateway_service,
    get_launchd_label,
    launchd_gateway_labels_for_install,
)
from gateway.restart import _get_parent_pid
from hermes_constants import get_hermes_home

_CAPTURE_TEXT = dict(capture_output=True, text=True, encoding="utf-8", errors="replace")


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_windows() -> bool:
    return sys.platform == "win32"


@dataclass(frozen=True)
class ProfileGatewayProcess:
    profile: str
    path: Path
    pid: int
    create_time: float = 0.0


@dataclass(frozen=True)
class WindowsGatewayService:
    """A real Windows service supervising a profile gateway process tree."""

    name: str
    profile: str
    service_pid: int
    gateway_pid: int
    descendant_pids: frozenset[int]
    descendant_identities: tuple[tuple[int, float], ...]
    service_create_time: float = 0.0
    gateway_create_time: float = 0.0

def _get_service_pids(all_profiles: bool = False) -> set:
    """PIDs managed by systemd/launchd gateway services (excluded from stale-process sweeps).

    Relies on the service manager committing the new PID before the restart command returns.
    ``all_profiles`` widens the current profile's unit/label to the whole ``hermes-gateway*`` /
    ``ai.hermes.gateway*`` fleet so update/reaper never kill a sibling's service gateway as "manual".

    ``all_profiles`` widens the launchd branch to every installed ``ai.hermes.gateway*`` LaunchAgent — the
    update path needs the whole fleet excluded from its sweep (#41403, #73626): sibling-profile launchd
    gateways found by the (BSD-fixed) ps scan must not be misclassified as manual processes and killed.
    Default-scope callers (``gateway status``, cron checks) keep seeing only the current profile's service;
    the orphan reaper passes all_profiles=True for the same friendly-fire reason. The systemd branch mirrors
    this: default scope filters to the current profile's exact unit name; ``all_profiles=True`` widens to
    the ``hermes-gateway*`` fleet glob.
    """
    pids: set = set()

    # --- systemd (Linux): user and system scopes ---
    if _systemd_runtime.supports_services():
        pattern = "hermes-gateway*" if all_profiles else _service_identity.service_name()
        for scope_args in [["systemctl", "--user"], ["systemctl"]]:
            try:
                # Belt-and-suspenders for the EXCLUDE use case (#74075): a bare ``launchctl list`` prefix
                # scan also catches ai.hermes.gateway* agents the label derivation can't map (renamed
                # profiles, other installs sharing this user). Over-inclusion is safe here — these PIDs are
                # only ever protected from the kill sweep, never targeted. Restart paths use the
                # label-derived set only.
                result = subprocess.run(
                    scope_args
                    + ["list-units", pattern, "--plain", "--no-legend", "--no-pager"],
                    timeout=5,
                    **_CAPTURE_TEXT,
                )
                for line in result.stdout.strip().splitlines():
                    parts = line.split()
                    if not parts or not parts[0].endswith(".service"):
                        continue
                    svc = parts[0]
                    try:
                        show = subprocess.run(
                            scope_args + ["show", svc, "--property=MainPID", "--value"],
                            timeout=5,
                            **_CAPTURE_TEXT,
                        )
                        pid = int(show.stdout.strip())
                        if pid > 0:
                            pids.add(pid)
                    except (ValueError, subprocess.TimeoutExpired):
                        pass
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

    # --- launchd (macOS) ---
    if is_macos():
        labels = {get_launchd_label()}
        if all_profiles:
            # Whole fleet, mirroring the systemd ``hermes-gateway*`` glob above.
            # Every gateway LaunchAgent, not just the invoking profile's — mirrors the systemd branch's
            # ``hermes-gateway*`` pattern above. The update path restarts the whole fleet, and its
            # stale-process sweep must not mistake a sibling service's fresh PID for a manual gateway it
            # should kill (#41403).
            labels.update(launchd_gateway_labels_for_install())
        for label in sorted(labels):
            try:
                _domain, pid = _locate_launchd_gateway_service(label)
            except subprocess.TimeoutExpired:
                continue
            if pid is not None and pid > 0:
                pids.add(pid)
        if all_profiles:
            # Prefix scan also catches ai.hermes.gateway* agents the label derivation can't map
            # (renamed profiles, other installs). Over-inclusion is safe: PIDs are only protected.
            try:
                result = subprocess.run(["launchctl", "list"], timeout=5, **_CAPTURE_TEXT)
                if result.returncode == 0:
                    for line in result.stdout.strip().splitlines():
                        parts = line.split()
                        if len(parts) >= 3 and parts[-1].startswith("ai.hermes.gateway"):
                            try:
                                pid = int(parts[0])
                                if pid > 0:
                                    pids.add(pid)
                            except ValueError:
                                pass
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

    return pids

def _get_ancestor_pids() -> set[int]:
    """PIDs of this process and its ancestors, so scans never count the invoking ``hermes`` CLI as a gateway.

    Walks from the current PID up to PID 1 (init) so that process-table scans never match the calling CLI
    process or any of its parents. This prevents ``hermes gateway status`` from falsely counting the
    ``hermes`` CLI that invoked it as a running gateway instance (see #13242).
    """
    ancestors: set[int] = set()
    pid = os.getpid()
    for _ in range(64):
        ancestors.add(pid)
        parent = _get_parent_pid(pid)
        if parent is None or parent <= 0 or parent in ancestors:
            break
        pid = parent
    return ancestors


def _append_unique_pid(pids: list[int], pid: int | None, exclude_pids: set[int]) -> None:
    if pid and pid > 0 and pid != os.getpid() and pid not in exclude_pids and pid not in pids:
        pids.append(pid)


def _iter_proc_cmdlines(exclude_pids: set[int]):
    """Yield ``(pid, cmdline)`` from ``/proc`` (Docker without procps); raises if /proc is unusable."""
    my_pid = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == my_pid or pid in exclude_pids:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as _f:
                cmdline = _f.read().decode("utf-8", errors="replace")
        except (OSError, PermissionError):
            continue
        yield pid, cmdline.replace("\x00", " ")


def _scan_gateway_pids(
    exclude_pids: set[int], all_profiles: bool = False, include_restart_managers: bool = False
) -> list[int]:
    """Best-effort process-table scan for gateway PIDs (backs up a stale/missing PID file; ``--all`` sweeps)."""
    # Exclude the entire ancestor chain so the CLI process that invoked this scan (e.g. ``hermes gateway
    # status``) is never mistaken for a running gateway. See #13242.
    exclude_pids = exclude_pids | _get_ancestor_pids()
    pids: list[int] = []
    # Strict matcher shared with gateway.status: requires a real ``gateway run`` argv, so
    # ``gateway status``/``dashboard`` siblings and ``python -m tui_gateway`` don't match.
    from gateway.status import (
        looks_like_gateway_command_line,
        looks_like_gateway_runtime_command_line,
        profile_flag_value,
        hermes_home_assignments,
        command_line_names_hermes_home,
    )
    current_home = str(get_hermes_home().resolve())
    # Forward slashes on both sides of the HERMES_HOME= match (mirrors gateway.status), and no
    # trailing separator: the assignments parser strips one, so the systemd ``Environment=``
    # spelling (``HERMES_HOME=/root/.hermes/``) compares equal to the resolved home.
    current_home_lc = current_home.lower().replace("\\", "/").rstrip("/")
    current_profile_arg = _service_identity.profile_arg(current_home)
    current_profile_name = current_profile_arg.split()[-1] if current_profile_arg else ""
    current_profile_name_lc = current_profile_name.lower()

    def _matches_current_profile(command: str) -> bool:
        command_lc = command.lower().replace("\\", "/")
        if current_profile_name:
            # Token equality, not substring: `-p ops` must not claim (or SIGTERM) an `-p ops-2` gateway.
            if profile_flag_value(command_lc) == current_profile_name_lc:
                return True
            return command_line_names_hermes_home(command_lc, current_home_lc)

        # Default profile: accept unless argv advertises another profile in any spelling the CLI
        # pre-parser accepts (``--profile=ops`` slipped past a substring test, so a default-profile
        # fallback stop could SIGTERM the named gateway). HERMES_HOME may come via env (invisible to
        # wmic/CIM), so only a non-matching explicit HERMES_HOME= disqualifies.
        if profile_flag_value(command_lc) is not None:
            return False
        return (not hermes_home_assignments(command_lc)
                or command_line_names_hermes_home(command_lc, current_home_lc))

    def _consider(pid: int, command: str) -> None:
        matches_runtime = looks_like_gateway_command_line(command) or (
            include_restart_managers and looks_like_gateway_runtime_command_line(command)
        )
        if matches_runtime and (all_profiles or _matches_current_profile(command)):
            _append_unique_pid(pids, pid, exclude_pids)

    try:
        if is_windows():
            listing = _windows_process_listing()
            if listing is None:
                return []
            for pid, command in _iter_windows_list_processes(listing):
                _consider(pid, command)
        else:
            # /proc first (Docker without procps), then `ps -Aww`.
            _found_via_proc = False
            if os.path.isdir("/proc"):
                try:
                    for pid, command in _iter_proc_cmdlines(exclude_pids):
                        _consider(pid, command)
                    _found_via_proc = True
                except Exception:
                    pass

            if not _found_via_proc:
                # ``-Aww`` not ``-A eww``: BSD/macOS ps rejects ``e``; ``-ww`` = unlimited width.
                result = subprocess.run(["ps", "-Aww", "-o", "pid=,command="], timeout=10, **_CAPTURE_TEXT)
                if result.returncode != 0:
                    return []
                for line in result.stdout.split("\n"):
                    parsed = _parse_ps_line(line)
                    if parsed is not None:
                        _consider(*parsed)
    except (OSError, subprocess.TimeoutExpired):
        return []

    # Windows: a venv ``pythonw.exe`` is a launcher stub that spawns the base Python with the same
    # command line, so each gateway yields two matched PIDs. Drop a matched PID that parents another.
    if is_windows() and len(pids) > 1:
        pids = _filter_venv_launcher_stubs(pids)

    return pids


def _parse_ps_line(line: str) -> tuple[int, str] | None:
    """``(pid, command)`` from one ``ps -o pid=,command=`` line; also accepts ``ps aux`` rows."""
    stripped = line.strip()
    if not stripped or "grep" in stripped:
        return None
    parts = stripped.split(None, 1)
    if len(parts) == 2:
        with contextlib.suppress(ValueError):
            return int(parts[0]), parts[1]
    aux_parts = stripped.split()
    if len(aux_parts) > 10 and aux_parts[1].isdigit():
        return int(aux_parts[1]), " ".join(aux_parts[10:])
    return None


def _iter_windows_list_processes(listing: str):
    """Yield ``(pid, command_line)`` from wmic/CIM ``/FORMAT:LIST`` output."""
    current_cmd = ""
    for line in listing.split("\n"):
        line = line.strip()
        if line.startswith("CommandLine="):
            current_cmd = line[len("CommandLine=") :]
        elif line.startswith("ProcessId="):
            with contextlib.suppress(ValueError):
                yield int(line[len("ProcessId=") :]), current_cmd
            current_cmd = ""


def _windows_process_listing() -> str | None:
    """``CommandLine=``/``ProcessId=`` LIST output for every Windows process (wmic, else Get-CimInstance), or None.
    ``bounded_probe_run``, NOT ``subprocess.run(timeout=...)``: run()'s post-timeout cleanup joins pipe
    readers unbounded and a conhost.exe holding duplicated handles wedges the caller forever; it also
    hides the console window this windowless pythonw backend would flash."""
    # Prefer wmic when present (fast, stable output format). On modern Windows 11 / Win 10 late builds, wmic
    # has been removed as part of the WMIC deprecation — fall back to PowerShell's Get-CimInstance. A spawn
    # failure or timeout (result is None) trips the fallback. ``hermes update`` hung exactly there on
    # slow-WMI machines where the full Win32_Process scan exceeds its budget (#87134). bounded_probe_run
    # also hides the console window: this scan runs inside the windowless pythonw.exe gateway/desktop
    # backend, so a bare wmic/powershell spawn would flash a conhost window on every watchdog probe.
    from runtime.subprocess_compat import bounded_probe_run
    wmic_path = shutil.which("wmic")
    result = None
    if wmic_path is not None:
        result = bounded_probe_run(
            [wmic_path, "process", "get", "ProcessId,CommandLine", "/FORMAT:LIST"], timeout=10, errors="ignore"
        )
    if result is None or result.returncode != 0 or not (result.stdout or ""):
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell is None:
            return None
        ps_cmd = (
            "Get-CimInstance Win32_Process | "
            "ForEach-Object { "
            "  'CommandLine=' + ($_.CommandLine -replace \"`r`n\",' ' -replace \"`n\",' '); "
            "  'ProcessId=' + $_.ProcessId; "
            "  '' "
            "}"
        )
        result = bounded_probe_run([powershell, "-NoProfile", "-Command", ps_cmd], timeout=15, errors="ignore")
        if result is None:
            return None
    return None if result.returncode != 0 or result.stdout is None else result.stdout


def _filter_venv_launcher_stubs(pids: list[int]) -> list[int]:
    """Drop venv-launcher ``pythonw.exe`` stubs that parent another matched PID (see ``_scan_gateway_pids``)."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return pids

    pid_set = set(pids)
    drop: set[int] = set()
    for pid in pids:
        try:
            ppid = psutil.Process(pid).ppid()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if ppid is not None and ppid in pid_set:
            drop.add(ppid)
    return [p for p in pids if p not in drop]


def find_gateway_pids(exclude_pids: set | None = None, all_profiles: bool = False) -> list:
    """Find running gateway PIDs for the current profile, or every profile with ``all_profiles`` (``hermes update``)."""
    _exclude = set(exclude_pids or set())
    pids: list[int] = []
    if not all_profiles:
        try:
            from gateway.status import get_running_pid
            _append_unique_pid(pids, get_running_pid(), _exclude)
        except Exception:
            pass
    for pid in _get_service_pids(all_profiles=all_profiles):
        _append_unique_pid(pids, pid, _exclude)
    try:
        include_restart_managers = not _systemd_runtime.supports_services()
    except Exception:
        include_restart_managers = False
    for pid in _scan_gateway_pids(_exclude, all_profiles=all_profiles, include_restart_managers=include_restart_managers):
        _append_unique_pid(pids, pid, _exclude)
    return pids

def find_profile_gateway_processes(exclude_pids: set | None = None, *, strict: bool = False) -> list[ProfileGatewayProcess]:
    """Return running gateway PIDs mapped to Hermes profiles via PID files."""
    _exclude = set(exclude_pids or set())
    processes: list[ProfileGatewayProcess] = []
    try:
        from gateway.status import get_running_pid, get_running_pid_identity_strict
        from profiles.paths import get_profile_dir
        from profiles.registry import list_profile_names
    except Exception:
        if strict:
            raise
        return processes

    seen: set[int] = set()
    try:
        profile_names = list_profile_names()
    except Exception:
        if strict:
            raise
        return processes
    for profile_name in profile_names:
        try:
            profile_path = get_profile_dir(profile_name)
            if strict:
                identity = get_running_pid_identity_strict(profile_path / "gateway.pid")
                pid = identity[0] if identity else None
                create_time = identity[1] if identity else 0.0
            else:
                pid = get_running_pid(profile_path / "gateway.pid", cleanup_stale=False)
                create_time = 0.0
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Could not inspect gateway PID for profile {profile_name}") from exc
            continue
        if pid is None or pid <= 0 or pid in _exclude or pid in seen:
            continue
        seen.add(pid)
        processes.append(ProfileGatewayProcess(profile=profile_name, path=profile_path, pid=pid, create_time=create_time))
    return processes


def _scm_service_field(service, field: str):
    """psutil ``WindowsService`` exposes getters as methods; ``as_dict()`` covers objects without them."""
    getter = getattr(service, field, None)
    return getter() if callable(getter) else service.as_dict().get(field)


def find_windows_gateway_services(
    *, psutil_module=None, profile_processes: list[ProfileGatewayProcess] | None = None
) -> list[WindowsGatewayService]:
    """Profile gateways supervised by real, Hermes-owned Windows services. Service-logon processes may
    hide their command lines, so identity = Hermes's own PID file + a parent chain ending at a running
    SCM service PID whose name or binary path is Hermes's (``gateway_windows.hermes_owns_windows_service``).
    The whole service subtree is returned so the Desktop preflight exempts exactly what the updater stops
    through the SCM; a gateway under any other service (a Scheduled Task's svchost) is a plain process."""
    if sys.platform != "win32":
        return []
    try:
        if psutil_module is None:
            import psutil as psutil_module  # type: ignore[no-redef]  # noqa: PLC0415
        if profile_processes is None:
            profile_processes = find_profile_gateway_processes(strict=True)
        from gateway.windows_service import hermes_owns_windows_service, hermes_service_roots

        hermes_roots = hermes_service_roots()
        service_names_by_pid: dict[int, set[str]] = {}
        indeterminate_services_by_pid: dict[int, list[tuple[str, object]]] = {}
        for service in psutil_module.win_service_iter():
            try:
                service_name = str(_scm_service_field(service, "name") or "")
                if not service_name:
                    raise RuntimeError("SCM service has an empty name")
                # Ownership before state: an OS service above the gateway (Task Scheduler's svchost for a
                # task-launched gateway, BITS mid-transition) is never its supervisor, so neither its
                # PID nor its status may steer the pause. Only Hermes-owned services reach the guards below.
                # The name alone settles Hermes-named services; binpath (QueryServiceConfig) is asked only
                # for the rest, and a service that refuses even that to this user is one this user could
                # not `sc stop` either — never Hermes's, never a reason to abort the enumeration.
                owned = hermes_owns_windows_service(service_name, "", hermes_roots)
                if not owned:
                    try:
                        service_binpath = str(_scm_service_field(service, "binpath") or "")
                    except (psutil_module.AccessDenied, OSError):
                        continue
                    owned = hermes_owns_windows_service(service_name, service_binpath, hermes_roots)
                if not owned:
                    continue
                service_status = _scm_service_field(service, "status")
                service_pid = int(_scm_service_field(service, "pid") or 0)
            except FileNotFoundError:
                # Deleted between enumeration and inspection.
                continue
            except Exception as exc:
                raise RuntimeError("SCM service inspection failed") from exc
            if service_status == "stopped":
                continue
            if service_status != "running":
                if service_pid > 0:
                    indeterminate_services_by_pid.setdefault(service_pid, []).append((service_name, service_status))
                continue
            if service_pid <= 0:
                raise RuntimeError(f"Running SCM service {service_name} has no valid process ID")
            service_names_by_pid.setdefault(service_pid, set()).add(service_name)
    except Exception as exc:
        raise RuntimeError("SCM service enumeration failed") from exc

    found: dict[str, WindowsGatewayService] = {}
    for profile_process in profile_processes:
        try:
            gateway_process = psutil_module.Process(int(profile_process.pid))
            gateway_create_time = float(gateway_process.create_time())
            if profile_process.create_time <= 0 or abs(gateway_create_time - profile_process.create_time) > 0.001:
                raise RuntimeError("Gateway process identity changed during SCM discovery")
            ancestor_pids = [int(parent.pid) for parent in gateway_process.parents()]
            for pid in ancestor_pids:
                indeterminate_services = indeterminate_services_by_pid.get(pid, [])
                if indeterminate_services:
                    service_name, service_status = indeterminate_services[0]
                    raise RuntimeError(f"SCM service {service_name} has indeterminate status: {service_status}")
            shared_service_pids = [pid for pid in ancestor_pids if len(service_names_by_pid.get(pid, set())) > 1]
            if shared_service_pids:
                raise RuntimeError(
                    "Gateway ownership is ambiguous under shared SCM host PID(s): "
                    + ", ".join(str(pid) for pid in shared_service_pids)
                )
            service_pid = next((pid for pid in ancestor_pids if len(service_names_by_pid.get(pid, set())) == 1), None)
            if service_pid is None:
                continue
            service_name = next(iter(service_names_by_pid[service_pid]))
            service_process = psutil_module.Process(service_pid)
            service_create_time = float(service_process.create_time())
            descendant_processes = service_process.children(recursive=True)
            descendants = frozenset(int(child.pid) for child in descendant_processes)
            if int(profile_process.pid) not in descendants:
                continue
            descendant_identities = tuple(
                sorted((int(child.pid), float(child.create_time())) for child in descendant_processes)
            )
            found[service_name] = WindowsGatewayService(
                name=service_name,
                profile=str(profile_process.profile),
                service_pid=service_pid,
                gateway_pid=int(profile_process.pid),
                descendant_pids=descendants,
                descendant_identities=descendant_identities,
                service_create_time=service_create_time,
                gateway_create_time=gateway_create_time,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Could not determine SCM ownership for gateway profile {profile_process.profile}") from exc
    return [found[name] for name in sorted(found)]

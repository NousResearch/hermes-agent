"""
Gateway subcommand for hermes CLI.

Handles: hermes gateway [run|start|stop|restart|status|install|uninstall|setup]
"""

import asyncio
import os
import shutil
import signal
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from gateway.status import terminate_pid
from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    GATEWAY_SERVICE_RESTART_EXIT_CODE,
    parse_restart_drain_timeout,
)
from hermes_cli.config import (
    get_env_value,
    get_hermes_home,
    is_managed,
    managed_error,
    read_raw_config,
    save_env_value,
)
# display_hermes_home is imported lazily at call sites to avoid ImportError
# when hermes_constants is cached from a pre-update version during `hermes update`.
from hermes_cli.setup import (
    print_header, print_info, print_success, print_warning, print_error,
    prompt, prompt_choice, prompt_yes_no,
)
from hermes_cli.colors import Colors, color


# =============================================================================
# Process Management (for manual gateway runs)
# =============================================================================


@dataclass(frozen=True)
class GatewayRuntimeSnapshot:
    manager: str
    service_installed: bool = False
    service_running: bool = False
    gateway_pids: tuple[int, ...] = ()
    service_scope: str | None = None

    @property
    def running(self) -> bool:
        return self.service_running or bool(self.gateway_pids)

    @property
    def has_process_service_mismatch(self) -> bool:
        return self.service_installed and self.running and not self.service_running


@dataclass(frozen=True)
class ProfileGatewayProcess:
    profile: str
    path: Path
    pid: int

def _get_service_pids() -> set:
    """Return PIDs currently managed by systemd or launchd gateway services.

    Used to avoid killing freshly-restarted service processes when sweeping
    for stale manual gateway processes after a service restart.  Relies on the
    service manager having committed the new PID before the restart command
    returns (true for both systemd and launchd in practice).
    """
    pids: set = set()

    # --- systemd (Linux): user and system scopes ---
    if supports_systemd_services():
        for scope_args in [["systemctl", "--user"], ["systemctl"]]:
            try:
                result = subprocess.run(
                    scope_args + ["list-units", "hermes-gateway*",
                                  "--plain", "--no-legend", "--no-pager"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.strip().splitlines():
                    parts = line.split()
                    if not parts or not parts[0].endswith(".service"):
                        continue
                    svc = parts[0]
                    try:
                        show = subprocess.run(
                            scope_args + ["show", svc,
                                          "--property=MainPID", "--value"],
                            capture_output=True, text=True, timeout=5,
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
        try:
            label = get_launchd_label()
            result = subprocess.run(
                ["launchctl", "list", label],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                # Output: "PID\tStatus\tLabel" header, then one data line
                for line in result.stdout.strip().splitlines():
                    parts = line.split()
                    if len(parts) >= 3 and parts[2] == label:
                        try:
                            pid = int(parts[0])
                            if pid > 0:
                                pids.add(pid)
                        except ValueError:
                            pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return pids


def _get_parent_pid(pid: int) -> int | None:
    """Return the parent PID for ``pid``, or ``None`` when unavailable."""
    if pid <= 1:
        return None
    try:
        result = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    try:
        parent_pid = int(raw.splitlines()[-1].strip())
    except ValueError:
        return None
    return parent_pid if parent_pid > 0 else None


def _is_pid_ancestor_of_current_process(target_pid: int) -> bool:
    """Return True when ``target_pid`` is this process or one of its ancestors."""
    if target_pid <= 0:
        return False

    pid = os.getpid()
    seen: set[int] = set()
    while pid and pid not in seen:
        if pid == target_pid:
            return True
        seen.add(pid)
        pid = _get_parent_pid(pid) or 0
    return False


def _request_gateway_self_restart(pid: int) -> bool:
    """Ask a running gateway ancestor to restart itself asynchronously."""
    if not hasattr(signal, "SIGUSR1"):
        return False
    if not _is_pid_ancestor_of_current_process(pid):
        return False
    try:
        os.kill(pid, signal.SIGUSR1)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def _graceful_restart_via_sigusr1(pid: int, drain_timeout: float) -> bool:
    """Send SIGUSR1 to a gateway PID and wait for it to exit gracefully.

    SIGUSR1 is wired in gateway/run.py to ``request_restart(via_service=True)``
    which drains in-flight agent runs (up to ``agent.restart_drain_timeout``
    seconds), then exits with code 75.  Both systemd (``Restart=always``
    + ``RestartForceExitStatus=75``) and launchd (``KeepAlive.SuccessfulExit
    = false``) relaunch the process after the graceful exit.

    This is the drain-aware alternative to ``systemctl restart`` / ``SIGTERM``,
    which SIGKILL in-flight agents after a short timeout.

    Args:
        pid: Gateway process PID (systemd MainPID, launchd PID, or bare
            process PID).
        drain_timeout: Seconds to wait for the process to exit after sending
            SIGUSR1.  Should be slightly larger than the gateway's
            ``agent.restart_drain_timeout`` to allow the drain loop to
            finish cleanly.

    Returns:
        True if the PID was signalled and exited within the timeout.
        False if SIGUSR1 couldn't be sent or the process didn't exit in
        time (caller should fall back to a harder restart path).
    """
    if not hasattr(signal, "SIGUSR1"):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, signal.SIGUSR1)
    except ProcessLookupError:
        # Already gone — nothing to drain.
        return True
    except (PermissionError, OSError):
        return False

    import time as _time

    deadline = _time.monotonic() + max(drain_timeout, 1.0)
    while _time.monotonic() < deadline:
        try:
            os.kill(pid, 0)  # signal 0 — probe liveness
        except ProcessLookupError:
            return True
        except PermissionError:
            # Process still exists but we can't signal it.  Treat as alive
            # so the caller falls back.
            pass
        _time.sleep(0.5)
    # Drain didn't finish in time.
    return False


def _get_ancestor_pids() -> set[int]:
    """Return the set of PIDs in the current process's ancestor chain.

    Walks from the current PID up to PID 1 (init) so that process-table scans
    never match the calling CLI process or any of its parents.  This prevents
    ``hermes gateway status`` from falsely counting the ``hermes`` CLI that
    invoked it as a running gateway instance (see #13242).
    """
    ancestors: set[int] = set()
    pid = os.getpid()
    # Cap iterations to avoid infinite loops on exotic platforms.
    for _ in range(64):
        ancestors.add(pid)
        parent = _get_parent_pid(pid)
        if parent is None or parent <= 0 or parent in ancestors:
            break
        pid = parent
    return ancestors


def _append_unique_pid(pids: list[int], pid: int | None, exclude_pids: set[int]) -> None:
    if pid is None or pid <= 0:
        return
    if pid == os.getpid() or pid in exclude_pids or pid in pids:
        return
    pids.append(pid)


def _scan_gateway_pids(exclude_pids: set[int], all_profiles: bool = False) -> list[int]:
    """Best-effort process-table scan for gateway PIDs.

    This supplements the profile-scoped PID file so status views can still spot
    a live gateway when the PID file is stale/missing, and ``--all`` sweeps can
    discover gateways outside the current profile.
    """
    # Exclude the entire ancestor chain so the CLI process that invoked this
    # scan (e.g. ``hermes gateway status``) is never mistaken for a running
    # gateway.  See #13242.
    exclude_pids = exclude_pids | _get_ancestor_pids()
    pids: list[int] = []
    patterns = [
        "hermes_cli.main gateway",
        "hermes_cli.main --profile",
        "hermes_cli.main -p",
        "hermes_cli/main.py gateway",
        "hermes_cli/main.py --profile",
        "hermes_cli/main.py -p",
        "hermes gateway",
        "gateway/run.py",
    ]
    current_home = str(get_hermes_home().resolve())
    current_profile_arg = _profile_arg(current_home)
    current_profile_name = current_profile_arg.split()[-1] if current_profile_arg else ""

    def _matches_current_profile(command: str) -> bool:
        if current_profile_name:
            return (
                f"--profile {current_profile_name}" in command
                or f"-p {current_profile_name}" in command
                or f"HERMES_HOME={current_home}" in command
            )

        if "--profile " in command or " -p " in command:
            return False
        if "HERMES_HOME=" in command and f"HERMES_HOME={current_home}" not in command:
            return False
        return True

    try:
        if is_windows():
            result = subprocess.run(
                ["wmic", "process", "get", "ProcessId,CommandLine", "/FORMAT:LIST"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=10,
            )
            if result.returncode != 0 or result.stdout is None:
                return []
            current_cmd = ""
            for line in result.stdout.split("\n"):
                line = line.strip()
                if line.startswith("CommandLine="):
                    current_cmd = line[len("CommandLine="):]
                elif line.startswith("ProcessId="):
                    pid_str = line[len("ProcessId="):]
                    if any(p in current_cmd for p in patterns) and (
                        all_profiles or _matches_current_profile(current_cmd)
                    ):
                        try:
                            _append_unique_pid(pids, int(pid_str), exclude_pids)
                        except ValueError:
                            pass
                    current_cmd = ""
        else:
            result = subprocess.run(
                ["ps", "-A", "eww", "-o", "pid=,command="],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return []
            for line in result.stdout.split("\n"):
                stripped = line.strip()
                if not stripped or "grep" in stripped:
                    continue

                pid = None
                command = ""

                parts = stripped.split(None, 1)
                if len(parts) == 2:
                    try:
                        pid = int(parts[0])
                        command = parts[1]
                    except ValueError:
                        pid = None

                if pid is None:
                    aux_parts = stripped.split()
                    if len(aux_parts) > 10 and aux_parts[1].isdigit():
                        pid = int(aux_parts[1])
                        command = " ".join(aux_parts[10:])

                if pid is None:
                    continue
                if any(pattern in command for pattern in patterns) and (
                    all_profiles or _matches_current_profile(command)
                ):
                    _append_unique_pid(pids, pid, exclude_pids)
    except (OSError, subprocess.TimeoutExpired):
        return []

    return pids


def find_gateway_pids(exclude_pids: set | None = None, all_profiles: bool = False) -> list:
    """Find PIDs of running gateway processes.

    Args:
        exclude_pids: PIDs to exclude from the result (e.g. service-managed
            PIDs that should not be killed during a stale-process sweep).
        all_profiles: When ``True``, return gateway PIDs across **all**
            profiles (the pre-7923 global behaviour).  ``hermes update``
            needs this because a code update affects every profile.
            When ``False`` (default), only PIDs belonging to the current
            Hermes profile are returned.
    """
    _exclude = set(exclude_pids or set())
    pids: list[int] = []
    if not all_profiles:
        try:
            from gateway.status import get_running_pid

            _append_unique_pid(pids, get_running_pid(), _exclude)
        except Exception:
            pass
    for pid in _get_service_pids():
        _append_unique_pid(pids, pid, _exclude)
    for pid in _scan_gateway_pids(_exclude, all_profiles=all_profiles):
        _append_unique_pid(pids, pid, _exclude)
    return pids


def find_profile_gateway_processes(
    exclude_pids: set | None = None,
) -> list[ProfileGatewayProcess]:
    """Return running gateway PIDs mapped to Hermes profiles via PID files."""
    _exclude = set(exclude_pids or set())
    processes: list[ProfileGatewayProcess] = []
    try:
        from gateway.status import get_running_pid
        from hermes_cli.profiles import list_profiles
    except Exception:
        return processes

    seen: set[int] = set()
    for profile in list_profiles():
        try:
            pid = get_running_pid(profile.path / "gateway.pid", cleanup_stale=False)
        except Exception:
            continue
        if pid is None or pid <= 0 or pid in _exclude or pid in seen:
            continue
        seen.add(pid)
        processes.append(ProfileGatewayProcess(profile=profile.name, path=profile.path, pid=pid))
    return processes


def _gateway_run_args_for_profile(profile: str) -> list[str]:
    args = [get_python_path(), "-m", "hermes_cli.main"]
    if profile != "default":
        args.extend(["--profile", profile])
    args.extend(["gateway", "run", "--replace"])
    return args


def launch_detached_profile_gateway_restart(profile: str, old_pid: int) -> bool:
    """Relaunch a manually-run profile gateway after its current PID exits."""
    if old_pid <= 0:
        return False

    watcher = textwrap.dedent(
        """
        import os
        import subprocess
        import sys
        import time

        pid = int(sys.argv[1])
        cmd = sys.argv[2:]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            except PermissionError:
                pass
            time.sleep(0.2)
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        """
    ).strip()

    try:
        subprocess.Popen(
            [sys.executable, "-c", watcher, str(old_pid), *_gateway_run_args_for_profile(profile)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False
    return True


def _probe_systemd_service_running(system: bool = False) -> tuple[bool, bool]:
    selected_system = _select_systemd_scope(system)
    unit_exists = get_systemd_unit_path(system=selected_system).exists()
    if not unit_exists:
        return selected_system, False
    try:
        result = _run_systemctl(
            ["is-active", get_service_name()],
            system=selected_system,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (RuntimeError, subprocess.TimeoutExpired):
        return selected_system, False
    return selected_system, result.stdout.strip() == "active"


def _read_systemd_unit_properties(
    system: bool = False,
    properties: tuple[str, ...] = (
        "ActiveState",
        "SubState",
        "Result",
        "ExecMainStatus",
        "MainPID",
    ),
) -> dict[str, str]:
    """Return selected ``systemctl show`` properties for the gateway unit."""
    selected_system = _select_systemd_scope(system)
    try:
        result = _run_systemctl(
            [
                "show",
                get_service_name(),
                "--no-pager",
                "--property",
                ",".join(properties),
            ],
            system=selected_system,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (RuntimeError, subprocess.TimeoutExpired, OSError):
        return {}

    if result.returncode != 0:
        return {}

    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key] = value.strip()
    return parsed


def _systemd_main_pid_from_props(props: dict[str, str]) -> int | None:
    try:
        pid = int(props.get("MainPID", "0") or "0")
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _systemd_main_pid(system: bool = False) -> int | None:
    return _systemd_main_pid_from_props(_read_systemd_unit_properties(system=system))


def _read_gateway_runtime_status() -> dict | None:
    try:
        from gateway.status import read_runtime_status

        state = read_runtime_status()
    except Exception:
        return None
    return state if isinstance(state, dict) else None


def _gateway_runtime_status_for_pid(pid: int | None) -> dict | None:
    if not pid:
        return None
    state = _read_gateway_runtime_status()
    if not state:
        return None
    try:
        state_pid = int(state.get("pid", 0) or 0)
    except (TypeError, ValueError):
        return None
    return state if state_pid == pid else None


def _wait_for_systemd_service_restart(
    *,
    system: bool = False,
    previous_pid: int | None = None,
    timeout: float = 60.0,
) -> bool:
    """Wait for the gateway service to become active after a restart handoff."""
    import time

    svc = get_service_name()
    scope_label = _service_scope_label(system).capitalize()
    deadline = time.time() + timeout
    printed_runtime_wait = False

    while time.time() < deadline:
        props = _read_systemd_unit_properties(system=system)
        active_state = props.get("ActiveState", "")
        sub_state = props.get("SubState", "")
        new_pid = None
        try:
            from gateway.status import get_running_pid

            new_pid = get_running_pid()
        except Exception:
            new_pid = None
        if not new_pid:
            new_pid = _systemd_main_pid_from_props(props)

        if active_state == "active":
            if new_pid and (previous_pid is None or new_pid != previous_pid):
                runtime_state = _gateway_runtime_status_for_pid(new_pid)
                gateway_state = (runtime_state or {}).get("gateway_state")
                if gateway_state == "running":
                    print(f"✓ {scope_label} service restarted (PID {new_pid})")
                    return True
                if gateway_state == "startup_failed":
                    reason = (runtime_state or {}).get("exit_reason") or "startup failed"
                    print(f"⚠ {scope_label} service process restarted (PID {new_pid}), but gateway startup failed: {reason}")
                    return False
                if not printed_runtime_wait:
                    print(f"⏳ {scope_label} service process started (PID {new_pid}); waiting for gateway runtime...")
                    printed_runtime_wait = True

        if active_state == "activating" and sub_state == "auto-restart":
            time.sleep(1)
            continue

        if _systemd_unit_is_start_limited(props):
            _print_systemd_start_limit_wait(system=system)
            return False

        time.sleep(2)

    print(
        f"⚠ {scope_label} service did not become active within {int(timeout)}s.\n"
        f"  Check status: {'sudo ' if system else ''}hermes gateway status\n"
        f"  Check logs:   journalctl {'--user ' if not system else ''}-u {svc} -l --since '2 min ago'"
    )
    return False


def _systemd_unit_is_start_limited(props: dict[str, str]) -> bool:
    result = props.get("Result", "").lower()
    sub_state = props.get("SubState", "").lower()
    return result == "start-limit-hit" or sub_state == "start-limit-hit"


def _systemd_error_indicates_start_limit(exc: subprocess.CalledProcessError) -> bool:
    parts: list[str] = []
    for attr in ("stderr", "stdout", "output"):
        value = getattr(exc, attr, None)
        if not value:
            continue
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        parts.append(str(value))
    text = "\n".join(parts).lower()
    return (
        "start-limit-hit" in text
        or "start request repeated too quickly" in text
        or "start-limit" in text
    )


def _systemd_service_is_start_limited(system: bool = False) -> bool:
    return _systemd_unit_is_start_limited(_read_systemd_unit_properties(system=system))


def _print_systemd_start_limit_wait(system: bool = False) -> None:
    svc = get_service_name()
    scope_label = _service_scope_label(system).capitalize()
    scope_flag = " --system" if system else ""
    systemctl_prefix = "systemctl " if system else "systemctl --user "
    journal_prefix = "journalctl " if system else "journalctl --user "
    print(f"⏳ {scope_label} service is temporarily rate-limited by systemd.")
    print("  systemd is refusing another immediate start after repeated exits.")
    print(f"  Wait for the start-limit window to expire, then run: {'sudo ' if system else ''}hermes gateway restart{scope_flag}")
    print(f"  Or clear the failed state manually: {systemctl_prefix}reset-failed {svc}")
    print(f"  Check logs: {journal_prefix}-u {svc} -l --since '5 min ago'")


def _recover_pending_systemd_restart(system: bool = False, previous_pid: int | None = None) -> bool:
    """Recover a planned service restart that is stuck in systemd state."""
    props = _read_systemd_unit_properties(system=system)
    if not props:
        return False

    try:
        from gateway.status import read_runtime_status
    except Exception:
        return False

    runtime_state = read_runtime_status() or {}
    if not runtime_state.get("restart_requested"):
        return False

    active_state = props.get("ActiveState", "")
    sub_state = props.get("SubState", "")
    exec_main_status = props.get("ExecMainStatus", "")
    result = props.get("Result", "")

    if active_state == "activating" and sub_state == "auto-restart":
        print("⏳ Service restart already pending — waiting for systemd relaunch...")
        return _wait_for_systemd_service_restart(
            system=system,
            previous_pid=previous_pid,
        )

    if active_state == "failed" and (
        exec_main_status == str(GATEWAY_SERVICE_RESTART_EXIT_CODE)
        or result == "exit-code"
    ):
        svc = get_service_name()
        scope_label = _service_scope_label(system).capitalize()
        print(f"↻ Clearing failed state for pending {scope_label.lower()} service restart...")
        _run_systemctl(
            ["reset-failed", svc],
            system=system,
            check=False,
            timeout=30,
        )
        _run_systemctl(
            ["start", svc],
            system=system,
            check=False,
            timeout=90,
        )
        return _wait_for_systemd_service_restart(
            system=system,
            previous_pid=previous_pid,
        )

    return False


def _probe_launchd_service_running() -> bool:
    if not get_launchd_plist_path().exists():
        return False
    try:
        result = subprocess.run(
            ["launchctl", "list", get_launchd_label()],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def get_gateway_runtime_snapshot(system: bool = False) -> GatewayRuntimeSnapshot:
    """Return a unified view of gateway liveness for the current profile."""
    gateway_pids = tuple(find_gateway_pids())
    if is_termux():
        return GatewayRuntimeSnapshot(
            manager="Termux / manual process",
            gateway_pids=gateway_pids,
        )

    from hermes_constants import is_container

    if is_linux() and is_container():
        return GatewayRuntimeSnapshot(
            manager="docker (foreground)",
            gateway_pids=gateway_pids,
        )

    if supports_systemd_services():
        selected_system, service_running = _probe_systemd_service_running(system=system)
        scope_label = _service_scope_label(selected_system)
        return GatewayRuntimeSnapshot(
            manager=f"systemd ({scope_label})",
            service_installed=get_systemd_unit_path(system=selected_system).exists(),
            service_running=service_running,
            gateway_pids=gateway_pids,
            service_scope=scope_label,
        )

    if is_macos():
        return GatewayRuntimeSnapshot(
            manager="launchd",
            service_installed=get_launchd_plist_path().exists(),
            service_running=_probe_launchd_service_running(),
            gateway_pids=gateway_pids,
            service_scope="launchd",
        )

    return GatewayRuntimeSnapshot(
        manager="manual process",
        gateway_pids=gateway_pids,
    )


def _format_gateway_pids(pids: tuple[int, ...] | list[int], *, limit: int | None = 3) -> str:
    rendered = [str(pid) for pid in pids[:limit] if pid > 0] if limit is not None else [str(pid) for pid in pids if pid > 0]
    if limit is not None and len(pids) > limit:
        rendered.append("...")
    return ", ".join(rendered)


def _print_gateway_process_mismatch(snapshot: GatewayRuntimeSnapshot) -> None:
    if not snapshot.has_process_service_mismatch:
        return
    print()
    print("⚠ Gateway process is running for this profile, but the service is not active")
    print(f"  PID(s): {_format_gateway_pids(snapshot.gateway_pids, limit=None)}")
    print("  This is usually a manual foreground/tmux/nohup run, so `hermes gateway`")
    print("  can refuse to start another copy until this process stops.")


def _print_other_profiles_gateway_status() -> None:
    """Print a summary of gateway status across all profiles.

    Shown at the bottom of ``hermes gateway status`` output so users with
    multiple profiles can tell at a glance which gateways are running and
    avoid confusing another profile's process with the current one.
    """
    try:
        from hermes_cli.profiles import get_active_profile_name

        current = get_active_profile_name()
        other_processes = [
            p for p in find_profile_gateway_processes()
            if p.profile != current
        ]
        if not other_processes:
            return

        print()
        print("Other profiles:")
        for proc in other_processes:
            print(f"  ✓ {proc.profile:<16s} — PID {proc.pid}")
    except Exception:
        pass


def kill_gateway_processes(force: bool = False, exclude_pids: set | None = None,
                           all_profiles: bool = False) -> int:
    """Kill any running gateway processes. Returns count killed.

    Args:
        force: Use the platform's force-kill mechanism instead of graceful terminate.
        exclude_pids: PIDs to skip (e.g. service-managed PIDs that were just
            restarted and should not be killed).
        all_profiles: When ``True``, kill across all profiles.  Passed
            through to :func:`find_gateway_pids`.
    """
    pids = find_gateway_pids(exclude_pids=exclude_pids, all_profiles=all_profiles)
    killed = 0
    
    for pid in pids:
        try:
            terminate_pid(pid, force=force)
            killed += 1
        except ProcessLookupError:
            # Process already gone
            pass
        except PermissionError:
            print(f"⚠ Permission denied to kill PID {pid}")
    
        except OSError as exc:
            print(f"Failed to kill PID {pid}: {exc}")
    return killed


def stop_profile_gateway() -> bool:
    """Stop only the gateway for the current profile (HERMES_HOME-scoped).

    Uses the PID file written by start_gateway(), so it only kills the
    gateway belonging to this profile — not gateways from other profiles.
    Returns True if a process was stopped, False if none was found.
    """
    try:
        from gateway.status import get_running_pid, remove_pid_file
    except ImportError:
        return False

    pid = get_running_pid()
    if pid is None:
        return False

    try:
        from gateway.status import write_planned_stop_marker
        write_planned_stop_marker(pid)
    except Exception:
        pass

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # Already gone
    except PermissionError:
        print(f"⚠ Permission denied to kill PID {pid}")
        return False

    # Wait briefly for it to exit
    import time as _time
    for _ in range(20):
        try:
            os.kill(pid, 0)
            _time.sleep(0.5)
        except (ProcessLookupError, PermissionError):
            break

    if get_running_pid() is None:
        remove_pid_file()
    return True


def is_linux() -> bool:
    return sys.platform.startswith('linux')


from hermes_constants import is_container, is_termux, is_wsl


def _wsl_systemd_operational() -> bool:
    """Check if systemd is actually running as PID 1 on WSL.

    WSL2 with ``systemd=true`` in wsl.conf has working systemd.
    WSL2 without it (or WSL1) does not — systemctl commands fail.
    """
    return _systemd_operational(system=True)


def _systemd_operational(system: bool = False) -> bool:
    """Return True when the requested systemd scope is usable."""
    try:
        result = _run_systemctl(
            ["is-system-running"],
            system=system,
            capture_output=True,
            text=True,
            timeout=5,
        )
        # "running", "degraded", "starting" all mean systemd is PID 1
        status = result.stdout.strip().lower()
        return status in ("running", "degraded", "starting", "initializing")
    except (RuntimeError, subprocess.TimeoutExpired, OSError):
        return False


def _container_systemd_operational() -> bool:
    """Return True when a container exposes working user or system systemd."""
    if _systemd_operational(system=False):
        return True
    if _systemd_operational(system=True):
        return True
    return False


def supports_systemd_services() -> bool:
    if not is_linux() or is_termux():
        return False
    if shutil.which("systemctl") is None:
        return False
    if is_wsl():
        return _wsl_systemd_operational()
    if is_container():
        return _container_systemd_operational()
    return True


def is_macos() -> bool:
    return sys.platform == 'darwin'

def is_windows() -> bool:
    return sys.platform == 'win32'


# =============================================================================
# Service Configuration
# =============================================================================

_SERVICE_BASE = "hermes-gateway"
SERVICE_DESCRIPTION = "Hermes Agent Gateway - Messaging Platform Integration"


def _profile_suffix() -> str:
    """Derive a service-name suffix from the current HERMES_HOME.

    Returns ``""`` for the default root, the profile name for
    ``<root>/profiles/<name>``, or a short hash for any other path.
    Works correctly in Docker (HERMES_HOME=/opt/data) and standard deployments.
    """
    import hashlib
    import re
    from hermes_constants import get_default_hermes_root
    home = get_hermes_home().resolve()
    default = get_default_hermes_root().resolve()
    if home == default:
        return ""
    # Detect <root>/profiles/<name> pattern → use the profile name
    profiles_root = (default / "profiles").resolve()
    try:
        rel = home.relative_to(profiles_root)
        parts = rel.parts
        if len(parts) == 1 and re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", parts[0]):
            return parts[0]
    except ValueError:
        pass
    # Fallback: short hash for arbitrary HERMES_HOME paths
    return hashlib.sha256(str(home).encode()).hexdigest()[:8]


def _profile_arg(hermes_home: str | None = None) -> str:
    """Return ``--profile <name>`` only when HERMES_HOME is a named profile.

    For ``~/.hermes/profiles/<name>``, returns ``"--profile <name>"``.
    For the default profile or hash-based custom paths, returns the empty string.

    Args:
        hermes_home: Optional explicit HERMES_HOME path. Defaults to the current
            ``get_hermes_home()`` value. Should be passed when generating a
            service definition for a different user (e.g. system service).
    """
    import re
    from hermes_constants import get_default_hermes_root
    home = Path(hermes_home or str(get_hermes_home())).resolve()
    default = get_default_hermes_root().resolve()
    if home == default:
        return ""
    profiles_root = (default / "profiles").resolve()
    try:
        rel = home.relative_to(profiles_root)
        parts = rel.parts
        if len(parts) == 1 and re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", parts[0]):
            return f"--profile {parts[0]}"
    except ValueError:
        pass
    return ""


def get_service_name() -> str:
    """Derive a systemd service name scoped to this HERMES_HOME.

    Default ``~/.hermes`` returns ``hermes-gateway`` (backward compatible).
    Profile ``~/.hermes/profiles/coder`` returns ``hermes-gateway-coder``.
    Any other HERMES_HOME appends a short hash for uniqueness.
    """
    suffix = _profile_suffix()
    if not suffix:
        return _SERVICE_BASE
    return f"{_SERVICE_BASE}-{suffix}"



def get_systemd_unit_path(system: bool = False) -> Path:
    name = get_service_name()
    if system:
        return Path("/etc/systemd/system") / f"{name}.service"
    return Path.home() / ".config" / "systemd" / "user" / f"{name}.service"


class UserSystemdUnavailableError(RuntimeError):
    """Raised when ``systemctl --user`` cannot reach the user D-Bus session.

    Typically hit on fresh RHEL/Debian SSH sessions where linger is disabled
    and no user@.service is running, so ``/run/user/$UID/bus`` never exists.
    Carries a user-facing remediation message in ``args[0]``.
    """


class SystemScopeRequiresRootError(RuntimeError):
    """Raised when a system-scope gateway operation is attempted as non-root.

    System-scope units live in ``/etc/systemd/system/`` and require root for
    install / uninstall / start / stop / restart via ``systemctl``. The
    previous behavior was ``sys.exit(1)`` which blew past the wizard's
    ``except Exception`` guards and dumped the user at a bare shell prompt
    with no guidance. Raising a typed exception lets callers that can
    recover (the setup wizard) print actionable remediation instead, while
    ``gateway_command`` still exits 1 with the same message for the direct
    CLI path.

    ``args[0]`` carries the user-facing message, ``args[1]`` the action name.
    ``str(e)`` returns only the message (not the tuple repr) so format
    strings like ``f"Failed: {e}"`` render cleanly.
    """

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


def _user_dbus_socket_path() -> Path:
    """Return the expected per-user D-Bus socket path (regardless of existence)."""
    xdg = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(xdg) / "bus"


def _user_systemd_private_socket_path() -> Path:
    """Return the per-user systemd private socket path (regardless of existence)."""
    xdg = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(xdg) / "systemd" / "private"


def _user_systemd_socket_ready() -> bool:
    """Return True when user-scope systemd has a reachable control socket.

    Some distros expose only the per-user systemd private socket even when the
    D-Bus session bus socket is absent. ``systemctl --user`` can still work in
    that configuration, so preflight checks must treat either socket as valid.
    """
    return _user_dbus_socket_path().exists() or _user_systemd_private_socket_path().exists()


def _ensure_user_systemd_env() -> None:
    """Ensure DBUS_SESSION_BUS_ADDRESS and XDG_RUNTIME_DIR are set for systemctl --user.

    On headless servers (SSH sessions), these env vars may be missing even when
    the user's systemd instance is running (via linger).  Without them,
    ``systemctl --user`` fails with "Failed to connect to bus: No medium found".
    We detect the standard socket path and set the vars so all subsequent
    subprocess calls inherit them.
    """
    uid = os.getuid()
    if "XDG_RUNTIME_DIR" not in os.environ:
        runtime_dir = f"/run/user/{uid}"
        if Path(runtime_dir).exists():
            os.environ["XDG_RUNTIME_DIR"] = runtime_dir

    if "DBUS_SESSION_BUS_ADDRESS" not in os.environ:
        xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{uid}")
        bus_path = Path(xdg_runtime) / "bus"
        if bus_path.exists():
            os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus_path}"


def _wait_for_user_dbus_socket(timeout: float = 3.0) -> bool:
    """Poll for the user systemd runtime socket(s), up to ``timeout`` seconds.

    Linger-enabled user@.service can take a second or two to spawn its control
    socket(s) after ``loginctl enable-linger`` runs. Returns True once either
    the user D-Bus socket or the per-user systemd private socket exists.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _user_systemd_socket_ready():
            _ensure_user_systemd_env()
            return True
        time.sleep(0.2)
    return _user_systemd_socket_ready()


def _preflight_user_systemd(*, auto_enable_linger: bool = True) -> None:
    """Ensure ``systemctl --user`` will reach the user-scope systemd instance.

    No-op when the user D-Bus socket or per-user systemd private socket is
    already there (the common case on desktops and linger-enabled servers). On
    fresh SSH sessions where both are missing:

    * If linger is already enabled, wait briefly for user@.service to spawn
      the socket.
    * If linger is disabled and ``auto_enable_linger`` is True, try
      ``loginctl enable-linger $USER`` (works as non-root when polkit permits
      it, otherwise needs sudo).
    * If the socket is still missing afterwards, raise
      :class:`UserSystemdUnavailableError` with a precise remediation message.

    Callers should treat the exception as a terminal condition for user-scope
    systemd operations and surface the message to the user.
    """
    _ensure_user_systemd_env()
    if _user_systemd_socket_ready():
        return

    import getpass

    username = getpass.getuser()
    linger_enabled, linger_detail = get_systemd_linger_status()

    if linger_enabled is True:
        if _wait_for_user_dbus_socket(timeout=3.0):
            return
        # Linger is on but socket still missing — unusual; fall through to error.
        _raise_user_systemd_unavailable(
            username,
            reason="User systemd control sockets are missing even though linger is enabled.",
            fix_hint=(
                f"  systemctl start user@{os.getuid()}.service\n"
                "  (may require sudo; try again after the command succeeds)"
            ),
        )

    if auto_enable_linger and shutil.which("loginctl"):
        try:
            result = subprocess.run(
                ["loginctl", "enable-linger", username],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except Exception as exc:
            _raise_user_systemd_unavailable(
                username,
                reason=f"loginctl enable-linger failed ({exc}).",
                fix_hint=f"  sudo loginctl enable-linger {username}",
            )
        else:
            if result.returncode == 0:
                if _wait_for_user_dbus_socket(timeout=5.0):
                    print(f"✓ Enabled linger for {username} — user D-Bus now available")
                    return
                # enable-linger succeeded but the socket never appeared.
                _raise_user_systemd_unavailable(
                    username,
                    reason="Linger was enabled, but the user D-Bus socket did not appear.",
                    fix_hint=(
                        "  Log out and log back in, then re-run the command.\n"
                        f"  Or reboot and run: systemctl --user start {get_service_name()}"
                    ),
                )
            detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
            _raise_user_systemd_unavailable(
                username,
                reason=f"loginctl enable-linger was denied: {detail}",
                fix_hint=f"  sudo loginctl enable-linger {username}",
            )

    _raise_user_systemd_unavailable(
        username,
        reason=(
            "User D-Bus session is not available "
            f"({linger_detail or 'linger disabled'})."
        ),
        fix_hint=f"  sudo loginctl enable-linger {username}",
    )


def _raise_user_systemd_unavailable(username: str, *, reason: str, fix_hint: str) -> None:
    """Build a user-facing error message and raise UserSystemdUnavailableError."""
    msg = (
        f"{reason}\n"
        "  systemctl --user cannot reach the user D-Bus session in this shell.\n"
        "\n"
        "  To fix:\n"
        f"{fix_hint}\n"
        "\n"
        "  Alternative: run the gateway in the foreground (stays up until\n"
        "  you exit / close the terminal):\n"
        "    hermes gateway run"
    )
    raise UserSystemdUnavailableError(msg)


def _systemctl_cmd(system: bool = False) -> list[str]:
    if not system:
        _ensure_user_systemd_env()
    return ["systemctl"] if system else ["systemctl", "--user"]


def _journalctl_cmd(system: bool = False) -> list[str]:
    return ["journalctl"] if system else ["journalctl", "--user"]


def _run_systemctl(args: list[str], *, system: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """Run a systemctl command, raising RuntimeError if systemctl is missing.

    Defense-in-depth: callers are gated by ``supports_systemd_services()``,
    but this ensures any future caller that bypasses the gate still gets a
    clear error instead of a raw ``FileNotFoundError`` traceback.
    """
    try:
        return subprocess.run(_systemctl_cmd(system) + args, **kwargs)
    except FileNotFoundError:
        raise RuntimeError(
            "systemctl is not available on this system"
        ) from None


def _service_scope_label(system: bool = False) -> str:
    return "system" if system else "user"


def get_installed_systemd_scopes() -> list[str]:
    scopes = []
    seen_paths: set[Path] = set()
    for system, label in ((False, "user"), (True, "system")):
        unit_path = get_systemd_unit_path(system=system)
        if unit_path in seen_paths:
            continue
        if unit_path.exists():
            scopes.append(label)
            seen_paths.add(unit_path)
    return scopes


def has_conflicting_systemd_units() -> bool:
    return len(get_installed_systemd_scopes()) > 1


# Legacy service names from older Hermes installs that predate the
# hermes-gateway rename. Kept as an explicit allowlist (NOT a glob) so
# profile units (hermes-gateway-*.service) and unrelated third-party
# "hermes" units are never matched.
_LEGACY_SERVICE_NAMES: tuple[str, ...] = ("hermes.service",)

# ExecStart content markers that identify a unit as running our gateway.
# A legacy unit is only flagged when its file contains one of these.
_LEGACY_UNIT_EXECSTART_MARKERS: tuple[str, ...] = (
    "hermes_cli.main gateway",
    "hermes_cli/main.py gateway",
    "gateway/run.py",
    " hermes gateway ",
    "/hermes gateway ",
)


def _legacy_unit_search_paths() -> list[tuple[bool, Path]]:
    """Return ``[(is_system, base_dir), ...]`` — directories to scan for legacy units.

    Factored out so tests can monkeypatch the search roots without touching
    real filesystem paths.
    """
    return [
        (False, Path.home() / ".config" / "systemd" / "user"),
        (True, Path("/etc/systemd/system")),
    ]


def _find_legacy_hermes_units() -> list[tuple[str, Path, bool]]:
    """Return ``[(unit_name, unit_path, is_system)]`` for legacy Hermes gateway units.

    Detects unit files installed by older Hermes versions that used a
    different service name (e.g. ``hermes.service`` before the rename to
    ``hermes-gateway.service``). When both a legacy unit and the current
    ``hermes-gateway.service`` are active, they fight over the same bot
    token — the PR #5646 signal-recovery change turns this into a 30-second
    SIGTERM flap loop.

    Safety guards:

    * Explicit allowlist of legacy names (no globbing). Profile units such
      as ``hermes-gateway-coder.service`` and unrelated third-party
      ``hermes-*`` services are never matched.
    * ExecStart content check — only flag units that invoke our gateway
      entrypoint. A user-created ``hermes.service`` running an unrelated
      binary is left untouched.
    * Results are returned purely for caller inspection; this function
      never mutates or removes anything.
    """
    results: list[tuple[str, Path, bool]] = []
    for is_system, base in _legacy_unit_search_paths():
        for name in _LEGACY_SERVICE_NAMES:
            unit_path = base / name
            try:
                if not unit_path.exists():
                    continue
                text = unit_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, PermissionError):
                continue
            if not any(marker in text for marker in _LEGACY_UNIT_EXECSTART_MARKERS):
                # Not our gateway — leave alone
                continue
            results.append((name, unit_path, is_system))
    return results


def has_legacy_hermes_units() -> bool:
    """Return True when any legacy Hermes gateway unit files exist."""
    return bool(_find_legacy_hermes_units())


def print_legacy_unit_warning() -> None:
    """Warn about legacy Hermes gateway unit files if any are installed.

    Idempotent: prints nothing when no legacy units are detected. Safe to
    call from any status/install/setup path.
    """
    legacy = _find_legacy_hermes_units()
    if not legacy:
        return
    print_warning("Legacy Hermes gateway unit(s) detected from an older install:")
    for name, path, is_system in legacy:
        scope = "system" if is_system else "user"
        print_info(f"    {path}  ({scope} scope)")
    print_info("  These run alongside the current hermes-gateway service and")
    print_info("  cause SIGTERM flap loops — both try to use the same bot token.")
    print_info("  Remove them with:")
    print_info("    hermes gateway migrate-legacy")


def remove_legacy_hermes_units(
    interactive: bool = True,
    dry_run: bool = False,
) -> tuple[int, list[Path]]:
    """Stop, disable, and remove legacy Hermes gateway unit files.

    Iterates over whatever ``_find_legacy_hermes_units()`` returns — which is
    an explicit allowlist of legacy names (not a glob). Profile units and
    unrelated third-party services are never touched.

    Args:
        interactive: When True, prompt before removing. When False, remove
            without asking (used when another prompt has already confirmed,
            e.g. from the install flow).
        dry_run: When True, list what would be removed and return.

    Returns:
        ``(removed_count, remaining_paths)`` — remaining includes units we
        couldn't remove (typically system-scope when not running as root).
    """
    legacy = _find_legacy_hermes_units()
    if not legacy:
        print("No legacy Hermes gateway units found.")
        return 0, []

    user_units = [(n, p) for n, p, is_sys in legacy if not is_sys]
    system_units = [(n, p) for n, p, is_sys in legacy if is_sys]

    print()
    print("Legacy Hermes gateway unit(s) found:")
    for name, path, is_system in legacy:
        scope = "system" if is_system else "user"
        print(f"  {path}  ({scope} scope)")
    print()

    if dry_run:
        print("(dry-run — nothing removed)")
        return 0, [p for _, p, _ in legacy]

    if interactive and not prompt_yes_no("Remove these legacy units?", True):
        print("Skipped. Run again with: hermes gateway migrate-legacy")
        return 0, [p for _, p, _ in legacy]

    removed = 0
    remaining: list[Path] = []

    # User-scope removal
    for name, path in user_units:
        try:
            _run_systemctl(["stop", name], system=False, check=False, timeout=90)
            _run_systemctl(["disable", name], system=False, check=False, timeout=30)
            path.unlink(missing_ok=True)
            print(f"  ✓ Removed {path}")
            removed += 1
        except (OSError, RuntimeError) as e:
            print(f"  ⚠ Could not remove {path}: {e}")
            remaining.append(path)

    if user_units:
        try:
            _run_systemctl(["daemon-reload"], system=False, check=False, timeout=30)
        except RuntimeError:
            pass

    # System-scope removal (needs root)
    if system_units:
        if os.geteuid() != 0:
            print()
            print_warning("System-scope legacy units require root to remove.")
            print_info("  Re-run with: sudo hermes gateway migrate-legacy")
            for _, path in system_units:
                remaining.append(path)
        else:
            for name, path in system_units:
                try:
                    _run_systemctl(["stop", name], system=True, check=False, timeout=90)
                    _run_systemctl(["disable", name], system=True, check=False, timeout=30)
                    path.unlink(missing_ok=True)
                    print(f"  ✓ Removed {path}")
                    removed += 1
                except (OSError, RuntimeError) as e:
                    print(f"  ⚠ Could not remove {path}: {e}")
                    remaining.append(path)

            try:
                _run_systemctl(["daemon-reload"], system=True, check=False, timeout=30)
            except RuntimeError:
                pass

    print()
    if remaining:
        print_warning(f"{len(remaining)} legacy unit(s) still present — see messages above.")
    else:
        print_success(f"Removed {removed} legacy unit(s).")

    return removed, remaining


def print_systemd_scope_conflict_warning() -> None:
    scopes = get_installed_systemd_scopes()
    if len(scopes) < 2:
        return

    rendered_scopes = " + ".join(scopes)
    print_warning(f"Both user and system gateway services are installed ({rendered_scopes}).")
    print_info("  This is confusing and can make start/stop/status behavior ambiguous.")
    print_info("  Default gateway commands target the user service unless you pass --system.")
    print_info("  Keep one of these:")
    print_info("    hermes gateway uninstall")
    print_info("    sudo hermes gateway uninstall --system")


def _require_root_for_system_service(action: str) -> None:
    if os.geteuid() != 0:
        raise SystemScopeRequiresRootError(
            f"System gateway {action} requires root. Re-run with sudo.",
            action,
        )


def _system_service_identity(run_as_user: str | None = None) -> tuple[str, str, str]:
    import getpass
    import grp
    import pwd

    username = (run_as_user or os.getenv("SUDO_USER") or os.getenv("USER") or os.getenv("LOGNAME") or getpass.getuser()).strip()
    if not username:
        raise ValueError("Could not determine which user the gateway service should run as")
    if username == "root" and not run_as_user:
        raise ValueError("Refusing to install the gateway system service as root; pass --run-as-user root to override (e.g. in LXC containers)")
    if username == "root":
        print_warning("Installing gateway service to run as root.")
        print_info("  This is fine for LXC/container environments but not recommended on bare-metal hosts.")

    try:
        user_info = pwd.getpwnam(username)
    except KeyError as e:
        raise ValueError(f"Unknown user: {username}") from e

    group_name = grp.getgrgid(user_info.pw_gid).gr_name
    return username, group_name, user_info.pw_dir


def _read_systemd_user_from_unit(unit_path: Path) -> str | None:
    if not unit_path.exists():
        return None

    for line in unit_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("User="):
            value = line.split("=", 1)[1].strip()
            return value or None
    return None


def _default_system_service_user() -> str | None:
    for candidate in (os.getenv("SUDO_USER"), os.getenv("USER"), os.getenv("LOGNAME")):
        if candidate and candidate.strip() and candidate.strip() != "root":
            return candidate.strip()
    return None


def prompt_linux_gateway_install_scope() -> str | None:
    choice = prompt_choice(
        "  Choose how the gateway should run in the background:",
        [
            "User service (no sudo; best for laptops/dev boxes; may need linger after logout)",
            "System service (starts on boot; requires sudo; still runs as your user)",
            "Skip service install for now",
        ],
        default=0,
    )
    return {0: "user", 1: "system", 2: None}[choice]


def install_linux_gateway_from_setup(force: bool = False) -> tuple[str | None, bool]:
    scope = prompt_linux_gateway_install_scope()
    if scope is None:
        return None, False

    if scope == "system":
        run_as_user = _default_system_service_user()
        if os.geteuid() != 0:
            print_warning("  System service install requires sudo, so Hermes can't create it from this user session.")
            if run_as_user:
                print_info(f"  After setup, run: sudo hermes gateway install --system --run-as-user {run_as_user}")
            else:
                print_info("  After setup, run: sudo hermes gateway install --system --run-as-user <your-user>")
            print_info("  Then start it with: sudo hermes gateway start --system")
            return scope, False

        if not run_as_user:
            while True:
                run_as_user = prompt("  Run the system gateway service as which user?", default="")
                run_as_user = (run_as_user or "").strip()
                if run_as_user:
                    break
                print_error("  Enter a username.")

        systemd_install(force=force, system=True, run_as_user=run_as_user)
        return scope, True

    systemd_install(force=force, system=False)
    return scope, True


def get_systemd_linger_status() -> tuple[bool | None, str]:
    """Return systemd linger status for the current user.

    Returns:
        (True, "") when linger is enabled.
        (False, "") when linger is disabled.
        (None, detail) when the status could not be determined.
    """
    if is_termux():
        return None, "not supported in Termux"
    if not is_linux():
        return None, "not supported on this platform"

    if not shutil.which("loginctl"):
        return None, "loginctl not found"

    username = os.getenv("USER") or os.getenv("LOGNAME")
    if not username:
        try:
            import pwd
            username = pwd.getpwuid(os.getuid()).pw_name
        except Exception:
            return None, "could not determine current user"

    try:
        result = subprocess.run(
            ["loginctl", "show-user", username, "--property=Linger", "--value"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception as e:
        return None, str(e)

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
        return None, detail or "loginctl query failed"

    value = (result.stdout or "").strip().lower()
    if value in {"yes", "true", "1"}:
        return True, ""
    if value in {"no", "false", "0"}:
        return False, ""

    rendered = value or "<empty>"
    return None, f"unexpected loginctl output: {rendered}"


def print_systemd_linger_guidance() -> None:
    """Print the current linger status and the fix when it is disabled."""
    linger_enabled, linger_detail = get_systemd_linger_status()
    if linger_enabled is True:
        print("✓ Systemd linger is enabled (service survives logout)")
    elif linger_enabled is False:
        print("⚠ Systemd linger is disabled (gateway may stop when you log out)")
        print("  Run: sudo loginctl enable-linger $USER")
    else:
        print(f"⚠ Could not verify systemd linger ({linger_detail})")
        print("  If you want the gateway user service to survive logout, run:")
        print("  sudo loginctl enable-linger $USER")

def _launchd_user_home() -> Path:
    """Return the real macOS user home for launchd artifacts.

    Profile-mode Hermes often sets ``HOME`` to a profile-scoped directory, but
    launchd user agents still live under the actual account home.
    """
    import pwd

    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def get_launchd_plist_path() -> Path:
    """Return the launchd plist path, scoped per profile.

    Default ``~/.hermes`` → ``ai.hermes.gateway.plist`` (backward compatible).
    Profile ``~/.hermes/profiles/coder`` → ``ai.hermes.gateway-coder.plist``.
    """
    suffix = _profile_suffix()
    name = f"ai.hermes.gateway-{suffix}" if suffix else "ai.hermes.gateway"
    return _launchd_user_home() / "Library" / "LaunchAgents" / f"{name}.plist"

def _detect_venv_dir() -> Path | None:
    """Detect the active virtualenv directory.

    Checks ``sys.prefix`` first (works regardless of the directory name),
    then ``VIRTUAL_ENV`` env var (covers uv-managed environments where
    sys.prefix == sys.base_prefix), then falls back to probing common
    directory names under PROJECT_ROOT.
    Returns ``None`` when no virtualenv can be found.
    """
    # If we're running inside a virtualenv, sys.prefix points to it.
    if sys.prefix != sys.base_prefix:
        venv = Path(sys.prefix)
        if venv.is_dir():
            return venv

    # uv and some other tools set VIRTUAL_ENV without changing sys.prefix.
    # This catches `uv run` where sys.prefix == sys.base_prefix but the
    # environment IS a venv.  (#8620)
    _virtual_env = os.environ.get("VIRTUAL_ENV")
    if _virtual_env:
        venv = Path(_virtual_env)
        if venv.is_dir():
            return venv

    # Fallback: check common virtualenv directory names under the project root.
    for candidate in (".venv", "venv"):
        venv = PROJECT_ROOT / candidate
        if venv.is_dir():
            return venv

    return None


def get_python_path() -> str:
    venv = _detect_venv_dir()
    if venv is not None:
        if is_windows():
            venv_python = venv / "Scripts" / "python.exe"
        else:
            venv_python = venv / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
    return sys.executable


# =============================================================================
# Systemd (Linux)
# =============================================================================

def _build_user_local_paths(home: Path, path_entries: list[str]) -> list[str]:
    """Return user-local bin dirs that exist and aren't already in *path_entries*."""
    candidates = [
        str(home / ".local" / "bin"),       # uv, uvx, pip-installed CLIs
        str(home / ".cargo" / "bin"),        # Rust/cargo tools
        str(home / "go" / "bin"),            # Go tools
        str(home / ".npm-global" / "bin"),   # npm global packages
    ]
    return [p for p in candidates if p not in path_entries and Path(p).exists()]


def _build_wsl_interop_paths(path_entries: list[str]) -> list[str]:
    """Return WSL Windows interop PATH entries for generated systemd units.

    WSL shells normally inherit Windows PATH entries such as
    ``/mnt/c/WINDOWS/System32``. systemd user services do not, so gateway tools
    that call ``powershell.exe``/``cmd.exe`` work in a terminal but fail in the
    background service unless we persist the relevant entries at install time.
    """
    if not is_wsl():
        return []

    candidates: list[str] = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry.startswith("/mnt/"):
            candidates.append(entry)

    for executable in ("powershell.exe", "cmd.exe", "explorer.exe", "wsl.exe"):
        resolved = shutil.which(executable)
        if resolved:
            candidates.append(str(Path(resolved).parent))

    for entry in (
        "/mnt/c/WINDOWS/system32",
        "/mnt/c/WINDOWS",
        "/mnt/c/WINDOWS/System32/Wbem",
        "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/",
        "/mnt/c/WINDOWS/System32/OpenSSH/",
    ):
        if Path(entry).exists():
            candidates.append(entry)

    result: list[str] = []
    seen = set(path_entries)
    for entry in candidates:
        if entry and entry not in seen:
            seen.add(entry)
            result.append(entry)
    return result


def _remap_path_for_user(path: str, target_home_dir: str) -> str:
    """Remap *path* from the current user's home to *target_home_dir*.

    If *path* lives under ``Path.home()`` the corresponding prefix is swapped
    to *target_home_dir*; otherwise the path is returned unchanged.

      /root/.hermes/hermes-agent  -> /home/alice/.hermes/hermes-agent
      /opt/hermes                 -> /opt/hermes  (kept as-is)

    Note: this function intentionally does NOT resolve symlinks. A venv's
    ``bin/python`` is typically a symlink to the base interpreter (e.g. a
    uv-managed CPython at ``~/.local/share/uv/python/.../python3.11``);
    resolving that symlink swaps the unit's ``ExecStart`` to a bare Python
    that has none of the venv's site-packages, so the service crashes on
    the first ``import``. Keep the symlinked path so the venv activates
    its own environment. Lexical expansion only via ``expanduser``.
    """
    current_home = Path.home()
    p = Path(path).expanduser()
    try:
        relative = p.relative_to(current_home)
        return str(Path(target_home_dir) / relative)
    except ValueError:
        return str(p)


def _hermes_home_for_target_user(target_home_dir: str) -> str:
    """Remap the current HERMES_HOME to the equivalent under a target user's home.

    When installing a system service via sudo, get_hermes_home() resolves to
    root's home.  This translates it to the target user's equivalent path:
      /root/.hermes                    → /home/alice/.hermes
      /root/.hermes/profiles/coder     → /home/alice/.hermes/profiles/coder
      /opt/custom-hermes               → /opt/custom-hermes  (kept as-is)
    """
    current_hermes = get_hermes_home().resolve()
    current_default = (Path.home() / ".hermes").resolve()
    target_default = Path(target_home_dir) / ".hermes"

    # Default ~/.hermes → remap to target user's default
    if current_hermes == current_default:
        return str(target_default)

    # Profile or subdir of ~/.hermes → preserve the relative structure
    try:
        relative = current_hermes.relative_to(current_default)
        return str(target_default / relative)
    except ValueError:
        # Completely custom path (not under ~/.hermes) — keep as-is
        return str(current_hermes)


def generate_systemd_unit(system: bool = False, run_as_user: str | None = None) -> str:
    python_path = get_python_path()
    working_dir = str(PROJECT_ROOT)
    detected_venv = _detect_venv_dir()
    venv_dir = str(detected_venv) if detected_venv else str(PROJECT_ROOT / "venv")
    venv_bin = str(detected_venv / "bin") if detected_venv else str(PROJECT_ROOT / "venv" / "bin")
    node_bin = str(PROJECT_ROOT / "node_modules" / ".bin")

    path_entries = [venv_bin, node_bin]
    resolved_node = shutil.which("node")
    if resolved_node:
        resolved_node_dir = str(Path(resolved_node).resolve().parent)
        if resolved_node_dir not in path_entries:
            path_entries.append(resolved_node_dir)

    common_bin_paths = ["/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin"]
    # systemd's TimeoutStopSec must exceed the gateway's drain_timeout so
    # there's budget left for post-interrupt cleanup (tool subprocess kill,
    # adapter disconnect, session DB close) before systemd escalates to
    # SIGKILL on the cgroup — otherwise bash/sleep tool-call children left
    # by a force-interrupted agent get reaped by systemd instead of us
    # (#8202). 30s of headroom covers the worst case we've observed.
    _drain_timeout = int(_get_restart_drain_timeout() or 0)
    restart_timeout = max(60, _drain_timeout) + 30

    if system:
        username, group_name, home_dir = _system_service_identity(run_as_user)
        hermes_home = _hermes_home_for_target_user(home_dir)
        profile_arg = _profile_arg(hermes_home)
        # Remap all paths that may resolve under the calling user's home
        # (e.g. /root/) to the target user's home so the service can
        # actually access them.
        python_path = _remap_path_for_user(python_path, home_dir)
        working_dir = _remap_path_for_user(working_dir, home_dir)
        venv_dir = _remap_path_for_user(venv_dir, home_dir)
        venv_bin = _remap_path_for_user(venv_bin, home_dir)
        node_bin = _remap_path_for_user(node_bin, home_dir)
        path_entries = [_remap_path_for_user(p, home_dir) for p in path_entries]
        path_entries.extend(_build_user_local_paths(Path(home_dir), path_entries))
        path_entries.extend(_build_wsl_interop_paths(path_entries))
        path_entries.extend(common_bin_paths)
        sane_path = ":".join(path_entries)
        return f"""[Unit]
Description={SERVICE_DESCRIPTION}
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User={username}
Group={group_name}
ExecStart={python_path} -m hermes_cli.main{f" {profile_arg}" if profile_arg else ""} gateway run --replace
WorkingDirectory={working_dir}
Environment="HOME={home_dir}"
Environment="USER={username}"
Environment="LOGNAME={username}"
Environment="PATH={sane_path}"
Environment="VIRTUAL_ENV={venv_dir}"
Environment="HERMES_HOME={hermes_home}"
Restart=always
RestartSec=60
RestartMaxDelaySec=300
RestartSteps=5
RestartForceExitStatus={GATEWAY_SERVICE_RESTART_EXIT_CODE}
KillMode=mixed
KillSignal=SIGTERM
ExecReload=/bin/kill -USR1 $MAINPID
TimeoutStopSec={restart_timeout}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

    hermes_home = str(get_hermes_home().resolve())
    profile_arg = _profile_arg(hermes_home)
    path_entries.extend(_build_user_local_paths(Path.home(), path_entries))
    path_entries.extend(_build_wsl_interop_paths(path_entries))
    path_entries.extend(common_bin_paths)
    sane_path = ":".join(path_entries)
    return f"""[Unit]
Description={SERVICE_DESCRIPTION}
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
ExecStart={python_path} -m hermes_cli.main{f" {profile_arg}" if profile_arg else ""} gateway run --replace
WorkingDirectory={working_dir}
Environment="PATH={sane_path}"
Environment="VIRTUAL_ENV={venv_dir}"
Environment="HERMES_HOME={hermes_home}"
Restart=always
RestartSec=60
RestartMaxDelaySec=300
RestartSteps=5
RestartForceExitStatus={GATEWAY_SERVICE_RESTART_EXIT_CODE}
KillMode=mixed
KillSignal=SIGTERM
ExecReload=/bin/kill -USR1 $MAINPID
TimeoutStopSec={restart_timeout}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
"""

def _normalize_service_definition(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _normalize_launchd_plist_for_comparison(text: str) -> str:
    """Normalize launchd plist text for staleness checks.

    The generated plist intentionally captures a broad PATH assembled from the
    invoking shell so user-installed tools remain reachable under launchd.
    That makes raw text comparison unstable across shells, so ignore the PATH
    payload when deciding whether the installed plist is stale.
    """
    import re

    normalized = _normalize_service_definition(text)
    return re.sub(
        r'(<key>PATH</key>\s*<string>)(.*?)(</string>)',
        r'\1__HERMES_PATH__\3',
        normalized,
        flags=re.S,
    )


def systemd_unit_is_current(system: bool = False) -> bool:
    unit_path = get_systemd_unit_path(system=system)
    if not unit_path.exists():
        return False

    installed = unit_path.read_text(encoding="utf-8")
    expected_user = _read_systemd_user_from_unit(unit_path) if system else None
    expected = generate_systemd_unit(system=system, run_as_user=expected_user)
    return _normalize_service_definition(installed) == _normalize_service_definition(expected)



def refresh_systemd_unit_if_needed(system: bool = False) -> bool:
    """Rewrite the installed systemd unit when the generated definition has changed."""
    unit_path = get_systemd_unit_path(system=system)
    if not unit_path.exists() or systemd_unit_is_current(system=system):
        return False

    expected_user = _read_systemd_user_from_unit(unit_path) if system else None
    unit_path.write_text(generate_systemd_unit(system=system, run_as_user=expected_user), encoding="utf-8")
    _run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    print(f"↻ Updated gateway {_service_scope_label(system)} service definition to match the current Hermes install")
    return True



def _print_linger_enable_warning(username: str, detail: str | None = None) -> None:
    print()
    print("⚠ Linger not enabled — gateway may stop when you close this terminal.")
    if detail:
        print(f"  Auto-enable failed: {detail}")
    print()
    print("  On headless servers (VPS, cloud instances) run:")
    print(f"    sudo loginctl enable-linger {username}")
    print()
    print("  Then restart the gateway:")
    print(f"    systemctl --user restart {get_service_name()}.service")
    print()



def _ensure_linger_enabled() -> None:
    """Enable linger when possible so the user gateway survives logout."""
    if is_termux() or not is_linux():
        return

    import getpass

    username = getpass.getuser()
    linger_file = Path(f"/var/lib/systemd/linger/{username}")
    if linger_file.exists():
        print("✓ Systemd linger is enabled (service survives logout)")
        return

    linger_enabled, linger_detail = get_systemd_linger_status()
    if linger_enabled is True:
        print("✓ Systemd linger is enabled (service survives logout)")
        return

    if not shutil.which("loginctl"):
        _print_linger_enable_warning(username, linger_detail or "loginctl not found")
        return

    print("Enabling linger so the gateway survives SSH logout...")
    try:
        result = subprocess.run(
            ["loginctl", "enable-linger", username],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception as e:
        _print_linger_enable_warning(username, str(e))
        return

    if result.returncode == 0:
        print("✓ Linger enabled — gateway will persist after logout")
        return

    detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
    _print_linger_enable_warning(username, detail or linger_detail)


def _select_systemd_scope(system: bool = False) -> bool:
    if system:
        return True
    return get_systemd_unit_path(system=True).exists() and not get_systemd_unit_path(system=False).exists()


def _system_scope_wizard_would_need_root(system: bool = False) -> bool:
    """True when the setup wizard is about to trigger a system-scope operation
    as a non-root user.

    Replicates the decision ``_select_systemd_scope`` makes inside
    ``systemd_start`` / ``systemd_restart`` / ``systemd_stop`` so the wizard
    can detect the dead-end BEFORE prompting, rather than letting
    ``SystemScopeRequiresRootError`` propagate out and leave the user
    staring at a bare shell.
    """
    if os.geteuid() == 0:
        return False
    return _select_systemd_scope(system=system)


def _print_system_scope_remediation(action: str) -> None:
    """Print actionable remediation when the wizard skips a system-scope
    prompt because the user isn't root. Keeps the wizard flowing instead of
    aborting.
    """
    svc = get_service_name()
    print_warning(
        f"Gateway is installed as a system-wide service — "
        f"{action} requires root."
    )
    print_info("  Options:")
    print_info(f"    1. {action.capitalize()} it this time:")
    if action == "start":
        print_info(f"         sudo systemctl start {svc}")
    elif action == "stop":
        print_info(f"         sudo systemctl stop {svc}")
    elif action == "restart":
        print_info(f"         sudo systemctl restart {svc}")
    else:
        print_info(f"         sudo systemctl {action} {svc}")
    print_info("    2. Switch to a per-user service (recommended for personal use):")
    print_info("         sudo hermes gateway uninstall --system")
    print_info("         hermes gateway install")
    print_info("         hermes gateway start")


def _get_restart_drain_timeout() -> float:
    """Return the configured gateway restart drain timeout in seconds."""
    raw = os.getenv("HERMES_RESTART_DRAIN_TIMEOUT", "").strip()
    if not raw:
        cfg = read_raw_config()
        agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
        raw = str(
            agent_cfg.get(
                "restart_drain_timeout", DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
            )
        )
    return parse_restart_drain_timeout(raw)


def systemd_install(force: bool = False, system: bool = False, run_as_user: str | None = None):
    if system:
        _require_root_for_system_service("install")

    # Offer to remove legacy units (hermes.service from pre-rename installs)
    # before installing the new hermes-gateway.service. If both remain, they
    # flap-fight for the Telegram bot token on every gateway startup.
    # Only removes units matching _LEGACY_SERVICE_NAMES + our ExecStart
    # signature — profile units are never touched.
    if has_legacy_hermes_units():
        print()
        print_legacy_unit_warning()
        print()
        if prompt_yes_no("Remove the legacy unit(s) before installing?", True):
            remove_legacy_hermes_units(interactive=False)
            print()

    unit_path = get_systemd_unit_path(system=system)
    scope_flag = " --system" if system else ""

    if unit_path.exists() and not force:
        if not systemd_unit_is_current(system=system):
            print(f"↻ Repairing outdated {_service_scope_label(system)} systemd service at: {unit_path}")
            refresh_systemd_unit_if_needed(system=system)
            _run_systemctl(["enable", get_service_name()], system=system, check=True, timeout=30)
            print(f"✓ {_service_scope_label(system).capitalize()} service definition updated")
            return
        print(f"Service already installed at: {unit_path}")
        print("Use --force to reinstall")
        return

    unit_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Installing {_service_scope_label(system)} systemd service to: {unit_path}")
    unit_path.write_text(generate_systemd_unit(system=system, run_as_user=run_as_user), encoding="utf-8")

    _run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    _run_systemctl(["enable", get_service_name()], system=system, check=True, timeout=30)

    print()
    print(f"✓ {_service_scope_label(system).capitalize()} service installed and enabled!")
    print()
    print("Next steps:")
    print(f"  {'sudo ' if system else ''}hermes gateway start{scope_flag}              # Start the service")
    print(f"  {'sudo ' if system else ''}hermes gateway status{scope_flag}             # Check status")
    print(f"  {'journalctl' if system else 'journalctl --user'} -u {get_service_name()} -f  # View logs")
    print()

    if system:
        configured_user = _read_systemd_user_from_unit(unit_path)
        if configured_user:
            print(f"Configured to run as: {configured_user}")
    else:
        _ensure_linger_enabled()

    print_systemd_scope_conflict_warning()
    print_legacy_unit_warning()


def systemd_uninstall(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("uninstall")

    _run_systemctl(["stop", get_service_name()], system=system, check=False, timeout=90)
    _run_systemctl(["disable", get_service_name()], system=system, check=False, timeout=30)

    unit_path = get_systemd_unit_path(system=system)
    if unit_path.exists():
        unit_path.unlink()
        print(f"✓ Removed {unit_path}")

    _run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    print(f"✓ {_service_scope_label(system).capitalize()} service uninstalled")


def _require_service_installed(action: str, system: bool = False) -> None:
    unit_path = get_systemd_unit_path(system=system)
    if not unit_path.exists():
        scope_flag = " --system" if system else ""
        print(f"✗ Gateway service is not installed")
        print(f"  Run: {'sudo ' if system else ''}hermes gateway install{scope_flag}")
        sys.exit(1)


def systemd_start(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("start")
    else:
        # Fail fast with actionable guidance if the user D-Bus session is not
        # reachable (common on fresh RHEL/Debian SSH sessions without linger).
        # Raises UserSystemdUnavailableError with a remediation message.
        _preflight_user_systemd()
    _require_service_installed("start", system=system)
    refresh_systemd_unit_if_needed(system=system)
    _run_systemctl(["start", get_service_name()], system=system, check=True, timeout=30)
    print(f"✓ {_service_scope_label(system).capitalize()} service started")



def systemd_stop(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("stop")
    _require_service_installed("stop", system=system)
    try:
        from gateway.status import get_running_pid, write_planned_stop_marker
        pid = get_running_pid(cleanup_stale=False)
        if pid is not None:
            write_planned_stop_marker(pid)
    except Exception:
        pass
    _run_systemctl(["stop", get_service_name()], system=system, check=True, timeout=90)
    print(f"✓ {_service_scope_label(system).capitalize()} service stopped")



def systemd_restart(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("restart")
    else:
        _preflight_user_systemd()
    _require_service_installed("restart", system=system)
    refresh_systemd_unit_if_needed(system=system)
    from gateway.status import get_running_pid

    pid = get_running_pid() or _systemd_main_pid(system=system)
    if pid is not None:
        scope_label = _service_scope_label(system).capitalize()
        svc = get_service_name()
        drain_timeout = _get_restart_drain_timeout()

        print(f"⏳ {scope_label} service restarting gracefully (PID {pid})...")
        if _graceful_restart_via_sigusr1(pid, drain_timeout + 5):
            # The gateway exits with code 75 for a planned service restart.
            # RestartSec can otherwise delay the relaunch even though the
            # operator asked for an immediate restart, so kick the unit once
            # the old PID has exited and then wait for the replacement PID.
            _run_systemctl(
                ["reset-failed", svc],
                system=system,
                check=False,
                timeout=30,
            )
            _run_systemctl(
                ["restart", svc],
                system=system,
                check=False,
                timeout=90,
            )
            if _wait_for_systemd_service_restart(system=system, previous_pid=pid):
                return
            if _systemd_service_is_start_limited(system=system):
                return

        print(
            f"⚠ Graceful restart did not complete within {int(drain_timeout + 5)}s; "
            "forcing a service restart..."
        )
        _run_systemctl(
            ["reset-failed", svc],
            system=system,
            check=False,
            timeout=30,
        )
        try:
            _run_systemctl(["restart", svc], system=system, check=True, timeout=90)
        except subprocess.CalledProcessError as exc:
            if _systemd_error_indicates_start_limit(exc) or _systemd_service_is_start_limited(system=system):
                _print_systemd_start_limit_wait(system=system)
                return
            raise
        _wait_for_systemd_service_restart(system=system, previous_pid=pid)
        return

    if _recover_pending_systemd_restart(system=system, previous_pid=pid):
        return

    _run_systemctl(
        ["reset-failed", get_service_name()],
        system=system,
        check=False,
        timeout=30,
    )
    try:
        _run_systemctl(["restart", get_service_name()], system=system, check=True, timeout=90)
    except subprocess.CalledProcessError as exc:
        if _systemd_error_indicates_start_limit(exc) or _systemd_service_is_start_limited(system=system):
            _print_systemd_start_limit_wait(system=system)
            return
        raise
    _wait_for_systemd_service_restart(system=system, previous_pid=pid)



def systemd_status(deep: bool = False, system: bool = False, full: bool = False):
    system = _select_systemd_scope(system)
    unit_path = get_systemd_unit_path(system=system)
    scope_flag = " --system" if system else ""

    if not unit_path.exists():
        print("✗ Gateway service is not installed")
        print(f"  Run: {'sudo ' if system else ''}hermes gateway install{scope_flag}")
        return

    if has_conflicting_systemd_units():
        print_systemd_scope_conflict_warning()
        print()

    if has_legacy_hermes_units():
        print_legacy_unit_warning()
        print()

    if not systemd_unit_is_current(system=system):
        print("⚠ Installed gateway service definition is outdated")
        print(f"  Run: {'sudo ' if system else ''}hermes gateway restart{scope_flag}  # auto-refreshes the unit")
        print()

    status_cmd = ["status", get_service_name(), "--no-pager"]
    if full:
        status_cmd.append("-l")

    _run_systemctl(
        status_cmd,
        system=system,
        capture_output=False,
        timeout=10,
    )

    result = _run_systemctl(
        ["is-active", get_service_name()],
        system=system,
        capture_output=True,
        text=True,
        timeout=10,
    )

    status = result.stdout.strip()

    if status == "active":
        print(f"✓ {_service_scope_label(system).capitalize()} gateway service is running")
    else:
        print(f"✗ {_service_scope_label(system).capitalize()} gateway service is stopped")
        print(f"  Run: {'sudo ' if system else ''}hermes gateway start{scope_flag}")

    configured_user = _read_systemd_user_from_unit(unit_path) if system else None
    if configured_user:
        print(f"Configured to run as: {configured_user}")

    runtime_lines = _runtime_health_lines()
    if runtime_lines:
        print()
        print("Recent gateway health:")
        for line in runtime_lines:
            print(f"  {line}")

    unit_props = _read_systemd_unit_properties(system=system)
    active_state = unit_props.get("ActiveState", "")
    sub_state = unit_props.get("SubState", "")
    exec_main_status = unit_props.get("ExecMainStatus", "")
    result_code = unit_props.get("Result", "")
    if active_state == "activating" and sub_state == "auto-restart":
        print("  ⏳ Restart pending: systemd is waiting to relaunch the gateway")
    elif _systemd_unit_is_start_limited(unit_props):
        print("  ⏳ Restart pending: systemd is temporarily rate-limiting starts")
        print(f"  Run after the start-limit window expires: {'sudo ' if system else ''}hermes gateway restart{scope_flag}")
        print(f"  Or clear it manually: systemctl {'--user ' if not system else ''}reset-failed {get_service_name()}")
    elif active_state == "failed" and exec_main_status == str(GATEWAY_SERVICE_RESTART_EXIT_CODE):
        print("  ⚠ Planned restart is stuck in systemd failed state (exit 75)")
        print(f"  Run: systemctl {'--user ' if not system else ''}reset-failed {get_service_name()} && {'sudo ' if system else ''}hermes gateway start{scope_flag}")
    elif active_state == "failed" and result_code:
        print(f"  ⚠ Systemd unit result: {result_code}")

    if system:
        print("✓ System service starts at boot without requiring systemd linger")
    elif deep:
        print_systemd_linger_guidance()
    else:
        linger_enabled, _ = get_systemd_linger_status()
        if linger_enabled is True:
            print("✓ Systemd linger is enabled (service survives logout)")
        elif linger_enabled is False:
            print("⚠ Systemd linger is disabled (gateway may stop when you log out)")
            print("  Run: sudo loginctl enable-linger $USER")

    if deep:
        print()
        print("Recent logs:")
        log_cmd = _journalctl_cmd(system) + ["-u", get_service_name(), "-n", "20", "--no-pager"]
        if full:
            log_cmd.append("-l")
        subprocess.run(log_cmd, timeout=10)


# =============================================================================
# Launchd (macOS)
# =============================================================================

def get_launchd_label() -> str:
    """Return the launchd service label, scoped per profile."""
    suffix = _profile_suffix()
    return f"ai.hermes.gateway-{suffix}" if suffix else "ai.hermes.gateway"


def _launchd_domain() -> str:
    return f"gui/{os.getuid()}"


def generate_launchd_plist() -> str:
    python_path = get_python_path()
    working_dir = str(PROJECT_ROOT)
    hermes_home = str(get_hermes_home().resolve())
    log_dir = get_hermes_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    label = get_launchd_label()
    profile_arg = _profile_arg(hermes_home)
    # Build a sane PATH for the launchd plist.  launchd provides only a
    # minimal default (/usr/bin:/bin:/usr/sbin:/sbin) which misses Homebrew,
    # nvm, cargo, etc.  We prepend venv/bin and node_modules/.bin (matching
    # the systemd unit), then capture the user's full shell PATH so every
    # user-installed tool (node, ffmpeg, …) is reachable.
    detected_venv = _detect_venv_dir()
    venv_bin = str(detected_venv / "bin") if detected_venv else str(PROJECT_ROOT / "venv" / "bin")
    venv_dir = str(detected_venv) if detected_venv else str(PROJECT_ROOT / "venv")
    node_bin = str(PROJECT_ROOT / "node_modules" / ".bin")
    # Resolve the directory containing the node binary (e.g. Homebrew, nvm)
    # so it's explicitly in PATH even if the user's shell PATH changes later.
    priority_dirs = [venv_bin, node_bin]
    resolved_node = shutil.which("node")
    if resolved_node:
        resolved_node_dir = str(Path(resolved_node).resolve().parent)
        if resolved_node_dir not in priority_dirs:
            priority_dirs.append(resolved_node_dir)
    sane_path = ":".join(
        dict.fromkeys(priority_dirs + [p for p in os.environ.get("PATH", "").split(":") if p])
    )

    # Build ProgramArguments array, including --profile when using a named profile
    prog_args = [
        f"<string>{python_path}</string>",
        "<string>-m</string>",
        "<string>hermes_cli.main</string>",
    ]
    if profile_arg:
        for part in profile_arg.split():
            prog_args.append(f"<string>{part}</string>")
    prog_args.extend([
        "<string>gateway</string>",
        "<string>run</string>",
        "<string>--replace</string>",
    ])
    prog_args_xml = "\n        ".join(prog_args)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
        {prog_args_xml}
    </array>
    
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{sane_path}</string>
        <key>VIRTUAL_ENV</key>
        <string>{venv_dir}</string>
        <key>HERMES_HOME</key>
        <string>{hermes_home}</string>
    </dict>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    
    <key>StandardOutPath</key>
    <string>{log_dir}/gateway.log</string>
    
    <key>StandardErrorPath</key>
    <string>{log_dir}/gateway.error.log</string>
</dict>
</plist>
"""

def launchd_plist_is_current() -> bool:
    """Check if the installed launchd plist matches the currently generated one."""
    plist_path = get_launchd_plist_path()
    if not plist_path.exists():
        return False

    installed = plist_path.read_text(encoding="utf-8")
    expected = generate_launchd_plist()
    return _normalize_launchd_plist_for_comparison(installed) == _normalize_launchd_plist_for_comparison(expected)


def refresh_launchd_plist_if_needed() -> bool:
    """Rewrite the installed launchd plist when the generated definition has changed.

    Unlike systemd, launchd picks up plist changes on the next ``launchctl kill``/
    ``launchctl kickstart`` cycle — no daemon-reload is needed. We still bootout/
    bootstrap to make launchd re-read the updated plist immediately.
    """
    plist_path = get_launchd_plist_path()
    if not plist_path.exists() or launchd_plist_is_current():
        return False

    plist_path.write_text(generate_launchd_plist(), encoding="utf-8")
    label = get_launchd_label()
    # Bootout/bootstrap so launchd picks up the new definition
    subprocess.run(["launchctl", "bootout", f"{_launchd_domain()}/{label}"], check=False, timeout=90)
    subprocess.run(["launchctl", "bootstrap", _launchd_domain(), str(plist_path)], check=False, timeout=30)
    print("↻ Updated gateway launchd service definition to match the current Hermes install")
    return True


def launchd_install(force: bool = False):
    plist_path = get_launchd_plist_path()
    
    if plist_path.exists() and not force:
        if not launchd_plist_is_current():
            print(f"↻ Repairing outdated launchd service at: {plist_path}")
            refresh_launchd_plist_if_needed()
            print("✓ Service definition updated")
            return
        print(f"Service already installed at: {plist_path}")
        print("Use --force to reinstall")
        return
    
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Installing launchd service to: {plist_path}")
    plist_path.write_text(generate_launchd_plist())
    
    subprocess.run(["launchctl", "bootstrap", _launchd_domain(), str(plist_path)], check=True, timeout=30)
    
    print()
    print("✓ Service installed and loaded!")
    print()
    print("Next steps:")
    print("  hermes gateway status             # Check status")
    from hermes_constants import display_hermes_home as _dhh
    print(f"  tail -f {_dhh()}/logs/gateway.log  # View logs")

def launchd_uninstall():
    plist_path = get_launchd_plist_path()
    label = get_launchd_label()
    subprocess.run(["launchctl", "bootout", f"{_launchd_domain()}/{label}"], check=False, timeout=90)
    
    if plist_path.exists():
        plist_path.unlink()
        print(f"✓ Removed {plist_path}")
    
    print("✓ Service uninstalled")

def launchd_start():
    plist_path = get_launchd_plist_path()
    label = get_launchd_label()

    # Self-heal if the plist is missing entirely (e.g., manual cleanup, failed upgrade)
    if not plist_path.exists():
        print("↻ launchd plist missing; regenerating service definition")
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_text(generate_launchd_plist(), encoding="utf-8")
        subprocess.run(["launchctl", "bootstrap", _launchd_domain(), str(plist_path)], check=True, timeout=30)
        subprocess.run(["launchctl", "kickstart", f"{_launchd_domain()}/{label}"], check=True, timeout=30)
        print("✓ Service started")
        return

    refresh_launchd_plist_if_needed()
    try:
        subprocess.run(["launchctl", "kickstart", f"{_launchd_domain()}/{label}"], check=True, timeout=30)
    except subprocess.CalledProcessError as e:
        if e.returncode not in (3, 113):
            raise
        print("↻ launchd job was unloaded; reloading service definition")
        subprocess.run(["launchctl", "bootstrap", _launchd_domain(), str(plist_path)], check=True, timeout=30)
        subprocess.run(["launchctl", "kickstart", f"{_launchd_domain()}/{label}"], check=True, timeout=30)
    print("✓ Service started")

def launchd_stop():
    label = get_launchd_label()
    target = f"{_launchd_domain()}/{label}"
    try:
        from gateway.status import get_running_pid, write_planned_stop_marker
        pid = get_running_pid(cleanup_stale=False)
        if pid is not None:
            write_planned_stop_marker(pid)
    except Exception:
        pass
    # bootout unloads the service definition so KeepAlive doesn't respawn
    # the process.  A plain `kill SIGTERM` only signals the process — launchd
    # immediately restarts it because KeepAlive.SuccessfulExit = false.
    # `hermes gateway start` re-bootstraps when it detects the job is unloaded.
    try:
        subprocess.run(["launchctl", "bootout", target], check=True, timeout=90)
    except subprocess.CalledProcessError as e:
        if e.returncode in (3, 113):
            pass  # Already unloaded — nothing to stop.
        else:
            raise
    _wait_for_gateway_exit(timeout=10.0, force_after=5.0)
    print("✓ Service stopped")

def _wait_for_gateway_exit(timeout: float = 10.0, force_after: float | None = 5.0) -> bool:
    """Wait for the gateway process (by saved PID) to exit.

    Uses the PID from the gateway.pid file — not launchd labels — so this
    works correctly when multiple gateway instances run under separate
    HERMES_HOME directories.

    Args:
        timeout: Total seconds to wait before giving up.
        force_after: Seconds of graceful waiting before escalating to force-kill.
    """
    import time
    from gateway.status import get_running_pid

    deadline = time.monotonic() + timeout
    force_deadline = (time.monotonic() + force_after) if force_after is not None else None
    force_sent = False

    while time.monotonic() < deadline:
        pid = get_running_pid()
        if pid is None:
            return True  # Process exited cleanly.

        if force_after is not None and not force_sent and time.monotonic() >= force_deadline:
            # Grace period expired — force-kill the specific PID.
            try:
                terminate_pid(pid, force=True)
                print(f"⚠ Gateway PID {pid} did not exit gracefully; sent SIGKILL")
            except (ProcessLookupError, PermissionError, OSError):
                return True  # Already gone or we can't touch it.
            force_sent = True

        time.sleep(0.3)

    # Timed out even after force-kill.
    remaining_pid = get_running_pid()
    if remaining_pid is not None:
        print(f"⚠ Gateway PID {remaining_pid} still running after {timeout}s — restart may fail")
        return False
    return True


def launchd_restart():
    label = get_launchd_label()
    target = f"{_launchd_domain()}/{label}"
    drain_timeout = _get_restart_drain_timeout()
    from gateway.status import get_running_pid

    try:
        pid = get_running_pid()
        if pid is not None and _request_gateway_self_restart(pid):
            print("✓ Service restart requested")
            return
        if pid is not None:
            try:
                terminate_pid(pid, force=False)
            except (ProcessLookupError, PermissionError, OSError):
                pid = None
            if pid is not None:
                exited = _wait_for_gateway_exit(timeout=drain_timeout, force_after=None)
                if not exited:
                    print(f"⚠ Gateway drain timed out after {drain_timeout:.0f}s — forcing launchd restart")
        subprocess.run(["launchctl", "kickstart", "-k", target], check=True, timeout=90)
        print("✓ Service restarted")
    except subprocess.CalledProcessError as e:
        if e.returncode not in (3, 113):
            raise
        # Job not loaded — bootstrap and start fresh
        print("↻ launchd job was unloaded; reloading")
        plist_path = get_launchd_plist_path()
        subprocess.run(["launchctl", "bootstrap", _launchd_domain(), str(plist_path)], check=True, timeout=30)
        subprocess.run(["launchctl", "kickstart", target], check=True, timeout=30)
        print("✓ Service restarted")

def launchd_status(deep: bool = False):
    plist_path = get_launchd_plist_path()
    label = get_launchd_label()
    try:
        result = subprocess.run(
            ["launchctl", "list", label],
            capture_output=True,
            text=True,
            timeout=10,
        )
        loaded = result.returncode == 0
        loaded_output = result.stdout
    except subprocess.TimeoutExpired:
        loaded = False
        loaded_output = ""

    print(f"Launchd plist: {plist_path}")
    if launchd_plist_is_current():
        print("✓ Service definition matches the current Hermes install")
    else:
        print("⚠ Service definition is stale relative to the current Hermes install")
        print("  Run: hermes gateway start")

    if loaded:
        print("✓ Gateway service is loaded")
        print(loaded_output)
    else:
        print("✗ Gateway service is not loaded")
        print("  Service definition exists locally but launchd has not loaded it.")
        print("  Run: hermes gateway start")
    
    if deep:
        log_file = get_hermes_home() / "logs" / "gateway.log"
        if log_file.exists():
            print()
            print("Recent logs:")
            subprocess.run(["tail", "-20", str(log_file)], timeout=10)


# =============================================================================
# Gateway Runner
# =============================================================================

def run_gateway(verbose: int = 0, quiet: bool = False, replace: bool = False):
    """Run the gateway in foreground.
    
    Args:
        verbose: Stderr log verbosity count added on top of default WARNING (0=WARNING, 1=INFO, 2+=DEBUG).
        quiet: Suppress all stderr log output.
        replace: If True, kill any existing gateway instance before starting.
                 This prevents systemd restart loops when the old process
                 hasn't fully exited yet.
    """
    sys.path.insert(0, str(PROJECT_ROOT))

    # Refresh the systemd unit definition on every boot so that restart
    # settings (RestartSec, StartLimitIntervalSec, etc.) stay current even
    # when the process was respawned via exit-code-75 (stale-code or
    # /restart) rather than through `hermes gateway restart` which already
    # calls refresh_systemd_unit_if_needed().  Without this, a code update
    # that ships new unit settings won't take effect until the next manual
    # `hermes gateway start/restart` — leaving the gateway vulnerable to
    # the exact failure mode the new settings were meant to prevent.
    if supports_systemd_services():
        try:
            refresh_systemd_unit_if_needed(system=False)
        except Exception:
            pass  # best-effort; don't block gateway startup
    
    from gateway.run import start_gateway
    
    print("┌─────────────────────────────────────────────────────────┐")
    print("│           ⚕ Hermes Gateway Starting...                 │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│  Messaging platforms + cron scheduler                    │")
    print("│  Press Ctrl+C to stop                                   │")
    print("└─────────────────────────────────────────────────────────┘")
    print()
    
    # Exit with code 1 if gateway fails to connect any platform,
    # so systemd Restart=always will retry on transient errors
    verbosity = None if quiet else verbose
    try:
        success = asyncio.run(start_gateway(replace=replace, verbosity=verbosity))
    except KeyboardInterrupt:
        print("\nGateway stopped.")
        return
    if not success:
        sys.exit(1)


# =============================================================================
# Gateway Setup (Interactive Messaging Platform Configuration)
# =============================================================================

# Per-platform config: each entry defines the env vars, setup instructions,
# and prompts needed to configure a messaging platform.
_PLATFORMS = [
    {
        "key": "telegram",
        "label": "Telegram",
        "emoji": "📱",
        "token_var": "TELEGRAM_BOT_TOKEN",
        "setup_instructions": [
            "1. Open Telegram and message @BotFather",
            "2. Send /newbot and follow the prompts to create your bot",
            "3. Copy the bot token BotFather gives you",
            "4. To find your user ID: message @userinfobot — it replies with your numeric ID",
        ],
        "vars": [
            {"name": "TELEGRAM_BOT_TOKEN", "prompt": "Bot token", "password": True,
             "help": "Paste the token from @BotFather (step 3 above)."},
            {"name": "TELEGRAM_ALLOWED_USERS", "prompt": "Allowed user IDs (comma-separated)", "password": False,
             "is_allowlist": True,
             "help": "Paste your user ID from step 4 above."},
            {"name": "TELEGRAM_HOME_CHANNEL", "prompt": "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)", "password": False,
             "help": "For DMs, this is your user ID. You can set it later by typing /set-home in chat."},
        ],
    },
    {
        "key": "discord",
        "label": "Discord",
        "emoji": "💬",
        "token_var": "DISCORD_BOT_TOKEN",
        "setup_instructions": [
            "1. Go to https://discord.com/developers/applications → New Application",
            "2. Go to Bot → Reset Token → copy the bot token",
            "3. Enable: Bot → Privileged Gateway Intents → Message Content Intent",
            "4. Invite the bot to your server:",
            "   OAuth2 → URL Generator → check BOTH scopes:",
            "     - bot",
            "     - applications.commands  (required for slash commands!)",
            "   Bot Permissions: Send Messages, Read Message History, Attach Files",
            "   Copy the URL and open it in your browser to invite.",
            "5. Get your user ID: enable Developer Mode in Discord settings,",
            "   then right-click your name → Copy ID",
        ],
        "vars": [
            {"name": "DISCORD_BOT_TOKEN", "prompt": "Bot token", "password": True,
             "help": "Paste the token from step 2 above."},
            {"name": "DISCORD_ALLOWED_USERS", "prompt": "Allowed user IDs or usernames (comma-separated)", "password": False,
             "is_allowlist": True,
             "help": "Paste your user ID from step 5 above."},
            {"name": "DISCORD_HOME_CHANNEL", "prompt": "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)", "password": False,
             "help": "Right-click a channel → Copy Channel ID (requires Developer Mode)."},
        ],
    },
    {
        "key": "slack",
        "label": "Slack",
        "emoji": "💼",
        "token_var": "SLACK_BOT_TOKEN",
        "setup_instructions": [
            "1. Go to https://api.slack.com/apps → Create New App → From Scratch",
            "2. Enable Socket Mode: Settings → Socket Mode → Enable",
            "   Create an App-Level Token with scope: connections:write → copy xapp-... token",
            "3. Add Bot Token Scopes: Features → OAuth & Permissions → Scopes",
            "   Required: chat:write, app_mentions:read, channels:history, channels:read,",
            "   groups:history, im:history, im:read, im:write, users:read, files:read, files:write",
            "4. Subscribe to Events: Features → Event Subscriptions → Enable",
            "   Required events: message.im, message.channels, app_mention",
            "   Optional: message.groups (for private channels)",
            "   ⚠ Without message.channels the bot will ONLY work in DMs!",
            "5. Install to Workspace: Settings → Install App → copy xoxb-... token",
            "6. Reinstall the app after any scope or event changes",
            "7. Find your user ID: click your profile → three dots → Copy member ID",
            "8. Invite the bot to channels: /invite @YourBot",
        ],
        "vars": [
            {"name": "SLACK_BOT_TOKEN", "prompt": "Bot Token (xoxb-...)", "password": True,
             "help": "Paste the bot token from step 3 above."},
            {"name": "SLACK_APP_TOKEN", "prompt": "App Token (xapp-...)", "password": True,
             "help": "Paste the app-level token from step 4 above."},
            {"name": "SLACK_ALLOWED_USERS", "prompt": "Allowed user IDs (comma-separated)", "password": False,
             "is_allowlist": True,
             "help": "Paste your member ID from step 7 above."},
        ],
    },
    {
        "key": "matrix",
        "label": "Matrix",
        "emoji": "🔐",
        "token_var": "MATRIX_ACCESS_TOKEN",
        "setup_instructions": [
            "1. Works with any Matrix homeserver (self-hosted Synapse/Conduit/Dendrite or matrix.org)",
            "2. Create a bot user on your homeserver, or use your own account",
            "3. Get an access token: Element → Settings → Help & About → Access Token",
            "   Or via API: curl -X POST https://your-server/_matrix/client/v3/login \\",
            "     -d '{\"type\":\"m.login.password\",\"user\":\"@bot:server\",\"password\":\"...\"}'",
            "4. Alternatively, provide user ID + password and Hermes will log in directly",
            "5. For E2EE: set MATRIX_ENCRYPTION=true (requires pip install 'mautrix[encryption]')",
            "6. To find your user ID: it's @username:your-server (shown in Element profile)",
        ],
        "vars": [
            {"name": "MATRIX_HOMESERVER", "prompt": "Homeserver URL (e.g. https://matrix.example.org)", "password": False,
             "help": "Your Matrix homeserver URL. Works with any self-hosted instance."},
            {"name": "MATRIX_ACCESS_TOKEN", "prompt": "Access token (leave empty to use password login instead)", "password": True,
             "help": "Paste your access token, or leave empty and provide user ID + password below."},
            {"name": "MATRIX_USER_ID", "prompt": "User ID (@bot:server — required for password login)", "password": False,
             "help": "Full Matrix user ID, e.g. @hermes:matrix.example.org"},
            {"name": "MATRIX_ALLOWED_USERS", "prompt": "Allowed user IDs (comma-separated, e.g. @you:server)", "password": False,
             "is_allowlist": True,
             "help": "Matrix user IDs who can interact with the bot."},
            {"name": "MATRIX_HOME_ROOM", "prompt": "Home room ID (for cron/notification delivery, or empty to set later with /set-home)", "password": False,
             "help": "Room ID (e.g. !abc123:server) for delivering cron results and notifications."},
        ],
    },
    {
        "key": "mattermost",
        "label": "Mattermost",
        "emoji": "💬",
        "token_var": "MATTERMOST_TOKEN",
        "setup_instructions": [
            "1. In Mattermost: Integrations → Bot Accounts → Add Bot Account",
            "   (System Console → Integrations → Bot Accounts must be enabled)",
            "2. Give it a username (e.g. hermes) and copy the bot token",
            "3. Works with any self-hosted Mattermost instance — enter your server URL",
            "4. To find your user ID: click your avatar (top-left) → Profile",
            "   Your user ID is displayed there — click it to copy.",
            "   ⚠ This is NOT your username — it's a 26-character alphanumeric ID.",
            "5. To get a channel ID: click the channel name → View Info → copy the ID",
        ],
        "vars": [
            {"name": "MATTERMOST_URL", "prompt": "Server URL (e.g. https://mm.example.com)", "password": False,
             "help": "Your Mattermost server URL. Works with any self-hosted instance."},
            {"name": "MATTERMOST_TOKEN", "prompt": "Bot token", "password": True,
             "help": "Paste the bot token from step 2 above."},
            {"name": "MATTERMOST_ALLOWED_USERS", "prompt": "Allowed user IDs (comma-separated)", "password": False,
             "is_allowlist": True,
             "help": "Your Mattermost user ID from step 4 above."},
            {"name": "MATTERMOST_HOME_CHANNEL", "prompt": "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)", "password": False,
             "help": "Channel ID where Hermes delivers cron results and notifications."},
            {"name": "MATTERMOST_REPLY_MODE", "prompt": "Reply mode — 'off' for flat messages, 'thread' for threaded replies (default: off)", "password": False,
             "help": "off = flat channel messages, thread = replies nest under your message."},
        ],
    },
    {
        "key": "whatsapp",
        "label": "WhatsApp",
        "emoji": "📲",
        "token_var": "WHATSAPP_ENABLED",
    },
    {
        "key": "signal",
        "label": "Signal",
        "emoji": "📡",
        "token_var": "SIGNAL_HTTP_URL",
    },
    {
        "key": "email",
        "label": "Email",
        "emoji": "📧",
        "token_var": "EMAIL_ADDRESS",
        "setup_instructions": [
            "1. Use a dedicated email account for your Hermes agent",
            "2. For Gmail: enable 2FA, then create an App Password at",
            "   https://myaccount.google.com/apppasswords",
            "3. For other providers: use your email password or app-specific password",
            "4. IMAP must be enabled on your email account",
        ],
        "vars": [
            {"name": "EMAIL_ADDRESS", "prompt": "Email address", "password": False,
             "help": "The email address Hermes will use (e.g., hermes@gmail.com)."},
            {"name": "EMAIL_PASSWORD", "prompt": "Email password (or app password)", "password": True,
             "help": "For Gmail, use an App Password (not your regular password)."},
            {"name": "EMAIL_IMAP_HOST", "prompt": "IMAP host", "password": False,
             "help": "e.g., imap.gmail.com for Gmail, outlook.office365.com for Outlook."},
            {"name": "EMAIL_SMTP_HOST", "prompt": "SMTP host", "password": False,
             "help": "e.g., smtp.gmail.com for Gmail, smtp.office365.com for Outlook."},
            {"name": "EMAIL_ALLOWED_USERS", "prompt": "Allowed sender emails (comma-separated)", "password": False,
             "is_allowlist": True,
             "help": "Only emails from these addresses will be processed."},
        ],
    },
    {
        "key": "sms",
        "label": "SMS (Twilio)",
        "emoji": "📱",
        "token_var": "TWILIO_ACCOUNT_SID",
        "setup_instructions": [
            "1. Create a Twilio account at https://www.twilio.com/",
            "2. Get your Account SID and Auth Token from the Twilio Console dashboard",
            "3. Buy or configure a phone number capable of sending SMS",
            "4. Set up your webhook URL for inbound SMS:",
            "   Twilio Console → Phone Numbers → Active Numbers → your number",
            "   → Messaging → A MESSAGE COMES IN → Webhook → https://your-server:8080/webhooks/twilio",
        ],
        "vars": [
            {"name": "TWILIO_ACCOUNT_SID", "prompt": "Twilio Account SID", "password": False,
             "help": "Found on the Twilio Console dashboard."},
            {"name": "TWILIO_AUTH_TOKEN", "prompt": "Twilio Auth Token", "password": True,
             "help": "Found on the Twilio Console dashboard (click to reveal)."},
            {"name": "TWILIO_PHONE_NUMBER", "prompt": "Twilio phone number (E.164 format, e.g. +15551234567)", "password": False,
             "help": "The Twilio phone number to send SMS from."},
            {"name": "SMS_ALLOWED_USERS", "prompt": "Allowed phone numbers (comma-separated, E.164 format)", "password": False,
             "is_allowlist": True,
             "help": "Only messages from these phone numbers will be processed."},
            {"name": "SMS_HOME_CHANNEL", "prompt": "Home channel phone number (for cron/notification delivery, or empty)", "password": False,
             "help": "Phone number to deliver cron job results and notifications to."},
        ],
    },
    {
        "key": "dingtalk",
        "label": "DingTalk",
        "emoji": "💬",
        "token_var": "DINGTALK_CLIENT_ID",
        "setup_instructions": [
            "1. Go to https://open-dev.dingtalk.com → Create Application",
            "2. Under 'Credentials', copy the AppKey (Client ID) and AppSecret (Client Secret)",
            "3. Enable 'Stream Mode' under the bot settings",
            "4. Add the bot to a group chat or message it directly",
        ],
        "vars": [
            {"name": "DINGTALK_CLIENT_ID", "prompt": "AppKey (Client ID)", "password": False,
             "help": "The AppKey from your DingTalk application credentials."},
            {"name": "DINGTALK_CLIENT_SECRET", "prompt": "AppSecret (Client Secret)", "password": True,
             "help": "The AppSecret from your DingTalk application credentials."},
        ],
    },
    {
        "key": "feishu",
        "label": "Feishu / Lark",
        "emoji": "🪽",
        "token_var": "FEISHU_APP_ID",
        "setup_instructions": [
            "1. Go to https://open.feishu.cn/ (or https://open.larksuite.com/ for Lark)",
            "2. Create an app and copy the App ID and App Secret",
            "3. Enable the Bot capability for the app",
            "4. Choose WebSocket (recommended) or Webhook connection mode",
            "5. Add the bot to a group chat or message it directly",
            "6. Restrict access with FEISHU_ALLOWED_USERS for production use",
        ],
        "vars": [
            {"name": "FEISHU_APP_ID", "prompt": "App ID", "password": False,
             "help": "The App ID from your Feishu/Lark application."},
            {"name": "FEISHU_APP_SECRET", "prompt": "App Secret", "password": True,
             "help": "The App Secret from your Feishu/Lark application."},
            {"name": "FEISHU_DOMAIN", "prompt": "Domain — feishu or lark (default: feishu)", "password": False,
             "help": "Use 'feishu' for Feishu China, or 'lark' for Lark international."},
            {"name": "FEISHU_CONNECTION_MODE", "prompt": "Connection mode — websocket or webhook (default: websocket)", "password": False,
             "help": "websocket is recommended unless you specifically need webhook mode."},
            {"name": "FEISHU_ALLOWED_USERS", "prompt": "Allowed user IDs (comma-separated, or empty)", "password": False,
             "is_allowlist": True,
             "help": "Restrict which Feishu/Lark users can interact with the bot."},
            {"name": "FEISHU_HOME_CHANNEL", "prompt": "Home chat ID (optional, for cron/notifications)", "password": False,
             "help": "Chat ID for scheduled results and notifications."},
        ],
    },
    {
        "key": "wecom",
        "label": "WeCom (Enterprise WeChat)",
        "emoji": "💬",
        "token_var": "WECOM_BOT_ID",
        "setup_instructions": [
            "1. Go to WeCom Admin Console → Applications → Create AI Bot",
            "2. Copy the Bot ID and Secret from the bot's credentials page",
            "3. The bot connects via WebSocket — no public endpoint needed",
            "4. Add the bot to a group chat or message it directly in WeCom",
            "5. Restrict access with WECOM_ALLOWED_USERS for production use",
        ],
        "vars": [
            {"name": "WECOM_BOT_ID", "prompt": "Bot ID", "password": False,
             "help": "The Bot ID from your WeCom AI Bot."},
            {"name": "WECOM_SECRET", "prompt": "Secret", "password": True,
             "help": "The secret from your WeCom AI Bot."},
            {"name": "WECOM_ALLOWED_USERS", "prompt": "Allowed user IDs (comma-separated, or empty)", "password": False,
             "is_allowlist": True,
             "help": "Restrict which WeCom users can interact with the bot."},
            {"name": "WECOM_HOME_CHANNEL", "prompt": "Home chat ID (optional, for cron/notifications)", "password": False,
             "help": "Chat ID for scheduled results and notifications."},
        ],
    },
    {
        "key": "wecom_callback",
        "label": "WeCom Callback (Self-Built App)",
        "emoji": "💬",
        "token_var": "WECOM_CALLBACK_CORP_ID",
        "setup_instructions": [
            "1. Go to WeCom Admin Console → Applications → Create Self-Built App",
            "2. Note the Corp ID (top of admin console) and create a Corp Secret",
            "3. Under Receive Messages, configure the callback URL to point to your server",
            "4. Copy the Token and EncodingAESKey from the callback configuration",
            "5. The adapter runs an HTTP server — ensure the port is reachable from WeCom",
            "6. Restrict access with WECOM_CALLBACK_ALLOWED_USERS for production use",
        ],
        "vars": [
            {"name": "WECOM_CALLBACK_CORP_ID", "prompt": "Corp ID", "password": False,
             "help": "Your WeCom enterprise Corp ID."},
            {"name": "WECOM_CALLBACK_CORP_SECRET", "prompt": "Corp Secret", "password": True,
             "help": "The secret for your self-built application."},
            {"name": "WECOM_CALLBACK_AGENT_ID", "prompt": "Agent ID", "password": False,
             "help": "The Agent ID of your self-built application."},
            {"name": "WECOM_CALLBACK_TOKEN", "prompt": "Callback Token", "password": True,
             "help": "The Token from your WeCom callback configuration."},
            {"name": "WECOM_CALLBACK_ENCODING_AES_KEY", "prompt": "Encoding AES Key", "password": True,
             "help": "The EncodingAESKey from your WeCom callback configuration."},
            {"name": "WECOM_CALLBACK_PORT", "prompt": "Callback server port (default: 8645)", "password": False,
             "help": "Port for the HTTP callback server."},
            {"name": "WECOM_CALLBACK_ALLOWED_USERS", "prompt": "Allowed user IDs (comma-separated, or empty)", "password": False,
             "is_allowlist": True,
             "help": "Restrict which WeCom users can interact with the app."},
        ],
    },
    {
        "key": "weixin",
        "label": "Weixin / WeChat",
        "emoji": "💬",
        "token_var": "WEIXIN_ACCOUNT_ID",
    },
    {
        "key": "bluebubbles",
        "label": "BlueBubbles (iMessage)",
        "emoji": "💬",
        "token_var": "BLUEBUBBLES_SERVER_URL",
        "setup_instructions": [
            "1. Install BlueBubbles on a Mac that will act as your iMessage server:",
            "   https://bluebubbles.app/",
            "2. Complete the BlueBubbles setup wizard — sign in with your Apple ID",
            "3. In BlueBubbles Settings → API, note the Server URL and password",
            "4. The server URL is typically http://<your-mac-ip>:1234",
            "5. Hermes connects via the BlueBubbles REST API and receives",
            "   incoming messages via a local webhook",
            "6. To authorize users, use DM pairing: hermes pairing generate bluebubbles",
            "   Share the code — the user sends it via iMessage to get approved",
        ],
        "vars": [
            {"name": "BLUEBUBBLES_SERVER_URL", "prompt": "BlueBubbles server URL (e.g. http://192.168.1.10:1234)", "password": False,
             "help": "The URL shown in BlueBubbles Settings → API."},
            {"name": "BLUEBUBBLES_PASSWORD", "prompt": "BlueBubbles server password", "password": True,
             "help": "The password shown in BlueBubbles Settings → API."},
            {"name": "BLUEBUBBLES_ALLOWED_USERS", "prompt": "Pre-authorized phone numbers or iMessage IDs (comma-separated, or leave empty for DM pairing)", "password": False,
             "is_allowlist": True,
             "help": "Optional — pre-authorize specific users. Leave empty to use DM pairing instead (recommended)."},
            {"name": "BLUEBUBBLES_HOME_CHANNEL", "prompt": "Home channel (phone number or iMessage ID for cron/notifications, or empty)", "password": False,
             "help": "Phone number or Apple ID to deliver cron results and notifications to."},
        ],
    },
    {
        "key": "qqbot",
        "label": "QQ Bot",
        "emoji": "🐧",
        "token_var": "QQ_APP_ID",
        "setup_instructions": [
            "1. Register a QQ Bot application at q.qq.com",
            "2. Note your App ID and App Secret from the application page",
            "3. Enable the required intents (C2C, Group, Guild messages)",
            "4. Configure sandbox or publish the bot",
        ],
        "vars": [
            {"name": "QQ_APP_ID", "prompt": "QQ Bot App ID", "password": False,
             "help": "Your QQ Bot App ID from q.qq.com."},
            {"name": "QQ_CLIENT_SECRET", "prompt": "QQ Bot App Secret", "password": True,
             "help": "Your QQ Bot App Secret from q.qq.com."},
            {"name": "QQ_ALLOWED_USERS", "prompt": "Allowed user OpenIDs (comma-separated, leave empty for open access)", "password": False,
             "is_allowlist": True,
             "help": "Optional — restrict DM access to specific user OpenIDs."},
            {"name": "QQBOT_HOME_CHANNEL", "prompt": "Home channel (user/group OpenID for cron delivery, or empty)", "password": False,
             "help": "OpenID to deliver cron results and notifications to."},
        ],
    },
    {
        "key": "yuanbao",
        "label": "Yuanbao",
        "emoji": "💎",
        "token_var": "YUANBAO_APP_ID",
        "setup_instructions": [
            "1. Download the Yuanbao app from https://yuanbao.tencent.com/",
            "2. In the app, go to PAI → My Bot and create a new bot",
            "3. After the bot is created, copy the App ID and App Secret",
            "4. Enter them below and Hermes will connect automatically over WebSocket",
        ],
        "vars": [
            {"name": "YUANBAO_APP_ID", "prompt": "App ID", "password": False,
             "help": "The App ID from your Yuanbao IM Bot credentials."},
            {"name": "YUANBAO_APP_SECRET", "prompt": "App Secret", "password": True,
             "help": "The App Secret (used for HMAC signing) from your Yuanbao IM Bot."},
        ],
    },
]
def _all_platforms() -> list[dict]:
    """Return the full list of platforms for setup menus.

    Combines the built-in ``_PLATFORMS`` with plugin platforms registered via
    ``platform_registry``. Plugins are discovered on first call so bundled
    platforms (like IRC, which auto-load via ``kind: platform``) appear in
    ``hermes setup gateway`` without needing the gateway to be running.
    Built-ins keep their dict shape; plugin entries are adapted to the same
    shape with ``_registry_entry`` holding the source.
    """
    # Populate the registry so plugin platforms are visible. Idempotent.
    # Bundled platform plugins (``kind: platform``) auto-load unconditionally,
    # so every shipped messaging channel appears in the setup menu by default.
    # User-installed platform plugins under ~/.hermes/plugins/ still require
    # opt-in via ``plugins.enabled`` (untrusted code).
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
    except Exception as e:
        logger.debug("plugin discovery failed during platform enumeration: %s", e)

    platforms = [dict(p) for p in _PLATFORMS]
    by_key = {p["key"]: p for p in platforms}

    try:
        from gateway.platform_registry import platform_registry
    except Exception:
        return platforms

    for entry in platform_registry.all_entries():
        if entry.name in by_key:
            continue  # built-in already covers it
        platforms.append({
            "key": entry.name,
            "label": entry.label,
            "emoji": entry.emoji,
            "token_var": entry.required_env[0] if entry.required_env else "",
            "install_hint": entry.install_hint,
            "_registry_entry": entry,
        })
    return platforms


def _platform_status(platform: dict) -> str:
    """Return a plain-text status string for a platform.

    Returns uncolored text so it can safely be embedded in
    curses menu items (ANSI codes break width calculation).
    """
    entry = platform.get("_registry_entry")
    if entry is not None:
        configured = False
        # Prefer is_connected (checks both env and config.yaml) over
        # check_fn (typically just dependency / env presence).
        if entry.is_connected is not None:
            try:
                from gateway.config import PlatformConfig
                synthetic = PlatformConfig(enabled=True)
                configured = bool(entry.is_connected(synthetic))
            except Exception:
                configured = False
        if not configured:
            try:
                configured = bool(entry.check_fn())
            except Exception:
                configured = False
        return "configured" if configured else "not configured"

    token_var = platform.get("token_var", "")
    if not token_var:
        return "not configured"
    val = get_env_value(token_var)
    if token_var == "WHATSAPP_ENABLED":
        if val and val.lower() == "true":
            session_file = get_hermes_home() / "whatsapp" / "session" / "creds.json"
            if session_file.exists():
                return "configured + paired"
            return "enabled, not paired"
        return "not configured"
    if platform.get("key") == "signal":
        account = get_env_value("SIGNAL_ACCOUNT")
        if val and account:
            return "configured"
        if val or account:
            return "partially configured"
        return "not configured"
    if platform.get("key") == "email":
        pwd = get_env_value("EMAIL_PASSWORD")
        imap = get_env_value("EMAIL_IMAP_HOST")
        smtp = get_env_value("EMAIL_SMTP_HOST")
        if all([val, pwd, imap, smtp]):
            return "configured"
        if any([val, pwd, imap, smtp]):
            return "partially configured"
        return "not configured"
    if platform.get("key") == "matrix":
        homeserver = get_env_value("MATRIX_HOMESERVER")
        password = get_env_value("MATRIX_PASSWORD")
        if (val or password) and homeserver:
            e2ee = get_env_value("MATRIX_ENCRYPTION")
            suffix = " + E2EE" if e2ee and e2ee.lower() in ("true", "1", "yes") else ""
            return f"configured{suffix}"
        if val or password or homeserver:
            return "partially configured"
        return "not configured"
    if platform.get("key") == "weixin":
        token = get_env_value("WEIXIN_TOKEN")
        if val and token:
            return "configured"
        if val or token:
            return "partially configured"
        return "not configured"
    if val:
        return "configured"
    return "not configured"


def _runtime_health_lines() -> list[str]:
    """Summarize the latest persisted gateway runtime health state."""
    try:
        from gateway.status import read_runtime_status
    except Exception:
        return []

    state = read_runtime_status()
    if not state:
        return []

    lines: list[str] = []
    gateway_state = state.get("gateway_state")
    exit_reason = state.get("exit_reason")
    active_agents = state.get("active_agents")
    restart_requested = state.get("restart_requested")
    platforms = state.get("platforms", {}) or {}

    for platform, pdata in platforms.items():
        if pdata.get("state") == "fatal":
            message = pdata.get("error_message") or "unknown error"
            lines.append(f"⚠ {platform}: {message}")

    if gateway_state == "startup_failed" and exit_reason:
        lines.append(f"⚠ Last startup issue: {exit_reason}")
    elif gateway_state == "draining":
        action = "restart" if restart_requested else "shutdown"
        count = int(active_agents or 0)
        lines.append(f"⏳ Gateway draining for {action} ({count} active agent(s))")
    elif gateway_state == "stopped" and exit_reason:
        lines.append(f"⚠ Last shutdown reason: {exit_reason}")

    return lines


def _setup_standard_platform(platform: dict):
    """Interactive setup for Telegram, Discord, or Slack."""
    emoji = platform["emoji"]
    label = platform["label"]
    token_var = platform["token_var"]

    print()
    print(color(f"  ─── {emoji} {label} Setup ───", Colors.CYAN))

    # Show step-by-step setup instructions if this platform has them
    instructions = platform.get("setup_instructions")
    if instructions:
        print()
        for line in instructions:
            print_info(f"  {line}")

    existing_token = get_env_value(token_var)
    if existing_token:
        print()
        print_success(f"{label} is already configured.")
        if not prompt_yes_no(f"  Reconfigure {label}?", False):
            return

    allowed_val_set = None  # Track if user set an allowlist (for home channel offer)

    for var in platform["vars"]:
        print()
        print_info(f"  {var['help']}")
        existing = get_env_value(var["name"])
        if existing and var["name"] != token_var:
            print_info(f"  Current: {existing}")

        # Allowlist fields get special handling for the deny-by-default security model
        if var.get("is_allowlist"):
            print_info("  The gateway DENIES all users by default for security.")
            print_info("  Enter user IDs to create an allowlist, or leave empty")
            print_info("  and you'll be asked about open access next.")
            value = prompt(f"  {var['prompt']}", password=False)
            if value:
                cleaned = value.replace(" ", "")
                # For Discord, strip common prefixes (user:123, <@123>, <@!123>)
                if "DISCORD" in var["name"]:
                    parts = []
                    for uid in cleaned.split(","):
                        uid = uid.strip()
                        if uid.startswith("<@") and uid.endswith(">"):
                            uid = uid.lstrip("<@!").rstrip(">")
                        if uid.lower().startswith("user:"):
                            uid = uid[5:]
                        if uid:
                            parts.append(uid)
                    cleaned = ",".join(parts)
                save_env_value(var["name"], cleaned)
                print_success("  Saved — only these users can interact with the bot.")
                allowed_val_set = cleaned
            else:
                # No allowlist — ask about open access vs DM pairing
                print()
                access_choices = [
                    "Enable open access (anyone can message the bot)",
                    "Use DM pairing (unknown users request access, you approve with 'hermes pairing approve')",
                    "Skip for now (bot will deny all users until configured)",
                ]
                access_idx = prompt_choice("  How should unauthorized users be handled?", access_choices, 1)
                if access_idx == 0:
                    save_env_value("GATEWAY_ALLOW_ALL_USERS", "true")
                    print_warning("  Open access enabled — anyone can use your bot!")
                elif access_idx == 1:
                    print_success("  DM pairing mode — users will receive a code to request access.")
                    print_info("  Approve with: hermes pairing approve <platform> <code>")
                else:
                    print_info("  Skipped — configure later with 'hermes gateway setup'")
            continue

        value = prompt(f"  {var['prompt']}", password=var.get("password", False))
        if value:
            save_env_value(var["name"], value)
            print_success(f"  Saved {var['name']}")
        elif var["name"] == token_var:
            print_warning(f"  Skipped — {label} won't work without this.")
            return
        else:
            print_info("  Skipped (can configure later)")

    # If an allowlist was set and home channel wasn't, offer to reuse
    # the first user ID (common for Telegram DMs).
    home_var = f"{label.upper()}_HOME_CHANNEL"
    home_val = get_env_value(home_var)
    if allowed_val_set and not home_val and label == "Telegram":
        first_id = allowed_val_set.split(",")[0].strip()
        if first_id and prompt_yes_no(f"  Use your user ID ({first_id}) as the home channel?", True):
            save_env_value(home_var, first_id)
            print_success(f"  Home channel set to {first_id}")

    print()
    print_success(f"{emoji} {label} configured!")


def _setup_whatsapp():
    """Delegate to the existing WhatsApp setup flow."""
    from hermes_cli.main import cmd_whatsapp
    import argparse
    cmd_whatsapp(argparse.Namespace())


def _setup_email():
    """Configure Email via the standard platform setup."""
    email_platform = next(p for p in _PLATFORMS if p["key"] == "email")
    _setup_standard_platform(email_platform)


def _setup_sms():
    """Configure SMS (Twilio) via the standard platform setup."""
    sms_platform = next(p for p in _PLATFORMS if p["key"] == "sms")
    _setup_standard_platform(sms_platform)


def _setup_dingtalk():
    """Configure DingTalk — QR scan (recommended) or manual credential entry."""
    from hermes_cli.setup import (
        prompt_choice, prompt_yes_no, print_success, print_warning,
    )

    dingtalk_platform = next(p for p in _PLATFORMS if p["key"] == "dingtalk")
    emoji = dingtalk_platform["emoji"]
    label = dingtalk_platform["label"]

    print()
    print(color(f"  ─── {emoji} {label} Setup ───", Colors.CYAN))

    existing = get_env_value("DINGTALK_CLIENT_ID")
    if existing:
        print()
        print_success(f"{label} is already configured (Client ID: {existing}).")
        if not prompt_yes_no(f"  Reconfigure {label}?", False):
            return

    print()
    method = prompt_choice(
        "  Choose setup method",
        [
            "QR Code Scan (Recommended, auto-obtain Client ID and Client Secret)",
            "Manual Input (Client ID and Client Secret)",
        ],
        default=0,
    )

    if method == 0:
        # ── QR-code device-flow authorization ──
        try:
            from hermes_cli.dingtalk_auth import dingtalk_qr_auth
        except ImportError as exc:
            print_warning(f"  QR auth module failed to load ({exc}), falling back to manual input.")
            _setup_standard_platform(dingtalk_platform)
            return

        result = dingtalk_qr_auth()
        if result is None:
            print_warning("  QR auth incomplete, falling back to manual input.")
            _setup_standard_platform(dingtalk_platform)
            return

        client_id, client_secret = result
        save_env_value("DINGTALK_CLIENT_ID", client_id)
        save_env_value("DINGTALK_CLIENT_SECRET", client_secret)
        save_env_value("DINGTALK_ALLOW_ALL_USERS", "true")
        print()
        print_success(f"{emoji} {label} configured via QR scan!")
    else:
        # ── Manual entry ──
        _setup_standard_platform(dingtalk_platform)
        # Also enable allow-all by default for convenience
        if get_env_value("DINGTALK_CLIENT_ID"):
            save_env_value("DINGTALK_ALLOW_ALL_USERS", "true")


def _setup_wecom():
    """Interactive setup for WeCom — scan QR code or manual credential input."""
    print()
    print(color("  ─── 💬 WeCom (Enterprise WeChat) Setup ───", Colors.CYAN))

    existing_bot_id = get_env_value("WECOM_BOT_ID")
    existing_secret = get_env_value("WECOM_SECRET")
    if existing_bot_id and existing_secret:
        print()
        print_success("WeCom is already configured.")
        if not prompt_yes_no("  Reconfigure WeCom?", False):
            return

    # ── Choose setup method ──
    print()
    method_choices = [
        "Scan QR code to obtain Bot ID and Secret automatically (recommended)",
        "Enter existing Bot ID and Secret manually",
    ]
    method_idx = prompt_choice("  How would you like to set up WeCom?", method_choices, 0)

    bot_id = None
    secret = None

    if method_idx == 0:
        # ── QR scan flow ──
        try:
            from gateway.platforms.wecom import qr_scan_for_bot_info
        except Exception as exc:
            print_error(f"  WeCom QR scan import failed: {exc}")
            qr_scan_for_bot_info = None

        if qr_scan_for_bot_info is not None:
            try:
                credentials = qr_scan_for_bot_info()
            except KeyboardInterrupt:
                print()
                print_warning("  WeCom setup cancelled.")
                return
            except Exception as exc:
                print_warning(f"  QR scan failed: {exc}")
                credentials = None
            if credentials:
                bot_id = credentials.get("bot_id", "")
                secret = credentials.get("secret", "")
                print_success("  ✔ QR scan successful! Bot ID and Secret obtained.")

        if not bot_id or not secret:
            print_info("  QR scan did not complete. Continuing with manual input.")
            bot_id = None
            secret = None

    # ── Manual credential input ──
    if not bot_id or not secret:
        print()
        print_info("  1. Go to WeCom Application → Workspace → Smart Robot -> Create smart robots")
        print_info("  2. Select API Mode")
        print_info("  3. Copy the Bot ID and Secret from the bot's credentials info")
        print_info("  4. The bot connects via WebSocket — no public endpoint needed")
        print()
        bot_id = prompt("  Bot ID", password=False)
        if not bot_id:
            print_warning("  Skipped — WeCom won't work without a Bot ID.")
            return
        secret = prompt("  Secret", password=True)
        if not secret:
            print_warning("  Skipped — WeCom won't work without a Secret.")
            return

    # ── Save core credentials ──
    save_env_value("WECOM_BOT_ID", bot_id)
    save_env_value("WECOM_SECRET", secret)

    # ── Allowed users (deny-by-default security) ──
    print()
    print_info("  The gateway DENIES all users by default for security.")
    print_info("  Enter user IDs to create an allowlist, or leave empty.")
    allowed = prompt("  Allowed user IDs (comma-separated, or empty)", password=False)
    if allowed:
        cleaned = allowed.replace(" ", "")
        save_env_value("WECOM_ALLOWED_USERS", cleaned)
        print_success("  Saved — only these users can interact with the bot.")
    else:
        print()
        access_choices = [
            "Enable open access (anyone can message the bot)",
            "Use DM pairing (unknown users request access, you approve with 'hermes pairing approve')",
            "Disable direct messages",
            "Skip for now (bot will deny all users until configured)",
        ]
        access_idx = prompt_choice("  How should unauthorized users be handled?", access_choices, 1)
        if access_idx == 0:
            save_env_value("WECOM_DM_POLICY", "open")
            save_env_value("GATEWAY_ALLOW_ALL_USERS", "true")
            print_warning("  Open access enabled — anyone can use your bot!")
        elif access_idx == 1:
            save_env_value("WECOM_DM_POLICY", "pairing")
            print_success("  DM pairing mode — users will receive a code to request access.")
            print_info("  Approve with: hermes pairing approve <platform> <code>")
        elif access_idx == 2:
            save_env_value("WECOM_DM_POLICY", "disabled")
            print_warning("  Direct messages disabled.")
        else:
            print_info("  Skipped — configure later with 'hermes gateway setup'")

    # ── Home channel (optional) ──
    print()
    print_info("  Chat ID for scheduled results and notifications.")
    home = prompt("  Home chat ID (optional, for cron/notifications)", password=False)
    if home:
        save_env_value("WECOM_HOME_CHANNEL", home)
        print_success(f"  Home channel set to {home}")

    print()
    print_success("💬 WeCom configured!")


def _setup_yuanbao():
    """Configure Yuanbao via the standard platform setup."""
    yuanbao_platform = next(p for p in _PLATFORMS if p["key"] == "yuanbao")
    _setup_standard_platform(yuanbao_platform)


def _is_service_installed() -> bool:
    """Check if the gateway is installed as a system service."""
    if supports_systemd_services():
        return get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()
    elif is_macos():
        return get_launchd_plist_path().exists()
    return False


def _is_service_running() -> bool:
    """Check if the gateway service is currently running."""
    if supports_systemd_services():
        user_unit_exists = get_systemd_unit_path(system=False).exists()
        system_unit_exists = get_systemd_unit_path(system=True).exists()

        if user_unit_exists:
            try:
                result = _run_systemctl(
                    ["is-active", get_service_name()],
                    system=False, capture_output=True, text=True, timeout=10,
                )
                if result.stdout.strip() == "active":
                    return True
            except (RuntimeError, subprocess.TimeoutExpired):
                pass

        if system_unit_exists:
            try:
                result = _run_systemctl(
                    ["is-active", get_service_name()],
                    system=True, capture_output=True, text=True, timeout=10,
                )
                if result.stdout.strip() == "active":
                    return True
            except (RuntimeError, subprocess.TimeoutExpired):
                pass

        return False
    elif is_macos() and get_launchd_plist_path().exists():
        try:
            result = subprocess.run(
                ["launchctl", "list", get_launchd_label()],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    # Check for manual processes
    return len(find_gateway_pids()) > 0


def _setup_weixin():
    """Interactive setup for Weixin / WeChat personal accounts."""
    print()
    print(color("  ─── 💬 Weixin / WeChat Setup ───", Colors.CYAN))
    print()
    print_info("  1. Hermes will open Tencent iLink QR login in this terminal.")
    print_info("  2. Use WeChat to scan and confirm the QR code.")
    print_info("  3. Hermes will store the returned account_id/token in ~/.hermes/.env.")
    print_info("  4. This adapter supports native text, image, video, and document delivery.")

    existing_account = get_env_value("WEIXIN_ACCOUNT_ID")
    existing_token = get_env_value("WEIXIN_TOKEN")
    if existing_account and existing_token:
        print()
        print_success("Weixin is already configured.")
        if not prompt_yes_no("  Reconfigure Weixin?", False):
            return

    try:
        from gateway.platforms.weixin import check_weixin_requirements, qr_login
    except Exception as exc:
        print_error(f"  Weixin adapter import failed: {exc}")
        print_info("  Install gateway dependencies first, then retry.")
        return

    if not check_weixin_requirements():
        print_error("  Missing dependencies: Weixin needs aiohttp and cryptography.")
        print_info("  Install them, then rerun `hermes gateway setup`.")
        return

    print()
    if not prompt_yes_no("  Start QR login now?", True):
        print_info("  Cancelled.")
        return

    import asyncio
    try:
        credentials = asyncio.run(qr_login(str(get_hermes_home())))
    except KeyboardInterrupt:
        print()
        print_warning("  Weixin setup cancelled.")
        return
    except Exception as exc:
        print_error(f"  QR login failed: {exc}")
        return

    if not credentials:
        print_warning("  QR login did not complete.")
        return

    account_id = credentials.get("account_id", "")
    token = credentials.get("token", "")
    base_url = credentials.get("base_url", "")
    user_id = credentials.get("user_id", "")

    save_env_value("WEIXIN_ACCOUNT_ID", account_id)
    save_env_value("WEIXIN_TOKEN", token)
    if base_url:
        save_env_value("WEIXIN_BASE_URL", base_url)
    save_env_value("WEIXIN_CDN_BASE_URL", get_env_value("WEIXIN_CDN_BASE_URL") or "https://novac2c.cdn.weixin.qq.com/c2c")

    print()
    access_choices = [
        "Use DM pairing approval (recommended)",
        "Allow all direct messages",
        "Only allow listed user IDs",
        "Disable direct messages",
    ]
    access_idx = prompt_choice("  How should direct messages be authorized?", access_choices, 0)
    if access_idx == 0:
        save_env_value("WEIXIN_DM_POLICY", "pairing")
        save_env_value("WEIXIN_ALLOW_ALL_USERS", "false")
        save_env_value("WEIXIN_ALLOWED_USERS", "")
        print_success("  DM pairing enabled.")
        print_info("  Unknown DM users can request access and you approve them with `hermes pairing approve`.")
    elif access_idx == 1:
        save_env_value("WEIXIN_DM_POLICY", "open")
        save_env_value("WEIXIN_ALLOW_ALL_USERS", "true")
        save_env_value("WEIXIN_ALLOWED_USERS", "")
        print_warning("  Open DM access enabled for Weixin.")
    elif access_idx == 2:
        default_allow = user_id or ""
        allowlist = prompt("  Allowed Weixin user IDs (comma-separated)", default_allow, password=False).replace(" ", "")
        save_env_value("WEIXIN_DM_POLICY", "allowlist")
        save_env_value("WEIXIN_ALLOW_ALL_USERS", "false")
        save_env_value("WEIXIN_ALLOWED_USERS", allowlist)
        print_success("  Weixin allowlist saved.")
    else:
        save_env_value("WEIXIN_DM_POLICY", "disabled")
        save_env_value("WEIXIN_ALLOW_ALL_USERS", "false")
        save_env_value("WEIXIN_ALLOWED_USERS", "")
        print_warning("  Direct messages disabled.")

    print()
    print_info("  Note: QR login connects an iLink bot identity (e.g. ...@im.bot), not a")
    print_info("  scriptable personal WeChat account. Ordinary WeChat groups typically cannot")
    print_info("  invite an @im.bot identity, and iLink does not deliver ordinary-group events")
    print_info("  to most bot accounts. The settings below only apply when iLink actually")
    print_info("  delivers group events for your account type — otherwise DM remains the only")
    print_info("  working channel regardless of this choice.")
    group_choices = [
        "Disable group chats (recommended)",
        "Allow all group chats",
        "Only allow listed group chat IDs",
    ]
    group_idx = prompt_choice("  How should group chats be handled?", group_choices, 0)
    if group_idx == 0:
        save_env_value("WEIXIN_GROUP_POLICY", "disabled")
        save_env_value("WEIXIN_GROUP_ALLOWED_USERS", "")
        print_info("  Group chats disabled.")
    elif group_idx == 1:
        save_env_value("WEIXIN_GROUP_POLICY", "open")
        save_env_value("WEIXIN_GROUP_ALLOWED_USERS", "")
        print_warning("  All group chats enabled (only takes effect if iLink delivers group events).")
    else:
        allow_groups = prompt("  Allowed group chat IDs (comma-separated, not member user IDs)", "", password=False).replace(" ", "")
        save_env_value("WEIXIN_GROUP_POLICY", "allowlist")
        save_env_value("WEIXIN_GROUP_ALLOWED_USERS", allow_groups)
        print_success("  Group allowlist saved (only takes effect if iLink delivers group events).")

    if user_id:
        print()
        if prompt_yes_no(f"  Use your Weixin user ID ({user_id}) as the home channel?", True):
            save_env_value("WEIXIN_HOME_CHANNEL", user_id)
            print_success(f"  Home channel set to {user_id}")

    print()
    print_success("Weixin configured!")
    print_info(f"  Account ID: {account_id}")
    if user_id:
        print_info(f"  User ID: {user_id}")


def _setup_feishu():
    """Interactive setup for Feishu / Lark — scan-to-create or manual credentials."""
    print()
    print(color("  ─── 🪽 Feishu / Lark Setup ───", Colors.CYAN))

    existing_app_id = get_env_value("FEISHU_APP_ID")
    existing_secret = get_env_value("FEISHU_APP_SECRET")
    if existing_app_id and existing_secret:
        print()
        print_success("Feishu / Lark is already configured.")
        if not prompt_yes_no("  Reconfigure Feishu / Lark?", False):
            return

    # ── Choose setup method ──
    print()
    method_choices = [
        "Scan QR code to create a new bot automatically (recommended)",
        "Enter existing App ID and App Secret manually",
    ]
    method_idx = prompt_choice("  How would you like to set up Feishu / Lark?", method_choices, 0)

    credentials = None
    used_qr = False

    if method_idx == 0:
        # ── QR scan-to-create ──
        try:
            from gateway.platforms.feishu import qr_register
        except Exception as exc:
            print_error(f"  Feishu / Lark onboard import failed: {exc}")
            qr_register = None

        if qr_register is not None:
            try:
                credentials = qr_register()
            except KeyboardInterrupt:
                print()
                print_warning("  Feishu / Lark setup cancelled.")
                return
            except Exception as exc:
                print_warning(f"  QR registration failed: {exc}")
        if credentials:
            used_qr = True
        if not credentials:
            print_info("  QR setup did not complete. Continuing with manual input.")

    # ── Manual credential input ──
    if not credentials:
        print()
        print_info("  Go to https://open.feishu.cn/ (or https://open.larksuite.com/ for Lark)")
        print_info("  Create an app, enable the Bot capability, and copy the credentials.")
        print()
        app_id = prompt("  App ID", password=False)
        if not app_id:
            print_warning("  Skipped — Feishu / Lark won't work without an App ID.")
            return
        app_secret = prompt("  App Secret", password=True)
        if not app_secret:
            print_warning("  Skipped — Feishu / Lark won't work without an App Secret.")
            return

        domain_choices = ["feishu (China)", "lark (International)"]
        domain_idx = prompt_choice("  Domain", domain_choices, 0)
        domain = "lark" if domain_idx == 1 else "feishu"

        # Try to probe the bot with manual credentials
        bot_name = None
        try:
            from gateway.platforms.feishu import probe_bot
            bot_info = probe_bot(app_id, app_secret, domain)
            if bot_info:
                bot_name = bot_info.get("bot_name")
                print_success(f"  Credentials verified — bot: {bot_name or 'unnamed'}")
            else:
                print_warning("  Could not verify bot connection. Credentials saved anyway.")
        except Exception as exc:
            print_warning(f"  Credential verification skipped: {exc}")

        credentials = {
            "app_id": app_id,
            "app_secret": app_secret,
            "domain": domain,
            "open_id": None,
            "bot_name": bot_name,
        }

    # ── Save core credentials ──
    app_id = credentials["app_id"]
    app_secret = credentials["app_secret"]
    domain = credentials.get("domain", "feishu")
    open_id = credentials.get("open_id")
    bot_name = credentials.get("bot_name")

    save_env_value("FEISHU_APP_ID", app_id)
    save_env_value("FEISHU_APP_SECRET", app_secret)
    save_env_value("FEISHU_DOMAIN", domain)
    # Bot identity is resolved at runtime via _hydrate_bot_identity().

    # ── Connection mode ──
    if used_qr:
        connection_mode = "websocket"
    else:
        print()
        mode_choices = [
            "WebSocket (recommended — no public URL needed)",
            "Webhook (requires a reachable HTTP endpoint)",
        ]
        mode_idx = prompt_choice("  Connection mode", mode_choices, 0)
        connection_mode = "webhook" if mode_idx == 1 else "websocket"
        if connection_mode == "webhook":
            print_info("  Webhook defaults: 127.0.0.1:8765/feishu/webhook")
            print_info("  Override with FEISHU_WEBHOOK_HOST / FEISHU_WEBHOOK_PORT / FEISHU_WEBHOOK_PATH")
            print_info("  For signature verification, set FEISHU_ENCRYPT_KEY and FEISHU_VERIFICATION_TOKEN")
    save_env_value("FEISHU_CONNECTION_MODE", connection_mode)

    if bot_name:
        print()
        print_success(f"  Bot created: {bot_name}")

    # ── DM security policy ──
    print()
    access_choices = [
        "Use DM pairing approval (recommended)",
        "Allow all direct messages",
        "Only allow listed user IDs",
    ]
    access_idx = prompt_choice("  How should direct messages be authorized?", access_choices, 0)
    if access_idx == 0:
        save_env_value("FEISHU_ALLOW_ALL_USERS", "false")
        save_env_value("FEISHU_ALLOWED_USERS", "")
        print_success("  DM pairing enabled.")
        print_info("  Unknown users can request access; approve with `hermes pairing approve`.")
    elif access_idx == 1:
        save_env_value("FEISHU_ALLOW_ALL_USERS", "true")
        save_env_value("FEISHU_ALLOWED_USERS", "")
        print_warning("  Open DM access enabled for Feishu / Lark.")
    else:
        save_env_value("FEISHU_ALLOW_ALL_USERS", "false")
        default_allow = open_id or ""
        allowlist = prompt("  Allowed user IDs (comma-separated)", default_allow, password=False).replace(" ", "")
        save_env_value("FEISHU_ALLOWED_USERS", allowlist)
        print_success("  Allowlist saved.")

    # ── Group policy ──
    print()
    group_choices = [
        "Respond only when @mentioned in groups (recommended)",
        "Disable group chats",
    ]
    group_idx = prompt_choice("  How should group chats be handled?", group_choices, 0)
    if group_idx == 0:
        save_env_value("FEISHU_GROUP_POLICY", "open")
        print_info("  Group chats enabled (bot must be @mentioned).")
    else:
        save_env_value("FEISHU_GROUP_POLICY", "disabled")
        print_info("  Group chats disabled.")

    # ── Home channel ──
    print()
    home_channel = prompt("  Home chat ID (optional, for cron/notifications)", password=False)
    if home_channel:
        save_env_value("FEISHU_HOME_CHANNEL", home_channel)
        print_success(f"  Home channel set to {home_channel}")

    print()
    print_success("🪽 Feishu / Lark configured!")
    print_info(f"  App ID: {app_id}")
    print_info(f"  Domain: {domain}")
    if bot_name:
        print_info(f"  Bot: {bot_name}")


def _setup_qqbot():
    """Interactive setup for QQ Bot — scan-to-configure or manual credentials."""
    print()
    print(color("  ─── 🐧 QQ Bot Setup ───", Colors.CYAN))

    existing_app_id = get_env_value("QQ_APP_ID")
    existing_secret = get_env_value("QQ_CLIENT_SECRET")
    if existing_app_id and existing_secret:
        print()
        print_success("QQ Bot is already configured.")
        if not prompt_yes_no("  Reconfigure QQ Bot?", False):
            return

    # ── Choose setup method ──
    print()
    method_choices = [
        "Scan QR code to add bot automatically (recommended)",
        "Enter existing App ID and App Secret manually",
    ]
    method_idx = prompt_choice("  How would you like to set up QQ Bot?", method_choices, 0)

    credentials = None

    if method_idx == 0:
        # ── QR scan-to-configure ──
        try:
            from gateway.platforms.qqbot import qr_register
            credentials = qr_register()
        except KeyboardInterrupt:
            print()
            print_warning("  QQ Bot setup cancelled.")
            return
        if not credentials:
            print_info("  QR setup did not complete. Continuing with manual input.")

    # ── Manual credential input ──
    if not credentials:
        print()
        print_info("  Go to https://q.qq.com to register a QQ Bot application.")
        print_info("  Note your App ID and App Secret from the application page.")
        print()
        app_id = prompt("  App ID", password=False)
        if not app_id:
            print_warning("  Skipped — QQ Bot won't work without an App ID.")
            return
        app_secret = prompt("  App Secret", password=True)
        if not app_secret:
            print_warning("  Skipped — QQ Bot won't work without an App Secret.")
            return
        credentials = {"app_id": app_id.strip(), "client_secret": app_secret.strip(), "user_openid": ""}

    # ── Save core credentials ──
    save_env_value("QQ_APP_ID", credentials["app_id"])
    save_env_value("QQ_CLIENT_SECRET", credentials["client_secret"])

    user_openid = credentials.get("user_openid", "")

    # ── DM security policy ──
    print()
    access_choices = [
        "Use DM pairing approval (recommended)",
        "Allow all direct messages",
        "Only allow listed user OpenIDs",
    ]
    access_idx = prompt_choice("  How should direct messages be authorized?", access_choices, 0)
    if access_idx == 0:
        save_env_value("QQ_ALLOW_ALL_USERS", "false")
        if user_openid:
            print()
            if prompt_yes_no(f"  Add yourself ({user_openid}) to the allow list?", True):
                save_env_value("QQ_ALLOWED_USERS", user_openid)
                print_success(f"  Allow list set to {user_openid}")
            else:
                save_env_value("QQ_ALLOWED_USERS", "")
        else:
            save_env_value("QQ_ALLOWED_USERS", "")
        print_success("  DM pairing enabled.")
        print_info("  Unknown users can request access; approve with `hermes pairing approve`.")
    elif access_idx == 1:
        save_env_value("QQ_ALLOW_ALL_USERS", "true")
        save_env_value("QQ_ALLOWED_USERS", "")
        print_warning("  Open DM access enabled for QQ Bot.")
    else:
        default_allow = user_openid or ""
        allowlist = prompt("  Allowed user OpenIDs (comma-separated)", default_allow, password=False).replace(" ", "")
        save_env_value("QQ_ALLOW_ALL_USERS", "false")
        save_env_value("QQ_ALLOWED_USERS", allowlist)
        print_success("  Allowlist saved.")

    # ── Home channel ──
    if user_openid:
        print()
        if prompt_yes_no(f"  Use your QQ user ID ({user_openid}) as the home channel?", True):
            save_env_value("QQBOT_HOME_CHANNEL", user_openid)
            print_success(f"  Home channel set to {user_openid}")
    else:
        print()
        home_channel = prompt("  Home channel OpenID (for cron/notifications, or empty)", password=False)
        if home_channel:
            save_env_value("QQBOT_HOME_CHANNEL", home_channel.strip())
            print_success(f"  Home channel set to {home_channel.strip()}")

    print()
    print_success("🐧 QQ Bot configured!")
    print_info(f"  App ID: {credentials['app_id']}")


def _setup_signal():
    """Interactive setup for Signal messenger."""
    import shutil

    print()
    print(color("  ─── 📡 Signal Setup ───", Colors.CYAN))

    existing_url = get_env_value("SIGNAL_HTTP_URL")
    existing_account = get_env_value("SIGNAL_ACCOUNT")
    if existing_url and existing_account:
        print()
        print_success("Signal is already configured.")
        if not prompt_yes_no("  Reconfigure Signal?", False):
            return

    # Check if signal-cli is available
    print()
    if shutil.which("signal-cli"):
        print_success("signal-cli found on PATH.")
    else:
        print_warning("signal-cli not found on PATH.")
        print_info("  Signal requires signal-cli running as an HTTP daemon.")
        print_info("  Install options:")
        print_info("    Linux:  download from https://github.com/AsamK/signal-cli/releases")
        print_info("    macOS:  brew install signal-cli")
        print_info("    Docker: bbernhard/signal-cli-rest-api")
        print()
        print_info("  After installing, link your account and start the daemon:")
        print_info("    signal-cli link -n \"HermesAgent\"")
        print_info("    signal-cli --account +YOURNUMBER daemon --http 127.0.0.1:8080")
        print()

    # HTTP URL
    print()
    print_info("  Enter the URL where signal-cli HTTP daemon is running.")
    default_url = existing_url or "http://127.0.0.1:8080"
    try:
        url = input(f"  HTTP URL [{default_url}]: ").strip() or default_url
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return

    # Test connectivity
    print_info("  Testing connection...")
    try:
        import httpx
        resp = httpx.get(f"{url.rstrip('/')}/api/v1/check", timeout=10.0)
        if resp.status_code == 200:
            print_success("  signal-cli daemon is reachable!")
        else:
            print_warning(f"  signal-cli responded with status {resp.status_code}.")
            if not prompt_yes_no("  Continue anyway?", False):
                return
    except Exception as e:
        print_warning(f"  Could not reach signal-cli at {url}: {e}")
        if not prompt_yes_no("  Save this URL anyway? (you can start signal-cli later)", True):
            return

    save_env_value("SIGNAL_HTTP_URL", url)

    # Account phone number
    print()
    print_info("  Enter your Signal account phone number in E.164 format.")
    print_info("  Example: +15551234567")
    default_account = existing_account or ""
    try:
        account = input(f"  Account number{f' [{default_account}]' if default_account else ''}: ").strip()
        if not account:
            account = default_account
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return

    if not account:
        print_error("  Account number is required.")
        return

    save_env_value("SIGNAL_ACCOUNT", account)

    # Allowed users
    print()
    print_info("  The gateway DENIES all users by default for security.")
    print_info("  Enter phone numbers or UUIDs of allowed users (comma-separated).")
    existing_allowed = get_env_value("SIGNAL_ALLOWED_USERS") or ""
    default_allowed = existing_allowed or account
    try:
        allowed = input(f"  Allowed users [{default_allowed}]: ").strip() or default_allowed
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return

    save_env_value("SIGNAL_ALLOWED_USERS", allowed)

    # Group messaging
    print()
    if prompt_yes_no("  Enable group messaging? (disabled by default for security)", False):
        print()
        print_info("  Enter group IDs to allow, or * for all groups.")
        existing_groups = get_env_value("SIGNAL_GROUP_ALLOWED_USERS") or ""
        try:
            groups = input(f"  Group IDs [{existing_groups or '*'}]: ").strip() or existing_groups or "*"
        except (EOFError, KeyboardInterrupt):
            print("\n  Setup cancelled.")
            return
        save_env_value("SIGNAL_GROUP_ALLOWED_USERS", groups)

    print()
    print_success("Signal configured!")
    print_info(f"  URL: {url}")
    print_info(f"  Account: {account}")
    print_info("  DM auth: via SIGNAL_ALLOWED_USERS + DM pairing")
    print_info(f"  Groups: {'enabled' if get_env_value('SIGNAL_GROUP_ALLOWED_USERS') else 'disabled'}")


def _builtin_setup_fn(key: str):
    """Resolve the interactive setup function for a built-in platform key.

    Late-bound to avoid a circular import with ``hermes_cli.setup`` (which
    imports from this module for the remaining bespoke flows).
    """
    from hermes_cli import setup as _s
    return {
        "telegram": _s._setup_telegram,
        "discord": _s._setup_discord,
        "slack": _s._setup_slack,
        "matrix": _s._setup_matrix,
        "mattermost": _s._setup_mattermost,
        "bluebubbles": _s._setup_bluebubbles,
        "webhooks": _s._setup_webhooks,
        "signal": _setup_signal,
        "whatsapp": _setup_whatsapp,
        "weixin": _setup_weixin,
        "dingtalk": _setup_dingtalk,
        "feishu": _setup_feishu,
        "wecom": _setup_wecom,
        "qqbot": _setup_qqbot,
    }.get(key)
def _configure_platform(platform: dict) -> None:
    """Run the interactive setup flow for a single platform.

    Dispatch order:
      1. Plugin-provided ``setup_fn`` on the registry entry.
      2. Built-in setup function matched by platform key.
      3. ``_setup_standard_platform`` when the entry has a ``vars`` schema.
      4. Env-var hint fallback for plugins that offer no setup helper.

    Bundled platform plugins (e.g. IRC) auto-load, so no plugin enable step
    is needed here. User-installed platform plugins under ~/.hermes/plugins/
    must already be in ``plugins.enabled`` before they appear in this menu.
    """
    entry = platform.get("_registry_entry")

    if entry is not None and entry.setup_fn is not None:
        entry.setup_fn()
        return

    fn = _builtin_setup_fn(platform["key"])
    if fn is not None:
        fn()
        return

    if platform.get("vars"):
        _setup_standard_platform(platform)
        return

    # Plugin with no setup helper — show env-var instructions.
    label = platform.get("label", platform["key"])
    emoji = platform.get("emoji", "🔌")
    print()
    print(color(f"  ─── {emoji} {label} Setup ───", Colors.CYAN))
    required = entry.required_env if entry else []
    if required:
        print_info(f"  Set these env vars in ~/.hermes/.env: {', '.join(required)}")
    else:
        print_info(f"  Configure {label} in config.yaml under gateway.platforms.{platform['key']}")
    if platform.get("install_hint"):
        print_info(f"  {platform['install_hint']}")


def gateway_setup():
    """Interactive setup for messaging platforms + gateway service."""
    if is_managed():
        managed_error("run gateway setup")
        return

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA))
    print(color("│             ⚕ Gateway Setup                            │", Colors.MAGENTA))
    print(color("├─────────────────────────────────────────────────────────┤", Colors.MAGENTA))
    print(color("│  Configure messaging platforms and the gateway service. │", Colors.MAGENTA))
    print(color("│  Press Ctrl+C at any time to exit.                     │", Colors.MAGENTA))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA))

    # ── Gateway service status ──
    print()
    service_installed = _is_service_installed()
    service_running = _is_service_running()

    if supports_systemd_services() and has_conflicting_systemd_units():
        print_systemd_scope_conflict_warning()
        print()

    if supports_systemd_services() and has_legacy_hermes_units():
        print_legacy_unit_warning()
        print()

    if service_installed and service_running:
        print_success("Gateway service is installed and running.")
    elif service_installed:
        print_warning("Gateway service is installed but not running.")
        if supports_systemd_services() and _system_scope_wizard_would_need_root():
            _print_system_scope_remediation("start")
        elif prompt_yes_no("  Start it now?", True):
            try:
                if supports_systemd_services():
                    systemd_start()
                elif is_macos():
                    launchd_start()
            except UserSystemdUnavailableError as e:
                print_error("  Failed to start — user systemd not reachable:")
                for line in str(e).splitlines():
                    print(f"  {line}")
            except SystemScopeRequiresRootError as e:
                # Defense in depth: the pre-check above should have caught
                # this, but handle the race/edge case gracefully instead of
                # letting the exception escape the wizard.
                print_error(f"  Failed to start: {e}")
                _print_system_scope_remediation("start")
            except subprocess.CalledProcessError as e:
                print_error(f"  Failed to start: {e}")
    else:
        print_info("Gateway service is not installed yet.")
        print_info("You'll be offered to install it after configuring platforms.")

    # ── Platform configuration loop ──
    while True:
        print()
        print_header("Messaging Platforms")

        platforms = _all_platforms()

        menu_items = [
            f"{p['emoji']} {p['label']}  ({_platform_status(p)})"
            for p in platforms
        ]
        menu_items.append("Done")

        choice = prompt_choice("Select a platform to configure:", menu_items, len(menu_items) - 1)
        if choice == len(platforms):
            break

        _configure_platform(platforms[choice])

    # ── Post-setup: offer to install/restart gateway ──
    # Consider any platform (built-in or plugin) where the user has made
    # meaningful progress.  ``_platform_status`` already handles plugin
    # entries via their check_fn and per-platform dual-states like
    # WhatsApp's "enabled, not paired".
    def _is_progress(status: str) -> bool:
        s = status.lower()
        return not (
            s == "not configured"
            or s.startswith("partially")
            or s.startswith("plugin disabled")
        )

    any_configured = any(
        _is_progress(_platform_status(p)) for p in _all_platforms()
    )

    if any_configured:
        print()
        print(color("─" * 58, Colors.DIM))
        service_installed = _is_service_installed()
        service_running = _is_service_running()

        if service_running:
            if supports_systemd_services() and _system_scope_wizard_would_need_root():
                _print_system_scope_remediation("restart")
            elif prompt_yes_no("  Restart the gateway to pick up changes?", True):
                try:
                    if supports_systemd_services():
                        systemd_restart()
                    elif is_macos():
                        launchd_restart()
                    else:
                        stop_profile_gateway()
                        print_info("Start manually: hermes gateway")
                except UserSystemdUnavailableError as e:
                    print_error("  Restart failed — user systemd not reachable:")
                    for line in str(e).splitlines():
                        print(f"  {line}")
                except SystemScopeRequiresRootError as e:
                    print_error(f"  Restart failed: {e}")
                    _print_system_scope_remediation("restart")
                except subprocess.CalledProcessError as e:
                    print_error(f"  Restart failed: {e}")
        elif service_installed:
            if supports_systemd_services() and _system_scope_wizard_would_need_root():
                _print_system_scope_remediation("start")
            elif prompt_yes_no("  Start the gateway service?", True):
                try:
                    if supports_systemd_services():
                        systemd_start()
                    elif is_macos():
                        launchd_start()
                except UserSystemdUnavailableError as e:
                    print_error("  Start failed — user systemd not reachable:")
                    for line in str(e).splitlines():
                        print(f"  {line}")
                except SystemScopeRequiresRootError as e:
                    print_error(f"  Start failed: {e}")
                    _print_system_scope_remediation("start")
                except subprocess.CalledProcessError as e:
                    print_error(f"  Start failed: {e}")
        else:
            print()
            if supports_systemd_services() or is_macos():
                platform_name = "systemd" if supports_systemd_services() else "launchd"
                wsl_note = " (note: services may not survive WSL restarts)" if is_wsl() else ""
                if prompt_yes_no(f"  Install the gateway as a {platform_name} service?{wsl_note} (runs in background, starts on boot)", True):
                    try:
                        installed_scope = None
                        did_install = False
                        if supports_systemd_services():
                            installed_scope, did_install = install_linux_gateway_from_setup(force=False)
                        else:
                            launchd_install(force=False)
                            did_install = True
                        print()
                        if did_install and prompt_yes_no("  Start the service now?", True):
                            try:
                                if supports_systemd_services():
                                    systemd_start(system=installed_scope == "system")
                                else:
                                    launchd_start()
                            except UserSystemdUnavailableError as e:
                                print_error("  Start failed — user systemd not reachable:")
                                for line in str(e).splitlines():
                                    print(f"  {line}")
                            except subprocess.CalledProcessError as e:
                                print_error(f"  Start failed: {e}")
                    except subprocess.CalledProcessError as e:
                        print_error(f"  Install failed: {e}")
                        print_info("  You can try manually: hermes gateway install")
                else:
                    print_info("  You can install later: hermes gateway install")
                    if supports_systemd_services():
                        print_info("  Or as a boot-time service: sudo hermes gateway install --system")
                    print_info("  Or run in foreground:  hermes gateway run")
            elif is_wsl():
                print_info("  WSL detected but systemd is not running.")
                print_info("  Run in foreground: hermes gateway run")
                print_info("  For persistence:   tmux new -s hermes 'hermes gateway run'")
                print_info("  To enable systemd: add systemd=true to /etc/wsl.conf, then 'wsl --shutdown'")
            else:
                if is_termux():
                    from hermes_constants import display_hermes_home as _dhh
                    print_info("  Termux does not use systemd/launchd services.")
                    print_info("  Run in foreground: hermes gateway run")
                    print_info(f"  Or start it manually in the background (best effort): nohup hermes gateway run >{_dhh()}/logs/gateway.log 2>&1 &")
                else:
                    print_info("  Service install not supported on this platform.")
                    print_info("  Run in foreground: hermes gateway run")
    else:
        print()
        print_info("No platforms configured. Run 'hermes gateway setup' when ready.")

    print()


# =============================================================================
# Main Command Handler
# =============================================================================

def gateway_command(args):
    """Handle gateway subcommands."""
    try:
        return _gateway_command_inner(args)
    except UserSystemdUnavailableError as e:
        # Clean, actionable message instead of a traceback when the user D-Bus
        # session is unreachable (fresh SSH shell, no linger, container, etc.).
        print_error("User systemd not reachable:")
        for line in str(e).splitlines():
            print(f"  {line}")
        sys.exit(1)
    except SystemScopeRequiresRootError as e:
        # The direct ``hermes gateway install|uninstall|start|stop|restart``
        # path lands here when the user typed a system-scope action without
        # sudo. Same exit code as before — just gives the wizard a way to
        # intercept the same condition with friendlier guidance before the
        # error is raised.
        print(str(e))
        sys.exit(1)


def _gateway_command_inner(args):
    subcmd = getattr(args, 'gateway_command', None)
    
    # Default to run if no subcommand
    if subcmd is None or subcmd == "run":
        verbose = getattr(args, 'verbose', 0)
        quiet = getattr(args, 'quiet', False)
        replace = getattr(args, 'replace', False)
        run_gateway(verbose, quiet=quiet, replace=replace)
        return

    if subcmd == "setup":
        gateway_setup()
        return

    # Service management commands
    if subcmd == "install":
        if is_managed():
            managed_error("install gateway service (managed by NixOS)")
            return
        force = getattr(args, 'force', False)
        system = getattr(args, 'system', False)
        run_as_user = getattr(args, 'run_as_user', None)
        if is_termux():
            print("Gateway service installation is not supported on Termux.")
            print("Run manually: hermes gateway")
            sys.exit(1)
        if supports_systemd_services():
            if is_wsl():
                print_warning("WSL detected — systemd services may not survive WSL restarts.")
                print_info("  Consider running in foreground instead: hermes gateway run")
                print_info("  Or use tmux/screen for persistence: tmux new -s hermes 'hermes gateway run'")
                print()
            systemd_install(force=force, system=system, run_as_user=run_as_user)
        elif is_macos():
            launchd_install(force)
        elif is_wsl():
            print("WSL detected but systemd is not running.")
            print("Either enable systemd (add systemd=true to /etc/wsl.conf and restart WSL)")
            print("or run the gateway in foreground mode:")
            print()
            print("  hermes gateway run                              # direct foreground")
            print("  tmux new -s hermes 'hermes gateway run'         # persistent via tmux")
            print("  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # background")
            sys.exit(1)
        elif is_container():
            print("Service installation is not needed inside a Docker container.")
            print("The container runtime is your service manager — use Docker restart policies instead:")
            print()
            print("  docker run --restart unless-stopped ...   # auto-restart on crash/reboot")
            print("  docker restart <container>                # manual restart")
            print()
            print("To run the gateway: hermes gateway run")
            sys.exit(0)
        else:
            print("Service installation not supported on this platform.")
            print("Run manually: hermes gateway run")
            sys.exit(1)
    
    elif subcmd == "uninstall":
        if is_managed():
            managed_error("uninstall gateway service (managed by NixOS)")
            return
        system = getattr(args, 'system', False)
        if is_termux():
            print("Gateway service uninstall is not supported on Termux because there is no managed service to remove.")
            print("Stop manual runs with: hermes gateway stop")
            sys.exit(1)
        if supports_systemd_services():
            systemd_uninstall(system=system)
        elif is_macos():
            launchd_uninstall()
        elif is_container():
            print("Service uninstall is not applicable inside a Docker container.")
            print("To stop the gateway, stop or remove the container:")
            print()
            print("  docker stop <container>")
            print("  docker rm <container>")
            sys.exit(0)
        else:
            print("Not supported on this platform.")
            sys.exit(1)

    elif subcmd == "start":
        system = getattr(args, 'system', False)
        start_all = getattr(args, 'all', False)

        if start_all:
            # Kill all stale gateway processes across all profiles before starting
            killed = kill_gateway_processes(all_profiles=True)
            if killed:
                print(f"✓ Killed {killed} stale gateway process(es) across all profiles")
                _wait_for_gateway_exit(timeout=10.0, force_after=5.0)

        if is_termux():
            print("Gateway service start is not supported on Termux because there is no system service manager.")
            print("Run manually: hermes gateway")
            sys.exit(1)
        if supports_systemd_services():
            systemd_start(system=system)
        elif is_macos():
            launchd_start()
        elif is_wsl():
            print("WSL detected but systemd is not available.")
            print("Run the gateway in foreground mode instead:")
            print()
            print("  hermes gateway run                              # direct foreground")
            print("  tmux new -s hermes 'hermes gateway run'         # persistent via tmux")
            print("  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # background")
            print()
            print("To enable systemd: add systemd=true to /etc/wsl.conf and run 'wsl --shutdown' from PowerShell.")
            sys.exit(1)
        elif is_container():
            print("Service start is not applicable inside a Docker container.")
            print("The gateway runs as the container's main process.")
            print()
            print("  docker start <container>     # start a stopped container")
            print("  docker restart <container>   # restart a running container")
            print()
            print("Or run the gateway directly: hermes gateway run")
            sys.exit(0)
        else:
            print("Not supported on this platform.")
            sys.exit(1)

    elif subcmd == "stop":
        stop_all = getattr(args, 'all', False)
        system = getattr(args, 'system', False)

        if stop_all:
            # --all: kill every gateway process on the machine
            service_available = False
            if supports_systemd_services() and (get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()):
                try:
                    systemd_stop(system=system)
                    service_available = True
                except subprocess.CalledProcessError:
                    pass
            elif is_macos() and get_launchd_plist_path().exists():
                try:
                    launchd_stop()
                    service_available = True
                except subprocess.CalledProcessError:
                    pass
            killed = kill_gateway_processes(all_profiles=True)
            total = killed + (1 if service_available else 0)
            if total:
                print(f"✓ Stopped {total} gateway process(es) across all profiles")
            else:
                print("✗ No gateway processes found")
        else:
            # Default: stop only the current profile's gateway
            service_available = False
            if supports_systemd_services() and (get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()):
                try:
                    systemd_stop(system=system)
                    service_available = True
                except subprocess.CalledProcessError:
                    pass
            elif is_macos() and get_launchd_plist_path().exists():
                try:
                    launchd_stop()
                    service_available = True
                except subprocess.CalledProcessError:
                    pass

            if not service_available:
                # No systemd/launchd — use profile-scoped PID file
                if stop_profile_gateway():
                    print("✓ Stopped gateway for this profile")
                else:
                    print("✗ No gateway running for this profile")
            else:
                print(f"✓ Stopped {get_service_name()} service")
    
    elif subcmd == "restart":
        # Try service first, fall back to killing and restarting
        service_available = False
        system = getattr(args, 'system', False)
        restart_all = getattr(args, 'all', False)
        service_configured = False

        if restart_all:
            # --all: stop every gateway process across all profiles, then start fresh
            service_stopped = False
            if supports_systemd_services() and (get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()):
                try:
                    systemd_stop(system=system)
                    service_stopped = True
                except subprocess.CalledProcessError:
                    pass
            elif is_macos() and get_launchd_plist_path().exists():
                try:
                    launchd_stop()
                    service_stopped = True
                except subprocess.CalledProcessError:
                    pass
            killed = kill_gateway_processes(all_profiles=True)
            total = killed + (1 if service_stopped else 0)
            if total:
                print(f"✓ Stopped {total} gateway process(es) across all profiles")
            _wait_for_gateway_exit(timeout=10.0, force_after=5.0)

            # Start the current profile's service fresh
            print("Starting gateway...")
            if supports_systemd_services() and (get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()):
                systemd_start(system=system)
            elif is_macos() and get_launchd_plist_path().exists():
                launchd_start()
            else:
                run_gateway(verbose=0)
            return
        
        if supports_systemd_services() and (get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()):
            service_configured = True
            try:
                systemd_restart(system=system)
                service_available = True
            except subprocess.CalledProcessError:
                pass
        elif is_macos() and get_launchd_plist_path().exists():
            service_configured = True
            try:
                launchd_restart()
                service_available = True
            except subprocess.CalledProcessError:
                pass
        
        if not service_available:
            # systemd/launchd restart failed — check if linger is the issue
            if supports_systemd_services():
                linger_ok, _detail = get_systemd_linger_status()
                if linger_ok is not True:
                    import getpass
                    _username = getpass.getuser()
                    print()
                    print("⚠ Cannot restart gateway as a service — linger is not enabled.")
                    print("  The gateway user service requires linger to function on headless servers.")
                    print()
                    print(f"  Run:  sudo loginctl enable-linger {_username}")
                    print()
                    print("  Then restart the gateway:")
                    print("    hermes gateway restart")
                    return

            if service_configured:
                print()
                print("✗ Gateway service restart failed.")
                print("  The service definition exists, but the service manager did not recover it.")
                print("  Fix the service, then retry: hermes gateway start")
                sys.exit(1)

            # Manual restart: stop only this profile's gateway
            if stop_profile_gateway():
                print("✓ Stopped gateway for this profile")

            _wait_for_gateway_exit(timeout=10.0, force_after=5.0)

            # Start fresh
            print("Starting gateway...")
            run_gateway(verbose=0)
    
    elif subcmd == "status":
        deep = getattr(args, 'deep', False)
        full = getattr(args, 'full', False)
        system = getattr(args, 'system', False)
        snapshot = get_gateway_runtime_snapshot(system=system)
        
        # Check for service first
        if supports_systemd_services() and (get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()):
            systemd_status(deep, system=system, full=full)
            _print_gateway_process_mismatch(snapshot)
        elif is_macos() and get_launchd_plist_path().exists():
            launchd_status(deep)
            _print_gateway_process_mismatch(snapshot)
        else:
            # Check for manually running processes
            pids = list(snapshot.gateway_pids)
            if pids:
                print(f"✓ Gateway is running (PID: {', '.join(map(str, pids))})")
                print("  (Running manually, not as a system service)")
                runtime_lines = _runtime_health_lines()
                if runtime_lines:
                    print()
                    print("Recent gateway health:")
                    for line in runtime_lines:
                        print(f"  {line}")
                print()
                if is_termux():
                    print("Termux note:")
                    print("  Android may stop background jobs when Termux is suspended")
                elif is_wsl():
                    print("WSL note:")
                    print("  The gateway is running in foreground/manual mode (recommended for WSL).")
                    print("  Use tmux or screen for persistence across terminal closes.")
                else:
                    print("To install as a service:")
                    print("  hermes gateway install")
                    print("  sudo hermes gateway install --system")
            else:
                print("✗ Gateway is not running")
                runtime_lines = _runtime_health_lines()
                if runtime_lines:
                    print()
                    print("Recent gateway health:")
                    for line in runtime_lines:
                        print(f"  {line}")
                print()
                print("To start:")
                print("  hermes gateway run      # Run in foreground")
                if is_termux():
                    print("  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # Best-effort background start")
                elif is_wsl():
                    print("  tmux new -s hermes 'hermes gateway run'         # persistent via tmux")
                    print("  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # background")
                else:
                    print("  hermes gateway install  # Install as user service")
                    print("  sudo hermes gateway install --system  # Install as boot-time system service")

        # Show other profiles' gateway status for multi-profile awareness
        _print_other_profiles_gateway_status()

    elif subcmd == "migrate-legacy":
        # Stop, disable, and remove legacy Hermes gateway unit files from
        # pre-rename installs (e.g. hermes.service). Profile units and
        # unrelated third-party services are never touched.
        dry_run = getattr(args, 'dry_run', False)
        yes = getattr(args, 'yes', False)
        if not supports_systemd_services() and not is_macos():
            print("Legacy unit migration only applies to systemd-based Linux hosts.")
            return
        remove_legacy_hermes_units(interactive=not yes, dry_run=dry_run)

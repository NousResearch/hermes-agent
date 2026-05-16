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
    """Return the parent PID for ``pid``, or ``None`` when unavailable.

    Uses psutil (core dependency) which works on every platform.  The
    older implementation shelled out to ``ps -o ppid= -p <pid>``, which
    silently fails on Windows (no ``ps``) so the ancestor walk terminated
    at self — the caller's dedup / exclude logic then couldn't distinguish
    "hermes CLI that invoked this scan" from "real gateway process".
    """
    if pid <= 1:
        return None
    try:
        import psutil  # type: ignore
        return psutil.Process(pid).ppid() or None
    except ImportError:
        pass
    except Exception:
        return None
    # Fallback: shell out to ps (POSIX only — bare ``ps`` doesn't exist on Windows).
    if not shutil.which("ps"):
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
        os.kill(pid, signal.SIGUSR1)  # windows-footgun: ok — POSIX signal, guarded by hasattr(signal, 'SIGUSR1') above
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
        os.kill(pid, signal.SIGUSR1)  # windows-footgun: ok — POSIX signal, guarded by hasattr(signal, 'SIGUSR1') above
    except ProcessLookupError:
        # Already gone — nothing to drain.
        return True
    except (PermissionError, OSError):
        return False

    import time as _time

    deadline = _time.monotonic() + max(drain_timeout, 1.0)
    # IMPORTANT Windows note: ``os.kill(pid, 0)`` is NOT a no-op on
    # Windows — Python's implementation calls ``TerminateProcess(handle, 0)``
    # for sig=0, hard-killing the target. Use the cross-platform
    # ``_pid_exists`` helper in gateway.status which does OpenProcess +
    # WaitForSingleObject on Windows.
    from gateway.status import _pid_exists

    while _time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
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

        # Default-profile case: no profile flag in argv. Accept as long as
        # the command doesn't advertise *some other* profile. HERMES_HOME
        # may be passed via env (not visible in wmic/CIM command line) so
        # its absence is NOT disqualifying — only a non-matching explicit
        # HERMES_HOME= in argv is.
        if "--profile " in command or " -p " in command:
            return False
        if "HERMES_HOME=" in command and f"HERMES_HOME={current_home}" not in command:
            return False
        return True

    try:
        if is_windows():
            # Prefer wmic when present (fast, stable output format).  On
            # modern Windows 11 / Win 10 late builds, wmic has been
            # removed as part of the WMIC deprecation — fall back to
            # PowerShell's Get-CimInstance.  Any OSError here (FileNotFoundError
            # on missing wmic) trips the fallback.
            wmic_path = shutil.which("wmic")
            used_fallback = False
            result = None
            if wmic_path is not None:
                try:
                    result = subprocess.run(
                        [wmic_path, "process", "get", "ProcessId,CommandLine", "/FORMAT:LIST"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="ignore",
                        timeout=10,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    result = None
            if result is None or result.returncode != 0 or not (result.stdout or ""):
                # Fallback: PowerShell Get-CimInstance, emit LIST-style output
                # so the downstream parser below doesn't need to branch.
                powershell = shutil.which("powershell") or shutil.which("pwsh")
                if powershell is None:
                    return []
                ps_cmd = (
                    "Get-CimInstance Win32_Process | "
                    "ForEach-Object { "
                    "  'CommandLine=' + ($_.CommandLine -replace \"`r`n\",' ' -replace \"`n\",' '); "
                    "  'ProcessId=' + $_.ProcessId; "
                    "  '' "
                    "}"
                )
                try:
                    result = subprocess.run(
                        [powershell, "-NoProfile", "-Command", ps_cmd],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="ignore",
                        timeout=15,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    return []
                used_fallback = True
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
            # Try /proc first (works in Docker without procps installed),
            # fall back to ps -A eww.
            _found_via_proc = False
            if os.path.isdir("/proc"):
                try:
                    my_pid = os.getpid()
                    for entry in os.listdir("/proc"):
                        if not entry.isdigit():
                            continue
                        pid = int(entry)
                        if pid == my_pid or pid in exclude_pids:
                            continue
                        try:
                            cmdline = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", errors="replace")
                            cmdline = cmdline.replace("\x00", " ")
                            if any(p in cmdline for p in patterns) and (
                                all_profiles or _matches_current_profile(cmdline)
                            ):
                                _append_unique_pid(pids, pid, exclude_pids)
                        except (OSError, PermissionError):
                            continue
                    _found_via_proc = True
                except Exception:
                    pass

            if not _found_via_proc:
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

    # Windows-specific: collapse venv launcher stubs.  A venv-built
    # ``pythonw.exe`` in ``<venv>/Scripts/`` is a ~100 KB launcher exe
    # that spawns the base Python (e.g. ``C:\Program Files\Python311\
    # pythonw.exe``) with the same command line, preserving the venv's
    # ``pyvenv.cfg`` context.  This is standard Windows CPython venv
    # behaviour — BUT it means every gateway run produces two pythonw
    # PIDs with identical command lines (one launcher stub, one actual
    # interpreter) which is confusing in ``gateway status`` output.
    # Filter the stub: if a PID in our result is the PARENT of another
    # PID in our result, and both are pythonw.exe, the parent is the
    # launcher stub — drop it, keep the child.
    if is_windows() and len(pids) > 1:
        pids = _filter_venv_launcher_stubs(pids)

    return pids


def _filter_venv_launcher_stubs(pids: list[int]) -> list[int]:
    """Drop venv-launcher ``pythonw.exe`` stubs that are parents of the real
    interpreter process.  See comment at the tail of ``_scan_gateway_pids``.

    Uses ``psutil`` (core dependency).  Safe on any platform; only invoked
    on Windows by the caller because the stub pattern is Windows-specific.
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        return pids

    pid_set = set(pids)
    # Collect each PID's parent so we can flag "child of another matched PID".
    parent_of: dict[int, int | None] = {}
    for pid in pids:
        try:
            parent_of[pid] = psutil.Process(pid).ppid()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            parent_of[pid] = None

    # For each child whose parent is also in our set, drop the parent.
    drop: set[int] = set()
    for pid, ppid in parent_of.items():
        if ppid is not None and ppid in pid_set:
            drop.add(ppid)

    return [p for p in pids if p not in drop]


def find_gateway_pids(exclude_pids: set | None = None, all_profiles: bool = False) -> list:
    """Find PIDs of running gateway processes.

    Args:
        exclude_pids: PIDs to exclude from the result (e.g. service-managed
            PIDs that should not be killed during a stale-process sweep).
        all_profiles: When ``True``, return gateway PIDs across **all**
            

... [OUTPUT TRUNCATED - 172004 chars omitted out of 222004 total] ...

ry else []
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
                    elif is_windows():
                        from hermes_cli import gateway_windows
                        gateway_windows.restart()
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
                    elif is_windows():
                        from hermes_cli import gateway_windows
                        gateway_windows.start()
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
            if supports_systemd_services() or is_macos() or is_windows():
                if supports_systemd_services():
                    platform_name = "systemd"
                elif is_macos():
                    platform_name = "launchd"
                else:
                    platform_name = "Scheduled Task"
                wsl_note = " (note: services may not survive WSL restarts)" if is_wsl() else ""
                if prompt_yes_no(f"  Install the gateway as a {platform_name} service?{wsl_note} (runs in background, starts on boot)", True):
                    try:
                        installed_scope = None
                        did_install = False
                        started_inline = False
                        if supports_systemd_services():
                            installed_scope, did_install = install_linux_gateway_from_setup(force=False)
                        elif is_macos():
                            launchd_install(force=False)
                            did_install = True
                        else:
                            # gateway_windows.install() registers the Scheduled
                            # Task AND starts it (schtasks /Run or direct-spawn
                            # fallback), so no separate start prompt is needed.
                            from hermes_cli import gateway_windows
                            gateway_windows.install(force=False)
                            did_install = True
                            started_inline = True
                        print()
                        if did_install and not started_inline and prompt_yes_no("  Start the service now?", True):
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
            elif is_termux():
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
        elif is_windows():
            from hermes_cli import gateway_windows
            gateway_windows.install(force=force)
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
        elif is_windows():
            from hermes_cli import gateway_windows
            gateway_windows.uninstall()
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
        elif is_windows():
            from hermes_cli import gateway_windows
            gateway_windows.start()
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
            elif is_windows():
                from hermes_cli import gateway_windows
                if gateway_windows.is_installed():
                    try:
                        gateway_windows.stop()
                        service_available = True
                    except (subprocess.CalledProcessError, RuntimeError):
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
            elif is_windows():
                from hermes_cli import gateway_windows
                if gateway_windows.is_installed():
                    try:
                        gateway_windows.stop()
                        service_available = True
                    except (subprocess.CalledProcessError, RuntimeError):
                        pass

            if not service_available:
                # No systemd/launchd/schtasks service — use profile-scoped PID file
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
            elif is_windows():
                from hermes_cli import gateway_windows
                if gateway_windows.is_installed():
                    try:
                        gateway_windows.stop()
                        service_stopped = True
                    except (subprocess.CalledProcessError, RuntimeError):
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
            elif is_windows():
                from hermes_cli import gateway_windows
                if gateway_windows.is_installed():
                    gateway_windows.start()
                else:
                    run_gateway(verbose=0)
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
        elif is_windows():
            from hermes_cli import gateway_windows
            if gateway_windows.is_installed():
                service_configured = True
                try:
                    gateway_windows.restart()
                    service_available = True
                except (subprocess.CalledProcessError, RuntimeError):
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
        _windows_service_installed = False
        if is_windows():
            from hermes_cli import gateway_windows
            _windows_service_installed = gateway_windows.is_installed()
        if supports_systemd_services() and (get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()):
            systemd_status(deep, system=system, full=full)
            _print_gateway_process_mismatch(snapshot)
        elif is_macos() and get_launchd_plist_path().exists():
            launchd_status(deep)
            _print_gateway_process_mismatch(snapshot)
        elif _windows_service_installed:
            from hermes_cli import gateway_windows
            gateway_windows.status(deep=deep)
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
                elif is_windows():
                    print("To install as a Windows Scheduled Task (auto-start on login):")
                    print("  hermes gateway install")
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
                elif is_windows():
                    print("  hermes gateway install  # Install as Windows Scheduled Task (auto-start on login)")
                else:
                    print("  hermes gateway install  # Install as user service")
                    print("  sudo hermes gateway install --system  # Install as boot-time system service")

        # Show other profiles' gateway status for multi-profile awareness
        _print_other_profiles_gateway_status()

    elif subcmd == "list":
        _gateway_list()

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
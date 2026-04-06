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
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from hermes_cli.config import get_env_value, get_hermes_home, save_env_value, is_managed, managed_error
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

def _get_service_pids() -> set:
    """Return PIDs currently managed by systemd or launchd gateway services.

    Used to avoid killing freshly-restarted service processes when sweeping
    for stale manual gateway processes after a service restart.
    """
    pids: set = set()

    # --- systemd (Linux) ---
    if is_linux():
        try:
            result = subprocess.run(
                ["systemctl", "--user", "list-units", "hermes-gateway*",
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
                        ["systemctl", "--user", "show", svc,
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

        # Also check system scope
        try:
            result = subprocess.run(
                ["systemctl", "list-units", "hermes-gateway*",
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
                        ["systemctl", "show", svc,
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
            from hermes_cli.gateway import get_launchd_label
            result = subprocess.run(
                ["launchctl", "list", get_launchd_label()],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                # Output format: "PID\tStatus\tLabel" header then data line
                for line in result.stdout.strip().splitlines():
                    parts = line.split()
                    if parts:
                        try:
                            pid = int(parts[0])
                            if pid > 0:
                                pids.add(pid)
                        except ValueError:
                            pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return pids


def find_gateway_pids(exclude_pids: set | None = None) -> list:
    """Find PIDs of running gateway processes.

    Args:
        exclude_pids: PIDs to exclude from the result (e.g. service-managed
            PIDs that should not be killed during a stale-process sweep).
    """
    pids = []
    _exclude = exclude_pids or set()
    patterns = [
        "hermes_cli.main gateway",
        "hermes_cli/main.py gateway",
        "hermes gateway",
        "gateway/run.py",
    ]

    try:
        if is_windows():
            # Windows: use wmic to search command lines
            result = subprocess.run(
                ["wmic", "process", "get", "ProcessId,CommandLine", "/FORMAT:LIST"],
                capture_output=True, text=True, timeout=10
            )
            # Parse WMIC LIST output: blocks of "CommandLine=...\nProcessId=...\n"
            current_cmd = ""
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line.startswith("CommandLine="):
                    current_cmd = line[len("CommandLine="):]
                elif line.startswith("ProcessId="):
                    pid_str = line[len("ProcessId="):]
                    if any(p in current_cmd for p in patterns):
                        try:
                            pid = int(pid_str)
                            if pid != os.getpid() and pid not in pids and pid not in _exclude:
                                pids.append(pid)
                        except ValueError:
                            pass
                    current_cmd = ""
        else:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.split('\n'):
                # Skip grep and current process
                if 'grep' in line or str(os.getpid()) in line:
                    continue
                for pattern in patterns:
                    if pattern in line:
                        parts = line.split()
                        if len(parts) > 1:
                            try:
                                pid = int(parts[1])
                                if pid not in pids and pid not in _exclude:
                                    pids.append(pid)
                            except ValueError:
                                continue
                        break
    except Exception:
        pass

    return pids


def kill_gateway_processes(force: bool = False, exclude_pids: set | None = None) -> int:
    """Kill any running gateway processes. Returns count killed.

    Args:
        force: Use SIGKILL instead of SIGTERM.
        exclude_pids: PIDs to skip (e.g. service-managed PIDs that were just
            restarted and should not be killed).
    """
    pids = find_gateway_pids(exclude_pids=exclude_pids)
    killed = 0
    
    for pid in pids:
        try:
            if force and not is_windows():
                os.kill(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            # Process already gone
            pass
        except PermissionError:
            print(f"⚠ Permission denied to kill PID {pid}")
    
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

    remove_pid_file()
    return True


def is_linux() -> bool:
    return sys.platform.startswith('linux')

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

    Returns ``""`` for the default ``~/.hermes``, the profile name for
    ``~/.hermes/profiles/<name>``, or a short hash for any other custom
    HERMES_HOME path.
    """
    import hashlib
    import re
    from pathlib import Path as _Path
    home = get_hermes_home().resolve()
    default = (_Path.home() / ".hermes").resolve()
    if home == default:
        return ""
    # Detect ~/.hermes/profiles/<name> pattern → use the profile name
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


SERVICE_NAME = _SERVICE_BASE  # backward-compat for external importers; prefer get_service_name()


def get_systemd_unit_path(system: bool = False) -> Path:
    name = get_service_name()
    if system:
        return Path("/etc/systemd/system") / f"{name}.service"
    return Path.home() / ".config" / "systemd" / "user" / f"{name}.service"


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


def _systemctl_cmd(system: bool = False) -> list[str]:
    if not system:
        _ensure_user_systemd_env()
    return ["systemctl"] if system else ["systemctl", "--user"]


def _journalctl_cmd(system: bool = False) -> list[str]:
    return ["journalctl"] if system else ["journalctl", "--user"]


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
        print(f"System gateway {action} requires root. Re-run with sudo.")
        sys.exit(1)


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
    """Return whether systemd user lingering is enabled for the current user.

    Returns:
        (True, "") when linger is enabled.
        (False, "") when linger is disabled.
        (None, detail) when the status could not be determined.
    """
    if not is_linux():
        return None, "not supported on this platform"

    import shutil

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

def get_launchd_plist_path() -> Path:
    """Return the launchd plist path, scoped per profile.

    Default ``~/.hermes`` → ``ai.hermes.gateway.plist`` (backward compatible).
    Profile ``~/.hermes/profiles/coder`` → ``ai.hermes.gateway-coder.plist``.
    """
    suffix = _profile_suffix()
    name = f"ai.hermes.gateway-{suffix}" if suffix else "ai.hermes.gateway"
    return Path.home() / "Library" / "LaunchAgents" / f"{name}.plist"

def _detect_venv_dir() -> Path | None:
    """Detect the active virtualenv directory.

    Checks ``sys.prefix`` first (works regardless of the directory name),
    then falls back to probing common directory names under PROJECT_ROOT.
    Returns ``None`` when no virtualenv can be found.
    """
    # If we're running inside a virtualenv, sys.prefix points to it.
    if sys.prefix != sys.base_prefix:
        venv = Path(sys.prefix)
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

def get_hermes_cli_path() -> str:
    """Get the path to the hermes CLI."""
    # Check if installed via pip
    import shutil
    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        return hermes_bin
    
    # Fallback to direct module execution
    return f"{get_python_path()} -m hermes_cli.main"


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

    if system:
        username, group_name, home_dir = _system_service_identity(run_as_user)
        hermes_home = _hermes_home_for_target_user(home_dir)
        path_entries.extend(_build_user_local_paths(Path(home_dir), path_entries))
        path_entries.extend(common_bin_paths)
        sane_path = ":".join(path_entries)
        return f"""[Unit]
Description={SERVICE_DESCRIPTION}
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=simple
User={username}
Group={group_name}
ExecStart={python_path} -m hermes_cli.main gateway run --replace
WorkingDirectory={working_dir}
Environment="HOME={home_dir}"
Environment="USER={username}"
Environment="LOGNAME={username}"
Environment="PATH={sane_path}"
Environment="VIRTUAL_ENV={venv_dir}"
Environment="HERMES_HOME={hermes_home}"
Restart=on-failure
RestartSec=30
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=60
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

    hermes_home = str(get_hermes_home().resolve())
    path_entries.extend(_build_user_local_paths(Path.home(), path_entries))
    path_entries.extend(common_bin_paths)
    sane_path = ":".join(path_entries)
    return f"""[Unit]
Description={SERVICE_DESCRIPTION}
After=network.target
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=simple
ExecStart={python_path} -m hermes_cli.main gateway run --replace
WorkingDirectory={working_dir}
Environment="PATH={sane_path}"
Environment="VIRTUAL_ENV={venv_dir}"
Environment="HERMES_HOME={hermes_home}"
Restart=on-failure
RestartSec=30
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=60
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
"""

def _normalize_service_definition(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


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
    subprocess.run(_systemctl_cmd(system) + ["daemon-reload"], check=True, timeout=30)
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
    if not is_linux():
        return

    import getpass
    import shutil

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


def systemd_install(force: bool = False, system: bool = False, run_as_user: str | None = None):
    if system:
        _require_root_for_system_service("install")

    unit_path = get_systemd_unit_path(system=system)
    scope_flag = " --system" if system else ""

    if unit_path.exists() and not force:
        if not systemd_unit_is_current(system=system):
            print(f"↻ Repairing outdated {_service_scope_label(system)} systemd service at: {unit_path}")
            refresh_systemd_unit_if_needed(system=system)
            subprocess.run(_systemctl_cmd(system) + ["enable", get_service_name()], check=True, timeout=30)
            print(f"✓ {_service_scope_label(system).capitalize()} service definition updated")
            return
        print(f"Service already installed at: {unit_path}")
        print("Use --force to reinstall")
        return

    unit_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Installing {_service_scope_label(system)} systemd service to: {unit_path}")
    unit_path.write_text(generate_systemd_unit(system=system, run_as_user=run_as_user), encoding="utf-8")

    subprocess.run(_systemctl_cmd(system) + ["daemon-reload"], check=True, timeout=30)
    subprocess.run(_systemctl_cmd(system) + ["enable", get_service_name()], check=True, timeout=30)

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


def systemd_uninstall(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("uninstall")

    subprocess.run(_systemctl_cmd(system) + ["stop", get_service_name()], check=False, timeout=90)
    subprocess.run(_systemctl_cmd(system) + ["disable", get_service_name()], check=False, timeout=30)

    unit_path = get_systemd_unit_path(system=system)
    if unit_path.exists():
        unit_path.unlink()
        print(f"✓ Removed {unit_path}")

    subprocess.run(_systemctl_cmd(system) + ["daemon-reload"], check=True, timeout=30)
    print(f"✓ {_service_scope_label(system).capitalize()} service uninstalled")


def systemd_start(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("start")
    refresh_systemd_unit_if_needed(system=system)
    subprocess.run(_systemctl_cmd(system) + ["start", get_service_name()], check=True, timeout=30)
    print(f"✓ {_service_scope_label(system).capitalize()} service started")



def systemd_stop(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("stop")
    subprocess.run(_systemctl_cmd(system) + ["stop", get_service_name()], check=True, timeout=90)
    print(f"✓ {_service_scope_label(system).capitalize()} service stopped")



def systemd_restart(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("restart")
    refresh_systemd_unit_if_needed(system=system)
    subprocess.run(_systemctl_cmd(system) + ["restart", get_service_name()], check=True, timeout=90)
    print(f"✓ {_service_scope_label(system).capitalize()} service restarted")



def systemd_status(deep: bool = False, system: bool = False):
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

    if not systemd_unit_is_current(system=system):
        print("⚠ Installed gateway service definition is outdated")
        print(f"  Run: {'sudo ' if system else ''}hermes gateway restart{scope_flag}  # auto-refreshes the unit")
        print()

    subprocess.run(
        _systemctl_cmd(system) + ["status", get_service_name(), "--no-pager"],
        capture_output=False,
        timeout=10,
    )

    result = subprocess.run(
        _systemctl_cmd(system) + ["is-active", get_service_name()],
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
        subprocess.run(_journalctl_cmd(system) + ["-u", get_service_name(), "-n", "20", "--no-pager"], timeout=10)


# =============================================================================
# Launchd (macOS)
# =============================================================================

def get_launchd_label() -> str:
    """Return the launchd service label, scoped per profile."""
    suffix = _profile_suffix()
    return f"ai.hermes.gateway-{suffix}" if suffix else "ai.hermes.gateway"


def _launchd_domain() -> str:
    import os
    return f"gui/{os.getuid()}"


def generate_launchd_plist() -> str:
    python_path = get_python_path()
    working_dir = str(PROJECT_ROOT)
    hermes_home = str(get_hermes_home().resolve())
    log_dir = get_hermes_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    label = get_launchd_label()
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

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>hermes_cli.main</string>
        <string>gateway</string>
        <string>run</string>
        <string>--replace</string>
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
    return _normalize_service_definition(installed) == _normalize_service_definition(expected)


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
        if e.returncode != 3:
            raise
        print("↻ launchd job was unloaded; reloading service definition")
        subprocess.run(["launchctl", "bootstrap", _launchd_domain(), str(plist_path)], check=True, timeout=30)
        subprocess.run(["launchctl", "kickstart", f"{_launchd_domain()}/{label}"], check=True, timeout=30)
    print("✓ Service started")

def launchd_stop():
    label = get_launchd_label()
    subprocess.run(["launchctl", "kill", "SIGTERM", f"{_launchd_domain()}/{label}"], check=True, timeout=30)
    print("✓ Service stopped")

def _wait_for_gateway_exit(timeout: float = 10.0, force_after: float = 5.0):
    """Wait for the gateway process (by saved PID) to exit.

    Uses the PID from the gateway.pid file — not launchd labels — so this
    works correctly when multiple gateway instances run under separate
    HERMES_HOME directories.

    Args:
        timeout: Total seconds to wait before giving up.
        force_after: Seconds of graceful waiting before sending SIGKILL.
    """
    import time
    from gateway.status import get_running_pid

    deadline = time.monotonic() + timeout
    force_deadline = time.monotonic() + force_after
    force_sent = False

    while time.monotonic() < deadline:
        pid = get_running_pid()
        if pid is None:
            return  # Process exited cleanly.

        if not force_sent and time.monotonic() >= force_deadline:
            # Grace period expired — force-kill the specific PID.
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"⚠ Gateway PID {pid} did not exit gracefully; sent SIGKILL")
            except (ProcessLookupError, PermissionError):
                return  # Already gone or we can't touch it.
            force_sent = True

        time.sleep(0.3)

    # Timed out even after SIGKILL.
    remaining_pid = get_running_pid()
    if remaining_pid is not None:
        print(f"⚠ Gateway PID {remaining_pid} still running after {timeout}s — restart may fail")


def launchd_restart():
    label = get_launchd_label()
    target = f"{_launchd_domain()}/{label}"
    # Use kickstart -k so launchd performs an atomic kill+restart.
    # A two-step stop/start from inside the gateway's own process tree
    # would kill the shell before the start command is reached.
    try:
        subprocess.run(["launchctl", "kickstart", "-k", target], check=True, timeout=90)
        print("✓ Service restarted")
    except subprocess.CalledProcessError as e:
        if e.returncode != 3:
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
    
    from gateway.run import start_gateway
    
    print("┌─────────────────────────────────────────────────────────┐")
    print("│           ⚕ Hermes Gateway Starting...                 │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│  Messaging platforms + cron scheduler                    │")
    print("│  Press Ctrl+C to stop                                   │")
    print("└─────────────────────────────────────────────────────────┘")
    print()
    
    # Exit with code 1 if gateway fails to connect any platform,
    # so systemd Restart=on-failure will retry on transient errors
    verbosity = None if quiet else verbose
    success = asyncio.run(start_gateway(replace=replace, verbosity=verbosity))
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
            "   groups:history, im:history, im:read, im:write, users:read, files:write",
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
            "5. For E2EE: set MATRIX_ENCRYPTION=true (requires pip install 'matrix-nio[e2e]')",
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
]


def _platform_status(platform: dict) -> str:
    """Return a plain-text status string for a platform.

    Returns uncolored text so it can safely be embedded in
    simple_term_menu items (ANSI codes break width calculation).
    """
    token_var = platform["token_var"]
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
    platforms = state.get("platforms", {}) or {}

    for platform, pdata in platforms.items():
        if pdata.get("state") == "fatal":
            message = pdata.get("error_message") or "unknown error"
            lines.append(f"⚠ {platform}: {message}")

    if gateway_state == "startup_failed" and exit_reason:
        lines.append(f"⚠ Last startup issue: {exit_reason}")
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
                    print_info("  Approve with: hermes pairing approve {platform} {code}")
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


def _is_service_installed() -> bool:
    """Check if the gateway is installed as a system service."""
    if is_linux():
        return get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()
    elif is_macos():
        return get_launchd_plist_path().exists()
    return False


def _is_service_running() -> bool:
    """Check if the gateway service is currently running."""
    if is_linux():
        user_unit_exists = get_systemd_unit_path(system=False).exists()
        system_unit_exists = get_systemd_unit_path(system=True).exists()

        if user_unit_exists:
            try:
                result = subprocess.run(
                    _systemctl_cmd(False) + ["is-active", get_service_name()],
                    capture_output=True, text=True, timeout=10,
                )
                if result.stdout.strip() == "active":
                    return True
            except subprocess.TimeoutExpired:
                pass

        if system_unit_exists:
            try:
                result = subprocess.run(
                    _systemctl_cmd(True) + ["is-active", get_service_name()],
                    capture_output=True, text=True, timeout=10,
                )
                if result.stdout.strip() == "active":
                    return True
            except subprocess.TimeoutExpired:
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
        print_info("    Linux:  sudo apt install signal-cli")
        print_info("            or download from https://github.com/AsamK/signal-cli")
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

    if is_linux() and has_conflicting_systemd_units():
        print_systemd_scope_conflict_warning()
        print()

    if service_installed and service_running:
        print_success("Gateway service is installed and running.")
    elif service_installed:
        print_warning("Gateway service is installed but not running.")
        if prompt_yes_no("  Start it now?", True):
            try:
                if is_linux():
                    systemd_start()
                elif is_macos():
                    launchd_start()
            except subprocess.CalledProcessError as e:
                print_error(f"  Failed to start: {e}")
    else:
        print_info("Gateway service is not installed yet.")
        print_info("You'll be offered to install it after configuring platforms.")

    # ── Platform configuration loop ──
    while True:
        print()
        print_header("Messaging Platforms")

        menu_items = []
        for plat in _PLATFORMS:
            status = _platform_status(plat)
            menu_items.append(f"{plat['label']}  ({status})")
        menu_items.append("Done")

        choice = prompt_choice("Select a platform to configure:", menu_items, len(menu_items) - 1)

        if choice == len(_PLATFORMS):
            break

        platform = _PLATFORMS[choice]

        if platform["key"] == "whatsapp":
            _setup_whatsapp()
        elif platform["key"] == "signal":
            _setup_signal()
        else:
            _setup_standard_platform(platform)

    # ── Post-setup: offer to install/restart gateway ──
    any_configured = any(
        bool(get_env_value(p["token_var"]))
        for p in _PLATFORMS
        if p["key"] != "whatsapp"
    ) or (get_env_value("WHATSAPP_ENABLED") or "").lower() == "true"

    if any_configured:
        print()
        print(color("─" * 58, Colors.DIM))
        service_installed = _is_service_installed()
        service_running = _is_service_running()

        if service_running:
            if prompt_yes_no("  Restart the gateway to pick up changes?", True):
                try:
                    if is_linux():
                        systemd_restart()
                    elif is_macos():
                        launchd_restart()
                    else:
                        stop_profile_gateway()
                        print_info("Start manually: hermes gateway")
                except subprocess.CalledProcessError as e:
                    print_error(f"  Restart failed: {e}")
        elif service_installed:
            if prompt_yes_no("  Start the gateway service?", True):
                try:
                    if is_linux():
                        systemd_start()
                    elif is_macos():
                        launchd_start()
                except subprocess.CalledProcessError as e:
                    print_error(f"  Start failed: {e}")
        else:
            print()
            if is_linux() or is_macos():
                platform_name = "systemd" if is_linux() else "launchd"
                if prompt_yes_no(f"  Install the gateway as a {platform_name} service? (runs in background, starts on boot)", True):
                    try:
                        installed_scope = None
                        did_install = False
                        if is_linux():
                            installed_scope, did_install = install_linux_gateway_from_setup(force=False)
                        else:
                            launchd_install(force=False)
                            did_install = True
                        print()
                        if did_install and prompt_yes_no("  Start the service now?", True):
                            try:
                                if is_linux():
                                    systemd_start(system=installed_scope == "system")
                                else:
                                    launchd_start()
                            except subprocess.CalledProcessError as e:
                                print_error(f"  Start failed: {e}")
                    except subprocess.CalledProcessError as e:
                        print_error(f"  Install failed: {e}")
                        print_info("  You can try manually: hermes gateway install")
                else:
                    print_info("  You can install later: hermes gateway install")
                    if is_linux():
                        print_info("  Or as a boot-time service: sudo hermes gateway install --system")
                    print_info("  Or run in foreground:  hermes gateway")
            else:
                print_info("  Service install not supported on this platform.")
                print_info("  Run in foreground: hermes gateway")
    else:
        print()
        print_info("No platforms configured. Run 'hermes gateway setup' when ready.")

    print()


# =============================================================================
# Main Command Handler
# =============================================================================

def gateway_command(args):
    """Handle gateway subcommands."""
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
        if is_linux():
            systemd_install(force=force, system=system, run_as_user=run_as_user)
        elif is_macos():
            launchd_install(force)
        else:
            print("Service installation not supported on this platform.")
            print("Run manually: hermes gateway run")
            sys.exit(1)
    
    elif subcmd == "uninstall":
        if is_managed():
            managed_error("uninstall gateway service (managed by NixOS)")
            return
        system = getattr(args, 'system', False)
        if is_linux():
            systemd_uninstall(system=system)
        elif is_macos():
            launchd_uninstall()
        else:
            print("Not supported on this platform.")
            sys.exit(1)
    
    elif subcmd == "start":
        system = getattr(args, 'system', False)
        if is_linux():
            systemd_start(system=system)
        elif is_macos():
            launchd_start()
        else:
            print("Not supported on this platform.")
            sys.exit(1)
    
    elif subcmd == "stop":
        stop_all = getattr(args, 'all', False)
        system = getattr(args, 'system', False)

        if stop_all:
            # --all: kill every gateway process on the machine
            service_available = False
            if is_linux() and (get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()):
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
            killed = kill_gateway_processes()
            total = killed + (1 if service_available else 0)
            if total:
                print(f"✓ Stopped {total} gateway process(es) across all profiles")
            else:
                print("✗ No gateway processes found")
        else:
            # Default: stop only the current profile's gateway
            service_available = False
            if is_linux() and (get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()):
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
        service_configured = False
        
        if is_linux() and (get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()):
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
            if is_linux():
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
        system = getattr(args, 'system', False)
        
        # Check for service first
        if is_linux() and (get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()):
            systemd_status(deep, system=system)
        elif is_macos() and get_launchd_plist_path().exists():
            launchd_status(deep)
        else:
            # Check for manually running processes
            pids = find_gateway_pids()
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
                print("  hermes gateway          # Run in foreground")
                print("  hermes gateway install  # Install as user service")
                print("  sudo hermes gateway install --system  # Install as boot-time system service")

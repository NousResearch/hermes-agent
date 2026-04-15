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
    _exclude = exclude_pids or set()
    pids = [pid for pid in _get_service_pids() if pid not in _exclude]
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
                capture_output=True, text=True, timeout=10
            )
            current_cmd = ""
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line.startswith("CommandLine="):
                    current_cmd = line[len("CommandLine="):]
                elif line.startswith("ProcessId="):
                    pid_str = line[len("ProcessId="):]
                    if any(p in current_cmd for p in patterns) and (all_profiles or _matches_current_profile(current_cmd)):
                        try:
                            pid = int(pid_str)
                            if pid != os.getpid() and pid not in pids and pid not in _exclude:
                                pids.append(pid)
                        except ValueError:
                            pass
                    current_cmd = ""
        else:
            result = subprocess.run(
                ["ps", "eww", "-ax", "-o", "pid=,command="],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.split('\n'):
                stripped = line.strip()
                if not stripped or 'grep' in stripped:
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
                if pid == os.getpid() or pid in pids or pid in _exclude:
                    continue
                if any(pattern in command for pattern in patterns) and (all_profiles or _matches_current_profile(command)):
                    pids.append(pid)
    except (OSError, subprocess.TimeoutExpired):
        pass

    return pids


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


from hermes_constants import is_container, is_termux, is_wsl


def _wsl_systemd_operational() -> bool:
    """Check if systemd is actually running as PID 1 on WSL.

    WSL2 with ``systemd=true`` in wsl.conf has working systemd.
    WSL2 without it (or WSL1) does not — systemctl commands fail.
    """
    try:
        result = subprocess.run(
            ["systemctl", "is-system-running"],
            capture_output=True, text=True, timeout=5,
        )
        # "running", "degraded", "starting" all mean systemd is PID 1
        status = result.stdout.strip().lower()
        return status in ("running", "degraded", "starting", "initializing")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def supports_systemd_services() -> bool:
    if not is_linux() or is_termux() or is_container():
        return False
    if shutil.which("systemctl") is None:
        return False
    if is_wsl():
        return _wsl_systemd_operational()
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
        "  게이트웨이를 백그라운드에서 어떻게 실행할지 선택해 주세요:",
        [
            "사용자 서비스 (sudo 불필요, 노트북/개발 환경에 적합, 로그아웃 후 linger 설정이 필요할 수 있음)",
            "시스템 서비스 (부팅 시 시작, sudo 필요, 실제 실행은 현재 사용자 계정으로 진행)",
            "지금은 서비스 설치 건너뛰기",
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
            print_warning("  시스템 서비스 설치에는 sudo가 필요해서, 현재 사용자 세션에서는 Hermes가 바로 생성할 수 없어요.")
            if run_as_user:
                print_info(f"  설정이 끝난 뒤 다음 명령을 실행하세요: sudo hermes gateway install --system --run-as-user {run_as_user}")
            else:
                print_info("  설정이 끝난 뒤 다음 명령을 실행하세요: sudo hermes gateway install --system --run-as-user <your-user>")
            print_info("  그다음 다음 명령으로 시작하세요: sudo hermes gateway start --system")
            return scope, False

        if not run_as_user:
            while True:
                run_as_user = prompt("  Run the system gateway service as which user?", default="")
                run_as_user = (run_as_user or "").strip()
                if run_as_user:
                    break
                print_error("  사용자 이름을 입력해 주세요.")

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
        print("✓ systemd linger가 활성화되어 있어요 (로그아웃 후에도 서비스가 유지돼요)")
    elif linger_enabled is False:
        print("⚠ systemd linger가 비활성화되어 있어요 (로그아웃하면 gateway가 멈출 수 있어요)")
        print("  실행: sudo loginctl enable-linger $USER")
    else:
        print(f"⚠ systemd linger 상태를 확인하지 못했어요 ({linger_detail})")
        print("  로그아웃 후에도 gateway 사용자 서비스를 유지하려면 다음 명령을 실행하세요:")
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
    restart_timeout = max(60, int(_get_restart_drain_timeout() or 0))

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
ExecStart={python_path} -m hermes_cli.main{f" {profile_arg}" if profile_arg else ""} gateway run --replace
WorkingDirectory={working_dir}
Environment="HOME={home_dir}"
Environment="USER={username}"
Environment="LOGNAME={username}"
Environment="PATH={sane_path}"
Environment="VIRTUAL_ENV={venv_dir}"
Environment="HERMES_HOME={hermes_home}"
Restart=on-failure
RestartSec=30
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
    path_entries.extend(common_bin_paths)
    sane_path = ":".join(path_entries)
    return f"""[Unit]
Description={SERVICE_DESCRIPTION}
After=network.target
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=simple
ExecStart={python_path} -m hermes_cli.main{f" {profile_arg}" if profile_arg else ""} gateway run --replace
WorkingDirectory={working_dir}
Environment="PATH={sane_path}"
Environment="VIRTUAL_ENV={venv_dir}"
Environment="HERMES_HOME={hermes_home}"
Restart=on-failure
RestartSec=30
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

    print("SSH 로그아웃 후에도 게이트웨이가 유지되도록 linger를 활성화하는 중...")
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
        print("✓ Linger를 활성화했어요 — 로그아웃 후에도 게이트웨이가 유지돼요")
        return

    detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
    _print_linger_enable_warning(username, detail or linger_detail)


def _select_systemd_scope(system: bool = False) -> bool:
    if system:
        return True
    return get_systemd_unit_path(system=True).exists() and not get_systemd_unit_path(system=False).exists()


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

    unit_path = get_systemd_unit_path(system=system)
    scope_flag = " --system" if system else ""

    if unit_path.exists() and not force:
        if not systemd_unit_is_current(system=system):
            print(f"↻ Repairing outdated {_service_scope_label(system)} systemd service at: {unit_path}")
            refresh_systemd_unit_if_needed(system=system)
            _run_systemctl(["enable", get_service_name()], system=system, check=True, timeout=30)
            print(f"✓ {_service_scope_label(system).capitalize()} service definition updated")
            return
        print(f"서비스가 이미 설치되어 있어요: {unit_path}")
        print("재설치하려면 --force 를 사용하세요")
        return

    unit_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{_service_scope_label(system)} systemd 서비스를 설치하는 중: {unit_path}")
    unit_path.write_text(generate_systemd_unit(system=system, run_as_user=run_as_user), encoding="utf-8")

    _run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    _run_systemctl(["enable", get_service_name()], system=system, check=True, timeout=30)

    print()
    print(f"✓ {_service_scope_label(system).capitalize()} 서비스를 설치하고 활성화했어요!")
    print()
    print("다음 단계:")
    print(f"  {'sudo ' if system else ''}hermes gateway start{scope_flag}              # 서비스 시작")
    print(f"  {'sudo ' if system else ''}hermes gateway status{scope_flag}             # 상태 확인")
    print(f"  {'journalctl' if system else 'journalctl --user'} -u {get_service_name()} -f  # 로그 보기")
    print()

    if system:
        configured_user = _read_systemd_user_from_unit(unit_path)
        if configured_user:
            print(f"실행 사용자: {configured_user}")
    else:
        _ensure_linger_enabled()

    print_systemd_scope_conflict_warning()


def systemd_uninstall(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("uninstall")

    _run_systemctl(["stop", get_service_name()], system=system, check=False, timeout=90)
    _run_systemctl(["disable", get_service_name()], system=system, check=False, timeout=30)

    unit_path = get_systemd_unit_path(system=system)
    if unit_path.exists():
        unit_path.unlink()
        print(f"✓ 제거했어요: {unit_path}")

    _run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    print(f"✓ {_service_scope_label(system).capitalize()} 서비스를 제거했어요")


def systemd_start(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("start")
    refresh_systemd_unit_if_needed(system=system)
    _run_systemctl(["start", get_service_name()], system=system, check=True, timeout=30)
    print(f"✓ {_service_scope_label(system).capitalize()} 서비스를 시작했어요")



def systemd_stop(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("stop")
    _run_systemctl(["stop", get_service_name()], system=system, check=True, timeout=90)
    print(f"✓ {_service_scope_label(system).capitalize()} 서비스를 중지했어요")



def systemd_restart(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("restart")
    refresh_systemd_unit_if_needed(system=system)
    from gateway.status import get_running_pid

    pid = get_running_pid()
    if pid is not None and _request_gateway_self_restart(pid):
        # SIGUSR1 sent — the gateway will drain active agents, exit with
        # code 75, and systemd will restart it after RestartSec (30s).
        # Wait for the old process to die and the new one to become active
        # so the CLI doesn't return while the service is still restarting.
        import time
        scope_label = _service_scope_label(system).capitalize()
        svc = get_service_name()
        scope_cmd = _systemctl_cmd(system)

        # Phase 1: wait for old process to exit (drain + shutdown)
        print(f"⏳ {scope_label} service draining active work...")
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
                time.sleep(1)
            except (ProcessLookupError, PermissionError):
                break  # old process is gone
        else:
            print(f"⚠ Old process (PID {pid}) still alive after 90s")

        # Phase 2: wait for systemd to start the new process
        print(f"⏳ Waiting for {svc} to restart...")
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                result = subprocess.run(
                    scope_cmd + ["is-active", svc],
                    capture_output=True, text=True, timeout=5,
                )
                if result.stdout.strip() == "active":
                    # Verify it's a NEW process, not the old one somehow
                    new_pid = get_running_pid()
                    if new_pid and new_pid != pid:
                        print(f"✓ {scope_label} service restarted (PID {new_pid})")
                        return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            time.sleep(2)

        # Timed out — check final state
        try:
            result = subprocess.run(
                scope_cmd + ["is-active", svc],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip() == "active":
                print(f"✓ {scope_label} service restarted")
                return
        except Exception:
            pass
        print(
            f"⚠ {scope_label} service did not become active within 60s.\n"
            f"  Check status: {'sudo ' if system else ''}hermes gateway status\n"
            f"  Check logs:   journalctl {'--user ' if not system else ''}-u {svc} --since '2 min ago'"
        )
        return
    _run_systemctl(["reload-or-restart", get_service_name()], system=system, check=True, timeout=90)
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

    _run_systemctl(
        ["status", get_service_name(), "--no-pager"],
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
            "1. Telegram에서 @BotFather에게 메시지를 보냅니다",
            "2. /newbot 을 보내고 안내에 따라 봇을 만듭니다",
            "3. BotFather가 알려주는 봇 토큰을 복사합니다",
            "4. 사용자 ID를 확인하려면 @userinfobot 에 메시지를 보내세요 — 숫자 ID를 답장해 줍니다",
        ],
        "vars": [
            {"name": "TELEGRAM_BOT_TOKEN", "prompt": "Telegram 봇 토큰", "password": True,
             "help": "위 3단계에서 받은 @BotFather 토큰을 붙여 넣으세요."},
            {"name": "TELEGRAM_ALLOWED_USERS", "prompt": "허용할 사용자 ID (쉼표로 구분)", "password": False,
             "is_allowlist": True,
             "help": "위 4단계에서 확인한 본인 사용자 ID를 붙여 넣으세요."},
            {"name": "TELEGRAM_HOME_CHANNEL", "prompt": "홈 채널 ID (cron/알림 전달용, 비워 두면 나중에 /set-home으로 설정)", "password": False,
             "help": "DM에서는 보통 본인 사용자 ID와 같습니다. 나중에 채팅에서 /set-home으로 설정할 수도 있습니다."},
        ],
    },
    {
        "key": "discord",
        "label": "Discord",
        "emoji": "💬",
        "token_var": "DISCORD_BOT_TOKEN",
        "setup_instructions": [
            "1. https://discord.com/developers/applications 로 이동 → New Application",
            "2. Bot → Reset Token 으로 이동해 봇 토큰을 복사합니다",
            "3. Bot → Privileged Gateway Intents → Message Content Intent 를 활성화합니다",
            "4. 봇을 서버에 초대합니다:",
            "   OAuth2 → URL Generator → 아래 두 scope를 모두 체크하세요:",
            "     - bot",
            "     - applications.commands  (슬래시 명령어에 필수)",
            "   Bot Permissions: Send Messages, Read Message History, Attach Files",
            "   생성된 URL을 복사해 브라우저에서 열고 초대하세요.",
            "5. 사용자 ID 확인: Discord 설정에서 Developer Mode를 켠 뒤",
            "   내 이름을 우클릭 → Copy ID",
        ],
        "vars": [
            {"name": "DISCORD_BOT_TOKEN", "prompt": "Discord 봇 토큰", "password": True,
             "help": "위 2단계에서 복사한 토큰을 붙여 넣으세요."},
            {"name": "DISCORD_ALLOWED_USERS", "prompt": "허용할 사용자 ID 또는 사용자명 (쉼표로 구분)", "password": False,
             "is_allowlist": True,
             "help": "위 5단계에서 확인한 본인 사용자 ID를 붙여 넣으세요."},
            {"name": "DISCORD_HOME_CHANNEL", "prompt": "홈 채널 ID (cron/알림 전달용, 비워 두면 나중에 /set-home으로 설정)", "password": False,
             "help": "채널을 우클릭 → Copy Channel ID (Developer Mode 필요)."},
        ],
    },
    {
        "key": "slack",
        "label": "Slack",
        "emoji": "💼",
        "token_var": "SLACK_BOT_TOKEN",
        "setup_instructions": [
            "1. https://api.slack.com/apps → Create New App → From Scratch 로 이동합니다",
            "2. Socket Mode 활성화: Settings → Socket Mode → Enable",
            "   scope가 connections:write 인 App-Level Token을 만든 뒤 xapp-... 토큰을 복사하세요",
            "3. Bot Token Scope 추가: Features → OAuth & Permissions → Scopes",
            "   필수: chat:write, app_mentions:read, channels:history, channels:read,",
            "   groups:history, im:history, im:read, im:write, users:read, files:read, files:write",
            "4. 이벤트 구독: Features → Event Subscriptions → Enable",
            "   필수 이벤트: message.im, message.channels, app_mention",
            "   선택: message.groups (비공개 채널용)",
            "   ⚠ message.channels 가 없으면 봇은 DM에서만 동작합니다!",
            "5. 워크스페이스에 설치: Settings → Install App → xoxb-... 토큰 복사",
            "6. scope나 이벤트를 바꾼 뒤에는 앱을 다시 설치하세요",
            "7. 사용자 ID 확인: 프로필 클릭 → 점 세 개 메뉴 → Copy member ID",
            "8. 채널에 봇 초대: /invite @YourBot",
        ],
        "vars": [
            {"name": "SLACK_BOT_TOKEN", "prompt": "Slack Bot Token (xoxb-...)", "password": True,
             "help": "위 5단계에서 복사한 봇 토큰을 붙여 넣으세요."},
            {"name": "SLACK_APP_TOKEN", "prompt": "Slack App Token (xapp-...)", "password": True,
             "help": "위 2단계에서 만든 app-level 토큰을 붙여 넣으세요."},
            {"name": "SLACK_ALLOWED_USERS", "prompt": "허용할 사용자 ID (쉼표로 구분)", "password": False,
             "is_allowlist": True,
             "help": "위 7단계에서 확인한 member ID를 붙여 넣으세요."},
        ],
    },
    {
        "key": "matrix",
        "label": "Matrix",
        "emoji": "🔐",
        "token_var": "MATRIX_ACCESS_TOKEN",
        "setup_instructions": [
            "1. 어떤 Matrix homeserver와도 사용할 수 있습니다 (자체 호스팅 Synapse/Conduit/Dendrite 또는 matrix.org)",
            "2. homeserver에 봇 계정을 만들거나 본인 계정을 사용할 수 있습니다",
            "3. access token 확인: Element → Settings → Help & About → Access Token",
            "   또는 API 사용: curl -X POST https://your-server/_matrix/client/v3/login \\",
            "     -d '{\"type\":\"m.login.password\",\"user\":\"@bot:server\",\"password\":\"...\"}'",
            "4. 또는 사용자 ID + 비밀번호를 입력하면 Hermes가 직접 로그인합니다",
            "5. E2EE를 쓰려면 MATRIX_ENCRYPTION=true 로 설정하세요 (pip install 'mautrix[encryption]' 필요)",
            "6. 사용자 ID는 @username:your-server 형식이며 Element 프로필에서 확인할 수 있습니다",
        ],
        "vars": [
            {"name": "MATRIX_HOMESERVER", "prompt": "Matrix homeserver URL (예: https://matrix.example.org)", "password": False,
             "help": "Matrix homeserver URL입니다. 어떤 자체 호스팅 인스턴스와도 사용할 수 있습니다."},
            {"name": "MATRIX_ACCESS_TOKEN", "prompt": "Access token (비워 두면 대신 비밀번호 로그인 사용)", "password": True,
             "help": "access token을 붙여 넣거나, 비워 두고 아래에 사용자 ID + 비밀번호를 입력하세요."},
            {"name": "MATRIX_USER_ID", "prompt": "사용자 ID (@bot:server — 비밀번호 로그인 시 필수)", "password": False,
             "help": "전체 Matrix 사용자 ID입니다. 예: @hermes:matrix.example.org"},
            {"name": "MATRIX_ALLOWED_USERS", "prompt": "허용할 사용자 ID (쉼표로 구분, 예: @you:server)", "password": False,
             "is_allowlist": True,
             "help": "봇과 상호작용할 수 있는 Matrix 사용자 ID 목록입니다."},
            {"name": "MATRIX_HOME_ROOM", "prompt": "홈 룸 ID (cron/알림 전달용, 비워 두면 나중에 /set-home으로 설정)", "password": False,
             "help": "cron 결과와 알림을 전달할 룸 ID입니다. 예: !abc123:server"},
        ],
    },
    {
        "key": "mattermost",
        "label": "Mattermost",
        "emoji": "💬",
        "token_var": "MATTERMOST_TOKEN",
        "setup_instructions": [
            "1. Mattermost에서 Integrations → Bot Accounts → Add Bot Account 로 이동합니다",
            "   (System Console → Integrations → Bot Accounts 가 활성화되어 있어야 합니다)",
            "2. 사용자명(예: hermes)을 지정하고 봇 토큰을 복사합니다",
            "3. 어떤 자체 호스팅 Mattermost 인스턴스와도 사용할 수 있습니다 — 서버 URL을 입력하세요",
            "4. 사용자 ID 확인: 아바타(좌상단) 클릭 → Profile",
            "   화면에 사용자 ID가 표시되며 클릭하면 복사됩니다.",
            "   ⚠ 이것은 사용자명이 아니라 26자리 영숫자 ID입니다.",
            "5. 채널 ID 확인: 채널 이름 클릭 → View Info → ID 복사",
        ],
        "vars": [
            {"name": "MATTERMOST_URL", "prompt": "Mattermost 서버 URL (예: https://mm.example.com)", "password": False,
             "help": "Mattermost 서버 URL입니다. 어떤 자체 호스팅 인스턴스와도 사용할 수 있습니다."},
            {"name": "MATTERMOST_TOKEN", "prompt": "Mattermost 봇 토큰", "password": True,
             "help": "위 2단계에서 복사한 봇 토큰을 붙여 넣으세요."},
            {"name": "MATTERMOST_ALLOWED_USERS", "prompt": "허용할 사용자 ID (쉼표로 구분)", "password": False,
             "is_allowlist": True,
             "help": "위 4단계에서 확인한 Mattermost 사용자 ID를 입력하세요."},
            {"name": "MATTERMOST_HOME_CHANNEL", "prompt": "홈 채널 ID (cron/알림 전달용, 비워 두면 나중에 /set-home으로 설정)", "password": False,
             "help": "Hermes가 cron 결과와 알림을 전달할 채널 ID입니다."},
            {"name": "MATTERMOST_REPLY_MODE", "prompt": "응답 모드 — 'off'는 일반 메시지, 'thread'는 스레드 응답 (기본값: off)", "password": False,
             "help": "off = 채널에 일반 메시지로 응답, thread = 사용자의 메시지 아래 스레드로 응답합니다."},
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
            "1. Hermes 전용 이메일 계정을 사용하는 것을 권장합니다",
            "2. Gmail은 2단계 인증을 켠 뒤 아래에서 앱 비밀번호를 만드세요",
            "   https://myaccount.google.com/apppasswords",
            "3. 다른 제공자는 일반 비밀번호 또는 앱 전용 비밀번호를 사용하세요",
            "4. 이메일 계정에서 IMAP가 활성화되어 있어야 합니다",
        ],
        "vars": [
            {"name": "EMAIL_ADDRESS", "prompt": "이메일 주소", "password": False,
             "help": "Hermes가 사용할 이메일 주소입니다. 예: hermes@gmail.com"},
            {"name": "EMAIL_PASSWORD", "prompt": "이메일 비밀번호 (또는 앱 비밀번호)", "password": True,
             "help": "Gmail은 일반 비밀번호 대신 앱 비밀번호를 사용하세요."},
            {"name": "EMAIL_IMAP_HOST", "prompt": "IMAP 호스트", "password": False,
             "help": "예: Gmail은 imap.gmail.com, Outlook은 outlook.office365.com"},
            {"name": "EMAIL_SMTP_HOST", "prompt": "SMTP 호스트", "password": False,
             "help": "예: Gmail은 smtp.gmail.com, Outlook은 smtp.office365.com"},
            {"name": "EMAIL_ALLOWED_USERS", "prompt": "허용할 발신자 이메일 (쉼표로 구분)", "password": False,
             "is_allowlist": True,
             "help": "이 주소들에서 온 이메일만 처리합니다."},
        ],
    },
    {
        "key": "sms",
        "label": "SMS (Twilio)",
        "emoji": "📱",
        "token_var": "TWILIO_ACCOUNT_SID",
        "setup_instructions": [
            "1. https://www.twilio.com/ 에서 Twilio 계정을 만듭니다",
            "2. Twilio Console 대시보드에서 Account SID와 Auth Token을 확인합니다",
            "3. SMS 발신이 가능한 전화번호를 구매하거나 설정합니다",
            "4. 수신 SMS용 webhook URL을 설정합니다:",
            "   Twilio Console → Phone Numbers → Active Numbers → 해당 번호",
            "   → Messaging → A MESSAGE COMES IN → Webhook → https://your-server:8080/webhooks/twilio",
        ],
        "vars": [
            {"name": "TWILIO_ACCOUNT_SID", "prompt": "Twilio Account SID", "password": False,
             "help": "Twilio Console 대시보드에서 확인할 수 있습니다."},
            {"name": "TWILIO_AUTH_TOKEN", "prompt": "Twilio Auth Token", "password": True,
             "help": "Twilio Console 대시보드에서 확인할 수 있습니다 (클릭하면 표시됨)."},
            {"name": "TWILIO_PHONE_NUMBER", "prompt": "Twilio 전화번호 (E.164 형식, 예: +155****4567)", "password": False,
             "help": "SMS를 발송할 Twilio 전화번호입니다."},
            {"name": "SMS_ALLOWED_USERS", "prompt": "허용할 전화번호 (쉼표로 구분, E.164 형식)", "password": False,
             "is_allowlist": True,
             "help": "이 번호들에서 온 메시지만 처리합니다."},
            {"name": "SMS_HOME_CHANNEL", "prompt": "홈 채널 전화번호 (cron/알림 전달용, 비워 둘 수 있음)", "password": False,
             "help": "cron 결과와 알림을 전달할 전화번호입니다."},
        ],
    },
    {
        "key": "dingtalk",
        "label": "DingTalk",
        "emoji": "💬",
        "token_var": "DINGTALK_CLIENT_ID",
        "setup_instructions": [
            "1. https://open-dev.dingtalk.com 으로 이동 → Create Application",
            "2. 'Credentials'에서 AppKey(Client ID)와 AppSecret(Client Secret)을 복사합니다",
            "3. 봇 설정에서 'Stream Mode'를 활성화합니다",
            "4. 봇을 그룹 채팅에 추가하거나 직접 메시지를 보냅니다",
        ],
        "vars": [
            {"name": "DINGTALK_CLIENT_ID", "prompt": "AppKey (Client ID)", "password": False,
             "help": "DingTalk 앱 자격 증명에 있는 AppKey입니다."},
            {"name": "DINGTALK_CLIENT_SECRET", "prompt": "AppSecret (Client Secret)", "password": True,
             "help": "DingTalk 앱 자격 증명에 있는 AppSecret입니다."},
        ],
    },
    {
        "key": "feishu",
        "label": "Feishu / Lark",
        "emoji": "🪽",
        "token_var": "FEISHU_APP_ID",
        "setup_instructions": [
            "1. https://open.feishu.cn/ (Lark는 https://open.larksuite.com/) 으로 이동합니다",
            "2. 앱을 생성하고 App ID와 App Secret을 복사합니다",
            "3. 앱에서 Bot 기능을 활성화합니다",
            "4. WebSocket(권장) 또는 Webhook 연결 모드를 선택합니다",
            "5. 봇을 그룹 채팅에 추가하거나 직접 메시지를 보냅니다",
            "6. 운영 환경에서는 FEISHU_ALLOWED_USERS 로 접근을 제한하세요",
        ],
        "vars": [
            {"name": "FEISHU_APP_ID", "prompt": "App ID", "password": False,
             "help": "Feishu/Lark 애플리케이션의 App ID입니다."},
            {"name": "FEISHU_APP_SECRET", "prompt": "App Secret", "password": True,
             "help": "Feishu/Lark 애플리케이션의 App Secret입니다."},
            {"name": "FEISHU_DOMAIN", "prompt": "도메인 — feishu 또는 lark (기본값: feishu)", "password": False,
             "help": "중국 Feishu는 'feishu', 국제 Lark는 'lark'를 사용하세요."},
            {"name": "FEISHU_CONNECTION_MODE", "prompt": "연결 모드 — websocket 또는 webhook (기본값: websocket)", "password": False,
             "help": "특별한 이유가 없다면 websocket을 권장합니다."},
            {"name": "FEISHU_ALLOWED_USERS", "prompt": "허용할 사용자 ID (쉼표로 구분, 비워 둘 수 있음)", "password": False,
             "is_allowlist": True,
             "help": "어떤 Feishu/Lark 사용자가 봇과 상호작용할 수 있는지 제한합니다."},
            {"name": "FEISHU_HOME_CHANNEL", "prompt": "홈 채팅 ID (선택 사항, cron/알림용)", "password": False,
             "help": "예약 실행 결과와 알림을 받을 채팅 ID입니다."},
        ],
    },
    {
        "key": "wecom",
        "label": "WeCom (Enterprise WeChat)",
        "emoji": "💬",
        "token_var": "WECOM_BOT_ID",
        "setup_instructions": [
            "1. WeCom 관리자 콘솔 → Applications → Create AI Bot 으로 이동합니다",
            "2. 봇 자격 증명 페이지에서 Bot ID와 Secret을 복사합니다",
            "3. 봇은 WebSocket으로 연결되므로 공개 엔드포인트가 필요 없습니다",
            "4. 봇을 그룹 채팅에 추가하거나 WeCom에서 직접 메시지를 보냅니다",
            "5. 운영 환경에서는 WECOM_ALLOWED_USERS 로 접근을 제한하세요",
        ],
        "vars": [
            {"name": "WECOM_BOT_ID", "prompt": "Bot ID", "password": False,
             "help": "WeCom AI Bot의 Bot ID입니다."},
            {"name": "WECOM_SECRET", "prompt": "Secret", "password": True,
             "help": "WeCom AI Bot의 secret입니다."},
            {"name": "WECOM_ALLOWED_USERS", "prompt": "허용할 사용자 ID (쉼표로 구분, 비워 둘 수 있음)", "password": False,
             "is_allowlist": True,
             "help": "어떤 WeCom 사용자가 봇과 상호작용할 수 있는지 제한합니다."},
            {"name": "WECOM_HOME_CHANNEL", "prompt": "홈 채팅 ID (선택 사항, cron/알림용)", "password": False,
             "help": "예약 실행 결과와 알림을 받을 채팅 ID입니다."},
        ],
    },
    {
        "key": "wecom_callback",
        "label": "WeCom Callback (Self-Built App)",
        "emoji": "💬",
        "token_var": "WECOM_CALLBACK_CORP_ID",
        "setup_instructions": [
            "1. WeCom 관리자 콘솔 → Applications → Create Self-Built App 으로 이동합니다",
            "2. Corp ID(관리자 콘솔 상단)를 확인하고 Corp Secret을 생성합니다",
            "3. Receive Messages에서 callback URL을 서버 주소로 설정합니다",
            "4. callback 설정에 있는 Token과 EncodingAESKey를 복사합니다",
            "5. 이 어댑터는 HTTP 서버를 실행하므로 해당 포트가 WeCom에서 접근 가능해야 합니다",
            "6. 운영 환경에서는 WECOM_CALLBACK_ALLOWED_USERS 로 접근을 제한하세요",
        ],
        "vars": [
            {"name": "WECOM_CALLBACK_CORP_ID", "prompt": "Corp ID", "password": False,
             "help": "WeCom 엔터프라이즈 Corp ID입니다."},
            {"name": "WECOM_CALLBACK_CORP_SECRET", "prompt": "Corp Secret", "password": True,
             "help": "Self-Built App용 secret입니다."},
            {"name": "WECOM_CALLBACK_AGENT_ID", "prompt": "Agent ID", "password": False,
             "help": "Self-Built App의 Agent ID입니다."},
            {"name": "WECOM_CALLBACK_TOKEN", "prompt": "Callback Token", "password": True,
             "help": "WeCom callback 설정에 있는 Token입니다."},
            {"name": "WECOM_CALLBACK_ENCODING_AES_KEY", "prompt": "Encoding AES Key", "password": True,
             "help": "WeCom callback 설정에 있는 EncodingAESKey입니다."},
            {"name": "WECOM_CALLBACK_PORT", "prompt": "Callback 서버 포트 (기본값: 8645)", "password": False,
             "help": "HTTP callback 서버가 사용할 포트입니다."},
            {"name": "WECOM_CALLBACK_ALLOWED_USERS", "prompt": "허용할 사용자 ID (쉼표로 구분, 비워 둘 수 있음)", "password": False,
             "is_allowlist": True,
             "help": "어떤 WeCom 사용자가 앱과 상호작용할 수 있는지 제한합니다."},
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
            "1. iMessage 서버 역할을 할 Mac에 BlueBubbles를 설치합니다:",
            "   https://bluebubbles.app/",
            "2. BlueBubbles 설정 마법사를 완료하고 Apple ID로 로그인합니다",
            "3. BlueBubbles Settings → API 에서 Server URL과 비밀번호를 확인합니다",
            "4. 서버 URL은 보통 http://<your-mac-ip>:1234 형식입니다",
            "5. Hermes는 BlueBubbles REST API로 연결하고",
            "   로컬 webhook을 통해 수신 메시지를 받습니다",
            "6. 사용자 승인에는 DM 페어링을 사용하세요: hermes pairing generate bluebubbles",
            "   생성된 코드를 공유하면 사용자가 iMessage로 보내 승인 요청을 할 수 있습니다",
        ],
        "vars": [
            {"name": "BLUEBUBBLES_SERVER_URL", "prompt": "BlueBubbles 서버 URL (예: http://192.168.1.10:1234)", "password": False,
             "help": "BlueBubbles Settings → API 에 표시되는 URL입니다."},
            {"name": "BLUEBUBBLES_PASSWORD", "prompt": "BlueBubbles 서버 비밀번호", "password": True,
             "help": "BlueBubbles Settings → API 에 표시되는 비밀번호입니다."},
            {"name": "BLUEBUBBLES_ALLOWED_USERS", "prompt": "미리 허용할 전화번호 또는 iMessage ID (쉼표로 구분, DM 페어링을 쓰려면 비워 두기)", "password": False,
             "is_allowlist": True,
             "help": "선택 사항입니다 — 특정 사용자를 미리 허용할 수 있습니다. 권장 방식은 비워 두고 DM 페어링을 사용하는 것입니다."},
            {"name": "BLUEBUBBLES_HOME_CHANNEL", "prompt": "홈 채널 (cron/알림용 전화번호 또는 iMessage ID, 비워 둘 수 있음)", "password": False,
             "help": "cron 결과와 알림을 받을 전화번호 또는 Apple ID입니다."},
        ],
    },
    {
        "key": "qqbot",
        "label": "QQ Bot",
        "emoji": "🐧",
        "token_var": "QQ_APP_ID",
        "setup_instructions": [
            "1. q.qq.com 에서 QQ Bot 애플리케이션을 등록합니다",
            "2. 애플리케이션 페이지에서 App ID와 App Secret을 확인합니다",
            "3. 필요한 intent(C2C, Group, Guild messages)를 활성화합니다",
            "4. sandbox를 설정하거나 봇을 배포합니다",
        ],
        "vars": [
            {"name": "QQ_APP_ID", "prompt": "QQ Bot App ID", "password": False,
             "help": "q.qq.com 에 있는 QQ Bot App ID입니다."},
            {"name": "QQ_CLIENT_SECRET", "prompt": "QQ Bot App Secret", "password": True,
             "help": "q.qq.com 에 있는 QQ Bot App Secret입니다."},
            {"name": "QQ_ALLOWED_USERS", "prompt": "허용할 사용자 OpenID (쉼표로 구분, 오픈 액세스를 원하면 비워 두기)", "password": False,
             "is_allowlist": True,
             "help": "선택 사항입니다 — 특정 사용자 OpenID만 DM 접근을 허용할 수 있습니다."},
            {"name": "QQ_HOME_CHANNEL", "prompt": "홈 채널 (cron 전달용 user/group OpenID, 비워 둘 수 있음)", "password": False,
             "help": "cron 결과와 알림을 받을 OpenID입니다."},
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
                return "설정됨 + 페어링됨"
            return "활성화됨, 아직 페어링 안 됨"
        return "설정 안 됨"
    if platform.get("key") == "signal":
        account = get_env_value("SIGNAL_ACCOUNT")
        if val and account:
            return "설정됨"
        if val or account:
            return "부분 설정됨"
        return "설정 안 됨"
    if platform.get("key") == "email":
        pwd = get_env_value("EMAIL_PASSWORD")
        imap = get_env_value("EMAIL_IMAP_HOST")
        smtp = get_env_value("EMAIL_SMTP_HOST")
        if all([val, pwd, imap, smtp]):
            return "설정됨"
        if any([val, pwd, imap, smtp]):
            return "부분 설정됨"
        return "설정 안 됨"
    if platform.get("key") == "matrix":
        homeserver = get_env_value("MATRIX_HOMESERVER")
        password = get_env_value("MATRIX_PASSWORD")
        if (val or password) and homeserver:
            e2ee = get_env_value("MATRIX_ENCRYPTION")
            suffix = " + E2EE" if e2ee and e2ee.lower() in ("true", "1", "yes") else ""
            return f"설정됨{suffix}"
        if val or password or homeserver:
            return "부분 설정됨"
        return "설정 안 됨"
    if platform.get("key") == "weixin":
        token = get_env_value("WEIXIN_TOKEN")
        if val and token:
            return "설정됨"
        if val or token:
            return "부분 설정됨"
        return "설정 안 됨"
    if val:
        return "설정됨"
    return "설정 안 됨"


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
    print(color(f"  ─── {emoji} {label} 설정 ───", Colors.CYAN))

    # Show step-by-step setup instructions if this platform has them
    instructions = platform.get("setup_instructions")
    if instructions:
        print()
        for line in instructions:
            print_info(f"  {line}")

    existing_token = get_env_value(token_var)
    if existing_token:
        print()
        print_success(f"{label}: 이미 설정되어 있음")
        if not prompt_yes_no(f"  {label}를 다시 설정할까요?", False):
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
            print_info("  보안을 위해 gateway는 기본적으로 모든 사용자를 거부합니다.")
            print_info("  허용 목록을 만들려면 사용자 ID를 입력하세요.")
            print_info("  비워 두면 다음 단계에서 오픈 액세스 여부를 물어봅니다.")
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
                print_success("  저장 완료 — 이 사용자들만 봇과 상호작용할 수 있습니다.")
                allowed_val_set = cleaned
            else:
                # No allowlist — ask about open access vs DM pairing
                print()
                access_choices = [
                    "오픈 액세스 사용 (누구나 봇에 메시지를 보낼 수 있음)",
                    "DM 페어링 사용 (알 수 없는 사용자가 접근 요청을 보내고, 'hermes pairing approve'로 승인)",
                    "지금은 건너뛰기 (설정 전까지 모든 사용자 거부)",
                ]
                access_idx = prompt_choice("  미승인 사용자를 어떻게 처리할까요?", access_choices, 1)
                if access_idx == 0:
                    save_env_value("GATEWAY_ALLOW_ALL_USERS", "true")
                    print_warning("  오픈 액세스 활성화 — 누구나 봇을 사용할 수 있습니다!")
                elif access_idx == 1:
                    print_success("  DM 페어링 모드 — 사용자는 코드를 받아 접근을 요청할 수 있습니다.")
                    print_info("  승인 명령: hermes pairing approve <platform> <code>")
                else:
                    print_info("  건너뜀 — 나중에 'hermes gateway setup'으로 설정할 수 있습니다")
            continue

        value = prompt(f"  {var['prompt']}", password=var.get("password", False))
        if value:
            save_env_value(var["name"], value)
            print_success(f"  저장 완료: {var['name']}")
        elif var["name"] == token_var:
            print_warning(f"  건너뜀 — {label}은(는) 이 값이 없으면 동작하지 않습니다.")
            return
        else:
            print_info("  건너뜀 (나중에 설정 가능)")

    # If an allowlist was set and home channel wasn't, offer to reuse
    # the first user ID (common for Telegram DMs).
    home_var = f"{label.upper()}_HOME_CHANNEL"
    home_val = get_env_value(home_var)
    if allowed_val_set and not home_val and label == "Telegram":
        first_id = allowed_val_set.split(",")[0].strip()
        if first_id and prompt_yes_no(f"  사용자 ID({first_id})를 홈 채널로 사용할까요?", True):
            save_env_value(home_var, first_id)
            print_success(f"  홈 채널을 {first_id}(으)로 설정했습니다")

    print()
    print_success(f"{emoji} {label} 설정 완료!")


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
    """Configure DingTalk via the standard platform setup."""
    dingtalk_platform = next(p for p in _PLATFORMS if p["key"] == "dingtalk")
    _setup_standard_platform(dingtalk_platform)


def _setup_wecom():
    """Configure WeCom (Enterprise WeChat) via the standard platform setup."""
    wecom_platform = next(p for p in _PLATFORMS if p["key"] == "wecom")
    _setup_standard_platform(wecom_platform)


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
        print_success("Weixin: 이미 설정되어 있음")
        if not prompt_yes_no("  Weixin을 다시 설정할까요?", False):
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
    access_idx = prompt_choice("  다이렉트 메시지 접근을 어떻게 승인할까요?", access_choices, 0)
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
    group_choices = [
        "Disable group chats (recommended)",
        "Allow all group chats",
        "Only allow listed group chat IDs",
    ]
    group_idx = prompt_choice("  그룹 채팅을 어떻게 처리할까요?", group_choices, 0)
    if group_idx == 0:
        save_env_value("WEIXIN_GROUP_POLICY", "disabled")
        save_env_value("WEIXIN_GROUP_ALLOWED_USERS", "")
        print_info("  그룹 채팅을 비활성화했습니다.")
    elif group_idx == 1:
        save_env_value("WEIXIN_GROUP_POLICY", "open")
        save_env_value("WEIXIN_GROUP_ALLOWED_USERS", "")
        print_warning("  All group chats enabled.")
    else:
        allow_groups = prompt("  Allowed group chat IDs (comma-separated)", "", password=False).replace(" ", "")
        save_env_value("WEIXIN_GROUP_POLICY", "allowlist")
        save_env_value("WEIXIN_GROUP_ALLOWED_USERS", allow_groups)
        print_success("  Group allowlist saved.")

    if user_id:
        print()
        if prompt_yes_no(f"  Use your Weixin user ID ({user_id}) as the home channel?", True):
            save_env_value("WEIXIN_HOME_CHANNEL", user_id)
            print_success(f"  홈 채널을 {user_id}(으)로 설정했습니다")

    print()
    print_success("Weixin 설정 완료!")
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
        print_success("Feishu / Lark: 이미 설정되어 있음")
        if not prompt_yes_no("  Feishu / Lark를 다시 설정할까요?", False):
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
        print_info("  https://open.feishu.cn/ (Lark는 https://open.larksuite.com/) 으로 이동하세요")
        print_info("  앱을 만들고 Bot 기능을 활성화한 뒤 자격 증명을 복사하세요.")
        print()
        app_id = prompt("  App ID", password=False)
        if not app_id:
            print_warning("  건너뜀 — Feishu / Lark는 App ID가 없으면 동작하지 않습니다.")
            return
        app_secret = prompt("  App Secret", password=True)
        if not app_secret:
            print_warning("  건너뜀 — Feishu / Lark는 App Secret이 없으면 동작하지 않습니다.")
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
    access_idx = prompt_choice("  다이렉트 메시지 접근을 어떻게 승인할까요?", access_choices, 0)
    if access_idx == 0:
        save_env_value("FEISHU_ALLOW_ALL_USERS", "false")
        save_env_value("FEISHU_ALLOWED_USERS", "")
        print_success("  DM 페어링을 활성화했습니다.")
        print_info("  알 수 없는 사용자가 접근을 요청할 수 있으며 `hermes pairing approve`로 승인할 수 있습니다.")
    elif access_idx == 1:
        save_env_value("FEISHU_ALLOW_ALL_USERS", "true")
        save_env_value("FEISHU_ALLOWED_USERS", "")
        print_warning("  Feishu / Lark의 오픈 DM 접근을 활성화했습니다.")
    else:
        save_env_value("FEISHU_ALLOW_ALL_USERS", "false")
        default_allow = open_id or ""
        allowlist = prompt("  허용할 사용자 ID (쉼표로 구분)", default_allow, password=False).replace(" ", "")
        save_env_value("FEISHU_ALLOWED_USERS", allowlist)
        print_success("  허용 목록을 저장했습니다.")

    # ── Group policy ──
    print()
    group_choices = [
        "그룹에서는 @멘션될 때만 응답 (권장)",
        "그룹 채팅 비활성화",
    ]
    group_idx = prompt_choice("  그룹 채팅을 어떻게 처리할까요?", group_choices, 0)
    if group_idx == 0:
        save_env_value("FEISHU_GROUP_POLICY", "open")
        print_info("  그룹 채팅을 활성화했습니다 (봇을 @멘션해야 응답).")
    else:
        save_env_value("FEISHU_GROUP_POLICY", "disabled")
        print_info("  그룹 채팅을 비활성화했습니다.")

    # ── Home channel ──
    print()
    home_channel = prompt("  홈 채팅 ID (선택 사항, cron/알림용)", password=False)
    if home_channel:
        save_env_value("FEISHU_HOME_CHANNEL", home_channel)
        print_success(f"  홈 채널을 {home_channel}(으)로 설정했습니다")

    print()
    print_success("🪽 Feishu / Lark 설정 완료!")
    print_info(f"  App ID: {app_id}")
    print_info(f"  Domain: {domain}")
    if bot_name:
        print_info(f"  Bot: {bot_name}")


def _setup_signal():
    """Interactive setup for Signal messenger."""
    import shutil

    print()
    print(color("  ─── 📡 Signal Setup ───", Colors.CYAN))

    existing_url = get_env_value("SIGNAL_HTTP_URL")
    existing_account = get_env_value("SIGNAL_ACCOUNT")
    if existing_url and existing_account:
        print()
        print_success("Signal: 이미 설정되어 있음")
        if not prompt_yes_no("  Signal을 다시 설정할까요?", False):
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
    print_success("Signal 설정 완료!")
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

    if supports_systemd_services() and has_conflicting_systemd_units():
        print_systemd_scope_conflict_warning()
        print()

    if service_installed and service_running:
        print_success("Gateway service is installed and running.")
    elif service_installed:
        print_warning("Gateway service is installed but not running.")
        if prompt_yes_no("  Start it now?", True):
            try:
                if supports_systemd_services():
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
        elif platform["key"] == "weixin":
            _setup_weixin()
        elif platform["key"] == "feishu":
            _setup_feishu()
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
                    if supports_systemd_services():
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
                    if supports_systemd_services():
                        systemd_start()
                    elif is_macos():
                        launchd_start()
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
        print_info("설정된 플랫폼이 없어요. 준비되면 'hermes gateway setup'를 실행하세요.")

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
        if is_termux():
            print("Termux에서는 게이트웨이 서비스를 설치할 수 없어요.")
            print("수동 실행: hermes gateway")
            sys.exit(1)
        if supports_systemd_services():
            if is_wsl():
                print_warning("WSL이 감지되었어요 — systemd 서비스가 WSL 재시작 후 유지되지 않을 수 있어요.")
                print_info("  대신 포그라운드 실행을 고려해 보세요: hermes gateway run")
                print_info("  또는 tmux/screen으로 유지하세요: tmux new -s hermes 'hermes gateway run'")
                print()
            systemd_install(force=force, system=system, run_as_user=run_as_user)
        elif is_macos():
            launchd_install(force)
        elif is_wsl():
            print("WSL이 감지되었지만 systemd가 실행 중이 아니에요.")
            print("systemd를 활성화(add systemd=true to /etc/wsl.conf 후 WSL 재시작)하거나")
            print("게이트웨이를 포그라운드 모드로 실행하세요:")
            print()
            print("  hermes gateway run                              # 직접 포그라운드 실행")
            print("  tmux new -s hermes 'hermes gateway run'         # tmux로 유지")
            print("  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # 백그라운드")
            sys.exit(1)
        elif is_container():
            print("Docker 컨테이너 안에서는 서비스 설치가 필요하지 않아요.")
            print("컨테이너 런타임이 서비스 관리자를 대신하므로 Docker 재시작 정책을 사용하세요:")
            print()
            print("  docker run --restart unless-stopped ...   # 크래시/재부팅 시 자동 재시작")
            print("  docker restart <container>                # 수동 재시작")
            print()
            print("게이트웨이 실행: hermes gateway run")
            sys.exit(0)
        else:
            print("이 플랫폼에서는 서비스 설치를 지원하지 않아요.")
            print("수동 실행: hermes gateway run")
            sys.exit(1)
    
    elif subcmd == "uninstall":
        if is_managed():
            managed_error("uninstall gateway service (managed by NixOS)")
            return
        system = getattr(args, 'system', False)
        if is_termux():
            print("Termux에는 제거할 관리형 서비스가 없어서 게이트웨이 서비스 제거를 지원하지 않아요.")
            print("수동 실행 중지: hermes gateway stop")
            sys.exit(1)
        if supports_systemd_services():
            systemd_uninstall(system=system)
        elif is_macos():
            launchd_uninstall()
        elif is_container():
            print("Docker 컨테이너 안에서는 서비스 제거가 해당되지 않아요.")
            print("게이트웨이를 중지하려면 컨테이너를 중지하거나 제거하세요:")
            print()
            print("  docker stop <container>")
            print("  docker rm <container>")
            sys.exit(0)
        else:
            print("이 플랫폼에서는 지원하지 않아요.")
            sys.exit(1)

    elif subcmd == "start":
        system = getattr(args, 'system', False)
        if is_termux():
            print("Termux에는 시스템 서비스 관리자가 없어서 게이트웨이 서비스 시작을 지원하지 않아요.")
            print("수동 실행: hermes gateway")
            sys.exit(1)
        if supports_systemd_services():
            systemd_start(system=system)
        elif is_macos():
            launchd_start()
        elif is_wsl():
            print("WSL이 감지되었지만 systemd를 사용할 수 없어요.")
            print("대신 게이트웨이를 포그라운드 모드로 실행하세요:")
            print()
            print("  hermes gateway run                              # 직접 포그라운드 실행")
            print("  tmux new -s hermes 'hermes gateway run'         # tmux로 유지")
            print("  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # 백그라운드")
            print()
            print("systemd를 활성화하려면 /etc/wsl.conf 에 systemd=true 를 추가하고 PowerShell에서 'wsl --shutdown'을 실행하세요.")
            sys.exit(1)
        elif is_container():
            print("Docker 컨테이너 안에서는 서비스 시작이 해당되지 않아요.")
            print("게이트웨이는 컨테이너의 메인 프로세스로 실행돼요.")
            print()
            print("  docker start <container>     # 중지된 컨테이너 시작")
            print("  docker restart <container>   # 실행 중인 컨테이너 재시작")
            print()
            print("또는 게이트웨이를 직접 실행하세요: hermes gateway run")
            sys.exit(0)
        else:
            print("이 플랫폼에서는 지원하지 않아요.")
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
                print(f"✓ 모든 프로필에서 게이트웨이 프로세스 {total}개를 중지했어요")
            else:
                print("✗ 게이트웨이 프로세스를 찾지 못했어요")
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
                    print("✓ 현재 프로필의 게이트웨이를 중지했어요")
                else:
                    print("✗ 현재 프로필에서 실행 중인 게이트웨이가 없어요")
            else:
                print(f"✓ {get_service_name()} 서비스를 중지했어요")
    
    elif subcmd == "restart":
        # Try service first, fall back to killing and restarting
        service_available = False
        system = getattr(args, 'system', False)
        service_configured = False
        
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
                    print("⚠ 서비스를 통해 게이트웨이를 재시작할 수 없어요 — linger가 활성화되어 있지 않아요.")
                    print("  헤드리스 서버에서 게이트웨이 사용자 서비스가 동작하려면 linger가 필요해요.")
                    print()
                    print(f"  실행:  sudo loginctl enable-linger {_username}")
                    print()
                    print("  그다음 게이트웨이를 다시 시작하세요:")
                    print("    hermes gateway restart")
                    return

            if service_configured:
                print()
                print("✗ 게이트웨이 서비스 재시작에 실패했어요.")
                print("  서비스 정의는 존재하지만 서비스 관리자가 복구하지 못했어요.")
                print("  서비스를 고친 뒤 다시 시도하세요: hermes gateway start")
                sys.exit(1)

            # Manual restart: stop only this profile's gateway
            if stop_profile_gateway():
                print("✓ 현재 프로필의 게이트웨이를 중지했어요")

            _wait_for_gateway_exit(timeout=10.0, force_after=5.0)

            # Start fresh
            print("게이트웨이를 시작하는 중...")
            run_gateway(verbose=0)
    
    elif subcmd == "status":
        deep = getattr(args, 'deep', False)
        system = getattr(args, 'system', False)
        
        # Check for service first
        if supports_systemd_services() and (get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()):
            systemd_status(deep, system=system)
        elif is_macos() and get_launchd_plist_path().exists():
            launchd_status(deep)
        else:
            # Check for manually running processes
            pids = find_gateway_pids()
            if pids:
                print(f"✓ 게이트웨이가 실행 중이에요 (PID: {', '.join(map(str, pids))})")
                print("  (시스템 서비스가 아니라 수동으로 실행 중이에요)")
                runtime_lines = _runtime_health_lines()
                if runtime_lines:
                    print()
                    print("최근 게이트웨이 상태:")
                    for line in runtime_lines:
                        print(f"  {line}")
                print()
                if is_termux():
                    print("Termux 참고:")
                    print("  Termux가 일시중지되면 Android가 백그라운드 작업을 멈출 수 있어요")
                elif is_wsl():
                    print("WSL 참고:")
                    print("  게이트웨이는 포그라운드/수동 모드로 실행 중이에요 (WSL 권장).")
                    print("  터미널을 닫아도 유지하려면 tmux 또는 screen을 사용하세요.")
                else:
                    print("서비스로 설치하려면:")
                    print("  hermes gateway install")
                    print("  sudo hermes gateway install --system")
            else:
                print("✗ 게이트웨이가 실행 중이 아니에요")
                runtime_lines = _runtime_health_lines()
                if runtime_lines:
                    print()
                    print("최근 게이트웨이 상태:")
                    for line in runtime_lines:
                        print(f"  {line}")
                print()
                print("시작하려면:")
                print("  hermes gateway run      # 포그라운드 실행")
                if is_termux():
                    print("  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # 최선의 백그라운드 시작")
                elif is_wsl():
                    print("  tmux new -s hermes 'hermes gateway run'         # tmux로 유지")
                    print("  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # 백그라운드")
                else:
                    print("  hermes gateway install  # 사용자 서비스로 설치")
                    print("  sudo hermes gateway install --system  # 부팅 시 시작되는 시스템 서비스로 설치")

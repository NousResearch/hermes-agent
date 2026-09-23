"""Gateway subcommand for hermes CLI.

Handles: hermes gateway [run|start|stop|restart|status|install|uninstall|setup]
"""

import asyncio
import contextlib
from hermes_cli.cli_output import line_input  # noqa: F401 — resolved lazily by siblings through the facade
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from hermes_cli import setup_platforms  # noqa: F401 — resolved lazily by siblings through the facade

# UV's bundled Python ships a minimal PATH; ensure launchctl/systemctl are discoverable.
if os.name == "posix":
    _sys_dirs = {"/bin", "/usr/bin", "/usr/sbin", "/sbin"}
    _path_dirs = set(os.environ.get("PATH", "").split(os.pathsep))
    _missing = _sys_dirs - _path_dirs
    if _missing:
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + os.pathsep.join(sorted(_missing))

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from gateway.config import coerce_systemd_watchdog_seconds, load_gateway_config  # noqa: F401 — resolved lazily by siblings through the facade
from gateway.status import terminate_pid
from gateway.restart import (  # noqa: F401 — resolved lazily by siblings through the facade
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    EXTERNAL_GATEWAY_SUPERVISOR_ENV,
    GATEWAY_FATAL_CONFIG_EXIT_CODE,
    GATEWAY_SERVICE_RESTART_EXIT_CODE,
    is_gateway_supervisor_process,
    parse_cron_drain_timeout,
    parse_restart_after_turn_timeout,
    parse_restart_drain_timeout,
    resolve_restart_exit_wait_budget,
    resolve_systemd_timeout_stop_sec,
)
from hermes_cli.config import (  # noqa: F401 — resolved lazily by siblings through the facade
    get_env_value,
    get_hermes_home,
    is_managed,
    managed_error,
    read_raw_config,
    save_env_value,
    write_platform_config_field,
)

# display_hermes_home is imported lazily: hermes_constants may be a cached pre-update version.
from hermes_cli.setup import (  # noqa: F401 — resolved lazily by siblings through the facade
    print_header,
    print_info,
    print_success,
    print_warning,
    print_error,
    prompt,
    prompt_choice,
    prompt_yes_no,
)
from hermes_cli.colors import Colors, color  # noqa: F401 — resolved lazily by siblings through the facade
from hermes_cli.gateway_service_unit import (  # noqa: F401 — resolved lazily by siblings through the facade
    _systemd_env_line,
    _installed_unit_ld_library_path,
    _ld_library_path_line,
    _hermes_home_for_target_user,
    _build_service_path_dirs,
    _stable_service_working_dir,
    _systemd_watchdog_seconds,
    _append_node_dir_for_service,
    _service_venv_dir,
    generate_systemd_unit,
    _normalize_service_definition,
    _SYSTEMD_OPTIONAL_DIRECTIVES,
    _strip_optional_systemd_directives,
    _normalize_launchd_plist_for_comparison,
    systemd_unit_is_current,
    _temp_home_in_service_definition,
    _refuse_temp_home_service_write,
    refresh_systemd_unit_if_needed,
)

logger = logging.getLogger(__name__)

# Shared ``subprocess.run`` kwargs for text-mode probes (stdout/stderr captured, decode-tolerant).
_CAPTURE_TEXT = dict(capture_output=True, text=True, encoding="utf-8", errors="replace")

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


from hermes_cli.gateway_process import (  # noqa: E402,F401 - facade exports and patch points
    _get_service_pids,
    _get_parent_pid,
    _is_pid_ancestor_of_current_process,
    _request_gateway_self_restart,
    _graceful_restart_via_sigusr1,
    _wait_for_pid_exit,
    GATEWAY_LOOP_ALIVE,
    GATEWAY_LOOP_WEDGED,
    GATEWAY_LOOP_UNKNOWN,
    DEFAULT_LOOP_LIVENESS_STALE_AFTER_S,
    _LOOP_TICK_ABSENT,
    _probe_loop_tick_socket,
    _ping_loop_tick_witness,
    _probe_loop_tick_tcp,
    _probe_loop_tick_socket_sustained,
    probe_gateway_loop_liveness,
    _escalate_wedged_gateway,
    _get_ancestor_pids,
    _append_unique_pid,
    _iter_proc_cmdlines,
    _scan_gateway_pids,
    _parse_ps_line,
    _iter_windows_list_processes,
    _windows_process_listing,
    _filter_venv_launcher_stubs,
    find_gateway_pids,
    find_profile_gateway_processes,
    _scm_service_field,
    find_windows_gateway_services,
    _gateway_run_args_for_profile,
    _capture_gateway_argv,
    _prepare_profile_gateway_update_restart,
    launch_detached_gateway_restart_by_cmdline,
    launch_detached_profile_gateway_restart,
    _spawn_gateway_restart_watcher,
)

# The facade owns the process dataclasses. Evaluate source-owned postponed annotations here,
# after those classes exist, without making direct sibling imports depend on facade import order.
from hermes_cli import gateway_process as _gateway_process  # noqa: E402
for _process_value in vars(_gateway_process).values():
    if getattr(_process_value, "__module__", None) == _gateway_process.__name__ and callable(_process_value):
        _process_value.__annotations__ = {
            key: eval(value, globals()) if isinstance(value, str) else value
            for key, value in _process_value.__annotations__.items()
        }
del _process_value, _gateway_process


from hermes_cli.gateway_runtime import (
    _systemd_unit_is_active,
    _probe_systemd_service_running,
    _parse_kv_pairs,
    _systemctl_show,
    _unit_environment_value,
    _hermes_home_pinned_by_unit,
    _hermes_home_from_systemd_unit_file,
    _sync_hermes_home_from_systemd_unit,
    _read_systemd_unit_properties,
    _positive_pid,
    _systemd_main_pid_from_props,
    _runtime_state_pid,
    _systemd_main_pid,
    _read_gateway_runtime_status,
    _systemd_cli_bits,
    _wait_for_systemd_service_restart,
    _systemd_restart_wait_timeout,
    _systemd_unit_is_start_limited,
    _systemd_error_indicates_start_limit,
    _systemd_service_is_start_limited,
    _print_systemd_start_limit_wait,
    _recover_pending_systemd_restart,
    _parse_launchd_pid_from_list_output,
    _parse_launchd_pid_from_print_output,
    _launchd_print_service_pid,
    _launchd_service_registered,
    _locate_launchd_gateway_service,
    _probe_launchd_service_running,
    _s6_gateway_snapshot,
    get_gateway_runtime_snapshot,
    _format_gateway_pids,
    _print_gateway_process_mismatch,
    _print_multiplex_standalone_reason,
    _print_served_ingress_urls,
    _print_unserved_shared_ingress,
    _print_other_profiles_gateway_status,
    _print_duplicate_credential_warnings,
    _gateway_list,
    kill_gateway_processes,
    _reaper_candidate_is_supervisor_owned,
    _reap_unsupervised_gateway_orphans,
    _reaper_exclusion_pids,
    _await_gateway_exit,
    _force_kill_survivors,
    _mark_planned_stop,
    stop_profile_gateway,
    _REAPER_SUPERVISOR_WALK_LIMIT,
    _ORPHAN_EXIT_GRACE_SECONDS,
    _ORPHAN_EXIT_POLL_SECONDS,
)




# The runtime shard may be imported before this facade. Its postponed annotations avoid
# importing a partially initialized facade; restore the original evaluated values here.
from hermes_cli import gateway_runtime as _gateway_runtime  # noqa: E402
for _runtime_value in vars(_gateway_runtime).values():
    if getattr(_runtime_value, "__module__", None) == _gateway_runtime.__name__ and callable(_runtime_value):
        _runtime_value.__annotations__ = {
            key: eval(value, globals()) if isinstance(value, str) else value
            for key, value in _runtime_value.__annotations__.items()
        }
del _runtime_value, _gateway_runtime


def is_linux() -> bool:
    return sys.platform.startswith("linux")


from hermes_constants import is_container, is_termux, is_wsl


def _wsl_systemd_operational() -> bool:
    """WSL2 with ``systemd=true`` in wsl.conf has working systemd; WSL1/without it does not."""
    return _systemd_operational(system=True)


def _systemd_operational(system: bool = False) -> bool:
    """Return True when the requested systemd scope is usable."""
    try:
        result = _run_systemctl(["is-system-running"], system=system, timeout=5, **_CAPTURE_TEXT)
    except (RuntimeError, subprocess.TimeoutExpired, OSError):
        return False
    # "running", "degraded", "starting" all mean systemd is PID 1
    return result.stdout.strip().lower() in {"running", "degraded", "starting", "initializing"}


def supports_systemd_services() -> bool:
    if not is_linux() or is_termux() or shutil.which("systemctl") is None:
        return False
    if is_wsl():
        return _wsl_systemd_operational()
    if is_container():
        # A container whose init is systemd (nspawn, some k8s pods) behaves like a host.
        return _systemd_operational(system=False) or _systemd_operational(system=True)
    return True


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_windows() -> bool:
    return sys.platform == "win32"


def _gw_windows():
    """Lazily import :mod:`hermes_cli.gateway_windows` (Windows-only service backend)."""
    from hermes_cli import gateway_windows
    return gateway_windows


# Task Scheduler states meaning "still supervised" (Ready = steady state after the launcher exits).
# Task Scheduler states that mean "this profile still has an official supervisor". Queued is a rare
# in-between. Disabled / MISSING are not supervisors. See #87001.
_WINDOWS_TASK_SUPERVISOR_STATES = frozenset({"Running", "Ready", "Queued"})


def _windows_scheduled_task_state(task_name: str) -> str | None:
    """Locale-independent Task Scheduler state, or None on failure.

    Query the COM API directly: Get-ScheduledTask auto-loads the CIM module,
    which can stall desktop backend startup for the entire ten-second timeout.
    Keep the existing supervisor semantics (Ready and Queued count as owned).
    """
    if not is_windows():
        return None
    quoted_name = task_name.replace("'", "''")
    ps_cmd = (
        "$ErrorActionPreference = 'Stop'; "
        "$s = [Activator]::CreateInstance([type]::GetTypeFromProgID('Schedule.Service')); "
        "$s.Connect(); "
        "try { "
        f"$t = $s.GetFolder('\\').GetTask('{quoted_name}'); "
        # TASK_STATE values are stable, unlike localized schtasks.exe output.
        "@('Unknown', 'Disabled', 'Queued', 'Ready', 'Running')[[int]$t.State] "
        "} catch { "
        "$e = $_.Exception; while ($e.InnerException) { $e = $e.InnerException }; "
        "if ($e.HResult -in @(-2147024894, -2147024893)) { 'MISSING' } else { throw } "
        "}"
    )
    try:
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell is None:
            return None
        result = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
            capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=10,
        )
        if result.returncode != 0:
            return None
        return (result.stdout or "").strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _windows_scheduled_task_supervises(task_name: str) -> bool:
    """True when Task Scheduler still owns this profile's gateway (Ready counts: the task is Ready, not
    Running, after bootstrap exits). Any failure returns False so callers fall back to pidfile / parent-chain.

    Used to treat Task Scheduler as a gateway supervisor on Windows: the orphan-reap sweep must not kill a
    gateway that a scheduled task launched and left detached. After the bootstrap exits the task is Ready,
    not Running; a Running-only check still writes the planned-stop marker, the gateway exits cleanly with
    code 0, and the scheduler never restarts it — silently killing A2A/messaging on every desktop-app launch
    (#86098, #87001).
    """
    return _windows_scheduled_task_state(task_name) in _WINDOWS_TASK_SUPERVISOR_STATES


def _gateway_detached_env() -> bool:
    return _truthy_env(os.getenv("HERMES_GATEWAY_DETACHED"))


def _stdin_is_tty() -> bool | None:
    """``sys.stdin.isatty()``; None when stdin is closed/invalid."""
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except (ValueError, OSError):
        return None


def _windows_gateway_should_absorb_console_controls() -> bool:
    """True for detached Windows gateway runs that should ignore Ctrl+C (``HERMES_GATEWAY_DETACHED=1``
    or no interactive stdin); foreground runs stay interruptible."""
    if not is_windows():
        return False
    if _gateway_detached_env():
        return True
    return not _stdin_is_tty()


def _windows_console_window_attached() -> bool | None:
    """Return whether Windows assigned this process a console window."""
    if not is_windows():
        return None
    try:
        import ctypes
        return bool(ctypes.windll.kernel32.GetConsoleWindow())  # type: ignore[attr-defined]
    except (OSError, AttributeError):
        return None


def _windows_gateway_breakaway_state() -> bool | None:
    """Consume private spawn metadata without guessing for older launchers."""
    if not is_windows():
        return None
    from hermes_cli._subprocess_compat import _WINDOWS_GATEWAY_BREAKAWAY_ENV
    return {"1": True, "0": False}.get(os.environ.pop(_WINDOWS_GATEWAY_BREAKAWAY_ENV, None))


# =============================================================================
# Service Configuration
# =============================================================================

_SERVICE_BASE = "hermes-gateway"
SERVICE_DESCRIPTION = "Hermes Agent Gateway - Messaging Platform Integration"

_SYSTEM_UNIT_DIR = Path("/etc/systemd/system")


def _profile_name_from_home(home: Path, default: Path) -> str | None:
    """Profile name when ``home`` is ``<default>/profiles/<name>`` with a service-safe name, else None."""
    import re
    try:
        parts = home.relative_to((default / "profiles").resolve()).parts
    except ValueError:
        return None
    if len(parts) == 1 and re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", parts[0]):
        return parts[0]
    return None


def _native_service_homes() -> set[Path]:
    """This process's native default home plus, when root under sudo, the invoking user's (see
    ``_profile_suffix`` for why sudo matters)."""
    from hermes_constants import _get_platform_default_hermes_home, sudo_invoker_default_home

    homes = {_get_platform_default_hermes_home().resolve()}
    sudo_home = sudo_invoker_default_home()
    if sudo_home is not None:
        homes.add(sudo_home.resolve())
    return homes


def _bare_unit_pinned_home() -> Path | None:
    """Resolved ``HERMES_HOME`` pinned by an installed ``hermes-gateway.service``, or None. The unit is the
    one naming basis that holds still across the sudo mid-command switch (see ``_profile_suffix``) and it
    covers every elevated identity — ``sudo -i`` and cron included, where SUDO_USER is absent.

    Linux- and root-gated: a systemd unit is not an identity authority for launchd labels, Windows
    scheduled tasks, or s6 slots, which share ``_profile_suffix()``, and only an elevated process ever
    operates the system unit — an unprivileged user-scope command must keep naming its own units, or a
    bare system unit pinning ``profiles/<name>`` would alias that profile onto the user's default unit.
    ``is_linux()`` is a plain ``sys.platform`` test; ``supports_systemd_services()`` would be wrong here,
    since it can shell out to ``systemctl is-system-running`` on WSL/containers and this runs on every
    name resolution.
    """
    if not is_linux() or os.geteuid() != 0:  # windows-footgun: ok — behind is_linux()
        return None
    pinned = _hermes_home_pinned_by_unit(_SYSTEM_UNIT_DIR / f"{_SERVICE_BASE}.service")
    if not pinned:
        return None
    try:
        return Path(pinned).expanduser().resolve()
    except (RuntimeError, ValueError):  # hand-edited unit: ``~nouser`` or an embedded NUL
        return None


def _profile_suffix() -> str:
    """Service-name suffix for HERMES_HOME: "" for a home that owns the bare name, the profile name for
    ``<root>/profiles/<name>``, else a short hash of the path.

    Bare-name owners: this process's platform-native default (``~/.hermes``), under sudo the invoking
    user's native default, and the home pinned by an installed ``hermes-gateway.service``. Under sudo the
    naming basis moves MID-COMMAND — sudo strips HERMES_HOME and sets HOME=/root, then
    ``_sync_hermes_home_from_systemd_unit()`` adopts the unit's own HERMES_HOME into ``os.environ`` — so a
    basis derived from the process alone names one unit before the adoption and another after it. The
    unit-pinned check must precede the profile branch: ``sudo hermes gateway install --system`` resolves
    the BARE name from root's default, then pins the invoking user's remapped home, so the bare unit
    legitimately carries a ``<root>/profiles/<name>`` home.

    The bare name is deliberately NOT tied to ``get_default_hermes_root()``: that helper treats any
    HERMES_HOME outside ``~/.hermes`` (Docker ``/opt/data``, a temp dir) as "the root itself", which let a
    temp-home harness resolve to the default profile's ``hermes-gateway`` unit and uninstall the
    production gateway. Service names are host-wide identities; a home with no installed bare unit and
    no native default keeps its own suffix.
    """
    import hashlib
    from hermes_constants import get_default_hermes_root
    home = get_hermes_home().resolve()
    if home in _native_service_homes() or home == _bare_unit_pinned_home():
        return ""
    name = _profile_name_from_home(home, get_default_hermes_root().resolve())
    return name or hashlib.sha256(str(home).encode()).hexdigest()[:8]


def _current_profile_name() -> str:
    """Profile id relative to the profile ROOT: ``default`` for the root itself (Docker's ``/opt/data``
    included), ``<name>`` for ``<root>/profiles/<name>``, else the service hash. s6 slots and the
    multiplexer ask which PROFILE this is; ``_profile_suffix()`` answers which HOST SERVICE this is."""
    from hermes_constants import profile_name_for_home
    return profile_name_for_home(get_hermes_home()) or _profile_suffix()


def _profile_arg(hermes_home: str | None = None, default_root: str | Path | None = None) -> str:
    """``--profile <name>`` for ``<root>/profiles/<name>``, else "". *hermes_home*/*default_root* let a
    sudo/root process generate a unit for another user (the defaults would refer to root)."""
    from hermes_constants import get_default_hermes_root
    home = Path(hermes_home or str(get_hermes_home())).resolve()
    default = Path(default_root).resolve() if default_root else get_default_hermes_root().resolve()
    if home == default:
        return ""
    name = _profile_name_from_home(home, default)
    return f"--profile {name}" if name else ""


def get_service_name() -> str:
    """Systemd service name: ``hermes-gateway`` for default HERMES_HOME, ``hermes-gateway-<profile>``
    or ``-<hash>`` otherwise."""
    suffix = _profile_suffix()
    return f"{_SERVICE_BASE}-{suffix}" if suffix else _SERVICE_BASE




class UserSystemdUnavailableError(RuntimeError):
    """``systemctl --user`` cannot reach the user D-Bus session (fresh SSH sessions with linger off,
    so ``/run/user/$UID/bus`` never exists). ``args[0]`` is a user-facing remediation message."""


class SystemScopeRequiresRootError(RuntimeError):
    """System-scope gateway operation attempted as non-root. Typed (not ``sys.exit(1)``) so the setup
    wizard can print remediation; ``args`` = (message, action) and ``str(e)`` is the message only."""

    def __str__(self) -> str:
        return self.args[0] if self.args else ""




def get_launchd_plist_path() -> Path:
    """``~/Library/LaunchAgents/ai.hermes.gateway[-<profile>].plist`` under the real account home."""
    import pwd
    suffix = _profile_suffix()
    name = f"ai.hermes.gateway-{suffix}" if suffix else "ai.hermes.gateway"
    # Real account home: profile mode may point HOME at a profile dir.
    home = Path(pwd.getpwuid(os.getuid()).pw_dir)  # windows-footgun: ok — POSIX launchd (macOS) helper, never invoked on Windows
    return home / "Library" / "LaunchAgents" / f"{name}.plist"


def launchd_gateway_labels_for_install() -> list[str]:
    """Launchd labels for every profile of THIS install (root first, then profiles by name). Derived from
    the profile layout, NOT by globbing ``~/Library/LaunchAgents``, so a sandboxed HERMES_HOME never
    restarts another install's fleet. Names that can't map to a suffix are skipped."""
    import re as _re
    from hermes_cli.profiles import list_profiles
    root_label: list[str] = []
    profile_labels: list[str] = []
    for profile in list_profiles():
        if profile.is_default:
            root_label.append("ai.hermes.gateway")
        elif _re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", profile.name):
            profile_labels.append(f"ai.hermes.gateway-{profile.name}")
    return root_label + sorted(profile_labels)


def legacy_launchd_labels_for_install(exclude=()) -> list[str]:
    """Launchd labels of THIS install that the profile-layout derivation can't map (#115254).

    A unit whose label predates the profile-name suffix scheme (``ai.hermes.gateway-<8hex>`` from the
    historical hash suffix) is invisible to ``launchd_gateway_labels_for_install()`` and therefore to
    the update restart pass. This reads the account's LaunchAgents and credits a plist only when its
    pinned ``HERMES_HOME`` is this install's root or one of its ``profiles/<name>`` homes — ownership
    judged from the plist's content, never from label shape or directory membership — so the
    derivation's boundary holds: a sandboxed HERMES_HOME (tests, side-by-side installs) never
    enumerates, let alone restarts, another install's fleet (#41403).
    """
    import plistlib
    import pwd

    from hermes_constants import get_default_hermes_root

    try:
        home = Path(pwd.getpwuid(os.getuid()).pw_dir)  # windows-footgun: ok — POSIX launchd (macOS) helper, never invoked on Windows
        root = get_default_hermes_root().resolve()
    except Exception:
        return []
    agents_dir = home / "Library" / "LaunchAgents"
    if not agents_dir.is_dir():
        return []
    excluded = set(exclude)
    labels: set[str] = set()
    for plist_path in sorted(agents_dir.glob("ai.hermes.gateway*.plist")):
        try:
            data = plistlib.loads(plist_path.read_bytes())
            label = data["Label"]
            pinned = Path(str(data["EnvironmentVariables"]["HERMES_HOME"])).expanduser().resolve()
            rel = pinned.relative_to(root).parts
        except Exception:
            continue  # unreadable plist, no pinned home, or a home outside this root: not ours — fail closed
        if not isinstance(label, str) or label in excluded or not label.startswith("ai.hermes.gateway"):
            continue
        if not rel or (len(rel) == 2 and rel[0] == "profiles"):
            labels.add(label)
    return sorted(labels)


from hermes_cli.gateway_systemd import (
    user_systemd_unit_dir,
    get_systemd_unit_path,
    _user_runtime_dir,
    _user_dbus_socket_path,
    _user_systemd_private_socket_path,
    _path_exists_safe,
    _runtime_dir_is_ours,
    _user_systemd_socket_ready,
    _ensure_user_systemd_env,
    _wait_for_user_dbus_socket,
    _wait_for_target_user_bus,
    _loginctl_enable_linger,
    _completed_process_detail,
    _preflight_user_systemd,
    _raise_user_systemd_unavailable,
    _systemctl_cmd,
    _run_systemctl,
    _service_scope_label,
    get_installed_systemd_scopes,
    has_conflicting_systemd_units,
    _legacy_unit_search_paths,
    _find_legacy_hermes_units,
    has_legacy_hermes_units,
    print_legacy_unit_warning,
    remove_legacy_hermes_units,
    print_systemd_scope_conflict_warning,
    refuses_container_user_scope_install,
    _require_root_for_system_service,
    _system_service_identity,
    _read_systemd_user_from_unit,
    _default_system_service_user,
    prompt_linux_gateway_install_scope,
    install_linux_gateway_from_setup,
    ensure_gateway_service,
    get_systemd_linger_status,
    _detect_venv_dir,
    get_python_path,
    _build_user_local_paths,
    _build_wsl_interop_paths,
    _remap_path_for_user,
    _print_linger_enable_warning,
    _ensure_linger_enabled,
    _ensure_system_service_linger,
    _select_systemd_scope,
    _system_scope_wizard_would_need_root,
    _print_system_scope_remediation,
    _get_restart_drain_timeout,
    _agent_timeout_setting,
    _get_cron_drain_timeout,
    _get_restart_exit_wait_budget,
    systemd_install,
    _systemd_scope_preamble,
    _systemd_unit_belongs_to_current_home,
    systemd_uninstall,
    _print_service_not_installed,
    _require_service_installed,
    systemd_start,
    systemd_stop,
    systemd_restart,
    _systemd_graceful_restart_action,
    _systemd_reset_and_run,
    systemd_status,
    _LEGACY_SERVICE_NAMES,
    _LEGACY_UNIT_EXECSTART_MARKERS,
)




# =============================================================================
# Launchd (macOS)
# =============================================================================


from hermes_cli.gateway_launchd import (  # noqa: E402,F401 — facade re-exports; tests patch here
    get_launchd_label,
    _probe_launchd_domain_for_label,
    _launchd_domain,
    _LAUNCHD_JOB_UNLOADED_EXIT_CODES,
    _LAUNCHCTL_DOMAIN_UNSUPPORTED_CODES,
    _launchd_error_indicates_unloaded,
    _launchctl_domain_unsupported,
    _LAUNCHCTL_BOOTSTRAP_EIO,
    _launchctl_bootstrap,
    _launchd_reload_log_path,
    _append_launchd_reload_log,
    _launchd_reload_budget,
    _launchctl_supervised_pid,
    _launchctl_label_supervising_process,
    _retry_launchctl_bootstrap_until_registered,
    _launchd_unsupported_marker_path,
    _write_launchd_unsupported_marker,
    _clear_launchd_unsupported_marker,
    _launchd_unsupported_marker_exists,
    _gateway_run_command,
    _timestamped_stderr_gateway_command,
    _spawn_detached_gateway,
    _launchd_fallback_to_detached,
    _launchd_degrade_or_raise,
    generate_launchd_plist,
    launchd_plist_is_current,
    _spawn_deferred_launchd_reload,
    refresh_launchd_plist_if_needed,
    launchd_install,
    launchd_uninstall,
    launchd_start,
    _launchctl_kickstart_current,
    _launchd_bootstrap_and_kickstart,
    _launchd_ok,
    launchd_stop,
    _launchd_kickstart,
    _wait_for_launchd_service_pid,
    launchd_restart,
    LAUNCHD_SUPERVISION_VERIFY_TIMEOUT,
    wait_for_launchd_gateway_supervision,
    launchd_status,
)


# Cached launchd domain — probe once per process invocation.
_resolved_launchd_domain: str | None = None



from hermes_cli.gateway_startup import (  # noqa: E402,F401 — facade re-exports; tests patch here
    _wait_for_gateway_exit,
    _wait_for_tcp_port_free,
    _wait_for_api_server_port_free,
    _truthy_env,
    _is_official_docker_checkout,
    _running_under_gateway_supervisor,
    host_multiplexer_serving,
    _served_by_another_host_gateway,
    named_profile_served_by_running_multiplexer,
    _served_profile_needs_no_service,
    _named_profile_refused_under_multiplexer,
    _guard_named_profile_under_multiplexer,
    _host_decision_exit_code,
    _attach_to_host_gateway_or_guard,
    _guard_supervised_gateway_conflict,
    _guard_existing_gateway_process_conflict,
    _guard_official_docker_root_gateway,
    _apply_startup_watchdog_config,
    _absorb_windows_console_controls,
    _make_exit_diag,
    _respawn_storm_backoff,
    run_gateway,
)


# =============================================================================
# Gateway Setup (Interactive Messaging Platform Configuration)
# =============================================================================

from hermes_cli.gateway_setup_wizard import (  # noqa: E402,F401 — facade re-exports; tests patch here
    _PLATFORMS,
    _all_platforms,
    _platform_status,
    _set_platform_unauthorized_dm_behavior,
    _print_setup_header,
    _confirm_reconfigure,
    _offer_home_channel,
    _save_env_values,
    _prompt_csv,
    _UNAUTHORIZED_ACCESS_CHOICES,
    _prompt_unauthorized_access,
    _telegram_auto_setup,
    _clean_discord_ids,
    _prompt_allowlist_var,
    _setup_standard_platform,
    _WEIXIN_DM_POLICIES,
    _WEIXIN_GROUP_NOTE,
    _setup_weixin,
    _setup_qqbot,
    _signal_line_input,
    _setup_signal,
    _builtin_setup_fn,
    _configure_platform,
    _wizard_offer_service_action,
    _setup_service_action,
    _WIZARD_BANNER,
    _WIZARD_BACKEND_LABELS,
    _WIZARD_NO_SERVICE_LINES,
    _wizard_service_status_block,
    _wizard_platform_loop,
    _wizard_install_service,
    _wizard_post_setup,
    gateway_setup,
)


# Operator wording for the out-of-loop watchdog exit reasons stamped by gateway/shutdown_watchdog.py.
_WATCHDOG_EXIT_REASONS = {
    "loop_liveness_watchdog": (
        "event loop stopped dispatching (housekeeping, cron and the kanban dispatcher froze); "
        f"the liveness watchdog exited with code {GATEWAY_SERVICE_RESTART_EXIT_CODE} for the supervisor to restart it"
    ),
    "shutdown_watchdog": "shutdown drain wedged; the shutdown watchdog forced the exit (see logs/gateway-shutdown-watchdog.log)",
}


def _runtime_health_lines() -> list[str]:
    """Summarize the latest persisted gateway runtime health state."""
    try:
        from gateway.status import (
            read_runtime_status, runtime_status_heartbeat_age_s, runtime_status_is_stale, runtime_status_pid_is_live)
    except Exception:
        return []

    state = read_runtime_status()
    if not state:
        return []

    gateway_state = state.get("gateway_state")
    exit_reason = state.get("exit_reason")
    lines = [
        f"⚠ {platform}: {pdata.get('error_message') or 'unknown error'}"
        for platform, pdata in (state.get("platforms", {}) or {}).items()
        if pdata.get("state") == "fatal"
    ]

    # A live-claiming snapshot can outlive an ungracefully killed gateway (taskkill /F, OOM). Past
    # the freshness TTL with the recorded PID gone, say so instead of rendering stale live state.
    if gateway_state in ("running", "degraded", "starting", "draining") and runtime_status_is_stale(state):
        if not runtime_status_pid_is_live(state):
            lines.append(
                f"⚠ Stale gateway_state.json: recorded state '{gateway_state}' but the "
                "recorded process is gone (likely an ungraceful shutdown)"
            )
            return lines
        # PID alive but housekeeping stopped re-stamping the file: the reporter's "not a crash" case
        # (#113372) — the process looks 'running' while housekeeping/cron/kanban dispatch are frozen.
        age = runtime_status_heartbeat_age_s(state)
        if gateway_state != "draining" and age is not None:
            lines.append(
                f"⚠ Gateway heartbeat stale: housekeeping has not refreshed gateway_state.json for {age} s "
                f"(event loop or housekeeping wedged; pid {state.get('pid')} alive) — restart the gateway"
            )

    if gateway_state == "startup_failed" and exit_reason:
        lines.append(f"⚠ Last startup issue: {exit_reason}")
    elif gateway_state == "degraded" and exit_reason:
        # An out-of-loop watchdog hard-exited the process (#113372): the loop stopped dispatching, so
        # housekeeping/cron/kanban froze together. Without this arm the file would read 'running'.
        lines.append(f"⚠ Gateway exited degraded: {_WATCHDOG_EXIT_REASONS.get(exit_reason, exit_reason)}")
    elif gateway_state == "draining":
        action = "restart" if state.get("restart_requested") else "shutdown"
        from gateway.status import parse_active_agents
        count = parse_active_agents(state.get("active_agents"))
        lines.append(f"⏳ Gateway draining for {action} ({count} active agent(s))")
        work = state.get("active_work")
        if isinstance(work, list) and work:
            from hermes_cli.update_cmd_drain_report import describe_active_work_unit
            lines.extend(f"     • {describe_active_work_unit(u)}" for u in work if isinstance(u, dict))
    elif gateway_state == "stopped" and exit_reason:
        lines.append(f"⚠ Last shutdown reason: {exit_reason}")

    return lines



def _print_info_lines(*lines: str) -> None:
    for line in lines:
        print_info(line)



# WhatsApp/DingTalk/WeCom/Feishu setup flows live in their plugins' adapter.py::interactive_setup.


def _running_under_s6() -> bool:
    from hermes_cli.service_manager import detect_service_manager
    return detect_service_manager() == "s6"


def _systemd_unit_installed() -> bool:
    return supports_systemd_services() and (
        get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()
    )


def _is_service_installed() -> bool:
    return _installed_service_kind() is not None


def _is_service_running() -> bool:
    """Check if the gateway service is currently running."""
    if supports_systemd_services():
        return _systemd_unit_is_active(False) or _systemd_unit_is_active(True)
    if is_macos() and get_launchd_plist_path().exists():
        try:
            return _launchd_service_registered(get_launchd_label(), timeout=10)
        except subprocess.TimeoutExpired:
            return False
    # Windows "installed" doesn't mean "running"; like manual runs, a live gateway process decides.
    return len(find_gateway_pids()) > 0



def _print_indented(text: str, emit=print) -> None:
    for line in text.splitlines():
        emit(f"  {line}")


def _service_backend(*, windows: bool = True) -> str | None:
    """Host service manager: ``"systemd"`` / ``"launchd"`` / ``"windows"`` / None, in the canonical
    predicate order every subcommand routes on. ``windows=False`` never probes ``is_windows()``."""
    if supports_systemd_services():
        return "systemd"
    if is_macos():
        return "launchd"
    if windows and is_windows():
        return "windows"
    return None


def _service_call(backend: str, verb: str, system: bool | None = False) -> None:
    """Run ``verb`` (start/stop/restart/uninstall) on ``backend``. Names resolve at call time so tests
    can monkeypatch them; only systemd takes a scope, and ``system=None`` omits it (wizard restart)."""
    if backend == "windows":
        return getattr(_gw_windows(), verb)()
    if backend == "launchd":
        return globals()[f"launchd_{verb}"]()
    fn = globals()[f"systemd_{verb}"]
    return fn() if system is None else fn(system=system)



# =============================================================================
# Main Command Handler
# =============================================================================

def _dispatch_via_service_manager_if_s6(action: str, profile: str | None = None) -> bool:
    """Dispatch start/stop/restart via s6 inside an s6 container; True iff dispatched (caller returns).
    Profile defaults to the current one; missing slot / s6 errors become actionable CLI messages."""
    from hermes_cli.service_manager import (
        GatewayNotRegisteredError, detect_service_manager, get_service_manager,
        register_unregistered_profile_gateway,
    )

    if detect_service_manager() != "s6":
        return False
    if profile is None:
        profile = _current_profile_name()  # root home (Docker /opt/data included) is gateway-default
    mgr = get_service_manager()
    if action not in ("start", "stop", "restart"):
        return False
    service = f"gateway-{profile}"
    try:
        try:
            getattr(mgr, action)(service)
        except GatewayNotRegisteredError:
            # A profile created from the HOST against a bind-mounted home has a directory but no
            # slot (`profile create` cannot reach the container's /run/service). Only `start`
            # repairs that; stop/restart on a missing slot stay an error.
            if action != "start" or not register_unregistered_profile_gateway(mgr, profile):
                raise
            print(f"✓ registered the s6 gateway slot for profile {profile!r}")
            mgr.start(service)
    except (RuntimeError, ValueError, OSError) as exc:  # S6Error is a RuntimeError
        print(f"✗ {exc}")
        sys.exit(1)
    return True


def _dispatch_all_via_service_manager_if_s6(action: str) -> bool:
    """Dispatch ``--all`` stop/restart to every registered profile gateway under s6; True iff dispatched.
    A bare pkill is seen by s6-supervise as a crash and restarted ~1s later; the service manager flips
    ``want up``/``want down`` correctly. ``start --all`` is not a CLI surface."""
    from hermes_cli.service_manager import (detect_service_manager, get_service_manager)
    if detect_service_manager() != "s6" or action not in ("stop", "restart"):
        return False
    mgr = get_service_manager()
    profiles = mgr.list_profile_gateways()
    if not profiles:
        print("✗ No profile gateways registered under s6")
        return True
    fn = mgr.stop if action == "stop" else mgr.restart
    errors: list[tuple[str, Exception]] = []
    for profile in profiles:
        try:
            fn(f"gateway-{profile}")
        except Exception as exc:  # noqa: BLE001 — report and continue
            errors.append((profile, exc))
    succeeded = len(profiles) - len(errors)
    verb = "stopped" if action == "stop" else "restarted"
    if succeeded:
        print(f"✓ {verb.capitalize()} {succeeded} profile gateway(s) under s6")
    for profile, exc in errors:
        print(f"✗ Could not {action} gateway-{profile}: {exc}")
    return True


def gateway_command(args):
    """Handle gateway subcommands."""
    try:
        return _gateway_command_inner(args)
    except UserSystemdUnavailableError as e:
        # Actionable message, not a traceback, when the user D-Bus session is unreachable.
        print_error("User systemd not reachable:")
        _print_indented(str(e))
        sys.exit(1)
    except SystemScopeRequiresRootError as e:
        # System-scope action typed without sudo; the wizard intercepts this earlier with guidance.
        print(str(e))
        sys.exit(1)
    except (subprocess.CalledProcessError, RuntimeError) as e:
        # systemctl exited non-zero or is missing entirely: guidance, not a traceback.
        from hermes_cli.gateway_command_errors import explain_service_failure
        lines = explain_service_failure(e)
        if lines is None:
            raise
        print_error(lines[0])
        _print_indented("\n".join(lines[1:]))
        sys.exit(1)


def _maybe_redirect_run_to_s6_supervision(args) -> bool:
    """Inside an s6 container, upgrade bare ``gateway run`` to the supervised s6 longrun; True iff dispatched.
    ``HERMES_S6_SUPERVISED_CHILD`` (set by ``S6ServiceManager._render_run_script``) marks the supervised
    child, which must run in foreground or we'd recurse run → start → run; ``--no-supervise`` /
    HERMES_GATEWAY_NO_SUPERVISE=1 opts out (CI smoke, debugging)."""
    no_supervise = getattr(args, "no_supervise", False) or \
        os.environ.get("HERMES_GATEWAY_NO_SUPERVISE", "").lower() in ("1", "true", "yes")
    # HERMES_S6_SUPERVISED_CHILD: we ARE the supervised child; fall through so the gateway starts.
    if no_supervise or os.environ.get("HERMES_S6_SUPERVISED_CHILD"):
        return False
    if not _dispatch_via_service_manager_if_s6("start"):
        return False
    # This process never reaches a GatewayRunner, so the watchdog armed by hermes_cli.main's argv
    # fast-path has no other disarm site: the in-process heartbeat below parks with zero CPU and no
    # progress lease, which the watchdog reads as a startup deadlock and os._exit(75)s the CMD process.
    from hermes_startup_watchdog import disarm_startup_watchdog

    disarm_startup_watchdog()
    # Breadcrumb on stderr (keep stdout clean for scripts); gateway logs follow via s6-log.
    print(
        "→ gateway is now running under s6 supervision (auto-restart on crash,\n"
        "  dashboard supervised alongside if HERMES_DASHBOARD is set).\n"
        "  This is the recommended setup for the s6 container image — the\n"
        "  gateway will keep running even if it crashes.\n"
        "  Use `--no-supervise` (or HERMES_GATEWAY_NO_SUPERVISE=1) to opt out\n"
        "  and get the pre-s6 foreground behavior instead.",
        file=sys.stderr,
        flush=True,
    )
    # Keep the CMD process alive as a heartbeat so the container survives gateway flaps (`docker stop`
    # SIGTERMs it). Prefer `sleep infinity` (frees the interpreter); execvp only returns by raising
    # (ENOENT with a clobbered PATH / no `sleep`), which used to crash containers.
    try:
        # The supervised gateway's lifetime is independent of this process — s6-supervise restarts it on
        # crash, and we don't want the container to exit when the gateway flaps. The CMD process keeps /init
        # alive until `docker stop` sends SIGTERM, at which point /init runs stage 3 shutdown (which tears
        # down the supervised gateway cleanly). Prefer `sleep infinity` (matches the static main-hermes
        # service's pattern in docker/s6-rc.d/main-hermes/run, and frees the Python interpreter — the
        # heartbeat is a tiny `sleep` process, not a resident interpreter). But `os.execvp` does a PATH
        # lookup for the `sleep` binary and historically crashed the whole container with FileNotFoundError
        # when PATH was empty/truncated/clobbered at this point — e.g. after user customizations rewrote
        # PATH, or on minimal images without `sleep` on PATH (issue #36208). Fall back to an in-process
        # block (no external binary, can't fail on PATH) so the container keeps running instead of dying
        # during boot.
        os.execvp("sleep", ["sleep", "infinity"])
    except OSError:
        print(
            "→ `sleep` is unavailable; keeping the s6 CMD process alive "
            "in-process until the container is stopped.",
            file=sys.stderr,
            flush=True,
        )
        _block_until_terminated()
    return True  # unreachable on the execvp success path


def _block_until_terminated() -> None:
    """Heartbeat when ``execvp("sleep")`` fails. SIGTERM exits 128+signum so ``docker stop`` is clean;
    ``Event().wait()`` covers platforms without ``signal.pause()``.

    Fallback heartbeat for when ``os.execvp("sleep", ...)`` can't run (``sleep`` missing from PATH — issue
    #36208). Installs a SIGTERM handler that exits with the conventional 128+signum code so ``docker stop``
    produces a clean, expected exit, then blocks on ``signal.pause()``. Windows) — although this path only
    runs inside the s6 Linux container image, the fallback keeps the helper safe to import and unit-test
    anywhere.
    """
    signal.signal(signal.SIGTERM, lambda signum, _frame: sys.exit(128 + signum))
    pause = getattr(signal, "pause", None)
    if pause is not None:
        while True:
            pause()
    else:  # pragma: no cover - non-Unix fallback, not exercised in the s6 image
        import threading
        threading.Event().wait()


def _installed_service_kind_for(windows) -> str | None:
    """``"systemd"`` / ``"launchd"`` when the unit/plist exists, else ``"windows"`` iff ``windows()``
    (a thunk so it runs last, like every caller's original ladder), else None."""
    if _systemd_unit_installed():
        return "systemd"
    if is_macos() and get_launchd_plist_path().exists():
        return "launchd"
    return "windows" if windows() else None


def _installed_service_kind() -> str | None:
    """Installed service kind; stricter than ``_service_backend`` (unit/plist/task must exist)."""
    return _installed_service_kind_for(lambda: is_windows() and _gw_windows().is_installed())


def _stop_installed_service(system: bool) -> bool:
    """Stop the installed systemd/launchd/Windows service. Returns True if one was stopped."""
    kind = _installed_service_kind()
    if kind is None:
        return False
    # SystemScopeRequiresRootError is a RuntimeError and must propagate from systemd_stop.
    try:
        _service_call(kind, "stop", system)
        return True
    except (subprocess.CalledProcessError, *((RuntimeError,) if kind == "windows" else ())):
        return False


def _refuse_from_inside_gateway(verb: str, reason: str) -> None:
    """Refuse self-targeting stop/restart/uninstall from inside the gateway process (#92560)."""
    from tools.process_registry import _is_supervised_gateway_process
    if _is_supervised_gateway_process():
        print_error(
            f"Refusing to {verb} the gateway from inside the gateway process.\n"
            f"This command was blocked to prevent {reason}.\n"
            f"Use `hermes gateway {verb}` from a shell outside the running gateway."
        )
        sys.exit(1)


def _print_lines(*lines: str) -> None:
    for line in lines:
        print(line)


def _print_runtime_health() -> None:
    runtime_lines = _runtime_health_lines()
    if runtime_lines:
        print()
        print("Recent gateway health:")
        for line in runtime_lines:
            print(f"  {line}")


def _cmd_run(args):
    if _maybe_redirect_run_to_s6_supervision(args):
        return  # unreachable; execvp doesn't return
    if getattr(args, "external_supervisor", False):
        os.environ[EXTERNAL_GATEWAY_SUPERVISOR_ENV] = "1"
    run_gateway(
        getattr(args, "verbose", 0), quiet=getattr(args, "quiet", False),
        replace=getattr(args, "replace", False), force=getattr(args, "force", False),
    )


def _cmd_setup(args):
    gateway_setup()


_WSL_FOREGROUND_HINT = (
    "", "  hermes gateway run                              # direct foreground",
    "  tmux new -s hermes 'hermes gateway run'         # persistent via tmux",
    "  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # background",
)
# ``(exit_code, *lines)`` when a subcommand has no service backend, keyed by (subcommand, reason).
# Reasons in check order: "termux", "wsl" (no operational systemd), "s6" / "container", "unsupported".
# ``None`` exit code means plain return.
_NO_BACKEND_MESSAGES = {
    ("install", "termux"): (1,
        "Gateway service installation is not supported on Termux.", "Run manually: hermes gateway"),
    ("install", "wsl"): (1,
        "WSL detected but systemd is not running.",
        "Either enable systemd (add systemd=true to /etc/wsl.conf and restart WSL)",
        "or run the gateway in foreground mode:", *_WSL_FOREGROUND_HINT),
    ("install", "s6"): (None,
        "Per-profile gateways are auto-registered when you create a profile.", "",
        "  hermes profile create <name>     # creates the s6 service slot",
        "  hermes -p <name> gateway start   # bring it up via s6",
        "  hermes status                    # see currently-supervised gateways"),
    ("install", "container"): (0,
        "Service installation is not needed inside a Docker container.",
        "The container runtime is your service manager — use Docker restart policies instead:", "",
        "  docker run --restart unless-stopped ...   # auto-restart on crash/reboot",
        "  docker restart <container>                # manual restart", "",
        "To run the gateway: hermes gateway run"),
    ("install", "unsupported"): (1,
        "Service installation not supported on this platform.", "Run manually: hermes gateway run"),
    ("uninstall", "termux"): (1,
        "Gateway service uninstall is not supported on Termux because there is no managed service to remove.",
        "Stop manual runs with: hermes gateway stop"),
    ("uninstall", "s6"): (None,
        "Per-profile gateways are auto-unregistered when you delete the profile.", "",
        "  hermes profile delete <name>     # tears down the s6 service slot",
        "  hermes -p <name> gateway stop    # stop without deleting the profile"),
    ("uninstall", "container"): (0,
        "Service uninstall is not applicable inside a Docker container.",
        "To stop the gateway, stop or remove the container:", "",
        "  docker stop <container>", "  docker rm <container>"),
    ("uninstall", "unsupported"): (1,
        "Running the gateway as a background service is not available on this platform "
        "(no systemd, launchd or Scheduled Tasks), so there is nothing to uninstall.",
        "Stop a manually started gateway with: hermes gateway stop"),
    ("start", "termux"): (1,
        "Gateway service start is not supported on Termux because there is no system service manager.",
        "Run manually: hermes gateway"),
    ("start", "wsl"): (1,
        "WSL detected but systemd is not available.",
        "Run the gateway in foreground mode instead:", *_WSL_FOREGROUND_HINT, "",
        "To enable systemd: add systemd=true to /etc/wsl.conf and run 'wsl --shutdown' from PowerShell."),
    ("start", "container"): (0,
        "Service start is not applicable inside a Docker container.",
        "The gateway runs as the container's main process.", "",
        "  docker start <container>     # start a stopped container",
        "  docker restart <container>   # restart a running container", "",
        "Or run the gateway directly: hermes gateway run"),
    ("start", "unsupported"): (1,
        "Running the gateway as a background service is not available on this platform "
        "(no systemd, launchd or Scheduled Tasks).",
        "Run it directly with: hermes gateway run"),
}


def _no_backend_exit(subcommand: str, reason: str) -> None:
    code, *lines = _NO_BACKEND_MESSAGES[(subcommand, reason)]
    _print_lines(*lines)
    if code is not None:
        sys.exit(code)


def _handle_no_backend(subcommand: str, *, wsl: bool, s6: bool) -> None:
    """Fallthrough when no service backend matched. Predicate order: WSL (only when ``wsl``) ->
    container (s6 slot hint only when ``s6``; ``start`` reaches here only when s6 isn't running) ->
    unsupported."""
    if wsl and is_wsl():
        reason = "wsl"
    elif is_container():
        reason = "s6" if s6 and _running_under_s6() else "container"
    else:
        reason = "unsupported"
    _no_backend_exit(subcommand, reason)


def _install_systemd_from_cli(args, *, force: bool, system: bool, run_as_user) -> None:
    if is_wsl():
        print_warning("WSL detected — systemd services may not survive WSL restarts.")
        _print_info_lines(
            "  Consider running in foreground instead: hermes gateway run",
            "  Or use tmux/screen for persistence: tmux new -s hermes 'hermes gateway run'",
        )
        print()
    # Honor --start-now/--start-on-login; else prompt on a TTY, default True headless.
    non_interactive = not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty())

    def _flag(name: str, question: str) -> bool:
        value = getattr(args, name, None)
        if value is not None:
            return value
        return prompt_yes_no(question, True) if not non_interactive else True

    start_now = _flag("start_now", "Start the gateway now after installing the service?")
    start_on_login = _flag("start_on_login", "Start the gateway automatically on login/boot with systemd?")
    systemd_install(
        force=force, system=system, run_as_user=run_as_user,
        enable_on_startup=start_on_login, non_interactive=non_interactive,
    )
    if start_now:
        systemd_start(system=system)


def _cmd_install(args):
    if is_managed():
        managed_error("install gateway service")
        return
    force = getattr(args, "force", False)
    # `--force` doubles as the reinstall flag here; a served profile's unit would only ever exit 78.
    _guard_named_profile_under_multiplexer(force=force)
    system = getattr(args, "system", False)
    run_as_user = getattr(args, "run_as_user", None)
    if is_termux():
        _no_backend_exit("install", "termux")
    backend = _service_backend()
    if backend == "systemd":
        if refuses_container_user_scope_install(system):
            sys.exit(1)
        _install_systemd_from_cli(args, force=force, system=system, run_as_user=run_as_user)
    elif backend == "launchd":
        launchd_install(force, start_now=getattr(args, "start_now", None) is not False)
    elif backend == "windows":
        _gw_windows().install(
            force=force,
            start_now=getattr(args, 'start_now', None),
            start_on_login=getattr(args, 'start_on_login', None),
            elevated_handoff=getattr(args, 'elevated_handoff', False),
        )
    else:
        _handle_no_backend("install", wsl=True, s6=True)


def _cmd_uninstall(args):
    _refuse_from_inside_gateway("uninstall", "the gateway from terminating itself")
    if is_managed():
        managed_error("uninstall gateway service")
        return
    system = getattr(args, "system", False)
    if is_termux():
        _no_backend_exit("uninstall", "termux")
    backend = _service_backend()
    if backend is not None:
        _service_call(backend, "uninstall", system)
    else:
        _handle_no_backend("uninstall", wsl=False, s6=True)


def _host_multiplexer_for_all_verb():
    """The live host gateway a ``--all`` verb must target instead of sweeping every profile.

    ``--all`` means "the ONE host multiplexer", not "every gateway process on this box": sweeping
    killed a multiplexer serving N profiles and started a single gateway in its place, so a
    per-profile command caused a host-wide outage and left exactly one profile served.
    """
    try:
        from gateway.host_attach import host_gateway
        return host_gateway()
    except Exception:
        logger.debug("Host gateway probe failed", exc_info=True)
        return None


def _host_multiplexer_is_ours(owner) -> bool:
    try:
        from gateway.status import _get_process_hermes_home, _same_hermes_home
        return bool(_same_hermes_home(owner.home, _get_process_hermes_home()))
    except Exception:
        return False


def _print_unfolded_gateway_note(owner) -> None:
    """Detect-and-converge: name the per-profile gateways `--all` no longer sweeps, never kill them."""
    try:
        others = [pid for pid in find_gateway_pids(all_profiles=True) if pid != owner.pid]
    except Exception:
        return
    if not others:
        return
    print(f"  {len(others)} per-profile gateway process(es) still run beside it "
          f"(PIDs: {', '.join(str(p) for p in others)}).")
    print("  They were left running; fold them in with: hermes gateway migrate --multiplex")


def _cmd_start(args):
    from hermes_cli.gateway_profile_lifecycle import profile_lifecycle
    if profile_lifecycle("start", args):
        return
    system = getattr(args, "system", False)
    start_all = getattr(args, "all", False)
    force = getattr(args, "force", False)
    _guard_named_profile_under_multiplexer(force=force)
    if not start_all and _dispatch_via_service_manager_if_s6("start"):
        return
    if start_all:
        owner = None if force else _host_multiplexer_for_all_verb()
        if owner is not None:
            # Already up: `--all` has nothing to start, and the old sweep here SIGTERMed this very
            # process before starting a single-profile replacement.
            print(f"✓ The host gateway is already running — {owner.describe()}")
            print("  One gateway per host serves every profile; nothing to start.")
            _print_unfolded_gateway_note(owner)
            return
        killed = kill_gateway_processes(all_profiles=True)
        if killed:
            print(f"✓ Killed {killed} stale gateway process(es) across all profiles")
            _wait_for_gateway_exit(timeout=10.0, force_after=5.0)
            _wait_for_api_server_port_free()

    if is_termux():
        _no_backend_exit("start", "termux")
    backend = _service_backend()
    if backend is not None:
        _service_call(backend, "start", system)
    else:
        _handle_no_backend("start", wsl=True, s6=False)


def _cmd_stop(args):
    _refuse_from_inside_gateway("stop", "restart loops")
    from hermes_cli.gateway_profile_lifecycle import profile_lifecycle
    if profile_lifecycle("stop", args):
        return
    stop_all = getattr(args, "all", False)
    system = getattr(args, "system", False)
    if not stop_all and not find_gateway_pids() and (
            _served_by_another_host_gateway() or named_profile_served_by_running_multiplexer()):
        # The launch/default-profile lifecycle still names the whole host.
        owner = _served_by_another_host_gateway()
        print_error(
            f"The host gateway serves profile '{_current_profile_name()}' — there is no separate "
            f"gateway for this profile to stop."
        )
        if owner is not None:
            print(f"  {owner.describe()}")
        print("  Stop or restart the host gateway instead:")
        print()
        owner_flag = f"-p {owner.profile_label} " if owner is not None else ""
        print(f"    hermes {owner_flag}gateway stop      # takes every served profile offline")
        print(f"    hermes {owner_flag}gateway restart")
        sys.exit(GATEWAY_FATAL_CONFIG_EXIT_CODE)
    # Under s6 a bare pkill is seen as a crash and restarted; go through the supervisor.
    if stop_all and _dispatch_all_via_service_manager_if_s6("stop"):
        return
    if not stop_all and _dispatch_via_service_manager_if_s6("stop"):
        return

    service_available = _stop_installed_service(system)
    if stop_all:
        total = kill_gateway_processes(all_profiles=True) + (1 if service_available else 0)
        if total:
            print(f"✓ Stopped {total} gateway process(es) across all profiles")
        else:
            print("✗ No gateway processes found")
    elif not service_available:
        if stop_profile_gateway():
            print("✓ Stopped gateway for this profile")
        else:
            print("✗ No gateway running for this profile")
    else:
        print(f"✓ Stopped {get_service_name()} service")


def _stop_host_multiplexer(owner) -> int:
    """SIGTERM the host gateway and nothing else; returns how many processes were signalled."""
    stopped = kill_gateway_processes()
    if stopped:
        return stopped
    try:
        terminate_pid(owner.pid, force=False)
        return 1
    except (ProcessLookupError, PermissionError, OSError):
        return 0


def _discard_dead_host_record() -> bool:
    """Drop the host rendezvous record once its owner is provably gone (after a confirmed stop)."""
    try:
        from gateway.host_rendezvous import ROLE_GATEWAY, discard_dead_record

        return discard_dead_record(ROLE_GATEWAY)
    except Exception:
        logger.debug("Host record retraction failed", exc_info=True)
        return False


def _restart_all(system: bool) -> None:
    owner = _host_multiplexer_for_all_verb()
    if owner is not None and not _host_multiplexer_is_ours(owner):
        # `--all` means "restart the ONE host multiplexer" — and this profile does not own it.
        # Sweeping every profile here took the host process down and replaced it with a gateway
        # serving only this profile.
        print_error(f"The host gateway runs under profile '{owner.profile_label}'.")
        print(f"  {owner.describe()}")
        print("  `--all` restarts the one host multiplexer, and this profile is not its owner.")
        print()
        print(f"    hermes -p {owner.profile_label} gateway restart --all")
        sys.exit(GATEWAY_FATAL_CONFIG_EXIT_CODE)

    service_stopped = _stop_installed_service(system)
    if owner is not None:
        # Stop ONLY the host process: its served set comes back with it, and any per-profile
        # gateway a pre-migration host still runs is left alone (never auto-SIGTERMed).
        stopped = _stop_host_multiplexer(owner)
        total = stopped + (1 if service_stopped else 0)
        if total:
            print(f"✓ Stopped the host gateway (PID {owner.pid}; serves "
                  f"{', '.join(owner.profiles) or 'unknown'})")
        _print_unfolded_gateway_note(owner)
    else:
        total = kill_gateway_processes(all_profiles=True) + (1 if service_stopped else 0)
        if total:
            print(f"✓ Stopped {total} gateway process(es) across all profiles")
    _wait_for_gateway_exit(timeout=10.0, force_after=5.0)
    _wait_for_api_server_port_free()
    # Retract the stopped owner's rendezvous record. Leaving it made this very function a silent
    # no-op: the re-entered `gateway run` below read the corpse's record and ATTACHED to it.
    _discard_dead_host_record()

    print("Starting gateway...")
    # Even without a registered task, gateway_windows.start() uses the detached launcher.
    kind = _installed_service_kind_for(is_windows)
    if kind is None:
        # replace=True: if the old owner is still draining (a long drain, an ineffective SIGKILL, a
        # foreign-home owner the scan never saw), take the host over instead of attaching to it.
        run_gateway(verbose=0, replace=True)
    else:
        _service_call(kind, "start", system)


def _cmd_restart(args):
    _refuse_from_inside_gateway("restart", "restart loops")
    from hermes_cli.gateway_profile_lifecycle import profile_lifecycle
    if profile_lifecycle("restart", args):
        return
    system = getattr(args, "system", False)
    restart_all = getattr(args, "all", False)
    force = getattr(args, "force", False)
    # `--all` targets the ONE host multiplexer and _restart_all does its own ownership check with
    # the right one-liner; running the generic named-profile guard first made that branch
    # unreachable for `-p X gateway restart --all` (it printed a bare `gateway restart` instead).
    if not restart_all:
        _guard_named_profile_under_multiplexer(force=force)
    if restart_all and _dispatch_all_via_service_manager_if_s6("restart"):
        return
    if not restart_all and _dispatch_via_service_manager_if_s6("restart"):
        return
    if restart_all:
        _restart_all(system)
        return

    # The Windows restart path handles both registered installs and detached restarts.
    kind = _installed_service_kind_for(is_windows)
    service_configured = kind is not None and (kind != "windows" or _gw_windows().is_installed())
    if kind is not None:
        swallow = (RuntimeError, OSError) if kind == "windows" else ()
        try:
            _service_call(kind, "restart", system)
            return
        except (subprocess.CalledProcessError, *swallow):
            pass

    # Linger only explains a FAILED systemd unit restart. Without an installed unit the
    # detached run below is the restart; bailing here left `hermes gateway restart` a
    # silent exit-0 no-op on any Linux login session (Desktop read it as success).
    if kind == "systemd" and supports_systemd_services():
        linger_ok, _detail = get_systemd_linger_status()
        if linger_ok is not True:
            import getpass
            _print_lines(
                "", "⚠ Cannot restart gateway as a service — linger is not enabled.",
                "  The gateway user service requires linger to function on headless servers.", "",
                f"  Run:  sudo loginctl enable-linger {getpass.getuser()}", "",
                "  Then restart the gateway:", "    hermes gateway restart",
            )
            return

    if service_configured:
        _print_lines(
            "", "✗ Gateway service restart failed.",
            "  The service definition exists, but the service manager did not recover it.",
            "  Fix the service, then retry: hermes gateway start",
        )
        sys.exit(1)

    # A gateway that declares an external supervisor (custom launchd agent / unit running
    # `gateway run --external-supervisor`) restarts by exiting back to it: the stop + foreground
    # run below would stamp this CLI's PID as the gateway and wedge every respawn (#110637).
    from gateway.status import get_running_pid
    from hermes_cli.gateway_supervised_restart import (
        gateway_declares_external_supervisor, restart_externally_supervised_gateway,
    )
    supervised_pid = get_running_pid()
    if supervised_pid and gateway_declares_external_supervisor(supervised_pid):
        restart_externally_supervised_gateway(supervised_pid)
        return

    if stop_profile_gateway():
        print("✓ Stopped gateway for this profile")
    _wait_for_gateway_exit(timeout=10.0, force_after=5.0)
    _wait_for_api_server_port_free()
    print("Starting gateway...")
    run_gateway(verbose=0, force=force)


# ``hermes gateway status`` hints for a manually-run / stopped gateway, keyed by host kind.
_STATUS_RUNNING_HINTS = {
    "termux": ("Termux note:", "  Android may stop background jobs when Termux is suspended"),
    "wsl": (
        "WSL note:", "  The gateway is running in foreground/manual mode (recommended for WSL).",
        "  Use tmux or screen for persistence across terminal closes.",
    ),
    "windows": ("To install as a Windows Scheduled Task (auto-start on login):", "  hermes gateway install"),
    "other": (
        "To install as a service:", "  hermes gateway install", "  sudo hermes gateway install --system",
    ),
}
_STATUS_STOPPED_HINTS = {
    "termux": (
        "  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # Best-effort background start",
    ),
    "wsl": (
        "  tmux new -s hermes 'hermes gateway run'         # persistent via tmux",
        "  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # background",
    ),
    "windows": ("  hermes gateway install  # Install as Windows Scheduled Task (auto-start on login)",),
    "other": (
        "  hermes gateway install  # Install as user service",
        "  sudo hermes gateway install --system  # Install as boot-time system service",
    ),
}


def _status_host_kind() -> str:
    if is_termux():
        return "termux"
    if is_wsl():
        return "wsl"
    return "windows" if is_windows() else "other"


def _cmd_status(args):
    from hermes_cli.gateway_profile_lifecycle import print_parked_status
    if print_parked_status():
        return
    deep = getattr(args, "deep", False)
    full = getattr(args, "full", False)
    system = getattr(args, "system", False)
    snapshot = get_gateway_runtime_snapshot(system=system)
    from hermes_cli.profiles import get_active_profile_name, profile_is_standalone

    active_standalone = ((get_active_profile_name() or "default") != "default"
                         and profile_is_standalone(get_hermes_home()))
    if active_standalone:
        from hermes_cli.gateway_multiplex_mode import STANDALONE_DEPRECATION_NOTICE
        print("standalone by config (gateway.standalone: true) — temporary compatibility shim")
        print(f"  {STANDALONE_DEPRECATION_NOTICE}")
    _windows_service_installed = is_windows() and _gw_windows().is_installed()
    if not active_standalone and not snapshot.running and named_profile_served_by_running_multiplexer():
        # Satellite profile: the default multiplexer is the live inbound process for it.
        print("✓ Gateway is running via the default-profile multiplexer")
        print("  Manage it from the default profile: hermes gateway status")
        _print_served_ingress_urls(get_active_profile_name())
        _print_unserved_shared_ingress(get_active_profile_name())
    elif (kind := _installed_service_kind_for(lambda: _windows_service_installed)) is not None:
        if kind == "systemd":
            systemd_status(deep, system=system, full=full)
        elif kind == "launchd":
            launchd_status(deep)
        else:
            _gw_windows().status(deep=deep)
        _print_gateway_process_mismatch(snapshot)
        _print_multiplex_standalone_reason()
        _print_served_ingress_urls()
    else:
        pids = list(snapshot.gateway_pids)
        if pids:
            print(f"✓ Gateway is running (PID: {', '.join(map(str, pids))})")
            print("  (Running manually, not as a system service)")
            _print_runtime_health()
            _print_multiplex_standalone_reason()
            _print_served_ingress_urls()
            print()
            _print_lines(*_STATUS_RUNNING_HINTS[_status_host_kind()])
        else:
            print("✗ Gateway is not running")
            _print_runtime_health()
            print()
            print("To start:")
            print("  hermes gateway run      # Run in foreground")
            _print_lines(*_STATUS_STOPPED_HINTS[_status_host_kind()])

    _print_duplicate_credential_warnings()
    _print_other_profiles_gateway_status()
    _print_standalone_by_config()


def _print_standalone_by_config() -> None:
    """Default-profile status: name the profiles that opted out of the host multiplexer by config,
    so the served set the host gateway reports is not mistaken for the installed roster."""
    from hermes_cli.profiles import get_active_profile_name, profiles_to_serve
    if (get_active_profile_name() or "default") != "default":
        return
    roster = {name for name, _home in profiles_to_serve(True, include_standalone=True)}
    served = {name for name, _home in profiles_to_serve(True)}
    names = sorted(roster - served - {"default"})
    if names:
        print(f"standalone by config (temporary compatibility shim): {', '.join(names)}")


def _cmd_list(args):
    _gateway_list()


def _cmd_migrate_legacy(args):
    """Stop, disable, and remove legacy Hermes gateway unit files (e.g. hermes.service)."""
    dry_run = getattr(args, "dry_run", False)
    yes = getattr(args, "yes", False)
    if not supports_systemd_services() and not is_macos():
        print("Legacy unit migration only applies to systemd-based Linux hosts.")
        return
    remove_legacy_hermes_units(interactive=not yes, dry_run=dry_run)


def _cmd_migrate(args):
    from hermes_cli.gateway_migrate import cmd_migrate
    cmd_migrate(args)


_GATEWAY_SUBCOMMANDS = {
    None: _cmd_run, "run": _cmd_run, "setup": _cmd_setup, "install": _cmd_install,
    "uninstall": _cmd_uninstall, "start": _cmd_start, "stop": _cmd_stop, "restart": _cmd_restart,
    "status": _cmd_status, "list": _cmd_list, "migrate-legacy": _cmd_migrate_legacy, "migrate": _cmd_migrate,
}


def _gateway_command_inner(args):
    handler = _GATEWAY_SUBCOMMANDS.get(getattr(args, "gateway_command", None))
    if handler is not None:
        handler(args)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

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


_PLUGIN_COMPAT_LAZY = {
    'DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT': ('gateway.restart', 'DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----

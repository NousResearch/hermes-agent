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
from gateway import host_topology as _host_topology
from gateway import restart as _gateway_restart
from gateway import process_liveness as _process_liveness
from gateway import process_discovery as _process_discovery
from gateway.process_discovery import (  # noqa: F401 - compatibility re-exports
    ProfileGatewayProcess,
    WindowsGatewayService,
    _get_service_pids,
    _scan_gateway_pids,
    _parse_ps_line,
    _iter_windows_list_processes,
    _windows_process_listing,
    _filter_venv_launcher_stubs,
    find_profile_gateway_processes,
    find_windows_gateway_services,
)
from gateway import service_identity as _service_identity
from gateway import service_process as _service_process
from gateway import signal_restart as _signal_restart
from gateway import systemd_identity as _systemd_identity
from gateway import systemd_legacy as _systemd_legacy
from gateway import systemd_lifecycle as _systemd_lifecycle
from gateway import systemd_restart as _systemd_restart
from gateway import systemd_restart_state as _systemd_restart_state
from gateway import systemd_runtime as _systemd_runtime
from gateway import systemd_unit_render as _systemd_unit_render
from gateway import systemd_unit_state as _systemd_unit_state
from profiles.paths import profile_name_from_home
from gateway.restart import (  # noqa: F401 — resolved lazily by siblings through the facade
    _get_parent_pid,
    _is_pid_ancestor_of_current_process,
    _request_gateway_self_restart,
    _wait_for_api_server_port_free,
    _wait_for_gateway_exit,
    _wait_for_tcp_port_free,
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


def find_gateway_pids(exclude_pids: set | None = None, all_profiles: bool = False) -> list:
    """Compatibility facade over Gateway-owned process discovery."""
    _exclude = set(exclude_pids or set())
    pids: list[int] = []
    if not all_profiles:
        try:
            from gateway.status import get_running_pid
            _process_discovery._append_unique_pid(pids, get_running_pid(), _exclude)
        except Exception:
            pass
    for pid in _get_service_pids(all_profiles=all_profiles):
        _process_discovery._append_unique_pid(pids, pid, _exclude)
    try:
        include_restart_managers = not _systemd_runtime.supports_services()
    except Exception:
        include_restart_managers = False
    for pid in _scan_gateway_pids(
        _exclude,
        all_profiles=all_profiles,
        include_restart_managers=include_restart_managers,
    ):
        _process_discovery._append_unique_pid(pids, pid, _exclude)
    return pids






# --- Wedged-gateway detection + bounded escalation ---------------------------
# A gateway whose asyncio loop is stalled cannot handle SIGTERM/SIGUSR1, so the drain wait burns
# its full budget and `hermes update` can deadlock. Two witnesses classify the loop BEFORE any
# drain wait: the heartbeat file ``state/gateway.heartbeat`` (rewritten every 30s on a thread, so
# staleness alone is not proof) and the loop-tick socket ``state/gateway.loop-tick.<pid>.sock``
# answered by the loop itself; the payload records whether the socket is armed (``loop_tick_socket``).
# ``alive``: socket answered, or fresh file not contradicted -> normal graceful drain. ``wedged``:
# heartbeat is this PID's, stale past several beats, AND the armed socket stays silent across
# ``tick_strikes`` consecutive misses -> callers may ``_escalate_wedged_gateway``; one silent probe
# is never authority. ``unknown``: no/unreadable heartbeat, PID mismatch, or witness conflict ->
# treated as alive; never escalate on ambiguity. Legacy payloads (no ``loop_tick_socket`` flag)
# wrote on-loop, so staleness alone remains proof.

# --- Wedged-gateway detection + bounded escalation (#81642) ----------------- A gateway whose asyncio loop
# is stalled (e.g. an in-loop compression pass, #72707) cannot process SIGTERM/SIGUSR1 shutdown: the drain
# wait then burns the full drain budget (180s by default), warns "still running after 180.0s — restart may
# fail", and `hermes update` can deadlock behind it. The loop publishes a liveness signal precisely for this
# case: an asyncio task rewrites ``state/gateway.heartbeat`` every 30s (#66892), so a frozen loop stops
# refreshing the file while a busy-but-alive loop keeps refreshing it. Since #90502 the heartbeat write runs
# on a thread (a stalling filesystem must not be able to block the loop the watchdog watches), which costs
# the file its status as *proof*: a stalled write or a saturated executor can age the file while the loop
# runs, and an off-loop write can land after the loop froze, keeping the file fresh for a dead loop. The
# loop therefore also arms a second witness — ``state/gateway.loop-tick.<pid>.sock``, a UNIX socket answered
# by the loop itself — and records whether it is armed in the heartbeat payload (``loop_tick_socket``).
# ``probe_gateway_loop_liveness`` reads both signals (a local stat + JSON read + a bounded socket ping,
# repeated up to ``tick_strikes`` times when a wedge is suspected — worst case ~3.4s, still far inside the
# 10s query tier of the subprocess timeout doc) and classifies the gateway BEFORE any drain wait begins: -
# ``alive``   — the loop answered the tick socket, or the file is fresh and the loop is not contradicted by
# the socket. Callers must take the normal graceful-drain path, which honours the in-flight cron drain floor
# (#86684). - ``wedged``  — the heartbeat belongs to this PID, is stale well past several missed beats, AND
# the tick socket is armed but stays silent across a sustained window of consecutive misses (default 3):
# both witnesses agree, sustained, that the loop is provably dead. One silent probe is never destructive
# authority — a transient synchronous stall can outlast a single recv timeout, so a lone miss falls to
# ``unknown``. Draining is pointless for a provably dead loop (nothing can run the drain), so callers may
# escalate immediately via ``_escalate_wedged_gateway``. - ``unknown`` — no heartbeat / unreadable / PID
# mismatch / witness conflict (fresh file with a silent loop, armed socket unreachable). Treated like
# ``alive``: never escalate on ambiguity. The distinction matters: only a *provably dead* loop may bypass
# the cron drain floor. A merely busy gateway still answers the probe (socket ping) and keeps its full drain
# budget — even when the filesystem is stalling the heartbeat write (the incident that motivated #90502).
# Legacy gateways (no ``loop_tick_socket`` flag in the payload) wrote the file on-loop, so their staleness
# remains proof and the old single-witness contract is unchanged.

# 3 missed 30s beats (gateway.shutdown_watchdog.DEFAULT_HEARTBEAT_INTERVAL_S): decisive, not one slow write.

# Sentinel for "the producer never wrote the witness flag" (legacy payload).


















def _gateway_run_args_for_profile(profile: str) -> list[str]:
    args = [_service_process.python_path(), "-m", "hermes_cli.main"]
    if profile != "default":
        args.extend(["--profile", profile])
    args.extend(["gateway", "run", "--replace"])
    return args


def _capture_gateway_argv(pid: int) -> list[str] | None:
    """Live argv of a running gateway (snapshotted before update kills so unmapped gateways can respawn);
    None if psutil is unavailable, the process is gone/denied, or the argv isn't a gateway command."""
    if pid <= 1:
        return None
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    try:
        argv = list(psutil.Process(pid).cmdline() or [])
    except Exception:  # NoSuchProcess / AccessDenied / ZombieProcess included
        return None
    if not argv:
        return None
    # Never respawn an unrelated process the scan happened to report.
    try:
        from gateway.status import looks_like_gateway_command_line
        if not looks_like_gateway_command_line(" ".join(argv)):
            return None
    except Exception:
        pass
    return argv


def _prepare_profile_gateway_update_restart(profile: str, pid: int) -> str | None:
    """Choose who relaunches a profile gateway after ``hermes update``: ``--external-supervisor`` gateways
    exit back to their manager (a detached watcher would race its replacement); otherwise arm the
    profile-derived detached watcher, falling back to replaying the captured command line.

    When the profile-derived relaunch cannot be armed -- typically because ``_gateway_run_args_for_profile``
    cannot rebuild a run argv for this profile -- fall back to replaying the process's own captured command
    line, which is what ``launch_detached_gateway_restart_by_cmdline`` exists for and what the Windows
    post-update path already does for its unmapped gateways. Without this the caller has no way to relaunch
    the process and (before #88654) silently left it running pre-update modules against post-update code on
    disk. ``argv`` is already captured above, so the fallback costs nothing extra.
    """
    argv = _capture_gateway_argv(pid)
    if argv and "--external-supervisor" in argv:
        return "external-supervisor"
    if launch_detached_profile_gateway_restart(profile, pid):
        return "detached"
    if argv and launch_detached_gateway_restart_by_cmdline(pid, list(argv)):
        return "detached-cmdline"
    return None


def launch_detached_gateway_restart_by_cmdline(old_pid: int, run_argv: list[str]) -> bool:
    """Relaunch a gateway with no profile→PID-file mapping by replaying its captured argv after exit."""
    return old_pid > 0 and bool(run_argv) and _spawn_gateway_restart_watcher(old_pid, list(run_argv))


def launch_detached_profile_gateway_restart(profile: str, old_pid: int) -> bool:
    """Relaunch a manually-run profile gateway after its current PID exits."""
    return old_pid > 0 and _spawn_gateway_restart_watcher(old_pid, _gateway_run_args_for_profile(profile))


def _spawn_gateway_restart_watcher(old_pid: int, run_argv: list[str]) -> bool:
    """Spawn the detached watcher that respawns ``run_argv`` once ``old_pid`` exits. Watcher and respawn
    both need platform-appropriate detach: POSIX setsid; on Windows ``start_new_session`` does NOT detach
    (the watcher would die with the CLI console), so ``windows_detach_popen_kwargs()`` supplies flags."""
    if old_pid <= 0 or not run_argv:
        return False
    from runtime.subprocess_compat import windows_detach_flags_without_breakaway, windows_detach_popen_kwargs

    # Windows: ``run_argv`` leads with the venv's console ``python.exe`` — the interpreter we want:
    # the watcher respawns it under CREATE_NO_WINDOW detach flags so the gateway owns one hidden
    # console all descendants inherit and nothing flashes (#54220/#56747). The spec helper
    # normalizes the interpreter and captures a stable cwd + env overlay (HERMES_HOME,
    # VIRTUAL_ENV, PYTHONPATH) so the respawn doesn't depend on the watcher's cwd. No-op on POSIX.
    respawn_cwd = ""
    # See gateway_windows.windowless_gateway_restart_spec. See #54220, #56747.
    respawn_env_overlay: dict[str, str] = {}
    if sys.platform == "win32":
        try:
            from gateway.windows_service import windowless_gateway_restart_spec
            run_argv, respawn_cwd, respawn_env_overlay = windowless_gateway_restart_spec(list(run_argv))
        except Exception:
            # Fall back to the original argv: a visible window beats a failed respawn.
            respawn_cwd = ""
            respawn_env_overlay = {}

    # cwd/env overlay are embedded as JSON literals in the watcher source (no extra argv plumbing).
    watcher = textwrap.dedent(
        """
        import os
        import subprocess
        import sys
        import time
        from gateway.windows_launch import _WINDOWS_GATEWAY_BREAKAWAY_ENV
        from runtime.subprocess_compat import (
            windows_detach_flags, windows_detach_flags_without_breakaway,
        )

        pid = int(sys.argv[1])
        cmd = sys.argv[2:]
        _respawn_cwd = {respawn_cwd_literal}
        _respawn_env_overlay = {respawn_env_literal}
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            # ``os.kill(pid, 0)`` is not a no-op on Windows — use the cross-platform existence check.
            from gateway.status import _pid_exists
            if not _pid_exists(pid):
                break
            time.sleep(0.2)

        # Route the respawned gateway's stray stdout/stderr to the same sidecar log _spawn_detached
        # uses: with DEVNULL a gateway killed moments after respawn (parent Job Object teardown when
        # breakaway is denied) left ZERO trace. Best-effort: DEVNULL when the log dir is unavailable.
        _stdio_target = subprocess.DEVNULL
        _stdio_fh = None
        try:
            from hermes_cli.config import get_hermes_home
            from pathlib import Path
            _log_dir = Path(get_hermes_home()) / "logs"
            _log_dir.mkdir(parents=True, exist_ok=True)
            _stdio_fh = open(_log_dir / "gateway-stdio.log", "ab", buffering=0)
            _stdio_target = _stdio_fh
        except Exception:
            pass

        # Platform-appropriate detach for the respawned gateway: POSIX start_new_session (setsid);
        # Windows needs explicit creationflags. CREATE_BREAKAWAY_FROM_JOB is critical: the watcher may
        # itself sit inside a job object (Electron/Tauri parent) and without breakaway the respawned
        # gateway dies when that job tears down. See runtime.subprocess_compat.windows_detach_flags().
        _popen_kwargs = {{"stdout": _stdio_target, "stderr": _stdio_target}}
        # Anchor at the stable working dir and overlay the env (VIRTUAL_ENV / PYTHONPATH /
        # HERMES_HOME) the windowless base interpreter needs to import hermes_cli. Empty on POSIX.
        if _respawn_cwd:
            _popen_kwargs["cwd"] = _respawn_cwd
        _base_env = {{**os.environ, **_respawn_env_overlay}}
        try:
            if sys.platform == "win32":
                try:
                    _popen_kwargs["creationflags"] = windows_detach_flags()
                    # Stamp the breakaway state exactly like gateway_windows._spawn_detached so the
                    # respawned gateway's exit-diag / lifecycle records show whether it escaped the
                    # parent Job Object (a job-teardown kill is otherwise indistinguishable).
                    _popen_kwargs["env"] = {{**_base_env, _WINDOWS_GATEWAY_BREAKAWAY_ENV: "1"}}
                    subprocess.Popen(cmd, **_popen_kwargs)
                except OSError:
                    # CREATE_BREAKAWAY_FROM_JOB is rejected with ERROR_ACCESS_DENIED when the parent's
                    # job object refuses breakaway; retry without it (mirrors _spawn_detached).
                    _popen_kwargs["creationflags"] = windows_detach_flags_without_breakaway()
                    _popen_kwargs["env"] = {{**_base_env, _WINDOWS_GATEWAY_BREAKAWAY_ENV: "0"}}
                    subprocess.Popen(cmd, **_popen_kwargs)
            else:
                if _respawn_env_overlay:
                    _popen_kwargs["env"] = _base_env
                _popen_kwargs["start_new_session"] = True
                subprocess.Popen(cmd, **_popen_kwargs)
        finally:
            if _stdio_fh is not None:
                try:
                    _stdio_fh.close()
                except OSError:
                    pass
        """
    ).strip().format(respawn_cwd_literal=json.dumps(respawn_cwd), respawn_env_literal=json.dumps(respawn_env_overlay))

    watcher_argv = [sys.executable, "-c", watcher, str(old_pid), *run_argv]
    devnull = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    # Same detach for the watcher itself, so closing the terminal doesn't kill it.
    try:
        subprocess.Popen(watcher_argv, **devnull, **windows_detach_popen_kwargs())
    except OSError:
        # Parent job object rejected CREATE_BREAKAWAY_FROM_JOB; retry without it (Windows only —
        # ``start_new_session=True`` cannot raise OSError on POSIX).
        fallback_kwargs: dict = (
            {"creationflags": windows_detach_flags_without_breakaway()} if sys.platform == "win32"
            else {"start_new_session": True}
        )
        try:
            subprocess.Popen(watcher_argv, **devnull, **fallback_kwargs)
        except OSError:
            return False
    return True






def _parse_kv_pairs(items) -> dict[str, str]:
    """``{key: value}`` from ``KEY=VALUE`` strings (later keys win; values stripped)."""
    return {k: v.strip() for k, v in (item.split("=", 1) for item in items if "=" in item)}














def _s6_gateway_snapshot(gateway_pids: tuple[int, ...]) -> GatewayRuntimeSnapshot | None:
    """Snapshot for an s6-supervised container gateway, or None when s6 isn't the service manager."""
    from gateway.service_manager import detect_service_manager, get_service_manager
    if detect_service_manager() != "s6":
        return None
    service_name = f"gateway-{_current_profile_name()}"
    mgr = get_service_manager()
    service_installed = service_running = False
    try:
        service_dir = getattr(mgr, "scandir", None)
        if service_dir is not None:
            service_installed = (service_dir / service_name).is_dir()
    except Exception:
        service_installed = False
    if service_installed:
        try:
            service_running = bool(mgr.is_running(service_name))
        except Exception:
            service_running = False
    return GatewayRuntimeSnapshot(
        manager="s6 (container supervisor)",
        service_installed=service_installed,
        service_running=service_running,
        gateway_pids=gateway_pids,
        service_scope="s6",
    )


def get_gateway_runtime_snapshot(system: bool = False) -> GatewayRuntimeSnapshot:
    """Return a unified view of gateway liveness for the current profile."""
    gateway_pids = tuple(find_gateway_pids())
    if is_termux():
        return GatewayRuntimeSnapshot(manager="Termux / manual process", gateway_pids=gateway_pids)

    from hermes_constants import is_container
    if is_linux() and is_container():
        # Report s6 supervision under our /init; other container runtimes keep "docker (foreground)".
        try:
            snapshot = _s6_gateway_snapshot(gateway_pids)
            if snapshot is not None:
                return snapshot
        except Exception:
            pass  # Fall through to the legacy label on any detection error.
        return GatewayRuntimeSnapshot(manager="docker (foreground)", gateway_pids=gateway_pids)

    if _systemd_runtime.supports_services():
        selected_system = _systemd_runtime.select_scope(system)
        service_running = _systemd_runtime.unit_is_active(selected_system)
        scope_label = _systemd_runtime.scope_label(selected_system)
        return GatewayRuntimeSnapshot(
            manager=f"systemd ({scope_label})",
            service_installed=_systemd_identity.unit_path(system=selected_system).exists(),
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

    return GatewayRuntimeSnapshot(manager="manual process", gateway_pids=gateway_pids)


def _format_gateway_pids(pids: tuple[int, ...] | list[int], *, limit: int | None = 3) -> str:
    rendered = [str(pid) for pid in (pids if limit is None else pids[:limit]) if pid > 0]
    if limit is not None and len(pids) > limit:
        rendered.append("...")
    return ", ".join(rendered)


def _print_gateway_process_mismatch(snapshot: GatewayRuntimeSnapshot) -> None:
    if not snapshot.has_process_service_mismatch:
        return
    print()
    pids_line = f"  PID(s): {_format_gateway_pids(snapshot.gateway_pids, limit=None)}"
    # Managed detached fallback (launchd exit-5 path) vs. a genuinely manual run.
    if _launchd_unsupported_marker_exists():
        print("⚠ Gateway is running as a detached fallback process — launchd cannot supervise it")
        print(pids_line)
        print("  Auto-start at login and auto-restart on crash are NOT available.")
        print("  Stop it with: hermes gateway stop")
    else:
        print("⚠ Gateway process is running for this profile, but the service is not active")
        print(pids_line)
        print("  This is usually a manual foreground/tmux/nohup run, so `hermes gateway`")
        print("  can refuse to start another copy until this process stops.")


def _print_multiplex_standalone_reason() -> None:
    """The boot guard kept an unset-default gateway standalone: say so in status, with the remedy."""
    from gateway.multiplex_mode import recorded_standalone_warning_lines
    for line in recorded_standalone_warning_lines():
        print(line)


def _print_served_ingress_urls(profile: str | None = None) -> None:
    """Callback URLs of inbound-port platforms the live multiplexer serves for secondary profiles
    (the value to paste into the Twilio / LINE / Teams / BlueBubbles console)."""
    try:
        from gateway.served_profiles import format_ingress_url_lines, served_profile_ingress_urls
        urls = served_profile_ingress_urls(profile)
    except Exception:
        return
    if not urls:
        return
    print()
    print("Inbound callback URLs on the shared listener:")
    for name, per_platform in sorted(urls.items()):
        for line in format_ingress_url_lines(per_platform, indent=f"  {name}/" if not profile else "  "):
            print(line)


def _print_unserved_shared_ingress(profile: str | None) -> None:
    """Shared-ingress platforms (WhatsApp/Relay) this served profile enabled that the multiplexer runs
    only on the default profile — the ``whatsapp: not served under multiplex`` line."""
    try:
        from gateway.served_profiles import served_profile_unserved_platforms
        unserved = served_profile_unserved_platforms(profile or "")
    except Exception:
        return
    if not unserved:
        return
    print()
    for platform, reason in sorted(unserved.items()):
        print(f"  ⚠ {platform}: {reason}")
    print("  Enable it on the default profile (shared ingress serves every profile), or disable it here.")


def _print_other_profiles_gateway_status() -> None:
    """Print other profiles' running gateways at the bottom of ``hermes gateway status``."""
    try:
        from profiles.current import get_active_profile_name
        current = get_active_profile_name()
        other_processes = [p for p in find_profile_gateway_processes() if p.profile != current]
        if not other_processes:
            return
        print()
        print("Other profiles:")
        for proc in other_processes:
            print(f"  ✓ {proc.profile:<16s} — PID {proc.pid}")
    except Exception:
        pass


def _print_duplicate_credential_warnings() -> None:
    """The migrate preflight's duplicate-credential findings, so ``gateway status`` explains a parked
    or racing bot (and why the fleet will not fold) with the same words as ``migrate --dry-run``."""
    with contextlib.suppress(Exception):
        from gateway.migration import duplicate_credential_findings
        lines = duplicate_credential_findings()
        if lines:
            print()
            for line in lines:
                print(f"⚠ {line}")


def _gateway_list() -> None:
    """List every profile and whether its gateway is running."""
    try:
        from profiles.current import get_active_profile_name
        from hermes_cli.profiles import list_profiles
    except Exception:
        print("Unable to list profiles.")
        return

    profiles = list_profiles()
    if not profiles:
        print("No profiles found.")
        return

    current = get_active_profile_name()

    print("Gateways:")
    for prof in profiles:
        marker = "✓" if prof.gateway_running else "✗"
        label = prof.name + (" (current)" if prof.name == current else "")
        parts = [f"  {marker} {label:<24s}"]
        if prof.gateway_running:
            pid = None
            try:
                from gateway.status import get_running_pid
                pid = get_running_pid(prof.path / "gateway.pid", cleanup_stale=False)
            except Exception:
                pass
            if pid:
                parts.append(f"PID {pid}")
            elif _host_topology.named_profile_served_by_running_multiplexer(prof.name):
                parts.append("served by the default multiplexer")
        else:
            parts.append("not running")
        print(" — ".join(parts))


def kill_gateway_processes(force: bool = False, exclude_pids: set | None = None, all_profiles: bool = False) -> int:
    """Kill running gateway processes (force-kill if ``force``); ``exclude_pids`` skips e.g. just-
    restarted service PIDs. Returns count killed."""
    killed = 0
    for pid in find_gateway_pids(exclude_pids=exclude_pids, all_profiles=all_profiles):
        try:
            expected_start_time = None
            if force:
                # Re-verify the LIVE cmdline at kill time: a PID recycled since the scan must never be tree-killed.
                # Re-verify at kill time, not just scan time: the cmdline match inside find_gateway_pids()
                # is stale by the time we get here, and a recycled PID could otherwise be tree-killed
                # (#89614 class). _capture_gateway_argv re-reads the LIVE cmdline and returns None for
                # anything that no longer looks like a gateway — refuse those.
                if _capture_gateway_argv(pid) is None:
                    continue
                from runtime.process_identity import get_process_start_time
                expected_start_time = get_process_start_time(pid)
            terminate_pid(pid, force=force, expected_start_time=expected_start_time)
            killed += 1
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"⚠ Permission denied to kill PID {pid}")
        except OSError as exc:
            print(f"Failed to kill PID {pid}: {exc}")
    return killed


_REAPER_SUPERVISOR_WALK_LIMIT = 12


def _reaper_candidate_is_supervisor_owned(pid: int) -> bool:
    """True when ``pid``'s parent chain reaches ``services.exe`` (Task Scheduler-owned gateway). Windows-only
    reaper backstop: ``_get_service_pids()`` is empty there, so a Scheduled-Task gateway with a stale
    pidfile would look like an orphan. Fail-open once the Task's bootstrap parent exits. Not applied
    on POSIX, where everything descends from PID 1 and would look supervised.

    See #83683, #86098.
    This check is deliberately NOT applied on POSIX: there, every process has PID 1 (launchd / init /
    systemd) in its ancestry — and a genuine orphan is *reparented directly to PID 1* — so supervisor-name
    ancestry carries zero signal and would spare every orphan the reaper exists to kill (#51325, 75936).
    POSIX supervised gateways are already covered pidfile- independently by the ``_get_service_pids()``
    exclusion.
    """
    if not is_windows():
        return False
    try:
        import psutil  # type: ignore
        parent = psutil.Process(pid).parent()
        for _ in range(_REAPER_SUPERVISOR_WALK_LIMIT):
            if parent is None:
                break
            with contextlib.suppress(Exception):
                if (parent.name() or "").lower() == "services.exe":
                    return True
            parent = parent.parent()
    except Exception:
        pass
    return False


def _reap_unsupervised_gateway_orphans(extra_exclude: set | None = None) -> bool:
    """Kill no-supervisor gateway orphans the pidfile/runtime record can't see. On WSL/no-systemd hosts
    the restart fallback runs the gateway in-process under a ``gateway restart`` argv; a stale pidfile
    then lets a live orphan keep the webhook port while a restart stacks a duplicate. No-op where a
    supervisor exists (there ``gateway restart`` is a transient command). ``extra_exclude``: already killed."""
    try:
        supervised_host = _systemd_runtime.supports_services()
    except Exception:
        supervised_host = True
    if supervised_host:
        return False

    # Task Scheduler is a supervisor too; its state beats a parent-chain walk (broken once the bootstrap exits).
    # A Scheduled Task gateway whose conhost/VBS bootstrap has already exited is invisible to
    # `_reaper_candidate_is_supervisor_owned` (the parent chain breaks before services.exe, fail-open), yet
    # it is alive and supervised. After that launcher exits the task is typically Ready, not Running —
    # treating only Running as supervised still kills the detached gateway on every desktop serve start
    # (#86098, #87001).
    if is_windows():
        try:
            from gateway.windows_service import get_task_name  # profile-aware task name
            _task_name = get_task_name()
        except Exception:
            _task_name = "Hermes_Gateway"
        if _windows_scheduled_task_supervises(_task_name):
            return False

    from gateway.status import _pid_exists, write_planned_stop_marker
    from runtime.process_identity import get_process_start_time
    own = _reaper_exclusion_pids(extra_exclude)
    try:
        # On Windows also drop Task Scheduler-owned candidates (the pidfile-less gap).
        orphans = [
            p for p in find_gateway_pids(exclude_pids=own) if p and p > 0 and not _reaper_candidate_is_supervisor_owned(p)
        ]
    except Exception:
        return False
    if not orphans:
        return False

    # Pin each orphan's start time now: the delayed SIGKILL must never hit a recycled PID.
    # Pin each orphan's identity NOW: the cmdline scan above matched at scan-time only, and the SIGKILL
    # escalation below fires seconds later. A PID recycled inside that window must never be force-killed
    # (#89614 class). Fingerprint capture is best-effort — SIGTERM below proceeds regardless (it targets the
    # process verified by the scan an instant ago), but the delayed SIGKILL requires a still-matching
    # fingerprint.
    orphan_identity: dict[int, int] = {}
    for pid in orphans:
        start = get_process_start_time(pid)
        if start is not None:
            orphan_identity[pid] = start

    reaped = False
    for pid in orphans:
        with contextlib.suppress(Exception):
            write_planned_stop_marker(pid)
        # ``os.kill(..., SIGTERM)`` maps to TerminateProcess on Windows, so it
        # would kill the gateway before its marker watcher can drain and close
        # state cleanly. Let the bounded survivor wait below escalate instead.
        if is_windows():
            reaped = True
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            print(f"⚠ Permission denied to kill orphaned gateway PID {pid}")
            continue
        reaped = True

    # Wait, then force-kill survivors so the replacement can bind the port cleanly.
    # Fail-closed: SIGKILL only a PID that still names the process fingerprinted at scan time.
    _force_kill_survivors([
        pid for pid in _await_gateway_exit(orphans, pid_exists=_pid_exists)
        if pid in orphan_identity and get_process_start_time(pid) == orphan_identity[pid]
    ])
    return reaped


def _reaper_exclusion_pids(extra_exclude: set | None) -> set[int]:
    """PIDs the orphan reaper must never kill: self, caller extras, service-managed, recorded."""
    own = {os.getpid()} | (extra_exclude or set())
    # Service-managed gateways are never orphans (on macOS _systemd_runtime.supports_services() is False, so a
    # launchd gateway would otherwise be SIGTERM'd); all_profiles because the scan sees siblings too.
    with contextlib.suppress(Exception):
        # This covers macOS launchd (_systemd_runtime.supports_services() is False there, so without this the launchd
        # gateway looks like an unsupervised orphan and gets SIGTERM'd, causing launchd to restart it — or
        # leaving it down under KeepAlive.SuccessfulExit=false) and any systemd unit reachable from a host
        # that got past the gate above (#83683, #85344).
        # all_profiles=True: the reaper's process scan sees every profile's gateway (and on macOS the
        # now-working ps fallback surfaces sibling launchd gateways, #73626), so the service exclusion must
        # cover the whole ai.hermes.gateway* fleet — not just the current profile's label — or a sibling
        # profile's launchd gateway is misclassified as an unsupervised orphan and reaped. Same class as the
        # update-sweep fix in #74075.
        own |= _get_service_pids(all_profiles=True)
    # Exempt the recorded gateway PID and its parent chain (on Windows the Scheduled-Task bootstrap's
    # ``gateway run`` argv matches the scan; killing it takes the gateway down). Use the RAW pidfile +
    # lock records, not only the validated probe: get_running_pid returns None on any validation
    # hiccup — exactly when a healthy standalone gateway would be hard-killed (Windows SIGTERM is
    # TerminateProcess, no drain). For a KILL exclusion list a stale PID at worst spares one process;
    # a false negative kills a live gateway. The probe still supplies the runtime-status fallback PID.
    try:
        from gateway.status import _pid_from_record, _read_gateway_lock_record, _read_pid_record, get_running_pid
        recorded_pids = {_pid_from_record(rec) for rec in (_read_pid_record(), _read_gateway_lock_record())}
        recorded_pids.add(get_running_pid(cleanup_stale=False))
        for recorded in recorded_pids:
            if not recorded or recorded <= 0:
                continue
            own.add(recorded)
            try:
                import psutil  # type: ignore
                parent = psutil.Process(recorded).parent()
                while parent is not None:
                    own.add(parent.pid)
                    parent = parent.parent()
            except Exception:
                pass
    except Exception:
        pass
    return own


# A retiring gateway runs a PASSIVE WAL checkpoint in ``SessionDB.close()``; a SIGKILL mid-checkpoint
# corrupts ``state.db``. It keeps serving while we wait, so a long grace only delays the port bind.
_ORPHAN_EXIT_GRACE_SECONDS = 30.0
_ORPHAN_EXIT_POLL_SECONDS = 0.2


def _await_gateway_exit(
    pids, *, pid_exists, sleep=None, grace_s: float = _ORPHAN_EXIT_GRACE_SECONDS, poll_s: float = _ORPHAN_EXIT_POLL_SECONDS
):
    """Poll up to *grace_s* for *pids* to exit; return survivors. ``pid_exists``/``sleep`` injectable for tests."""
    if sleep is None:
        sleep = time.sleep
    survivors = list(pids)
    for _ in range(max(1, int(grace_s / poll_s))):
        survivors = [p for p in survivors if pid_exists(p)]
        if not survivors:
            break
        sleep(poll_s)
    else:
        # Re-check after the LAST sleep, or a recycled PID could get the SIGKILL.
        survivors = [p for p in survivors if pid_exists(p)]
    return survivors


def _force_kill_survivors(survivors, *, kill=None) -> None:
    """SIGKILL processes that outlasted the grace period, loudly — a force-kill can tear the store, so
    it must leave a trace."""
    kill = kill or os.kill
    for pid in survivors:
        logger.warning(
            "Gateway PID %s did not exit within %.0fs of the stop request (SIGTERM, or the planned-stop "
            "marker on Windows) — sending "
            "SIGKILL. A kill during a WAL checkpoint can corrupt state.db; "
            "the next start will run an integrity check.",
            pid, _ORPHAN_EXIT_GRACE_SECONDS,
        )
        with contextlib.suppress((ProcessLookupError, PermissionError, OSError)):
            kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))




def stop_profile_gateway() -> bool:
    """Stop only this profile's gateway via its PID file; True if a process was stopped. Without a
    supervisor the pidfile can be stale while a live orphan holds the webhook port, so fall back to
    the orphan-aware scan rather than stacking a duplicate.

    Even when the pid file is valid and points to the current gateway, older orphans may linger from prior
    restarts that overwrote the pid file before the old process exited. After killing the recorded PID, also
    sweep for any remaining orphans so each restart produces at most one live gateway (#75936).
    """
    try:
        from gateway.status import get_running_pid, remove_pid_file
    except ImportError:
        return False

    pid = get_running_pid()
    if pid is None:
        return _reap_unsupervised_gateway_orphans()

    if is_windows():
        # Windows maps SIGTERM to TerminateProcess. The marker watcher is the
        # gateway's graceful-stop IPC, so wait for it before force-killing a
        # wedged process.
        from runtime.process_identity import get_process_start_time
        from gateway.windows_service import (
            _drain_gateway_pid,
            _force_terminate_known_gateway_pids,
            _windows_stop_drain_timeout,
        )

        # Capture identity BEFORE the drain (as _escalate_wedged_gateway does): if the PID is
        # recycled during the wait, terminate_pid's start-time mismatch refuses the taskkill.
        expected_start_time = get_process_start_time(pid)
        if not _drain_gateway_pid(pid, _windows_stop_drain_timeout()):
            _force_terminate_known_gateway_pids({pid: expected_start_time})
    else:
        _signal_restart._mark_planned_stop(pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # Already gone
        except PermissionError:
            print(f"⚠ Permission denied to kill PID {pid}")
            return False

    # ``_pid_exists``, NOT ``os.kill(pid, 0)`` (TerminateProcess on Windows).
    from gateway.status import _pid_exists
    for _ in range(20):
        if not _pid_exists(pid):
            break
        time.sleep(0.5)

    if get_running_pid() is None:
        remove_pid_file()

    # Reap orphans from prior restarts whose pidfile entry was overwritten; skip the PID just killed.
    try:
        # Exclude the PID we just killed so the sweep doesn't double-kill a process that's still tearing
        # down — _reap_unsupervised_gateway_orphans already excludes our own PID. See #75936.
        _reap_unsupervised_gateway_orphans(extra_exclude={pid} if pid else None)
    except Exception as exc:
        logger.debug("orphan reap after stop_profile_gateway failed: %s", exc)
    return True


def is_linux() -> bool:
    return sys.platform.startswith("linux")


from hermes_constants import is_container, is_termux, is_wsl








def is_macos() -> bool:
    return sys.platform == "darwin"


def is_windows() -> bool:
    return sys.platform == "win32"


def _gw_windows():
    """Lazily import :mod:`gateway.windows_service` (Windows-only service backend)."""
    from gateway import windows_service as gateway_windows
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
    from gateway.windows_launch import _WINDOWS_GATEWAY_BREAKAWAY_ENV
    return {"1": True, "0": False}.get(os.environ.pop(_WINDOWS_GATEWAY_BREAKAWAY_ENV, None))


# =============================================================================
# Service Configuration
# =============================================================================

SERVICE_DESCRIPTION = "Hermes Agent Gateway - Messaging Platform Integration"









def _current_profile_name() -> str:
    """Profile id relative to the profile ROOT: ``default`` for the root itself (Docker's ``/opt/data``
    included), ``<name>`` for ``<root>/profiles/<name>``, else the service hash. s6 slots and the
    multiplexer ask which PROFILE this is; ``_service_identity.service_suffix()`` answers which HOST SERVICE this is."""
    from hermes_constants import profile_name_for_home
    return profile_name_for_home(get_hermes_home()) or _service_identity.service_suffix()


















































# Legacy pre-rename names: explicit allowlist (NOT a glob) so profile and third-party units never match.

# ExecStart markers identifying a unit as running our gateway; a legacy unit is flagged only if one matches.








def print_legacy_unit_warning() -> None:
    """Warn about installed legacy gateway units; prints nothing when there are none."""
    legacy = _systemd_legacy.find_units()
    if not legacy:
        return
    print_warning("Legacy Hermes gateway unit(s) detected from an older install:")
    for _name, path, is_system in legacy:
        print_info(f"    {path}  ({_systemd_runtime.scope_label(is_system)} scope)")
    print_info("  These run alongside the current hermes-gateway service and")
    print_info("  cause SIGTERM flap loops — both try to use the same bot token.")
    print_info("  Remove them with:")
    print_info("    hermes gateway migrate-legacy")


def remove_legacy_hermes_units(
    interactive: bool = True,
    dry_run: bool = False,
) -> tuple[int, list[Path]]:
    """Prompt in the CLI; delegate mutation to the systemd legacy owner."""
    legacy = _systemd_legacy.find_units()
    if not legacy:
        print("No legacy Hermes gateway units found.")
        return 0, []

    print()
    print("Legacy Hermes gateway unit(s) found:")
    for _name, path, is_system in legacy:
        print(f"  {path}  ({_systemd_runtime.scope_label(is_system)} scope)")
    print()

    if dry_run:
        print("(dry-run — nothing removed)")
        return 0, [path for _, path, _ in legacy]
    if interactive and not prompt_yes_no("Remove these legacy units?", True):
        print("Skipped. Run again with: hermes gateway migrate-legacy")
        return 0, [path for _, path, _ in legacy]

    removed, remaining = _systemd_legacy.remove_units()
    for _name, path, _is_system in legacy:
        if path not in remaining and not path.exists():
            print(f"  ✓ Removed {path}")
    for path in remaining:
        print(f"  ⚠ Could not remove {path}")
    print()
    if remaining:
        print_warning(f"{len(remaining)} legacy unit(s) still present — see messages above.")
    else:
        print_success(f"Removed {removed} legacy unit(s).")
    return removed, remaining


def print_systemd_scope_conflict_warning() -> None:
    scopes = _systemd_runtime.installed_scopes()
    if len(scopes) < 2:
        return

    print_warning(f"Both user and system gateway services are installed ({' + '.join(scopes)}).")
    print_info("  This is confusing and can make start/stop/status behavior ambiguous.")
    print_info("  Default gateway commands target the user service unless you pass --system.")
    print_info("  Keep one of these:")
    print_info("    hermes gateway uninstall")
    print_info("    sudo hermes gateway uninstall --system")


def refuses_container_user_scope_install(system: bool) -> bool:
    """True (after printing the guidance) when a fresh USER-scope unit was requested inside a container.

    A systemd container passes ``_systemd_runtime.supports_services()`` on purpose so ``--system`` keeps working,
    but a user unit there is not container-scoped: the unit file and its ``default.target.wants`` symlink
    land in ``~/.config/systemd/user`` — commonly the host's own home bind-mounted in — so the host's
    ``systemd --user`` enables it too and a second gateway polls the same bot token outside the container.
    Callers decide between ``sys.exit(1)`` (CLI) and skipping the install (wizard)."""
    if system or not is_container():
        return False
    print_error("Refusing to install a user-scope systemd gateway service inside a container.")
    _print_info_lines(
        "The unit file and its enable symlink would be written to the home directory, which is",
        "commonly the host's own home bind-mounted in — the host's user manager then enables and",
        "starts the same unit, so a second gateway polls the same bot token outside the container",
        "(Telegram: 'Conflict: terminated by other getUpdates request').",
        "",
        "  hermes gateway run                                # run as the container's main process",
        "  docker run --restart unless-stopped ...           # container restart policy",
        "",
        "If systemd manages this container (systemd as PID 1), install an isolated system service instead:",
        "  sudo hermes gateway install --system --run-as-user <user>",
    )
    return True








def _default_system_service_user() -> str | None:
    for candidate in (os.getenv("SUDO_USER"), os.getenv("USER"), os.getenv("LOGNAME")):
        candidate = (candidate or "").strip()
        if candidate and candidate != "root":
            return candidate
    return None


def prompt_linux_gateway_install_scope() -> str | None:
    # Only root can create a boot-time system service; never hand a non-root user a "re-run under sudo" recipe.
    is_root = os.geteuid() == 0  # windows-footgun: ok — Linux systemd install wizard, never invoked on Windows
    options = ["User service (no sudo; best for laptops/dev boxes; may need linger after logout)"]
    values: list[str | None] = ["user"]
    if is_root:
        options.append("System service (starts on boot; runs as your chosen user)")
        values.append("system")
    options.append("Skip service install for now")
    values.append(None)
    choice = prompt_choice("  Choose how the gateway should run in the background:", options, default=0)
    if not is_root and choice == 0:
        print_info("  Tip: for a boot-time system service, re-run setup as root (e.g. from a root shell or `sudo -i`).")
    return values[choice]


def install_linux_gateway_from_setup(force: bool = False, enable_on_startup: bool = True) -> tuple[str | None, bool]:
    scope = prompt_linux_gateway_install_scope()
    if scope is None:
        return None, False

    if scope == "system":
        run_as_user = _default_system_service_user()
        if os.geteuid() != 0:  # windows-footgun: ok — Linux systemd install wizard, never invoked on Windows
            # Unreachable from the wizard (system scope only offered to root); defensive guard for direct callers.
            print_warning(
                "  System service install requires root. Re-run setup from a "
                "root shell, or install a user service instead: hermes gateway install"
            )
            return scope, False

        while not run_as_user:
            run_as_user = (prompt("  Run the system gateway service as which user?", default="") or "").strip()
            if not run_as_user:
                print_error("  Enter a username.")

        _systemd_lifecycle.install(force=force, system=True, run_as_user=run_as_user, enable_on_startup=enable_on_startup)
        return scope, True

    if refuses_container_user_scope_install(system=False):
        return scope, False
    _systemd_lifecycle.install(force=force, system=False, enable_on_startup=enable_on_startup)
    return scope, True





# =============================================================================
# Systemd (Linux)
# =============================================================================


















def _system_scope_wizard_would_need_root(system: bool = False) -> bool:
    """True when the wizard would trigger a system-scope operation as non-root — mirrors
    ``_select_systemd_scope`` so the dead-end is detected BEFORE prompting."""
    if os.geteuid() == 0:  # windows-footgun: ok — systemd scope wizard decision, never invoked on Windows
        return False
    return _systemd_runtime.select_scope(system=system)


def _print_system_scope_remediation(action: str) -> None:
    """Print remediation when the wizard skips a system-scope action because the user isn't root."""
    print_warning(f"Gateway is installed as a system-wide service — {action} requires root.")
    print_info("  Options:")
    print_info(f"    1. {action.capitalize()} it this time:")
    print_info(f"         sudo systemctl {action} {_service_identity.service_name()}")
    print_info("    2. Switch to a per-user service (recommended for personal use):")
    print_info("         sudo hermes gateway uninstall --system")
    print_info("         hermes gateway install")
    print_info("         hermes gateway start")
















def _print_service_not_installed(system: bool) -> None:
    sudo, scope_flag, _ = (("sudo ", " --system", "") if system else ("", "", "--user "))
    print("✗ Gateway service is not installed")
    print(f"  Run: {sudo}hermes gateway install{scope_flag}")


def _require_service_installed(action: str, system: bool = False) -> None:
    if not _systemd_identity.unit_path(system=system).exists():
        _print_service_not_installed(system)
        sys.exit(1)












def systemd_status(deep: bool = False, system: bool = False, full: bool = False):
    system = _systemd_runtime.select_scope(system)
    unit_path = _systemd_identity.unit_path(system=system)
    svc = _service_identity.service_name()
    scope_label = _systemd_runtime.scope_label(system).capitalize()
    sudo, scope_flag, user_flag = (("sudo ", " --system", "") if system else ("", "", "--user "))

    if not unit_path.exists():
        _print_service_not_installed(system)
        return

    if _systemd_runtime.has_conflicting_units():
        print_systemd_scope_conflict_warning()
        print()

    if _systemd_legacy.has_units():
        print_legacy_unit_warning()
        print()

    if not _systemd_unit_state.unit_is_current(system=system):
        print("⚠ Installed gateway service definition is outdated")
        print(f"  Run: {sudo}hermes gateway restart{scope_flag}  # auto-refreshes the unit")
        print()

    status_cmd = ["status", svc, "--no-pager"] + (["-l"] if full else [])
    _systemd_runtime.run_systemctl(status_cmd, system=system, capture_output=False, timeout=10)
    result = _systemd_runtime.run_systemctl(["is-active", svc], system=system, timeout=10, **_CAPTURE_TEXT)
    if result.stdout.strip() == "active":
        print(f"✓ {scope_label} gateway service is running")
    else:
        print(f"✗ {scope_label} gateway service is stopped")
        print(f"  Run: {sudo}hermes gateway start{scope_flag}")

    configured_user = _systemd_identity.read_unit_user(unit_path) if system else None
    if configured_user:
        print(f"Configured to run as: {configured_user}")

    _print_runtime_health()

    unit_props = _systemd_restart_state._read_systemd_unit_properties(system=system)
    active_state = unit_props.get("ActiveState", "")
    result_code = unit_props.get("Result", "")
    if active_state == "activating" and unit_props.get("SubState", "") == "auto-restart":
        print("  ⏳ Restart pending: systemd is waiting to relaunch the gateway")
    elif _systemd_restart_state._systemd_unit_is_start_limited(unit_props):
        print("  ⏳ Restart pending: systemd is temporarily rate-limiting starts")
        print(f"  Run after the start-limit window expires: {sudo}hermes gateway restart{scope_flag}")
        print(f"  Or clear it manually: systemctl {user_flag}reset-failed {svc}")
    elif active_state == "failed" and unit_props.get("ExecMainStatus", "") == str(GATEWAY_SERVICE_RESTART_EXIT_CODE):
        print("  ⚠ Planned restart is stuck in systemd failed state (exit 75)")
        print(f"  Run: systemctl {user_flag}reset-failed {svc} && {sudo}hermes gateway start{scope_flag}")
    elif active_state == "failed" and result_code:
        print(f"  ⚠ Systemd unit result: {result_code}")

    if system:
        print("✓ System service starts at boot without requiring systemd linger")
    else:
        linger_enabled, linger_detail = _systemd_runtime.linger_status()
        if linger_enabled is True:
            print("✓ Systemd linger is enabled (service survives logout)")
        elif linger_enabled is False:
            print("⚠ Systemd linger is disabled (gateway may stop when you log out)")
            print("  Run: sudo loginctl enable-linger $USER")
        elif deep:
            print(f"⚠ Could not verify systemd linger ({linger_detail})")
            print("  If you want the gateway user service to survive logout, run:")
            print("  sudo loginctl enable-linger $USER")

    if deep:
        print()
        print("Recent logs:")
        log_cmd = ["journalctl"] + ([] if system else ["--user"]) + ["-u", svc, "-n", "20", "--no-pager"]
        if full:
            log_cmd.append("-l")
        subprocess.run(log_cmd, timeout=10)


# =============================================================================
# Launchd (macOS)
# =============================================================================


from gateway.launchd_service import (  # noqa: E402,F401 - compatibility re-exports
    get_launchd_label,
    get_launchd_plist_path,
    launchd_gateway_labels_for_install,
    legacy_launchd_labels_for_install,
    _parse_launchd_pid_from_list_output,
    _parse_launchd_pid_from_print_output,
    _launchd_print_service_pid,
    _launchd_service_registered,
    _locate_launchd_gateway_service,
    _probe_launchd_service_running,
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



# =============================================================================
# Gateway Runner
# =============================================================================


def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_official_docker_checkout() -> bool:
    return str(PROJECT_ROOT) == "/opt/hermes" and (PROJECT_ROOT / "docker" / "entrypoint.sh").is_file()


def _running_under_gateway_supervisor() -> bool:
    """True when this process IS the supervisor-launched gateway, so the conflict guard never wedges
    the service into a respawn/refuse loop. Markers: systemd INVOCATION_ID, launchd XPC_SERVICE_NAME
    (shells inherit "0"), s6 HERMES_S6_SUPERVISED_CHILD, or ``--external-supervisor``."""
    return is_gateway_supervisor_process()


def _served_profile_needs_no_service() -> bool:
    """Print the "already served" note and return True when a setup flow must not install a standalone
    service: a live multiplexing default gateway already serves this named profile, so the unit/plist it
    would register can only sit dead (the start guard refuses it) or double-bind its platforms.
    Shared by ``hermes setup gateway`` / ``hermes setup`` / ``hermes import`` (``ensure_gateway_service``)
    and the ``hermes gateway setup`` wizard. See #111958."""
    if not _host_topology.named_profile_served_by_running_multiplexer():
        # Not served (yet): a named profile still gets no service of its own — same rule and text
        # as `gateway install`, so `hermes -p X setup` cannot grow a fleet member the verb refuses.
        return _named_profile_refused_under_multiplexer()
    from gateway.profile_serving import profile_is_standalone
    if profile_is_standalone(get_hermes_home()):
        from gateway.host_attach import standalone_rescan_message
        print_info(standalone_rescan_message(_current_profile_name()))
        return True
    print_success(
        f"Profile '{_current_profile_name()}' is already served by the default multiplexer."
    )
    print_info("  (served now by the running multiplexed gateway — add its bot token and it connects)")
    print_info("  No standalone gateway service was installed or started.")
    return True


def _named_profile_refused_under_multiplexer(force: bool = False) -> bool:
    """Print the refusal and return True when a NAMED profile must not get a gateway of its own.

    One gateway per host serves every profile, so a ``<root>/profiles/<name>`` home never installs or
    starts a standalone gateway: either the host gateway already serves it (a second one would
    double-bind its platforms: two pollers on one token, port fights) or no host gateway runs yet and
    the DEFAULT profile is where it is installed. Refusing only the served case let a host with no
    multiplexer running (or one that had not rescanned yet) grow a brand-new per-profile fleet member.
    ``--force`` is the one escape (a fleet split across UNIX users or a ``HERMES_HOME`` outside
    ``profiles/``); a service it already installed stays startable without it. Shared by ``run`` and the service verbs (``start``/``install``/``restart``): a
    refusal only inside ``gateway run`` leaves the service manager to discover it — systemd parks the
    unit on exit 78 while the CLI prints "started"; launchd (KeepAlive, no exit-status gating)
    respawns it every ThrottleInterval forever."""
    if force:
        return False
    try:
        suffix = _current_profile_name()
        from hermes_constants import profile_name_for_home
        from gateway.profile_serving import profile_is_standalone
        # A profile that authored gateway.standalone: true opted out of the host multiplexer: it is
        # allowed a gateway of its own without --force. Only a RUNNING host record that still lists
        # it (the host has not rescanned since the key was set) is refused with the rescan remedy.
        standalone = (profile_name_for_home(get_hermes_home()) not in (None, "default")
                      and profile_is_standalone(get_hermes_home()))
        # A unit/plist/task already registered for this home was installed with --force: that fleet
        # member (and the supervisor relaunching it, whose ExecStart carries no --force) is not NEW.
        new_standalone = (profile_name_for_home(get_hermes_home()) not in (None, "default")
                          and not _is_service_installed())
    except Exception:
        return False
    owner = _host_topology.served_by_another_host_gateway()
    served = owner is not None or _host_topology.named_profile_served_by_running_multiplexer()
    if standalone:
        if not served:
            return False
        from gateway.host_attach import standalone_rescan_message
        print_error(standalone_rescan_message(suffix))
        return True
    if not served and not new_standalone:
        return False

    if served:
        print_error(f"The host gateway already serves profile '{suffix}'.")
        if owner is not None:
            print(f"  {owner.describe()}")
    else:
        print_error(f"Profile '{suffix}' does not get a gateway of its own.")
    print(
        "  Exactly one gateway per host is the inbound process for every\n"
        "  profile. Starting a separate gateway for this profile would\n"
        "  double-bind its platforms (two pollers on one bot token, port\n"
        "  conflicts).\n"
    )
    if served:
        print("  Manage the host gateway instead:")
        print()
        print(f"    hermes -p {owner.profile_label if owner is not None else 'default'} gateway restart")
    else:
        print("  Install or start the host gateway from the default profile; it serves this one too:")
        print()
        print("    hermes gateway install")
        print()
        print("  Or fold an existing per-profile fleet onto one host gateway:")
        print()
        print("    hermes gateway migrate --multiplex")
    print()
    print("  A separate per-profile gateway (for a fleet split across UNIX users or a")
    print(f"  HERMES_HOME outside profiles/) needs --force:  hermes -p {suffix} gateway install --force")
    print()
    from hermes_constants import display_hermes_home
    from gateway.multiplex_mode import STANDALONE_DEPRECATION_NOTICE
    print("  Temporary compatibility path while multiplexing gaps are closed: set")
    print(f"  gateway.standalone: true in {display_hermes_home(get_hermes_home())}/config.yaml,")
    print("  then wait for the host gateway to rescan (<=30s) or send its rescan-profiles control verb.")
    print(f"  ({STANDALONE_DEPRECATION_NOTICE})")
    return True


def _guard_named_profile_under_multiplexer(force: bool = False) -> None:
    """Exit-78 form of ``_named_profile_refused_under_multiplexer`` for the CLI entry points."""
    if not _named_profile_refused_under_multiplexer(force=force):
        return
    # EX_CONFIG, not 1: the refusal is decided purely by config, so it is permanent. The systemd unit
    # (Restart=always, StartLimitIntervalSec=0) relies on RestartPreventExitStatus=78 as its only
    # backstop — exit 1 turned a correct refusal into an unbounded restart loop; s6 maps 78 to
    # "permanent failure" too.
    # This refusal is decided entirely by configuration (multiplex_profiles plus the allowlist), so it is
    # permanent: no number of retries can change the answer. Exiting 1 made it look transient to a service
    # manager -- and the systemd unit this module generates pairs Restart=always/RestartSec=5 with
    # StartLimitIntervalSec=0, deliberately trading systemd's generic start-rate limiter for the specific
    # RestartPreventExitStatus=GATEWAY_FATAL_CONFIG_EXIT_CODE backstop declared beside it. Returning 1 left
    # that backstop unarmed with the limiter already off, so a correct refusal became an unbounded restart
    # loop. 78 also reaches the s6 finish script's 125 "permanent failure" translation (see #51228), the
    # same path the other fatal-config exits take.
    sys.exit(GATEWAY_FATAL_CONFIG_EXIT_CODE)


def _host_decision_exit_code(decision) -> int:
    """Exit code for a host-attach verdict a supervisor may be watching.

    ``GATEWAY_FATAL_CONFIG_EXIT_CODE`` (78) is the PERMANENT refusal: systemd parks the unit on it
    (``RestartPreventExitStatus``), the s6 finish script maps it to 125, launchd maps it to a
    deliberate stop. That is right for a config-derived refusal and wrong for a runtime one — "some
    other process serves me right now" ends the moment that process goes away, and parking the unit
    on it strands the profile until a human notices. Transient verdicts therefore use
    ``GATEWAY_SERVICE_RESTART_EXIT_CODE`` (75, EX_TEMPFAIL), which every supervisor we generate
    already retries: systemd has ``RestartForceExitStatus=75`` with ``RestartSec=5``, the s6 finish
    script passes it through, and launchd relaunches a non-78 failure. Exit 0 would NOT do: s6
    parks a clean exit too.
    """
    if getattr(decision, "transient", False):
        return GATEWAY_SERVICE_RESTART_EXIT_CODE
    return GATEWAY_FATAL_CONFIG_EXIT_CODE


def _attach_to_host_gateway_or_guard(force: bool = False, replace: bool = False) -> None:
    """``gateway run`` against the ONE host gateway: attach, rescan-then-attach, replace, or refuse.

    A profile the host process already serves has nothing to run: print who serves it and exit 0
    without spawning anything. Under a service supervisor the SAME situation exits 75 instead, so
    the unit is RETRIED rather than parked (see :func:`_host_decision_exit_code`).

    ``--replace`` and ``--force`` are the two escape hatches this guard must not eat: both return
    here so ``start_gateway`` can act on them (it owns the signalling and the PID claim).
    """
    if force:
        return
    try:
        from gateway.host_attach import ATTACH, REFUSE, REPLACE_HOST, decide
        decision = decide(get_hermes_home(), replace=replace)
    except Exception:
        logger.debug("Host gateway attach probe failed", exc_info=True)
        decision = None
    if decision is not None and decision.outcome == REPLACE_HOST:
        return  # start_gateway replaces the owner; the config guard below must not pre-empt it
    if decision is not None and decision.outcome in (ATTACH, REFUSE):
        print(decision.message)
        if decision.outcome == REFUSE:
            code = _host_decision_exit_code(decision)
            # stdout goes to the supervisor's unit log; under launchd a permanent refusal is then
            # mapped to a clean exit and the unit is parked. The profile's own logs (errors.log,
            # WARNING+) are where a parked fleet is diagnosed, so name the verdict and the remedy there.
            logger.warning("gateway run refused (exit %d): %s", code, decision.message)
            sys.exit(code)
        if _running_under_gateway_supervisor():
            sys.exit(_host_decision_exit_code(decision))
        sys.exit(0)
    # No host record (older gateway, unwritable lock dir): the config-derived refusal still applies.
    _guard_named_profile_under_multiplexer(force=force)


def _guard_supervised_gateway_conflict(force: bool = False) -> None:
    """Refuse a foreground gateway when a service manager already supervises one: a shell-launched run
    becomes a second dispatcher that escapes the cgroup, survives ``systemctl restart``, and writes the
    shared kanban DB concurrently (multi-writer SQLite WAL corruption). ``--force`` starts anyway.

    See #35240.
    """
    if force or _running_under_gateway_supervisor():
        return
    try:
        snapshot = get_gateway_runtime_snapshot()
    except Exception:
        logger.debug("Supervised-gateway conflict probe failed", exc_info=True)
        return
    if not (snapshot.service_installed and snapshot.service_running):
        return

    print_error(f"A gateway is already running under {snapshot.manager} for this profile.")
    print(
        "  Starting another one from a shell leaves an orphan dispatcher that\n"
        "  escapes the service, survives restarts, and writes to the same kanban\n"
        "  DB concurrently — which can corrupt it. Restart the supervised gateway\n"
        "  instead:"
    )
    print()
    print("    hermes gateway restart")
    print()
    print(
        "  Pass --force to start a foreground gateway anyway (not recommended\n"
        "  while the service is running)."
    )
    sys.exit(1)


def _guard_existing_gateway_process_conflict(replace: bool = False) -> None:
    """Cheap PID-file preflight before the expensive ``gateway.run`` import (the authoritative lock check):
    supervisor loops re-running bare ``gateway run`` burned memory on plugin discovery just to fail
    "already running". Same user-facing contract; never scans other HERMES_HOME roots."""
    if replace or _running_under_gateway_supervisor():
        return
    try:
        from gateway.status import get_running_pid
        pid = get_running_pid()
    except Exception:
        logger.debug("Existing-gateway process probe failed", exc_info=True)
        return
    if pid is None:
        # get_running_pid() filters by the current profile's HERMES_HOME; warn if the PID file
        # belongs to another profile (user switched profiles while the old gateway still runs).
        try:
            from gateway.status import _read_pid_record, _pid_record_belongs_to_current_profile
            stale = _read_pid_record()
            if stale is not None and not _pid_record_belongs_to_current_profile(stale):
                logger.warning(
                    "PID file belongs to another profile (hermes_home=%s). "
                    "The old gateway may still be running under that profile.",
                    stale.get("hermes_home", "<unknown>"),
                )
        except Exception:
            pass
        return

    print_error(f"A gateway is already running (PID {pid}), so your bots are most likely online already.")
    print("  Check with `hermes gateway status`.")
    print("  To restart it: `hermes gateway restart`. To stop it: `hermes gateway stop`.")
    print("  To replace it from here: `hermes gateway run --replace`.")
    sys.exit(1)


def _guard_official_docker_root_gateway() -> None:
    """Refuse gateway startup when the official Docker privilege drop was bypassed."""
    if not hasattr(os, "geteuid") or os.geteuid() != 0 or _truthy_env(os.getenv("HERMES_ALLOW_ROOT_GATEWAY")):
        return
    if not _is_official_docker_checkout():
        return

    print_error("Refusing to run the Hermes gateway as root inside the official Docker image.")
    print(
        "  The image entrypoint normally drops privileges to the 'hermes' user. "
        "If you override entrypoint in Docker Compose, include "
        "/opt/hermes/docker/entrypoint.sh before the Hermes command."
    )
    print(
        "  Running the gateway as root can leave root-owned files in "
        "$HERMES_HOME and break later non-root dashboard/gateway runs."
    )
    print("  Set HERMES_ALLOW_ROOT_GATEWAY=1 only if you intentionally accept this risk.")
    sys.exit(1)


def _apply_startup_watchdog_config() -> None:
    """Idempotent backstop arming of the startup-liveness watchdog. Must run AFTER the conflict guards (a
    --replace loser must not arm one). config.yaml gateway.startup_watchdog* is the user surface; env
    vars bridge it because the argv fast-path arms before config loads, and explicit env wins. arm() is
    idempotent, so a config timeout needs disarm+re-arm. GatewayRunner disarms once the loop is live."""
    try:
        from hermes_startup_watchdog import (
            ENV_STARTUP_WATCHDOG, ENV_STARTUP_WATCHDOG_TIMEOUT_S, arm_startup_watchdog,
            disarm_startup_watchdog, startup_watchdog_disabled,
        )
        _sw_timeout_bridged = False
        try:
            from hermes_cli.config import load_config as _sw_load_config
            _gw_cfg = (_sw_load_config() or {}).get("gateway", {}) or {}
            if ENV_STARTUP_WATCHDOG not in os.environ and not _gw_cfg.get("startup_watchdog", True):
                os.environ[ENV_STARTUP_WATCHDOG] = "0"
            _sw_timeout = _gw_cfg.get("startup_watchdog_timeout_seconds")
            if ENV_STARTUP_WATCHDOG_TIMEOUT_S not in os.environ and _sw_timeout is not None:
                os.environ[ENV_STARTUP_WATCHDOG_TIMEOUT_S] = str(_sw_timeout)
                _sw_timeout_bridged = True
        except Exception:
            pass
        if startup_watchdog_disabled():
            disarm_startup_watchdog()
        else:
            if _sw_timeout_bridged:
                disarm_startup_watchdog()
            arm_startup_watchdog()
    except Exception:
        pass


def _absorb_windows_console_controls() -> None:
    """Make a detached Windows gateway ignore console-control broadcasts from sibling CLIs."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    except (OSError, ValueError):
        pass  # SetConsoleCtrlHandler unavailable (rare) — best-effort
    # signal only hooks SIGINT/SIGBREAK; SetConsoleCtrlHandler(NULL, TRUE) ignores ALL console
    # control events (CTRL_CLOSE/CTRL_LOGOFF included), as background services should.
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCtrlHandler(None, 1)  # type: ignore[attr-defined]
    except (OSError, AttributeError):
        pass


def _make_exit_diag():
    """``_exit_diag(tag, **extra)`` recorder writing ``logs/gateway-exit-diag.log`` — captures every way
    ``asyncio.run()`` can return, for chasing silent Windows gateway deaths. HERMES_GATEWAY_EXIT_DIAG=0 opts out."""
    from datetime import datetime as _dt, timezone as _tz

    def _exit_diag(tag: str, **extra: object) -> None:
        if os.environ.get("HERMES_GATEWAY_EXIT_DIAG", "1") != "1":
            return
        try:
            from hermes_constants import get_hermes_home as _ghh
            log_dir = _ghh() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            line = {
                "ts": _dt.now(_tz.utc).isoformat(), "tag": tag, "pid": os.getpid(),
                "python": sys.version.split()[0], "platform": sys.platform, **extra,
            }
            with open(log_dir / "gateway-exit-diag.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(line, default=str) + "\n")
        except Exception:
            pass  # never let the diagnostic itself crash the gateway

    return _exit_diag


def _respawn_storm_backoff() -> None:
    """Portable app-level respawn-storm breaker (for supervisors without a floor). Defaults mirror
    DEFAULT_CONFIG ``gateway.respawn_storm``; HERMES_GATEWAY_MAX_STARTS / HERMES_GATEWAY_START_WINDOW_S
    override; max_starts <= 0 disables. Never blocks startup."""
    try:
        from gateway.status import record_start_and_check_storm
        _max_starts = 5
        _win = 120.0
        try:
            from hermes_cli.config import load_config
            _cfg = load_config()
            _gw = _cfg.get("gateway") if isinstance(_cfg, dict) else None
            _rs = _gw.get("respawn_storm") if isinstance(_gw, dict) else None
            if isinstance(_rs, dict):
                if isinstance(_rs.get("max_starts"), int):
                    _max_starts = _rs["max_starts"]
                if isinstance(_rs.get("window_seconds"), (int, float)):
                    _win = float(_rs["window_seconds"])
        except Exception:
            pass
        try:
            _max_starts = int(os.environ["HERMES_GATEWAY_MAX_STARTS"])
        except (KeyError, ValueError):
            pass
        try:
            _win = float(os.environ["HERMES_GATEWAY_START_WINDOW_S"])
        except (KeyError, ValueError):
            pass
        _storm = record_start_and_check_storm(max_starts=_max_starts, window_s=_win) if _max_starts > 0 else None
        if _storm is not None:
            logger.warning(
                "Gateway (re)started %d times in %.0fs — backing off %.0fs to break a respawn storm.",
                _storm.count, _storm.window_s, _storm.backoff_s,
            )
            # Tell the startup watchdog the backoff sleep is intentional, not a parked deadlock.
            try:
                from hermes_startup_watchdog import kick_startup_watchdog
                kick_startup_watchdog(extra_s=_storm.backoff_s)
            except Exception:
                pass
            time.sleep(_storm.backoff_s)
    except Exception as _be:
        logger.debug("respawn-storm breaker check failed (non-fatal): %s", _be)


def run_gateway(verbose: int = 0, quiet: bool = False, replace: bool = False, force: bool = False):
    """Run the gateway in foreground. verbose 1=INFO/2+=DEBUG on stderr; quiet: no stderr logs; replace:
    kill an existing instance first (avoids systemd restart loops); force: skip the supervised guard."""
    _guard_official_docker_root_gateway()
    _attach_to_host_gateway_or_guard(force=force, replace=replace)
    _guard_supervised_gateway_conflict(force=force)
    _guard_existing_gateway_process_conflict(replace=replace)
    sys.path.insert(0, str(PROJECT_ROOT))
    _apply_startup_watchdog_config()

    # Detached Windows runs (HERMES_GATEWAY_DETACHED=1, or non-TTY for older wrappers) ignore
    # console-control broadcasts from sibling CLIs; foreground runs keep Ctrl+C-to-stop.
    stdin_is_tty = bool(_stdin_is_tty())
    _console_window_attached = _windows_console_window_attached()
    _breakaway = _windows_gateway_breakaway_state()
    _absorb = _windows_gateway_should_absorb_console_controls()
    if _absorb:
        _absorb_windows_console_controls()

    # A system-level unit execs us without XDG_RUNTIME_DIR/DBUS_SESSION_BUS_ADDRESS; adopt our own
    # user bus before any worker env snapshot so `systemd-run --user --scope` works (#104893).
    if is_linux() and os.environ.get("INVOCATION_ID"):
        _systemd_runtime.ensure_user_env()

    # Refresh the systemd unit on every boot so restart settings stay current even after an
    # exit-code-75 respawn (stale-code or /restart), which bypasses `hermes gateway restart`.
    if _systemd_runtime.supports_services():
        try:
            _systemd_unit_state.refresh_if_needed(system=False)
        except Exception:
            pass  # best-effort; don't block gateway startup

    from gateway.run import start_gateway
    print("┌─────────────────────────────────────────────────────────┐")
    print("│           ☤ Hermes Gateway Starting...                 │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│  Messaging platforms + cron scheduler                    │")
    print("│  Press Ctrl+C to stop                                   │")
    print("└─────────────────────────────────────────────────────────┘")
    print()

    # Exit 1 if no platform connects so systemd Restart=always retries transient errors.
    verbosity = None if quiet else verbose

    import atexit as _atexit
    import traceback as _traceback
    _exit_diag = _make_exit_diag()
    _exit_diag(
        "gateway.start", replace=replace, argv=sys.argv, stdin_is_tty=stdin_is_tty,
        console_window_attached=_console_window_attached, detached=_gateway_detached_env(),
        breakaway=_breakaway, absorb_windows_console_controls=_absorb,
    )
    _atexit.register(lambda: _exit_diag("atexit.hook", sys_exc=repr(sys.exc_info())))

    _respawn_storm_backoff()

    def _hard_exit_after_gateway_teardown(code: int) -> None:
        # Mirror gateway.run.main()'s wedge-proof exit: bypass Python finalization so non-daemon
        # threads (in-flight cron jobs) can't delay a /restart by minutes.
        from gateway.run import _exit_after_graceful_shutdown
        _exit_after_graceful_shutdown(code)

    success = False
    try:
        success = asyncio.run(start_gateway(replace=replace, force=force, verbosity=verbosity))
        _exit_diag("asyncio.run.returned", success=success)
    except KeyboardInterrupt:
        # Detached Windows runs absorb SIGINT above; keep the handler for console runs.
        _exit_diag("asyncio.run.KeyboardInterrupt", traceback=_traceback.format_exc())
        print("\nGateway stopped.")
        _hard_exit_after_gateway_teardown(0)
        return  # unreachable in production (os._exit); guard for test stubs
    except SystemExit as e:
        _exit_diag("asyncio.run.SystemExit", code=e.code, traceback=_traceback.format_exc())
        _hard_exit_after_gateway_teardown(0 if e.code is None else e.code if isinstance(e.code, int) else 1)
    except BaseException as e:
        # Everything else (CancelledError, exotic BaseExceptions): log the cause, then re-raise.
        _exit_diag("asyncio.run.exception", exc_type=type(e).__name__, exc_repr=repr(e), traceback=_traceback.format_exc())
        raise
    if not success:
        _exit_diag("gateway.exit_nonzero")
        _hard_exit_after_gateway_teardown(1)
    _exit_diag("gateway.exit_clean")
    _hard_exit_after_gateway_teardown(0)


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
            from gateway.drain_report import describe_active_work_unit
            lines.extend(f"     • {describe_active_work_unit(u)}" for u in work if isinstance(u, dict))
    elif gateway_state == "stopped" and exit_reason:
        lines.append(f"⚠ Last shutdown reason: {exit_reason}")

    return lines



def _print_info_lines(*lines: str) -> None:
    for line in lines:
        print_info(line)



# WhatsApp/DingTalk/WeCom/Feishu setup flows live in their plugins' adapter.py::interactive_setup.


def _running_under_s6() -> bool:
    from gateway.service_manager import detect_service_manager
    return detect_service_manager() == "s6"


def _systemd_unit_installed() -> bool:
    return _systemd_runtime.supports_services() and (
        _systemd_identity.unit_path(system=False).exists() or _systemd_identity.unit_path(system=True).exists()
    )


def _is_service_installed() -> bool:
    return _installed_service_kind() is not None


def _is_service_running() -> bool:
    """Check if the gateway service is currently running."""
    if _systemd_runtime.supports_services():
        return _systemd_runtime.unit_is_active(False) or _systemd_runtime.unit_is_active(True)
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
    if _systemd_runtime.supports_services():
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
    return _systemd_lifecycle.service_call(verb, False if system is None else system)



# =============================================================================
# Main Command Handler
# =============================================================================

def _dispatch_via_service_manager_if_s6(action: str, profile: str | None = None) -> bool:
    """Dispatch start/stop/restart via s6 inside an s6 container; True iff dispatched (caller returns).
    Profile defaults to the current one; missing slot / s6 errors become actionable CLI messages."""
    from gateway.service_manager import detect_service_manager, get_service_manager
    from gateway.s6_manager import GatewayNotRegisteredError, register_unregistered_profile_gateway

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
    from gateway.service_manager import detect_service_manager, get_service_manager
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
    except _systemd_runtime.UserSystemdUnavailableError as e:
        # Actionable message, not a traceback, when the user D-Bus session is unreachable.
        print_error("User systemd not reachable:")
        _print_indented(str(e))
        sys.exit(1)
    except _systemd_identity.SystemScopeRequiresRootError as e:
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
    # _systemd_identity.SystemScopeRequiresRootError is a RuntimeError and must propagate from systemd_stop.
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
    if _systemd_legacy.has_units():
        print()
        print_legacy_unit_warning()
        print()
        if non_interactive or prompt_yes_no("Remove the legacy unit(s) before installing?", True):
            remove_legacy_hermes_units(interactive=False)
            print()
    _systemd_lifecycle.install(
        force=force, system=system, run_as_user=run_as_user,
        enable_on_startup=start_on_login,
    )
    if start_now:
        _systemd_lifecycle.start(system=system)


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
    if backend is not None and _is_service_installed():
        from hermes_cli.gateway_setup_service import record_service_choice
        record_service_choice("install")


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
            _host_topology.served_by_another_host_gateway()
            or _host_topology.named_profile_served_by_running_multiplexer()):
        # The launch/default-profile lifecycle still names the whole host.
        owner = _host_topology.served_by_another_host_gateway()
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
        print(f"✓ Stopped {_service_identity.service_name()} service")


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
    if kind == "systemd" and _systemd_runtime.supports_services():
        linger_ok, _detail = _systemd_runtime.linger_status()
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
    from profiles.current import get_active_profile_name
    from gateway.profile_serving import profile_is_standalone

    active_standalone = ((get_active_profile_name() or "default") != "default"
                         and profile_is_standalone(get_hermes_home()))
    if active_standalone:
        from gateway.multiplex_mode import STANDALONE_DEPRECATION_NOTICE
        print("standalone by config (gateway.standalone: true) — temporary compatibility shim")
        print(f"  {STANDALONE_DEPRECATION_NOTICE}")
    _windows_service_installed = is_windows() and _gw_windows().is_installed()
    if (not active_standalone and not snapshot.running
            and _host_topology.named_profile_served_by_running_multiplexer()):
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
    from profiles.current import get_active_profile_name
    from gateway.profile_serving import profiles_to_serve
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
    if not _systemd_runtime.supports_services() and not is_macos():
        print("Legacy unit migration only applies to systemd-based Linux hosts.")
        return
    remove_legacy_hermes_units(interactive=not yes, dry_run=dry_run)


def _cmd_migrate(args):
    from nous_cli.gateway_migrate import cmd_migrate
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
    linger_enabled, linger_detail = _systemd_runtime.linger_status()
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

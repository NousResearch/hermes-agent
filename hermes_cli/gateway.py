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
    if supports_systemd_services():
        pattern = "hermes-gateway*" if all_profiles else get_service_name()
        # LOCAL PATCH (ai-hub, 2026-09-09). The all_profiles glob assumes every
        # gateway unit is named hermes-gateway*. On this box the live gateway unit
        # is `hermes-primary.service` (hosted-primary migration), so the glob
        # matched NOTHING: _get_service_pids(all_profiles=True) returned an empty
        # set and the orphan reaper saw the live gateway as a manual process. The
        # sweep guard measured EXCL=[] TARGETED=[<live pid>].
        # The fix must NOT widen to "hermes*": that would protect 8 unrelated
        # hermes-* services from every sweep. Instead ask the units themselves
        # which ones are gateways, by matching the ExecStart that starts one.
        extra_units: list[str] = []
        if all_profiles:
            try:
                listing = subprocess.run(
                    scope_args_probe := ["systemctl", "list-units", "hermes-*",
                                        "--plain", "--no-legend", "--no-pager"],
                    timeout=5, **_CAPTURE_TEXT)
                for _line in listing.stdout.strip().splitlines():
                    _parts = _line.split()
                    if not _parts or not _parts[0].endswith(".service"):
                        continue
                    _svc = _parts[0]
                    if _svc.startswith("hermes-gateway"):
                        continue  # already covered by the glob
                    _ex = subprocess.run(
                        ["systemctl", "show", _svc, "--property=ExecStart",
                         "--value"], timeout=5, **_CAPTURE_TEXT)
                    # Only a unit whose ExecStart literally runs the gateway.
                    if "gateway run" in (_ex.stdout or ""):
                        extra_units.append(_svc)
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass
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
                _lines = result.stdout.strip().splitlines()
                # LOCAL PATCH (see above): fold in gateway units whose name does not
                # match the hermes-gateway* glob, discovered by their ExecStart.
                _lines = _lines + ["%s x" % u for u in extra_units]
                for line in _lines:
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


def _get_parent_pid(pid: int) -> int | None:
    """Parent PID for ``pid``, or None. psutil first (works on Windows, where ``ps`` doesn't)."""
    if pid <= 1:
        return None
    try:
        import psutil  # type: ignore
        return psutil.Process(pid).ppid() or None
    except ImportError:
        pass
    except Exception:
        return None
    # ps fallback, POSIX only: Git Bash's ps.exe would flash a console from the windowless backend.
    if is_windows() or not shutil.which("ps"):
        return None
    try:
        result = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], timeout=5, **_CAPTURE_TEXT)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    raw = result.stdout.strip()
    if result.returncode != 0 or not raw:
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
    if not hasattr(signal, "SIGUSR1") or not _is_pid_ancestor_of_current_process(pid):
        return False
    try:
        os.kill(pid, signal.SIGUSR1)  # windows-footgun: ok — POSIX signal, guarded by hasattr(signal, 'SIGUSR1') above
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def _graceful_restart_via_sigusr1(pid: int, drain_timeout: float, *, on_progress=None) -> bool:
    """SIGUSR1 (drain-aware restart) a gateway PID and wait for exit; False if unsent or it outlived the timeout.

    gateway/run.py maps SIGUSR1 to ``request_restart(via_service=True)``: refuse new turns, drain,
    ``stop()``, exit; the supervisor relaunches. ``drain_timeout`` must cover after-turn wait + drain
    — pass ``resolve_restart_exit_wait_budget(...)``. ``on_progress`` (zero-arg) runs on every poll so
    a long wait can report what the gateway is still holding for (``update_cmd_drain_report``).
    """
    if not hasattr(signal, "SIGUSR1") or pid <= 0:
        return False
    try:
        os.kill(pid, signal.SIGUSR1)  # windows-footgun: ok — POSIX signal, guarded by hasattr(signal, 'SIGUSR1') above
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False

    return _wait_for_pid_exit(pid, max(drain_timeout, 1.0), on_progress=on_progress)


def _wait_for_pid_exit(pid: int, timeout: float, *, on_progress=None) -> bool:
    """Wait up to ``timeout``s for ``pid`` to exit; True once gone. (``launchctl bootstrap`` fails EIO
    while the previous instance still drains, so teardown callers must wait for the real exit.)"""
    if pid <= 0:
        return True
    # ``os.kill(pid, 0)`` hard-kills on Windows (TerminateProcess); use _pid_exists instead.
    from gateway.status import _pid_exists
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        if not _pid_exists(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        if on_progress is not None:
            on_progress()
        time.sleep(0.5)


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
GATEWAY_LOOP_ALIVE = "alive"
GATEWAY_LOOP_WEDGED = "wedged"
GATEWAY_LOOP_UNKNOWN = "unknown"

# 3 missed 30s beats (gateway.shutdown_watchdog.DEFAULT_HEARTBEAT_INTERVAL_S): decisive, not one slow write.
DEFAULT_LOOP_LIVENESS_STALE_AFTER_S = 90.0

# Sentinel for "the producer never wrote the witness flag" (legacy payload).
_LOOP_TICK_ABSENT = object()


def _probe_loop_tick_socket(pid: int, home: Path | None, timeout: float = 1.0) -> bool | None:
    """Ping the loop-tick witness socket: True answered, False node present but silent, None no node (not evidence)."""
    try:
        from gateway.shutdown_watchdog import get_loop_tick_socket_path
        path = get_loop_tick_socket_path(home, pid)
        if not path.is_socket():
            return None
    except Exception:
        return None
    return _ping_loop_tick_witness(socket.AF_UNIX, str(path), timeout)


def _ping_loop_tick_witness(family: int, address, timeout: float) -> bool:
    """Connect to a loop-tick witness and expect one byte ``"1"``; False on refusal/timeout/any error."""
    sock = None
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.settimeout(max(float(timeout), 0.0))
        sock.connect(address)
        return sock.recv(1) == b"1"
    except Exception:
        return False
    finally:
        if sock is not None:
            with contextlib.suppress(Exception):
                sock.close()


def _probe_loop_tick_tcp(port: int, timeout: float = 1.0) -> bool | None:
    """TCP-loopback variant of the tick probe for Windows (no AF_UNIX in asyncio); same semantics, None
    on invalid port."""
    try:
        port_num = int(port)
        if port_num <= 0 or port_num > 65535:
            return None
    except (TypeError, ValueError):
        return None
    return _ping_loop_tick_witness(socket.AF_INET, ("127.0.0.1", port_num), timeout)


def _probe_loop_tick_socket_sustained(
    pid: int, home: Path | None, *, timeout: float = 1.0, strikes: int = 3, gap_s: float = 0.2,
    tcp_port: int | None = None,
) -> bool | None:
    """Probe the tick socket up to ``strikes`` times, ``gap_s`` apart: True once answered, False if a node
    stayed silent the whole window, None if the node vanished (not evidence). One silent probe is not
    destructive evidence — a transient synchronous stall can outlast one recv timeout.

    A single silent probe is NOT destructive evidence: the loop may be in a short transient synchronous
    stall (a reconnect storm, a heavy synchronous callback, scheduler delay) that outlasts one recv timeout.
    Killing a gateway on that would be a false wedge — the exact class of false positive #90502 exists to
    prevent. Destructive authority therefore requires the loop to fail to answer across a bounded window of
    ``strikes`` consecutive misses, ``gap_s`` apart; any answer inside the window proves the loop is
    dispatching and returns ``True``.
    """
    total = max(int(strikes), 0)
    for attempt in range(total):
        if tcp_port is not None:
            result = _probe_loop_tick_tcp(tcp_port, timeout=timeout)
        else:
            result = _probe_loop_tick_socket(pid, home, timeout=timeout)
        if result is True:
            return True
        if result is None:
            # No node: ambiguity, never a wedge — absence is not a miss.
            return None
        if attempt < total - 1 and gap_s > 0:
            time.sleep(gap_s)
    return False


def probe_gateway_loop_liveness(
    pid: int, *, stale_after: float = DEFAULT_LOOP_LIVENESS_STALE_AFTER_S, home: Path | None = None,
    tick_timeout: float = 1.0, tick_strikes: int = 3, tick_gap_s: float = 0.2,
) -> str:
    """Classify a gateway PID's event loop as alive / wedged / unknown (see block comment above).
    Stale heartbeat is ``wedged`` only when the payload declares the tick socket armed AND it stays
    silent across ``tick_strikes`` misses; any answer is ``alive``; ambiguity is ``unknown``.

    - the loop-tick socket (``state/gateway.loop-tick.<pid>.sock``): answered by the gateway loop itself, so
    a reply is direct proof that the loop is dispatching. It is never refreshed by the heartbeat executor
    thread and never stalled by a filesystem that is slow to fsync. - the heartbeat file
    (``state/gateway.heartbeat``): rewritten every 30s on a thread since #90502, so freshness alone is no
    longer proof of loop schedulability — a stalled write (measured at 112.6s max on the incident box) or a
    saturated executor can age the file while the loop runs, and a write can land after the loop froze.
    """
    try:
        stale_budget = max(float(stale_after), 0.0)
    except (TypeError, ValueError):
        stale_budget = DEFAULT_LOOP_LIVENESS_STALE_AFTER_S
    try:
        from gateway.shutdown_watchdog import get_loop_heartbeat_path
        path = get_loop_heartbeat_path(home)
        mtime = path.stat().st_mtime
        payload = json.loads(path.read_text(encoding="utf-8"))
        heartbeat_pid = int(payload.get("pid", 0))
    except Exception:
        return GATEWAY_LOOP_UNKNOWN
    if heartbeat_pid <= 0 or int(pid) <= 0 or heartbeat_pid != int(pid):
        # Heartbeat is not this process's (old version, starting up, stale file): not evidence.
        return GATEWAY_LOOP_UNKNOWN

    # TCP loopback witness (Windows) takes priority when published; else the AF_UNIX socket.
    tcp_port = payload.get("loop_tick_tcp_port")
    try:
        tcp_port_int = int(tcp_port) if tcp_port is not None else None
    except (TypeError, ValueError):
        tcp_port_int = None

    if tcp_port_int is not None and tcp_port_int > 0:
        witness = _probe_loop_tick_tcp(tcp_port_int, timeout=tick_timeout)
        tick_armed = True
    else:
        witness = _probe_loop_tick_socket(pid, home, timeout=tick_timeout)
        tick_armed = payload.get("loop_tick_socket", _LOOP_TICK_ABSENT)
    if witness is True:
        # Loop answered: a stale file is a stalled write, not a wedge.
        return GATEWAY_LOOP_ALIVE
    # The loop answered a ping — it is dispatching right now. See #90502.
    age = time.time() - mtime
    if age <= stale_budget:
        if witness is False:
            # Fresh file but silent loop: an off-loop write can land after the loop froze.
            return GATEWAY_LOOP_UNKNOWN
        return GATEWAY_LOOP_ALIVE

    # Stale past the budget; the verdict depends on what the producer promised about its witness.
    if tick_armed is _LOOP_TICK_ABSENT:
        # Legacy on-loop writer: staleness proves the loop stopped scheduling.
        return GATEWAY_LOOP_WEDGED
    if tick_armed is not True:
        # Witness could not be armed (bind failed); off-loop write means staleness is not proof.
        return GATEWAY_LOOP_UNKNOWN
    if witness is False:
        # First miss. The probe above is miss #1, so ``tick_strikes - 1`` more attempts follow.
        # One silent probe is NOT destructive authority: a short transient synchronous stall can outlast a
        # single recv timeout, and killing a live gateway on it would be the exact false wedge #90502 exists
        # to prevent.
        sustained = _probe_loop_tick_socket_sustained(
            pid, home, timeout=tick_timeout, strikes=tick_strikes - 1, gap_s=tick_gap_s, tcp_port=tcp_port_int
        )
        if sustained is False:
            return GATEWAY_LOOP_WEDGED
        if sustained is True:
            return GATEWAY_LOOP_ALIVE  # Transient stall, not a wedge.
        return GATEWAY_LOOP_UNKNOWN  # Witness vanished mid-window: ambiguity — never kill on it.
    return GATEWAY_LOOP_UNKNOWN  # Armed but unreachable socket: ambiguity — never kill on it.


def _escalate_wedged_gateway(pid: int, *, term_grace: float = 5.0, kill_wait: float = 5.0) -> bool:
    """Bounded stop (SIGTERM, ``term_grace``, SIGKILL, ``kill_wait``) for a provably dead loop; True once gone.
    Callers MUST have classified ``GATEWAY_LOOP_WEDGED`` first: escalating a merely busy gateway
    bypasses the cron drain floor and SIGKILLs live work.

    See #86684.
    """
    from gateway.status import get_process_start_time
    expected_start_time = get_process_start_time(pid)
    try:
        terminate_pid(pid, force=False)
    except (ProcessLookupError, PermissionError, OSError):
        return _wait_for_pid_exit(pid, 1.0)
    if _wait_for_pid_exit(pid, max(float(term_grace), 0.0)):
        return True
    try:
        terminate_pid(pid, force=True, expected_start_time=expected_start_time)
        print(f"⚠ Gateway PID {pid} unresponsive to SIGTERM; sent SIGKILL")
    except (ProcessLookupError, PermissionError, OSError):
        pass
    return _wait_for_pid_exit(pid, max(float(kill_wait), 0.0))


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
    current_profile_arg = _profile_arg(current_home)
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
    from hermes_cli._subprocess_compat import bounded_probe_run
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
        include_restart_managers = not supports_systemd_services()
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
        from hermes_cli.profiles import list_profiles
    except Exception:
        if strict:
            raise
        return processes

    seen: set[int] = set()
    try:
        profiles = list_profiles()
    except Exception:
        if strict:
            raise
        return processes
    for profile in profiles:
        try:
            if strict:
                identity = get_running_pid_identity_strict(profile.path / "gateway.pid")
                pid = identity[0] if identity else None
                create_time = identity[1] if identity else 0.0
            else:
                pid = get_running_pid(profile.path / "gateway.pid", cleanup_stale=False)
                create_time = 0.0
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Could not inspect gateway PID for profile {profile.name}") from exc
            continue
        if pid is None or pid <= 0 or pid in _exclude or pid in seen:
            continue
        seen.add(pid)
        processes.append(ProfileGatewayProcess(profile=profile.name, path=profile.path, pid=pid, create_time=create_time))
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
        from hermes_cli.gateway_windows import hermes_owns_windows_service, hermes_service_roots

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


def _gateway_run_args_for_profile(profile: str) -> list[str]:
    args = [get_python_path(), "-m", "hermes_cli.main"]
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
    from hermes_cli._subprocess_compat import windows_detach_flags_without_breakaway, windows_detach_popen_kwargs

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
            from hermes_cli.gateway_windows import windowless_gateway_restart_spec
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
        from hermes_cli._subprocess_compat import (
            _WINDOWS_GATEWAY_BREAKAWAY_ENV, windows_detach_flags, windows_detach_flags_without_breakaway,
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
        # gateway dies when that job tears down. See _subprocess_compat.windows_detach_flags().
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


def _systemd_unit_is_active(system: bool) -> bool:
    """``systemctl is-active`` == "active" for the installed unit in ``system`` scope, else False."""
    if not get_systemd_unit_path(system=system).exists():
        return False
    try:
        result = _run_systemctl(["is-active", get_service_name()], system=system, timeout=10, **_CAPTURE_TEXT)
    except (RuntimeError, subprocess.TimeoutExpired):
        return False
    return result.stdout.strip() == "active"


def _probe_systemd_service_running(system: bool = False) -> tuple[bool, bool]:
    selected_system = _select_systemd_scope(system)
    return selected_system, _systemd_unit_is_active(selected_system)


def _parse_kv_pairs(items) -> dict[str, str]:
    """``{key: value}`` from ``KEY=VALUE`` strings (later keys win; values stripped)."""
    return {k: v.strip() for k, v in (item.split("=", 1) for item in items if "=" in item)}


def _systemctl_show(properties: tuple[str, ...], *, system: bool) -> dict[str, str]:
    """``systemctl show --property a,b`` for the gateway unit as ``{key: value}``; {} on failure."""
    try:
        result = _run_systemctl(
            ["show", get_service_name(), "--no-pager", "--property", ",".join(properties)],
            system=_select_systemd_scope(system), timeout=10, **_CAPTURE_TEXT,
        )
    except (RuntimeError, subprocess.TimeoutExpired, OSError):
        return {}
    return _parse_kv_pairs(result.stdout.splitlines()) if result.returncode == 0 else {}


def _unit_environment_value(unit_path: Path, name: str) -> str | None:
    """Value of one ``Environment="NAME=…"`` directive in the unit file at *unit_path*, with
    systemd's ``\\"``/``\\\\``/``%%`` quoting undone; None when the file or the key is absent."""
    try:
        text = unit_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        body = line.strip()
        if not body.startswith("Environment="):
            continue
        body = body[len("Environment=") :].strip()
        if body.startswith('"') and body.endswith('"'):
            body = body[1:-1].replace('\\"', '"').replace("\\\\", "\\").replace("%%", "%")
        if body.startswith(f"{name}="):
            return body.split("=", 1)[1].strip() or None
    return None


def _hermes_home_pinned_by_unit(unit_path: Path) -> str | None:
    """``HERMES_HOME`` pinned by the unit file at *unit_path*, or None when absent/unreadable."""
    return _unit_environment_value(unit_path, "HERMES_HOME")


def _hermes_home_from_systemd_unit_file(system: bool = False) -> str | None:
    """``HERMES_HOME`` from the on-disk unit file - what refresh/compare already read, and reliable under ``sudo``."""
    return _hermes_home_pinned_by_unit(get_systemd_unit_path(system=system))


def _sync_hermes_home_from_systemd_unit(system: bool) -> None:
    """Adopt a system-scope unit's ``HERMES_HOME``: under ``sudo`` it is stripped and HOME=/root, so
    get_hermes_home() would pick the wrong profile for runtime-status/PID reads."""
    if not system:
        return
    # On-disk unit first; ``systemctl show`` for units that only exist in the manager.
    unit_home = (_hermes_home_from_systemd_unit_file(system=True) or "").strip()
    if not unit_home:
        env_line = _systemctl_show(("Environment",), system=True).get("Environment", "")
        unit_home = _parse_kv_pairs(env_line.split()).get("HERMES_HOME", "").strip()
    if unit_home and os.environ.get("HERMES_HOME", "").strip() != unit_home:
        os.environ["HERMES_HOME"] = unit_home


def _read_systemd_unit_properties(
    system: bool = False,
    properties: tuple[str, ...] = ("ActiveState", "SubState", "Result", "ExecMainStatus", "MainPID"),
) -> dict[str, str]:
    """Return selected ``systemctl show`` properties for the gateway unit."""
    return _systemctl_show(properties, system=system)


def _positive_pid(value) -> int | None:
    """``int(value)`` when it parses and is > 0, else None."""
    try:
        pid = int(value or 0)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _systemd_main_pid_from_props(props: dict[str, str]) -> int | None:
    return _positive_pid(props.get("MainPID", "0") or "0")


def _runtime_state_pid(state: dict | None) -> int:
    """``pid`` recorded in a runtime-status dict; 0 when absent, unparsable, or non-positive."""
    return _positive_pid((state or {}).get("pid", 0)) or 0


def _systemd_main_pid(system: bool = False) -> int | None:
    return _systemd_main_pid_from_props(_read_systemd_unit_properties(system=system))


def _read_gateway_runtime_status() -> dict | None:
    try:
        from gateway.status import read_runtime_status
        state = read_runtime_status()
    except Exception:
        return None
    return state if isinstance(state, dict) else None


def _systemd_cli_bits(system: bool) -> tuple[str, str, str]:
    """``(sudo_prefix, scope_flag, user_flag)`` for printed hints: ``("sudo ", " --system", "")`` in
    system scope, ``("", "", "--user ")`` in user scope."""
    return ("sudo ", " --system", "") if system else ("", "", "--user ")


def _wait_for_systemd_service_restart(
    *,
    system: bool = False,
    previous_pid: int | None = None,
    timeout: float | None = None,
    replacement_observed: list[bool] | None = None,
) -> bool:
    """Wait for the gateway service to become active after a restart handoff."""
    svc = get_service_name()
    scope_label = _service_scope_label(system).capitalize()
    if timeout is None:
        timeout = _systemd_restart_wait_timeout(system=system)
    deadline = time.monotonic() + timeout
    printed_runtime_wait = False

    while time.monotonic() < deadline:
        props = _read_systemd_unit_properties(system=system)
        active_state = props.get("ActiveState", "")
        sub_state = props.get("SubState", "")
        try:
            from gateway.status import get_running_pid
            new_pid = get_running_pid()
        except Exception:
            new_pid = None
        new_pid = new_pid or _systemd_main_pid_from_props(props)

        runtime_state = _read_gateway_runtime_status()
        runtime_pid = _runtime_state_pid(runtime_state)
        if (
            previous_pid is not None
            and replacement_observed is not None
            and not replacement_observed
            and any(p > 0 and p != previous_pid for p in (new_pid or 0, runtime_pid))
        ):
            replacement_observed.append(True)

        if active_state == "active" and new_pid and (previous_pid is None or new_pid != previous_pid):
            if runtime_pid != new_pid:
                runtime_state = _read_gateway_runtime_status()
                if runtime_state and _runtime_state_pid(runtime_state) != new_pid:
                    runtime_state = None
            gateway_state = (runtime_state or {}).get("gateway_state")
            if gateway_state in ("running", "degraded"):
                print(f"✓ {scope_label} service restarted (PID {new_pid})")
                if gateway_state == "degraded":
                    # Serving, but a configured platform is parked or retrying: a real restart, not a
                    # failure — say so instead of waiting out the timeout and reporting one.
                    print(f"⚠ {scope_label} gateway is DEGRADED — see `hermes gateway status`")
                return True
            if gateway_state == "startup_failed":
                reason = (runtime_state or {}).get("exit_reason") or "startup failed"
                print(
                    f"⚠ {scope_label} service process restarted (PID {new_pid}), but gateway startup failed: {reason}"
                )
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

    sudo, _, user_flag = _systemd_cli_bits(system)
    print(
        f"⚠ {scope_label} service did not become active within {int(timeout)}s.\n"
        f"  Check status: {sudo}hermes gateway status\n"
        f"  Check logs:   journalctl {user_flag}-u {svc} -l --since '2 min ago'"
    )
    return False


def _systemd_restart_wait_timeout(system: bool = False) -> float:
    """Cover systemd's relaunch delays before applying the runtime wait floor."""
    from gateway.shutdown_forensics import parse_systemd_duration_to_us
    props = _read_systemd_unit_properties(system=system, properties=("RestartUSec", "TimeoutStartUSec"))
    supervisor_budget = 0.0
    for name in ("RestartUSec", "TimeoutStartUSec"):
        raw = props.get(name, "")
        duration_us = int(raw) if raw.isdigit() else parse_systemd_duration_to_us(raw)
        if duration_us is not None:
            supervisor_budget += duration_us / 1_000_000
    return 60.0 + supervisor_budget


def _systemd_unit_is_start_limited(props: dict[str, str]) -> bool:
    return "start-limit-hit" in (props.get("Result", "").lower(), props.get("SubState", "").lower())


def _systemd_error_indicates_start_limit(exc: subprocess.CalledProcessError) -> bool:
    parts: list[str] = []
    for attr in ("stderr", "stdout", "output"):
        value = getattr(exc, attr, None)
        if value:
            parts.append(value.decode(errors="replace") if isinstance(value, bytes) else str(value))
    text = "\n".join(parts).lower()
    return "start-limit-hit" in text or "start request repeated too quickly" in text or "start-limit" in text


def _systemd_service_is_start_limited(system: bool = False) -> bool:
    return _systemd_unit_is_start_limited(_read_systemd_unit_properties(system=system))


def _print_systemd_start_limit_wait(system: bool = False) -> None:
    svc = get_service_name()
    scope_label = _service_scope_label(system).capitalize()
    sudo, scope_flag, user_flag = _systemd_cli_bits(system)
    print(f"⏳ {scope_label} service is temporarily rate-limited by systemd.")
    print("  systemd is refusing another immediate start after repeated exits.")
    print(f"  Wait for the start-limit window to expire, then run: {sudo}hermes gateway restart{scope_flag}")
    print(f"  Or clear the failed state manually: systemctl {user_flag}reset-failed {svc}")
    print(f"  Check logs: journalctl {user_flag}-u {svc} -l --since '5 min ago'")


def _recover_pending_systemd_restart(system: bool = False, previous_pid: int | None = None) -> bool:
    """Recover a planned service restart that is stuck in systemd state."""
    props = _read_systemd_unit_properties(system=system)
    if not props:
        return False

    try:
        from gateway.status import read_runtime_status
    except Exception:
        return False

    if not (read_runtime_status() or {}).get("restart_requested"):
        return False

    active_state = props.get("ActiveState", "")
    if active_state == "activating" and props.get("SubState", "") == "auto-restart":
        print("⏳ Service restart already pending — waiting for systemd relaunch...")
        return _wait_for_systemd_service_restart(system=system, previous_pid=previous_pid)

    if active_state == "failed" and (
        props.get("ExecMainStatus", "") == str(GATEWAY_SERVICE_RESTART_EXIT_CODE)
        or props.get("Result", "") == "exit-code"
    ):
        svc = get_service_name()
        print(f"↻ Clearing failed state for pending {_service_scope_label(system)} service restart...")
        _run_systemctl(["reset-failed", svc], system=system, check=False, timeout=30)
        _run_systemctl(["start", svc], system=system, check=False, timeout=90)
        return _wait_for_systemd_service_restart(system=system, previous_pid=previous_pid)

    return False


def _parse_launchd_pid_from_list_output(output: str) -> int | None:
    """PID from ``launchctl list <label>`` (``"PID" = <n>;``); None if absent (registered, not running)
    or non-positive (crashed)."""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(('"PID"', "PID")) and "=" in stripped:
            return _positive_pid(stripped.split("=", 1)[1].strip().rstrip(";").strip('"'))
    return None


def _parse_launchd_pid_from_print_output(output: str) -> int | None:
    """Live PID from ``launchctl print`` (first ``pid = <N>`` line wins); None if absent or non-positive."""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("pid = "):
            return _positive_pid(stripped[len("pid = "):].strip())
    return None


def _launchd_print_service_pid(domain: str, label: str) -> tuple[bool, int | None]:
    """``(loaded, pid)`` for ``domain/label`` via ``launchctl print`` (domain-explicit; ``launchctl list``
    infers it from caller context). ``TimeoutExpired`` propagates: a wedged launchctl is not "unloaded".

    Domain-explicit on purpose: legacy ``launchctl list`` infers its domain from the caller's execution
    context, which is exactly the ambiguity that sank the first fleet-restart attempt (#41403 review).
    ``TimeoutExpired`` propagates — fleet-restart callers own per-label failure accounting (a wedged
    launchctl call must be reported, not read as "unloaded").
    """
    try:
        result = subprocess.run(["launchctl", "print", f"{domain}/{label}"], timeout=5, **_CAPTURE_TEXT)
    except FileNotFoundError:
        return (False, None)
    if result.returncode != 0:
        return (False, None)
    return (True, _parse_launchd_pid_from_print_output(result.stdout))


def _launchd_service_registered(label: str, *, timeout: int = 5) -> bool:
    """True when launchd knows ``label`` (``launchctl list`` exit 0). Domain-agnostic, so still true on
    macOS 26+ hosts whose per-user domains reject management. FileNotFoundError/TimeoutExpired propagate."""
    result = subprocess.run(["launchctl", "list", label], timeout=timeout, **_CAPTURE_TEXT)
    return result.returncode == 0


def _locate_launchd_gateway_service(label: str) -> tuple[str | None, int | None]:
    """``(domain, pid)`` for ``label``, probing ``gui/<uid>`` then ``user/<uid>``. Never uses the current
    profile's cached ``_launchd_domain()`` — a fleet can mix domains. ``TimeoutExpired`` propagates."""
    uid = os.getuid()  # windows-footgun: ok — POSIX launchd (macOS) helper, never invoked on Windows
    for domain in (f"gui/{uid}", f"user/{uid}"):
        loaded, pid = _launchd_print_service_pid(domain, label)
        if loaded:
            return (domain, pid)
    return (None, None)


def _probe_launchd_service_running() -> bool:
    """True when the plist exists AND launchd is running a process for the current label."""
    return get_launchd_plist_path().exists() and _launchctl_label_supervising_process(get_launchd_label())


def _s6_gateway_snapshot(gateway_pids: tuple[int, ...]) -> GatewayRuntimeSnapshot | None:
    """Snapshot for an s6-supervised container gateway, or None when s6 isn't the service manager."""
    from hermes_cli.service_manager import detect_service_manager, get_service_manager
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
    try:
        from gateway.status import read_runtime_status
        reason = (read_runtime_status() or {}).get("multiplex_standalone_reason")
    except Exception:
        return
    if reason:
        print(f"⚠ Serving the default profile only: {reason}")
        print("  Fold every profile onto this gateway: hermes gateway migrate --multiplex")


def _print_served_ingress_urls(profile: str | None = None) -> None:
    """Callback URLs of inbound-port platforms the live multiplexer serves for secondary profiles
    (the value to paste into the Twilio / LINE / Teams / BlueBubbles console)."""
    try:
        from hermes_cli.gateway_multiplex_served import format_ingress_url_lines, served_profile_ingress_urls
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
        from hermes_cli.gateway_multiplex_served import served_profile_unserved_platforms
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
        from hermes_cli.profiles import get_active_profile_name
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
        from hermes_cli.gateway_migrate import duplicate_credential_findings
        lines = duplicate_credential_findings()
        if lines:
            print()
            for line in lines:
                print(f"⚠ {line}")


def _gateway_list() -> None:
    """List every profile and whether its gateway is running."""
    try:
        from hermes_cli.profiles import list_profiles, get_active_profile_name
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
            elif named_profile_served_by_running_multiplexer(prof.name):
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
                from gateway.status import get_process_start_time
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
        supervised_host = supports_systemd_services()
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
            from hermes_cli.gateway_windows import get_task_name  # profile-aware task name
            _task_name = get_task_name()
        except Exception:
            _task_name = "Hermes_Gateway"
        if _windows_scheduled_task_supervises(_task_name):
            return False

    from gateway.status import _pid_exists, get_process_start_time, write_planned_stop_marker
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
    # Service-managed gateways are never orphans (on macOS supports_systemd_services() is False, so a
    # launchd gateway would otherwise be SIGTERM'd); all_profiles because the scan sees siblings too.
    with contextlib.suppress(Exception):
        # This covers macOS launchd (supports_systemd_services() is False there, so without this the launchd
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


def _mark_planned_stop(pid: int | None = None) -> None:
    """Best-effort planned-stop marker for ``pid`` (default: the recorded gateway PID)."""
    try:
        from gateway.status import get_running_pid, write_planned_stop_marker
        if pid is None:
            pid = get_running_pid(cleanup_stale=False)
        if pid is not None:
            write_planned_stop_marker(pid)
    except Exception:
        pass


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
        from gateway.status import get_process_start_time
        from hermes_cli.gateway_windows import (
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
        _mark_planned_stop(pid)
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


def user_systemd_unit_dir() -> Path:
    """``$XDG_CONFIG_HOME/systemd/user`` (``~/.config`` only as the spec's default).

    Hardcoding ``~/.config`` made every unit probe silently false-negative on a host that moves
    XDG_CONFIG_HOME — doctor then reported no gateway unit while systemd was running ours.
    """
    config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "systemd" / "user"


def get_systemd_unit_path(system: bool = False) -> Path:
    name = get_service_name()
    if system:
        return _SYSTEM_UNIT_DIR / f"{name}.service"
    return user_systemd_unit_dir() / f"{name}.service"


def _adopted_systemd_unit_name() -> str | None:
    """Name of a systemd unit that RUNS this gateway but is not named hermes-gateway*.

    LOCAL PATCH (ai-hub, 2026-09-09). Companion to the _get_service_pids
    discovery: patch 232 taught the orphan REAPER to find such units, but the
    restart arm still decided "is there a supervisor?" purely from
    get_systemd_unit_path().exists(). On this box the live gateway runs under a
    hand-written hermes-primary.service and no hermes-gateway.service file
    exists, so that test returns False and the restart arm falls through to the
    MANUAL branch: it signals the running gateway and starts an unsupervised
    replacement in the caller's own foreground, racing Restart=always.

    Identify by ExecStart, never by name glob: matching hermes* would adopt 8
    unrelated hermes-* services. Returns the unit name, or None when the
    standard unit is present or nothing on the box runs a gateway.
    """
    if not supports_systemd_services():
        return None
    _pin = os.environ.get("HERMES_GATEWAY_SYSTEMD_UNIT")
    if _pin:
        return _pin
    try:
        listing = subprocess.run(
            ["systemctl", "list-units", "hermes-*", "--plain", "--no-legend",
             "--no-pager"],
            timeout=5, **_CAPTURE_TEXT)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    for line in (listing.stdout or "").strip().splitlines():
        parts = line.split()
        if not parts or not parts[0].endswith(".service"):
            continue
        svc = parts[0]
        if svc.startswith("hermes-gateway"):
            continue
        try:
            ex = subprocess.run(
                ["systemctl", "show", svc, "--property=ExecStart", "--value"],
                timeout=5, **_CAPTURE_TEXT)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
        if "gateway run" in (ex.stdout or ""):
            return svc
    return None

class UserSystemdUnavailableError(RuntimeError):
    """``systemctl --user`` cannot reach the user D-Bus session (fresh SSH sessions with linger off,
    so ``/run/user/$UID/bus`` never exists). ``args[0]`` is a user-facing remediation message."""


class SystemScopeRequiresRootError(RuntimeError):
    """System-scope gateway operation attempted as non-root. Typed (not ``sys.exit(1)``) so the setup
    wizard can print remediation; ``args`` = (message, action) and ``str(e)`` is the message only."""

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


def _user_runtime_dir() -> Path:
    """``$XDG_RUNTIME_DIR`` or ``/run/user/<uid>`` (regardless of existence)."""
    return Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows


def _user_dbus_socket_path() -> Path:
    """Return the expected per-user D-Bus socket path (regardless of existence)."""
    return _user_runtime_dir() / "bus"


def _user_systemd_private_socket_path() -> Path:
    """Return the per-user systemd private socket path (regardless of existence)."""
    return _user_runtime_dir() / "systemd" / "private"


def _path_exists_safe(path: Path) -> bool:
    """``Path.exists()`` treating an inaccessible path as absent: a leaked ``XDG_RUNTIME_DIR`` from
    another user (``/run/user/0`` is 0700) would otherwise crash the preflight with EACCES.

    ``Path.exists()`` only swallows a subset of ``OSError`` (ENOENT/ENOTDIR/ EBADF/ELOOP); ``EACCES`` still
    propagates. When ``XDG_RUNTIME_DIR`` leaks from another user — the classic ``su``/``sudo -u`` from a
    root shell case, where ``/run/user/0`` is ``0700 root:root`` — stat-ing a socket underneath it raises
    ``PermissionError`` that escapes the systemd preflight as a raw traceback (#86558). An unreadable path
    is, for our purposes, not reachable.
    """
    try:
        return path.exists()
    except OSError:  # e.g. EACCES on another user's runtime dir
        return False


def _runtime_dir_is_ours(runtime_dir: str) -> bool:
    """True when *runtime_dir* exists and is owned by our uid (a leaked foreign XDG_RUNTIME_DIR must not be trusted)."""
    try:
        return Path(runtime_dir).stat().st_uid == os.getuid()  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
    except OSError:
        return False


def _user_systemd_socket_ready() -> bool:
    """True when the user D-Bus socket OR the per-user systemd private socket exists (some distros
    expose only the latter and ``systemctl --user`` still works). Inaccessible counts as not-ready."""
    return _path_exists_safe(_user_dbus_socket_path()) or _path_exists_safe(_user_systemd_private_socket_path())


def _ensure_user_systemd_env() -> None:
    """Set XDG_RUNTIME_DIR / DBUS_SESSION_BUS_ADDRESS so ``systemctl --user`` works on headless (SSH)
    hosts; an XDG_RUNTIME_DIR leaked from another user is replaced with our own ``/run/user/{uid}``.

    An ``XDG_RUNTIME_DIR`` that leaked from another user (``su``/``sudo -u`` from root, where the env still
    points at ``/run/user/0``) is dropped in favour of our own ``/run/user/{uid}`` so ``systemctl --user``
    targets the right instance instead of an unreadable foreign socket (#86558).
    """
    uid = os.getuid()  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if (not xdg or not _runtime_dir_is_ours(xdg)) and _runtime_dir_is_ours(f"/run/user/{uid}"):
        os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{uid}"

    if "DBUS_SESSION_BUS_ADDRESS" not in os.environ:
        bus_path = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{uid}")) / "bus"
        if _path_exists_safe(bus_path):
            os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus_path}"


def _wait_for_user_dbus_socket(timeout: float = 3.0) -> bool:
    """Poll up to ``timeout`` s for a user systemd control socket (user@.service takes a moment after enable-linger)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _user_systemd_socket_ready():
            _ensure_user_systemd_env()
            return True
        time.sleep(0.2)
    return _user_systemd_socket_ready()


def _wait_for_target_user_bus(uid: int, timeout: float = 5.0) -> bool:
    """Poll for ``/run/user/<uid>/bus`` of ANOTHER account (the system unit's ``User=`` while root installs).
    Only the D-Bus socket counts — ``systemd/private`` alone is enough for ``systemctl --user`` but not for
    the ``systemd-run --user`` that restart-safe workers need. Never adopts anything into our env."""
    bus = Path(f"/run/user/{uid}/bus")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _path_exists_safe(bus):
            return True
        time.sleep(0.2)
    return _path_exists_safe(bus)


def _loginctl_enable_linger(username: str) -> subprocess.CompletedProcess:
    """``loginctl enable-linger <username>`` (check=False, 30s); exceptions propagate to the caller."""
    return subprocess.run(["loginctl", "enable-linger", username], check=False, timeout=30, **_CAPTURE_TEXT)


def _completed_process_detail(result) -> str:
    """stderr, else stdout, else ``exit <rc>`` — stripped."""
    return (result.stderr or result.stdout or f"exit {result.returncode}").strip()


def _preflight_user_systemd(*, auto_enable_linger: bool = True) -> None:
    """Ensure ``systemctl --user`` can reach user-scope systemd; raise UserSystemdUnavailableError otherwise.
    No-op when a control socket exists; else wait briefly if linger is on, or (``auto_enable_linger``)
    try ``loginctl enable-linger`` (non-root works when polkit permits)."""
    _ensure_user_systemd_env()
    if _user_systemd_socket_ready():
        return

    import getpass
    username = getpass.getuser()
    linger_enabled, linger_detail = get_systemd_linger_status()
    sudo_hint = f"  sudo loginctl enable-linger {username}"

    if linger_enabled is True:
        if _wait_for_user_dbus_socket(timeout=3.0):
            return
        # Linger is on but socket still missing — unusual; fall through to error.
        _raise_user_systemd_unavailable(
            username,
            reason="User systemd control sockets are missing even though linger is enabled.",
            fix_hint=(
                f"  systemctl start user@{os.getuid()}.service\n"  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
                "  (may require sudo; try again after the command succeeds)"
            ),
        )

    if auto_enable_linger and shutil.which("loginctl"):
        try:
            result = _loginctl_enable_linger(username)
        except Exception as exc:
            _raise_user_systemd_unavailable(
                username, reason=f"loginctl enable-linger failed ({exc}).", fix_hint=sudo_hint
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
            _raise_user_systemd_unavailable(
                username,
                reason=f"loginctl enable-linger was denied: {_completed_process_detail(result)}",
                fix_hint=sudo_hint,
            )

    _raise_user_systemd_unavailable(
        username,
        reason=f"User D-Bus session is not available ({linger_detail or 'linger disabled'}).",
        fix_hint=sudo_hint,
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


def _run_systemctl(args: list[str], *, system: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """Run systemctl; raise RuntimeError (not raw FileNotFoundError) if missing, for callers bypassing
    ``supports_systemd_services()``."""
    try:
        return subprocess.run(_systemctl_cmd(system) + args, **kwargs)
    except FileNotFoundError:
        from hermes_cli.gateway_command_errors import SystemctlUnavailableError
        raise SystemctlUnavailableError() from None


def _service_scope_label(system: bool = False) -> str:
    return "system" if system else "user"


def get_installed_systemd_scopes() -> list[str]:
    scopes: list[str] = []
    seen_paths: set[Path] = set()
    for system, label in ((False, "user"), (True, "system")):
        unit_path = get_systemd_unit_path(system=system)
        if unit_path not in seen_paths and unit_path.exists():
            scopes.append(label)
            seen_paths.add(unit_path)
    return scopes


def has_conflicting_systemd_units() -> bool:
    return len(get_installed_systemd_scopes()) > 1


# Legacy pre-rename names: explicit allowlist (NOT a glob) so profile and third-party units never match.
_LEGACY_SERVICE_NAMES: tuple[str, ...] = ("hermes.service",)

# ExecStart markers identifying a unit as running our gateway; a legacy unit is flagged only if one matches.
_LEGACY_UNIT_EXECSTART_MARKERS: tuple[str, ...] = (
    "hermes_cli.main gateway",
    "hermes_cli/main.py gateway",
    "gateway/run.py",
    " hermes gateway ",
    "/hermes gateway ",
)


def _legacy_unit_search_paths() -> list[tuple[bool, Path]]:
    """``[(is_system, base_dir), ...]`` to scan for legacy units; factored out so tests can monkeypatch."""
    return [(False, user_systemd_unit_dir()), (True, _SYSTEM_UNIT_DIR)]


def _find_legacy_hermes_units() -> list[tuple[str, Path, bool]]:
    """``[(unit_name, unit_path, is_system)]`` for legacy gateway units (e.g. ``hermes.service``), which
    fight the current unit for the bot token (SIGTERM flap loop). Explicit name allowlist + ExecStart
    marker check so profile/third-party units never match; no mutation.

    Detects unit files installed by older Hermes versions that used a different service name (e.g. When both
    a legacy unit and the current ``hermes-gateway.service`` are active, they fight over the same bot token
    — the PR #5646 signal-recovery change turns this into a 30-second SIGTERM flap loop.
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
            if any(marker in text for marker in _LEGACY_UNIT_EXECSTART_MARKERS):
                results.append((name, unit_path, is_system))
    return results


def has_legacy_hermes_units() -> bool:
    """Return True when any legacy Hermes gateway unit files exist."""
    return bool(_find_legacy_hermes_units())


def print_legacy_unit_warning() -> None:
    """Warn about installed legacy gateway units; prints nothing when there are none."""
    legacy = _find_legacy_hermes_units()
    if not legacy:
        return
    print_warning("Legacy Hermes gateway unit(s) detected from an older install:")
    for name, path, is_system in legacy:
        print_info(f"    {path}  ({_service_scope_label(is_system)} scope)")
    print_info("  These run alongside the current hermes-gateway service and")
    print_info("  cause SIGTERM flap loops — both try to use the same bot token.")
    print_info("  Remove them with:")
    print_info("    hermes gateway migrate-legacy")


def remove_legacy_hermes_units(interactive: bool = True, dry_run: bool = False) -> tuple[int, list[Path]]:
    """Stop, disable, and remove legacy gateway units. ``interactive=False`` skips the prompt; ``dry_run``
    only lists. Returns ``(removed_count, remaining_paths)`` (remaining: e.g. system-scope when not root)."""
    legacy = _find_legacy_hermes_units()
    if not legacy:
        print("No legacy Hermes gateway units found.")
        return 0, []

    print()
    print("Legacy Hermes gateway unit(s) found:")
    for name, path, is_system in legacy:
        print(f"  {path}  ({_service_scope_label(is_system)} scope)")
    print()

    if dry_run:
        print("(dry-run — nothing removed)")
        return 0, [p for _, p, _ in legacy]

    if interactive and not prompt_yes_no("Remove these legacy units?", True):
        print("Skipped. Run again with: hermes gateway migrate-legacy")
        return 0, [p for _, p, _ in legacy]

    removed = 0
    remaining: list[Path] = []

    def _remove_units(units: list[tuple[str, Path]], *, system: bool) -> None:
        nonlocal removed
        for name, path in units:
            try:
                _run_systemctl(["stop", name], system=system, check=False, timeout=90)
                _run_systemctl(["disable", name], system=system, check=False, timeout=30)
                path.unlink(missing_ok=True)
                print(f"  ✓ Removed {path}")
                removed += 1
            except (OSError, RuntimeError) as e:
                print(f"  ⚠ Could not remove {path}: {e}")
                remaining.append(path)
        with contextlib.suppress(RuntimeError):
            _run_systemctl(["daemon-reload"], system=system, check=False, timeout=30)

    user_units = [(n, p) for n, p, is_sys in legacy if not is_sys]
    system_units = [(n, p) for n, p, is_sys in legacy if is_sys]
    if user_units:
        _remove_units(user_units, system=False)

    # System-scope removal (needs root)
    if system_units:
        if os.geteuid() != 0:  # windows-footgun: ok — Linux systemd removal path, guarded by `if system == "Linux"` / systemd-only branch
            print()
            print_warning("System-scope legacy units require root to remove.")
            print_info("  Re-run with: sudo hermes gateway migrate-legacy")
            remaining.extend(path for _, path in system_units)
        else:
            _remove_units(system_units, system=True)

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

    print_warning(f"Both user and system gateway services are installed ({' + '.join(scopes)}).")
    print_info("  This is confusing and can make start/stop/status behavior ambiguous.")
    print_info("  Default gateway commands target the user service unless you pass --system.")
    print_info("  Keep one of these:")
    print_info("    hermes gateway uninstall")
    print_info("    sudo hermes gateway uninstall --system")


def refuses_container_user_scope_install(system: bool) -> bool:
    """True (after printing the guidance) when a fresh USER-scope unit was requested inside a container.

    A systemd container passes ``supports_systemd_services()`` on purpose so ``--system`` keeps working,
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


def _require_root_for_system_service(action: str) -> None:
    if os.geteuid() != 0:  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
        raise SystemScopeRequiresRootError(f"System gateway {action} requires root. Re-run with sudo.", action)


def _system_service_identity(run_as_user: str | None = None) -> tuple[str, str, str, int]:
    import getpass
    import grp
    import pwd
    username = (
        run_as_user or os.getenv("SUDO_USER") or os.getenv("USER") or os.getenv("LOGNAME") or getpass.getuser()
    ).strip()
    if not username:
        raise ValueError("Could not determine which user the gateway service should run as")
    if username == "root" and not run_as_user:
        raise ValueError(
            "Refusing to install the gateway system service as root; pass --run-as-user root to override (e.g. in LXC containers)"
        )
    if username == "root":
        print_warning("Installing gateway service to run as root.")
        print_info("  This is fine for LXC/container environments but not recommended on bare-metal hosts.")

    try:
        user_info = pwd.getpwnam(username)
    except KeyError as e:
        raise ValueError(f"Unknown user: {username}") from e
    return username, grp.getgrgid(user_info.pw_gid).gr_name, user_info.pw_dir, user_info.pw_uid


def _read_systemd_user_from_unit(unit_path: Path) -> str | None:
    if not unit_path.exists():
        return None
    for line in unit_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("User="):
            return line.split("=", 1)[1].strip() or None
    return None


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

        systemd_install(force=force, system=True, run_as_user=run_as_user, enable_on_startup=enable_on_startup)
        return scope, True

    if refuses_container_user_scope_install(system=False):
        return scope, False
    systemd_install(force=force, system=False, enable_on_startup=enable_on_startup)
    return scope, True


def ensure_gateway_service(context: str = "setup") -> bool:
    """Install and start a user-scope gateway service without prompting (``hermes setup``/``import``).
    A zero-platform gateway is a supported degraded mode (cron runs), so this never gates on messaging
    config. Never raises; True when a service is installed and running."""
    from hermes_constants import is_container
    if is_container():
        # Containers use restart policies, not service managers.
        print_info("Start the gateway to bring your bots online:")
        print_info("   hermes gateway run          # Run as container main process")
        print_info("")
        print_info("For automatic restarts, use a Docker restart policy:")
        print_info("   docker run --restart unless-stopped ...")
        return False

    supports_systemd = supports_systemd_services()
    if not (supports_systemd or is_macos() or is_windows()):
        print_info("  No supported service manager found on this host.")
        print_info("  Run the gateway in the foreground with: hermes gateway")
        return False

    try:
        if _is_service_running():
            return True
        if _served_profile_needs_no_service():
            return True
        if not _is_service_installed():
            if supports_systemd and has_conflicting_systemd_units():
                # Both units would fight over bot tokens; don't pile a fresh install onto a conflicted state.
                print_systemd_scope_conflict_warning()
                return False
            print_info("  Installing the gateway background service ...")
            if supports_systemd:
                systemd_install(force=False, non_interactive=True)
            elif is_macos():
                launchd_install(force=False)
            else:
                _gw_windows().install(force=False)  # Registers the Scheduled Task AND starts it.
                print_success("  Gateway service installed and started.")
                return True
        if supports_systemd:
            systemd_start()
        elif is_macos():
            launchd_start()
        else:
            _gw_windows().start()
        print_success("  Gateway service running (cron jobs + messaging platforms).")
        return True
    except UserSystemdUnavailableError as e:
        print_warning("  Could not reach user systemd to start the gateway service:")
        _print_indented(str(e), print_info)
    except SystemScopeRequiresRootError as e:
        print_warning(f"  Gateway service needs root for this scope: {e}")
        _print_system_scope_remediation("start")
    except SystemExit:
        # Some install/start paths sys.exit() on hard failures (temp-HOME guard); never abort setup/import.
        print_warning("  Gateway service install did not complete.")
        print_info("  You can retry manually: hermes gateway install")
    except Exception as e:
        print_warning(f"  Gateway service install failed: {e}")
        print_info("  You can retry manually: hermes gateway install")
    return False


def get_systemd_linger_status(username: str | None = None) -> tuple[bool | None, str]:
    """Linger status for *username* or the current user when omitted.

    System-scope gateway installation runs as root but the service runs as a
    configured target user, so querying the caller would validate the wrong
    user manager.
    """
    if is_termux():
        return None, "not supported in Termux"
    if not is_linux():
        return None, "not supported on this platform"
    if not shutil.which("loginctl"):
        return None, "loginctl not found"

    if username is None:
        username = os.getenv("USER") or os.getenv("LOGNAME")
    if not username:
        try:
            import pwd
            username = pwd.getpwuid(os.getuid()).pw_name  # windows-footgun: ok — POSIX loginctl helper, never invoked on Windows
        except Exception:
            return None, "could not determine current user"

    try:
        result = subprocess.run(
            ["loginctl", "show-user", username, "--property=Linger", "--value"],
            check=False, timeout=10, **_CAPTURE_TEXT,
        )
    except Exception as e:
        return None, str(e)

    if result.returncode != 0:
        return None, _completed_process_detail(result) or "loginctl query failed"

    value = (result.stdout or "").strip().lower()
    if value in {"yes", "true", "1"}:
        return True, ""
    if value in {"no", "false", "0"}:
        return False, ""
    return None, f"unexpected loginctl output: {value or '<empty>'}"


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


def _detect_venv_dir() -> Path | None:
    """Active virtualenv dir: ``sys.prefix``, then ``VIRTUAL_ENV`` (uv sets it without changing
    sys.prefix), then .venv/venv under PROJECT_ROOT; None if none found."""
    candidates: list[Path] = []
    if sys.prefix != sys.base_prefix:
        candidates.append(Path(sys.prefix))
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(Path(os.environ["VIRTUAL_ENV"]))
    candidates += [PROJECT_ROOT / ".venv", PROJECT_ROOT / "venv"]
    return next((venv for venv in candidates if venv.is_dir()), None)


def get_python_path() -> str:
    venv = _detect_venv_dir()
    if venv is not None:
        try:
            from hermes_constants import venv_python_path
        except ImportError:
            # Update-boundary: a gateway restarted mid-update can hold a stale hermes_constants
            # without this symbol; see _reload_hermes_constants() in hermes_cli/managed_uv.py.
            from hermes_cli.managed_uv import _reload_hermes_constants
            venv_python_path = _reload_hermes_constants().venv_python_path

        venv_python = venv_python_path(venv, windows=is_windows())
        if venv_python.exists():
            return str(venv_python)
    return sys.executable


# =============================================================================
# Systemd (Linux)
# =============================================================================


def _build_user_local_paths(home: Path, path_entries: list[str]) -> list[str]:
    """Return user-local bin dirs that exist and aren't already in *path_entries*."""
    candidates = [
        str(home / ".local" / "bin"),  # uv, uvx, pip-installed CLIs
        str(home / ".cargo" / "bin"),  # Rust/cargo tools
        str(home / "go" / "bin"),  # Go tools
        str(home / ".npm-global" / "bin"),  # npm global packages
    ]
    return [p for p in candidates if p not in path_entries and Path(p).exists()]


def _build_wsl_interop_paths(path_entries: list[str]) -> list[str]:
    """WSL Windows-interop PATH entries for generated units: systemd services don't inherit the
    Windows PATH (``/mnt/c/WINDOWS/System32``…), so ``powershell.exe``/``cmd.exe`` break unless persisted."""
    if not is_wsl():
        return []

    candidates = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry.startswith("/mnt/")]
    for executable in ("powershell.exe", "cmd.exe", "explorer.exe", "wsl.exe"):
        resolved = shutil.which(executable)
        if resolved:
            candidates.append(str(Path(resolved).parent))
    candidates += [
        entry
        for entry in (
            "/mnt/c/WINDOWS/system32",
            "/mnt/c/WINDOWS",
            "/mnt/c/WINDOWS/System32/Wbem",
            "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/",
            "/mnt/c/WINDOWS/System32/OpenSSH/",
        )
        if Path(entry).exists()
    ]

    result: list[str] = []
    seen = set(path_entries)
    for entry in candidates:
        if entry and entry not in seen:
            seen.add(entry)
            result.append(entry)
    return result


def _remap_path_for_user(path: str, target_home_dir: str) -> str:
    """Swap the ``Path.home()`` prefix of *path* for *target_home_dir*; other paths return unchanged.
    Intentionally does NOT resolve symlinks."""
    current_home = Path.home()
    p = Path(path).expanduser()
    try:
        relative = p.relative_to(current_home)
        return str(Path(target_home_dir) / relative)
    except ValueError:
        return str(p)




def _print_linger_enable_warning(username: str, detail: str | None = None, *, system: bool = False) -> None:
    print()
    if system:
        print(f"⚠ Linger not enabled for {username} — cron and Kanban workers cannot start (no user D-Bus).")
    else:
        print("⚠ Linger not enabled — gateway may stop when you close this terminal.")
    if detail:
        print(f"  Auto-enable failed: {detail}")
    print()
    print("  Enable it manually:" if system else "  On headless servers (VPS, cloud instances) run:")
    print(f"    sudo loginctl enable-linger {username}")
    print()
    print("  Then restart the gateway:")
    sudo, _, user_flag = _systemd_cli_bits(system)
    print(f"    {sudo}systemctl {user_flag}restart {get_service_name()}.service")
    print()


def _ensure_linger_enabled(username: str | None = None, *, system: bool = False) -> bool:
    """Enable linger for *username* (default: the current user) when possible.

    A user unit needs linger so the gateway survives logout. A system unit (``system=True``) needs
    it for its ``User=`` so ``user@<uid>.service`` provides the D-Bus that ``systemd-run --user
    --scope`` — every restart-safe cron/Kanban worker — connects to (#104893). Returns True only
    when linger was enabled by this call.
    """
    if is_termux() or not is_linux():
        return False

    if username is None:
        import getpass
        username = getpass.getuser()
    enabled_msg = (
        f"✓ Systemd linger is enabled for {username} (worker D-Bus available)" if system
        else "✓ Systemd linger is enabled (service survives logout)"
    )
    if Path(f"/var/lib/systemd/linger/{username}").exists():
        print(enabled_msg)
        return False

    linger_enabled, linger_detail = get_systemd_linger_status(username)
    if linger_enabled is True:
        print(enabled_msg)
        return False

    if not shutil.which("loginctl"):
        _print_linger_enable_warning(username, linger_detail or "loginctl not found", system=system)
        return False

    if system:
        print(f"Enabling linger for {username} so cron and Kanban workers can reach systemd-run --user...")
    else:
        print("Enabling linger so the gateway survives SSH logout...")
    try:
        result = _loginctl_enable_linger(username)
    except Exception as e:
        _print_linger_enable_warning(username, str(e), system=system)
        return False

    if result.returncode != 0:
        _print_linger_enable_warning(username, _completed_process_detail(result) or linger_detail, system=system)
        return False
    print(f"✓ Enabled linger for {username}" if system else "✓ Linger enabled — gateway will persist after logout")
    return True


def _ensure_system_service_linger(username: str) -> None:
    """Enable linger for the installed unit's ``User=`` (root included: restart-safe workers always cross
    ``systemd-run --user``, so a root gateway needs ``user@0.service`` just the same).

    After a fresh enable, wait for the TARGET user's bus: logind starts ``user@<uid>.service``
    asynchronously and ``--start-now`` boots the gateway immediately. A gateway that was already running
    keeps its bus-less environment and ``systemctl start`` on an active unit is a no-op — say so rather
    than let the repair silently not take."""
    if not _ensure_linger_enabled(username, system=True):
        return
    import pwd
    uid = pwd.getpwnam(username).pw_uid  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
    if _wait_for_target_user_bus(uid):
        print(f"✓ /run/user/{uid}/bus is up — cron and Kanban workers can use systemd-run --user")
    else:
        print(f"⚠ /run/user/{uid}/bus did not appear within 5s.")
        print(f"  Start the user manager: sudo systemctl start user@{uid}.service")
    if _systemd_unit_is_active(system=True):
        print("  The running gateway was started without a user D-Bus; restart it to pick one up:")
        print(f"    sudo systemctl restart {get_service_name()}.service")


def _select_systemd_scope(system: bool = False) -> bool:
    return system or (get_systemd_unit_path(system=True).exists() and not get_systemd_unit_path(system=False).exists())


def _system_scope_wizard_would_need_root(system: bool = False) -> bool:
    """True when the wizard would trigger a system-scope operation as non-root — mirrors
    ``_select_systemd_scope`` so the dead-end is detected BEFORE prompting."""
    if os.geteuid() == 0:  # windows-footgun: ok — systemd scope wizard decision, never invoked on Windows
        return False
    return _select_systemd_scope(system=system)


def _print_system_scope_remediation(action: str) -> None:
    """Print remediation when the wizard skips a system-scope action because the user isn't root."""
    print_warning(f"Gateway is installed as a system-wide service — {action} requires root.")
    print_info("  Options:")
    print_info(f"    1. {action.capitalize()} it this time:")
    print_info(f"         sudo systemctl {action} {get_service_name()}")
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
        raw = str(agent_cfg.get("restart_drain_timeout", DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT))
    return parse_restart_drain_timeout(raw)


def _agent_timeout_setting(env_var: str, key: str, parse) -> float:
    """``parse(env)`` when the env var is non-empty, else ``parse(agent.<key>)`` (None if unset)."""
    env_raw = os.getenv(env_var)
    if env_raw is not None and str(env_raw).strip() != "":
        return parse(env_raw)
    cfg = read_raw_config()
    agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
    if isinstance(agent_cfg, dict) and key in agent_cfg:
        return parse(agent_cfg.get(key))
    return parse(None)


def _get_cron_drain_timeout() -> float:
    """Return the configured cron-only drain floor in seconds.

    See #82161.
    """
    return _agent_timeout_setting("HERMES_CRON_DRAIN_TIMEOUT", "cron_drain_timeout", parse_cron_drain_timeout)


def _get_restart_exit_wait_budget() -> float:
    """CLI wait for gateway exit after SIGUSR1 / self-restart (#77184)."""
    return resolve_restart_exit_wait_budget(
        # TimeoutStopSec must cover the full stop budget, not just restart_drain_timeout. Cron work can
        # legally wait cron_drain_timeout plus cleanup reserve before interrupt/teardown, and systemd
        # SIGKILLs if the unit's deadline is shorter (#94759). 30s of post-drain headroom is preserved on
        # top, with a 60s floor.
        _get_restart_drain_timeout(),
        _agent_timeout_setting(
            "HERMES_RESTART_AFTER_TURN_TIMEOUT", "restart_after_turn_timeout", parse_restart_after_turn_timeout
        ),
    )


def systemd_install(
    force: bool = False,
    system: bool = False,
    run_as_user: str | None = None,
    enable_on_startup: bool = True,
    non_interactive: bool = False,
):
    if system:
        _require_root_for_system_service("install")

    # Offer to remove legacy units first: alongside the new unit they flap-fight for the bot token.
    if has_legacy_hermes_units():
        print()
        print_legacy_unit_warning()
        print()
        if non_interactive or prompt_yes_no("Remove the legacy unit(s) before installing?", True):
            remove_legacy_hermes_units(interactive=False)
            print()

    unit_path = get_systemd_unit_path(system=system)
    scope_label = _service_scope_label(system)
    sudo, scope_flag, user_flag = _systemd_cli_bits(system)

    # Existing system units already pin HERMES_HOME; adopt it before any regenerate.
    if unit_path.exists():
        _sync_hermes_home_from_systemd_unit(system=system)

    if unit_path.exists() and not force:
        if not systemd_unit_is_current(system=system):
            print(f"↻ Repairing outdated {scope_label} systemd service at: {unit_path}")
            refresh_systemd_unit_if_needed(system=system)
            if enable_on_startup:
                _run_systemctl(["enable", get_service_name()], system=system, check=True, timeout=30)
            print(f"✓ {scope_label.capitalize()} service definition updated")
        else:
            print(f"Service already installed at: {unit_path}")
            print("Use --force to reinstall")
        # Same post-install guarantee as a fresh install: a repaired user unit must survive logout too.
        configured_user = _read_systemd_user_from_unit(unit_path) if system else None
        if configured_user:
            _ensure_system_service_linger(configured_user)
        elif not system:
            _ensure_linger_enabled()
        return

    unit_path.parent.mkdir(parents=True, exist_ok=True)
    new_unit = generate_systemd_unit(system=system, run_as_user=run_as_user)
    if _refuse_temp_home_service_write(new_unit, "systemd unit"):
        return
    print(f"Installing {scope_label} systemd service to: {unit_path}")
    unit_path.write_text(new_unit, encoding="utf-8")

    _run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    if enable_on_startup:
        _run_systemctl(["enable", get_service_name()], system=system, check=True, timeout=30)

    print()
    print(f"✓ {scope_label.capitalize()} service {'installed and enabled' if enable_on_startup else 'installed'}!")
    print()
    print("Next steps:")
    print(f"  {sudo}hermes gateway start{scope_flag}              # Start the service")
    print(f"  {sudo}hermes gateway status{scope_flag}             # Check status")
    print(f"  journalctl {user_flag}-u {get_service_name()} -f  # View logs")
    print()

    if system:
        configured_user = _read_systemd_user_from_unit(unit_path)
        if configured_user:
            print(f"Configured to run as: {configured_user}")
            _ensure_system_service_linger(configured_user)
    else:
        _ensure_linger_enabled()

    print_systemd_scope_conflict_warning()
    print_legacy_unit_warning()


def _systemd_scope_preamble(
    action: str, system: bool, *, require_installed: bool = True, preflight_user: bool = False
) -> bool:
    """Resolve the effective scope, then enforce root (system) / user D-Bus reachability (user, when
    ``preflight_user``) and — when ``require_installed`` — that the unit exists. Returns the scope."""
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service(action)
    elif preflight_user:
        # Fail fast with guidance when the user D-Bus session is unreachable (raises UserSystemdUnavailableError).
        _preflight_user_systemd()
    if require_installed:
        _require_service_installed(action, system=system)
    return system


def _systemd_unit_belongs_to_current_home(system: bool = False) -> bool:
    """False (with a warning) when the installed unit pins a HERMES_HOME other than this process's: the
    service name then resolved to ANOTHER install's gateway, and stop/disable/unlink would take it down."""
    _sync_hermes_home_from_systemd_unit(system=system)  # sudo strips HERMES_HOME; adopt the unit's first
    unit_home = _hermes_home_from_systemd_unit_file(system=system)
    if unit_home is None or Path(unit_home).expanduser().resolve() == get_hermes_home().resolve():
        return True
    print_warning(
        f"Refusing to remove {get_systemd_unit_path(system=system)}: it runs HERMES_HOME={unit_home}, "
        f"but this process has HERMES_HOME={get_hermes_home()}"
    )
    return False


def systemd_uninstall(system: bool = False):
    system = _systemd_scope_preamble("uninstall", system, require_installed=False)
    if not _systemd_unit_belongs_to_current_home(system):
        return
    _run_systemctl(["stop", get_service_name()], system=system, check=False, timeout=90)
    _run_systemctl(["disable", get_service_name()], system=system, check=False, timeout=30)

    unit_path = get_systemd_unit_path(system=system)
    if unit_path.exists():
        unit_path.unlink()
        print(f"✓ Removed {unit_path}")

    _run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    print(f"✓ {_service_scope_label(system).capitalize()} service uninstalled")


def _print_service_not_installed(system: bool) -> None:
    sudo, scope_flag, _ = _systemd_cli_bits(system)
    print("✗ Gateway service is not installed")
    print(f"  Run: {sudo}hermes gateway install{scope_flag}")


def _require_service_installed(action: str, system: bool = False) -> None:
    if not get_systemd_unit_path(system=system).exists():
        _print_service_not_installed(system)
        sys.exit(1)


def systemd_start(system: bool = False):
    system = _systemd_scope_preamble("start", system, preflight_user=True)
    # HERMES_HOME sync happens in refresh's systemd_unit_is_current gate; the unit is guaranteed to exist here.
    refresh_systemd_unit_if_needed(system=system)
    _run_systemctl(["start", get_service_name()], system=system, check=True, timeout=30)
    print(f"✓ {_service_scope_label(system).capitalize()} service started")


def systemd_stop(system: bool = False):
    system = _systemd_scope_preamble("stop", system)
    _sync_hermes_home_from_systemd_unit(system=system)
    _mark_planned_stop()
    try:
        _run_systemctl(["stop", get_service_name()], system=system, check=True, timeout=90)
    except subprocess.TimeoutExpired:
        print(
            f"Gateway {_service_scope_label(system)} service is still stopping after 90s; "
            "check `hermes gateway status` or logs for final shutdown state."
        )
        return
    print(f"✓ {_service_scope_label(system).capitalize()} service stopped")


def systemd_restart(system: bool = False):
    system = _systemd_scope_preamble("restart", system, preflight_user=True)
    # HERMES_HOME sync happens in refresh's systemd_unit_is_current gate; its os.environ mutation
    # persists for the get_running_pid / drain-timeout reads below.
    refresh_systemd_unit_if_needed(system=system)
    from gateway.status import get_running_pid
    pid = get_running_pid() or _systemd_main_pid(system=system)
    if pid is not None and probe_gateway_loop_liveness(pid) == GATEWAY_LOOP_WEDGED:
        # Event loop provably dead: SIGUSR1 can't drain it, so bounded SIGTERM → SIGKILL and let systemd relaunch.
        print(
            # Health probe says the event loop is provably dead (#81642): SIGUSR1 can never drain it, so the
            # graceful wait below would burn the full budget. A busy-but-alive gateway (fresh heartbeat)
            # never takes this path — its in-flight work, including the #86684 cron drain floor, keeps the
            # full graceful budget.
            # Health probe says the event loop is provably dead (#81642): the gateway cannot process a
            # graceful shutdown, so waiting the full drain budget only stalls the restart (and `hermes
            # update` behind it) for 180s. Bounded escalation instead: SIGTERM grace → SIGKILL → proceed,
            # ~10s worst case. Never taken for a busy-but-alive gateway — a fresh heartbeat keeps the drain
            # path (and the #86684 cron drain floor) fully intact.
            f"⚠ Gateway PID {pid} event loop is unresponsive — "
            "skipping graceful drain and forcing a bounded stop..."
        )
        _escalate_wedged_gateway(pid)
        svc = get_service_name()
        _run_systemctl(["reset-failed", svc], system=system, check=False, timeout=30)
        _run_systemctl(["restart", svc], system=system, check=False, timeout=90)
        _wait_for_systemd_service_restart(system=system, previous_pid=pid)
        return
    if pid is not None:
        service_action = _systemd_graceful_restart_action(system, pid)
        if service_action:
            _systemd_reset_and_run(service_action, system=system, previous_pid=pid)
        return

    if _recover_pending_systemd_restart(system=system, previous_pid=pid):
        return
    _systemd_reset_and_run("restart", system=system, previous_pid=pid)


def _systemd_graceful_restart_action(system: bool, pid: int) -> str | None:
    """SIGUSR1-drain the live gateway ``pid``; return the follow-up ``systemctl`` verb (``"start"`` /
    ``"restart"``) the caller must still issue, or None when systemd already owns the relaunch."""
    scope_label = _service_scope_label(system).capitalize()
    # Graceful in-band restart, mirroring the systemd branch. Previously this sent a bare SIGTERM and waited
    # ``_get_restart_drain_timeout()`` — which defaults to 0, so the wait could never succeed and every
    # restart fell through to ``kickstart -k``. A bare SIGTERM also leaves ``restart_requested`` False, so
    # the gateway exits 1 instead of 75 and reports itself to chat as "shutting down" rather than
    # "restarting", losing the resume_pending handoff. SIGUSR1 is the drain-aware path: refuse new turns,
    # wait for in-flight work (``agent.restart_after_turn_timeout``), then stop() within
    # ``agent.restart_drain_timeout``. The wait budget must cover BOTH phases plus headroom (#77184) — the
    # raw drain timeout covers only the second. Announce the wait BEFORE it runs: it can last the full
    # budget while the old gateway finishes in-flight agent runs, and it streams into surfaces with no other
    # feedback — the desktop updater's live output most of all, where a silent stop here reads as "update
    # stuck" (#44515).
    wait_budget = _get_restart_exit_wait_budget()
    print(
        f"⏳ {scope_label} service restarting gracefully (PID {pid}) — "
        f"waiting up to {wait_budget:.0f}s for in-flight turns + drain..."
    )
    from hermes_cli.update_cmd_drain_report import drain_progress_reporter
    if not _graceful_restart_via_sigusr1(pid, wait_budget, on_progress=drain_progress_reporter(budget_s=wait_budget)):
        print(f"⚠ Graceful restart did not complete within {int(wait_budget)}s; forcing a service restart...")
        return "restart"

    # Exit 75 hands restart ownership to systemd; observe that replacement rather than restarting again.
    replacement_observed: list[bool] = []
    if _wait_for_systemd_service_restart(system=system, previous_pid=pid, replacement_observed=replacement_observed):
        return None
    if replacement_observed or _systemd_service_is_start_limited(system=system):
        return None

    # A replacement may have started but not reached runtime readiness in time; never stop that generation.
    props = _read_systemd_unit_properties(system=system)
    if not props:
        return None
    replacement_pid = _systemd_main_pid_from_props(props)
    if (
        props.get("ActiveState") in {"active", "activating", "reloading"}
        or props.get("SubState") == "auto-restart"
        or (replacement_pid is not None and replacement_pid != pid)
    ):
        return None

    print("⚠ Systemd did not relaunch the gateway after its graceful exit; starting the inactive service...")
    # ``start`` is intentionally idempotent: a replacement appearing after the snapshot must not be stopped.
    return "start"


def _systemd_reset_and_run(action: str, *, system: bool, previous_pid) -> None:
    """``reset-failed`` then ``systemctl <action>``, then wait for the relaunch. Start-limit
    rejection prints the wait hint instead of raising; a 90s timeout prints where to look."""
    svc = get_service_name()
    _run_systemctl(["reset-failed", svc], system=system, check=False, timeout=30)
    try:
        _run_systemctl([action, svc], system=system, check=True, timeout=90)
    except subprocess.CalledProcessError as exc:
        if _systemd_error_indicates_start_limit(exc) or _systemd_service_is_start_limited(system=system):
            _print_systemd_start_limit_wait(system=system)
            return
        raise
    except subprocess.TimeoutExpired:
        print(
            f"Gateway {_service_scope_label(system)} service is still restarting after 90s; "
            "check `hermes gateway status` or logs for final state."
        )
        return
    _wait_for_systemd_service_restart(system=system, previous_pid=previous_pid)


def systemd_status(deep: bool = False, system: bool = False, full: bool = False):
    system = _select_systemd_scope(system)
    unit_path = get_systemd_unit_path(system=system)
    svc = get_service_name()
    scope_label = _service_scope_label(system).capitalize()
    sudo, scope_flag, user_flag = _systemd_cli_bits(system)

    if not unit_path.exists():
        _print_service_not_installed(system)
        return

    if has_conflicting_systemd_units():
        print_systemd_scope_conflict_warning()
        print()

    if has_legacy_hermes_units():
        print_legacy_unit_warning()
        print()

    if not systemd_unit_is_current(system=system):
        print("⚠ Installed gateway service definition is outdated")
        print(f"  Run: {sudo}hermes gateway restart{scope_flag}  # auto-refreshes the unit")
        print()

    status_cmd = ["status", svc, "--no-pager"] + (["-l"] if full else [])
    _run_systemctl(status_cmd, system=system, capture_output=False, timeout=10)
    result = _run_systemctl(["is-active", svc], system=system, timeout=10, **_CAPTURE_TEXT)
    if result.stdout.strip() == "active":
        print(f"✓ {scope_label} gateway service is running")
    else:
        print(f"✗ {scope_label} gateway service is stopped")
        print(f"  Run: {sudo}hermes gateway start{scope_flag}")

    configured_user = _read_systemd_user_from_unit(unit_path) if system else None
    if configured_user:
        print(f"Configured to run as: {configured_user}")

    _print_runtime_health()

    unit_props = _read_systemd_unit_properties(system=system)
    active_state = unit_props.get("ActiveState", "")
    result_code = unit_props.get("Result", "")
    if active_state == "activating" and unit_props.get("SubState", "") == "auto-restart":
        print("  ⏳ Restart pending: systemd is waiting to relaunch the gateway")
    elif _systemd_unit_is_start_limited(unit_props):
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
        linger_enabled, linger_detail = get_systemd_linger_status()
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



def _wait_for_gateway_exit(timeout: float = 10.0, force_after: float | None = 5.0) -> bool:
    """Wait up to ``timeout`` s for the gateway (by gateway.pid, not launchd labels, so multiple
    HERMES_HOMEs work) to exit; SIGKILL it after ``force_after`` s of graceful waiting."""
    from gateway.status import get_process_start_time, get_running_pid
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
                terminate_pid(pid, force=True, expected_start_time=get_process_start_time(pid))
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


def _wait_for_tcp_port_free(host: str, port: int, *, timeout: float = 10.0) -> bool:
    """Wait until nothing accepts TCP connections on host:port.

    PID exit is not enough on macOS: api_server disables SO_REUSEADDR, so a restart that wins
    the race logs EADDRINUSE and keeps running with no API. Connection-refused means the
    listener is gone; a timed-out connect is a live listener with a slow accept queue.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                pass
        except ConnectionRefusedError:
            return True
        except TimeoutError:
            pass  # a slow accept queue is still a live listener
        except OSError:
            return True  # unresolvable/unreachable address: nothing to wait for; the bind retry covers it
        time.sleep(0.1)
    return False


def _wait_for_api_server_port_free(*, timeout: float = 10.0) -> bool:
    """Wait for the configured api_server listen address to stop accepting.

    Only when api_server is enabled: with the platform off, a foreign listener on the default
    port is nobody's race and must not delay the restart."""
    from gateway.config import Platform
    from gateway.platforms.api_server import listen_address
    pconfig = load_gateway_config().platforms.get(Platform.API_SERVER)
    if pconfig is None or not pconfig.enabled:
        return True
    host, port = listen_address(pconfig.extra or {})
    freed = _wait_for_tcp_port_free(host, port, timeout=timeout)
    if not freed:
        print(
            f"⚠ {host}:{port} still accepting connections — "
            "new api_server may fail to bind"
        )
    return freed



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


def host_multiplexer_serving(profile_name: str | None = None):
    """The ONE live host gateway when it serves ``profile_name`` (default: the current profile).

    Companion to :func:`named_profile_served_by_running_multiplexer`, which answers the narrower
    "is this a SATELLITE of the default's multiplexer" and is hard-False for ``default`` — the
    profile that usually owns the shared process. Every lifecycle guard built on it was therefore
    blind to the host process itself. This one is true for ``default`` too, and it reports WHICH
    profiles the host process serves. Returns a ``gateway.host_attach.HostGateway`` or None.
    """
    try:
        from gateway.host_attach import host_gateway_serving
        name = profile_name if profile_name is not None else _current_profile_name()
        return host_gateway_serving(name or "default")
    except Exception:
        logger.debug("Host multiplexer probe failed", exc_info=True)
        return None


def _served_by_another_host_gateway(profile_name: str | None = None):
    """The host gateway serving ``profile_name`` when it is NOT this home's own process.

    The owner must never be guarded out of restarting itself: a refusal keyed on "something serves
    you" would make `hermes gateway restart` impossible for the profile that launched the host
    process. Guards want "ANOTHER process already serves you", which is this.
    """
    gateway = host_multiplexer_serving(profile_name)
    if gateway is None:
        return None
    try:
        from gateway.status import _get_process_hermes_home, _same_hermes_home
        if _same_hermes_home(gateway.home, _get_process_hermes_home()):
            return None
    except Exception:
        logger.debug("Host multiplexer home comparison failed", exc_info=True)
    return gateway


def named_profile_served_by_running_multiplexer(profile_name: str | None = None) -> bool:
    """True when a live default multiplexer already ticks this named profile (a satellite profile has no
    gateway.pid; the multiplexer fires its jobs and serves its platforms). Defaults to the current profile.

    See #97120.
    """
    try:
        suffix = profile_name if profile_name is not None else _current_profile_name()
    except Exception:
        return False
    if not suffix or suffix == "default":
        return False

    # The host record answers first: it names the live host process whatever home launched it, so a
    # multiplexer started by a named profile is visible here too.
    if host_multiplexer_serving(suffix) is not None:
        return True

    try:
        from hermes_constants import get_default_hermes_root
        default_root = get_default_hermes_root()
    except Exception:
        return False

    try:
        from hermes_cli.gateway_multiplex_served import live_default_gateway_pid, recorded_served_profiles
        if live_default_gateway_pid() is None:
            return False
        from hermes_cli.profiles import normalize_profile_name
        # The live gateway's own record wins: the CLI process cannot see an env-only opt-in on the
        # default profile (`hermes -p X` loads X's .env) and a config edit after start is not live yet.
        # Only a record without the key (pre-multiplex writer) falls through to config derivation.
        recorded = recorded_served_profiles(default_root)
        if recorded is not None:
            return normalize_profile_name(suffix) in {normalize_profile_name(p) for p in recorded}

        # No record (older gateway): only an EXPLICIT opt-in counts. The unset default is settled by
        # the gateway at boot (it may have stayed standalone); a CLI process must not guess it on.
        from hermes_cli.gateway_multiplex_mode import explicit_multiplex_flag
        return explicit_multiplex_flag(default_root) is True  # a multiplexer serves every named profile
    except Exception:
        logger.debug("Multiplexer-serving probe failed", exc_info=True)
        return False


def _served_profile_needs_no_service() -> bool:
    """Print the "already served" note and return True when a setup flow must not install a standalone
    service: a live multiplexing default gateway already serves this named profile, so the unit/plist it
    would register can only sit dead (the start guard refuses it) or double-bind its platforms.
    Shared by ``hermes setup gateway`` / ``hermes setup`` / ``hermes import`` (``ensure_gateway_service``)
    and the ``hermes gateway setup`` wizard. See #111958."""
    if not named_profile_served_by_running_multiplexer():
        # Not served (yet): a named profile still gets no service of its own — same rule and text
        # as `gateway install`, so `hermes -p X setup` cannot grow a fleet member the verb refuses.
        return _named_profile_refused_under_multiplexer()
    from hermes_cli.profiles import profile_is_standalone
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
        from hermes_cli.profiles import profile_is_standalone
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
    owner = _served_by_another_host_gateway()
    served = owner is not None or named_profile_served_by_running_multiplexer()
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
    from hermes_cli.gateway_multiplex_mode import STANDALONE_DEPRECATION_NOTICE
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
        _ensure_user_systemd_env()

    # Refresh the systemd unit on every boot so restart settings stay current even after an
    # exit-code-75 respawn (stale-code or /restart), which bypasses `hermes gateway restart`.
    if supports_systemd_services():
        try:
            refresh_systemd_unit_if_needed(system=False)
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
    # LOCAL PATCH (ai-hub, 2026-09-09). A unit FILE named hermes-gateway*.service is
    # not the only supervisor. This box runs the gateway under a hand-written
    # hermes-primary.service, so both path checks are False and every caller
    # concluded "no service installed": _cmd_restart then skipped the systemd verb
    # and took the manual branch (signal the live gateway, start an unsupervised
    # replacement in the caller's foreground) while Restart=always respawned it.
    # Patch 232 taught the orphan reaper the same lesson via ExecStart discovery;
    # this is the restart-arm half. Identify by ExecStart, never by name glob.
    return supports_systemd_services() and (
        get_systemd_unit_path(system=False).exists()
        or get_systemd_unit_path(system=True).exists()
        or _adopted_systemd_unit_name() is not None
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
        launchd_install(force)
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
    stop_all = getattr(args, "all", False)
    system = getattr(args, "system", False)
    if not stop_all and not find_gateway_pids() and (
            _served_by_another_host_gateway() or named_profile_served_by_running_multiplexer()):
        # A served profile owns no gateway to stop; "No gateway running for this profile" (exit 0) would
        # contradict `gateway status` ("running via the host multiplexer") on the same profile.
        # A `--force`-started separate gateway HAS a pid of its own and is stopped normally.
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

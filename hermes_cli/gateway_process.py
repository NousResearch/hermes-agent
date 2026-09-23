"""Gateway process discovery, loop liveness, and detached restart helpers.

Extracted from hermes_cli.gateway; each moved body reads patchable facade names at call time.
"""

from __future__ import annotations

from pathlib import Path


def _get_service_pids(all_profiles: bool = False, *, require_complete: bool = False) -> set:
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
    from hermes_cli.gateway import (
        _CAPTURE_TEXT,
        _locate_launchd_gateway_service,
        get_launchd_label,
        get_service_name,
        is_macos,
        launchd_gateway_labels_for_install,
        subprocess,
        supports_systemd_services,
    )
    pids: set = set()

    # --- systemd (Linux): user and system scopes ---
    if supports_systemd_services():
        pattern = "hermes-gateway*" if all_profiles else get_service_name()
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
                if require_complete and result.returncode != 0:
                    raise RuntimeError("systemd gateway inventory failed")
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
                        if require_complete and show.returncode != 0:
                            raise RuntimeError("systemd gateway PID inspection failed")
                        pid = int(show.stdout.strip())
                        if pid > 0:
                            pids.add(pid)
                    except (ValueError, subprocess.TimeoutExpired) as exc:
                        if require_complete:
                            raise RuntimeError("systemd gateway PID inspection failed") from exc
                        pass
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                if require_complete:
                    raise RuntimeError("systemd gateway inventory failed") from exc
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
                _domain, pid = (_locate_launchd_gateway_service(label, require_complete=True)
                                if require_complete else _locate_launchd_gateway_service(label))
            except subprocess.TimeoutExpired:
                if require_complete:
                    raise RuntimeError("launchd gateway inventory timed out")
                continue
            if pid is not None and pid > 0:
                pids.add(pid)
        if all_profiles:
            # Prefix scan also catches ai.hermes.gateway* agents the label derivation can't map
            # (renamed profiles, other installs). Over-inclusion is safe: PIDs are only protected.
            try:
                result = subprocess.run(["launchctl", "list"], timeout=5, **_CAPTURE_TEXT)
                if require_complete and result.returncode != 0:
                    raise RuntimeError("launchd gateway inventory failed")
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
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                if require_complete:
                    raise RuntimeError("launchd gateway inventory failed") from exc
                pass

    return pids


def _get_parent_pid(pid: int) -> int | None:
    """Parent PID for ``pid``, or None. psutil first (works on Windows, where ``ps`` doesn't)."""
    from hermes_cli.gateway import (
        _CAPTURE_TEXT,
        is_windows,
        shutil,
        subprocess,
    )
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
    from hermes_cli.gateway import (
        _get_parent_pid,
        os,
    )
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
    from hermes_cli.gateway import (
        _is_pid_ancestor_of_current_process,
        os,
        signal,
    )
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
    from hermes_cli.gateway import (
        _wait_for_pid_exit,
        os,
        signal,
    )
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
    from hermes_cli.gateway import (
        time,
    )
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
    from hermes_cli.gateway import (
        _ping_loop_tick_witness,
        socket,
    )
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
    from hermes_cli.gateway import (
        contextlib,
        socket,
    )
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
    from hermes_cli.gateway import (
        _ping_loop_tick_witness,
        socket,
    )
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
    from hermes_cli.gateway import (
        _probe_loop_tick_socket,
        _probe_loop_tick_tcp,
        time,
    )
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
    from hermes_cli.gateway import (
        DEFAULT_LOOP_LIVENESS_STALE_AFTER_S,
        GATEWAY_LOOP_ALIVE,
        GATEWAY_LOOP_UNKNOWN,
        GATEWAY_LOOP_WEDGED,
        _LOOP_TICK_ABSENT,
        _probe_loop_tick_socket,
        _probe_loop_tick_socket_sustained,
        _probe_loop_tick_tcp,
        json,
        time,
    )
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
    from hermes_cli.gateway import (
        _wait_for_pid_exit,
        terminate_pid,
    )
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
    from hermes_cli.gateway import (
        _get_parent_pid,
        os,
    )
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
    from hermes_cli.gateway import (
        os,
    )
    if pid and pid > 0 and pid != os.getpid() and pid not in exclude_pids and pid not in pids:
        pids.append(pid)


def _iter_proc_cmdlines(exclude_pids: set[int]):
    """Yield ``(pid, cmdline)`` from ``/proc`` (Docker without procps); raises if /proc is unusable."""
    from hermes_cli.gateway import (
        os,
    )
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
    from hermes_cli.gateway import (
        _CAPTURE_TEXT,
        _filter_venv_launcher_stubs,
        _get_ancestor_pids,
        _iter_proc_cmdlines,
        _iter_windows_list_processes,
        _parse_ps_line,
        _profile_arg,
        _windows_process_listing,
        get_hermes_home,
        is_windows,
        os,
        subprocess,
    )
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
    from hermes_cli.gateway import (
        contextlib,
    )
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
    from hermes_cli.gateway import (
        contextlib,
    )
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
    from hermes_cli.gateway import (
        shutil,
    )
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
    from hermes_cli.gateway import (
        _append_unique_pid,
        _get_service_pids,
        _scan_gateway_pids,
        supports_systemd_services,
    )
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
    from hermes_cli.gateway import (
        ProfileGatewayProcess,
    )
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
    from hermes_cli.gateway import (
        WindowsGatewayService,
        _scm_service_field,
        find_profile_gateway_processes,
        sys,
    )
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
    from hermes_cli.gateway import (
        get_python_path,
    )
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
    from hermes_cli.gateway import (
        _capture_gateway_argv,
        launch_detached_gateway_restart_by_cmdline,
        launch_detached_profile_gateway_restart,
    )
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
    from hermes_cli.gateway import (
        _spawn_gateway_restart_watcher,
    )
    return old_pid > 0 and bool(run_argv) and _spawn_gateway_restart_watcher(old_pid, list(run_argv))


def launch_detached_profile_gateway_restart(profile: str, old_pid: int) -> bool:
    """Relaunch a manually-run profile gateway after its current PID exits."""
    from hermes_cli.gateway import (
        _gateway_run_args_for_profile,
        _spawn_gateway_restart_watcher,
    )
    return old_pid > 0 and _spawn_gateway_restart_watcher(old_pid, _gateway_run_args_for_profile(profile))


def _spawn_gateway_restart_watcher(old_pid: int, run_argv: list[str]) -> bool:
    """Spawn the detached watcher that respawns ``run_argv`` once ``old_pid`` exits. Watcher and respawn
    both need platform-appropriate detach: POSIX setsid; on Windows ``start_new_session`` does NOT detach
    (the watcher would die with the CLI console), so ``windows_detach_popen_kwargs()`` supplies flags."""
    from hermes_cli.gateway import (
        json,
        subprocess,
        sys,
        textwrap,
    )
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

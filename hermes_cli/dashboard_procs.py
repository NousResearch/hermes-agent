"""Dashboard process-hygiene helpers — extracted from ``hermes_cli/main.py``.

Mechanical move (main.py decomposition): the three leaf process-hygiene
helpers (``_scan_dashboard_processes``, ``_kill_stale_dashboard_processes``,
``_detect_concurrent_hermes_instances``) are lifted verbatim. References to
helpers that STAY in ``hermes_cli.main`` (``_find_stale_dashboard_pids``,
``_respawn_dashboard_processes``, ``_is_windows``, ...) are routed through a
lazy ``_m()`` main reference so existing test monkeypatches on
``hermes_cli.main.<name>`` keep reaching this code path, and imports stay
one-way at import time (main.py imports this module, never the reverse).
``main.py`` re-exports all three names (``# noqa: F401``) so callers and test
patches on ``hermes_cli.main`` resolve unchanged.
"""

import os
import subprocess
import sys
from pathlib import Path


def _m():
    """Lazy ``hermes_cli.main`` reference (call-time; keeps patches working)."""
    from hermes_cli import main

    return main


def _scan_dashboard_processes(
    *,
    exclude_pids: set[int] | None = None,
) -> list[tuple[int, str]]:
    """Return matching ``dashboard``/``serve`` processes with their cmdlines.

    ``hermes dashboard`` is a long-lived server process commonly started and
    forgotten.  When ``hermes update`` replaces files on disk, the running
    process keeps the old Python backend in memory while the JS bundle on
    disk is updated, causing a silent frontend/backend mismatch (e.g. new
    auth headers the old backend doesn't recognise → every API call 401s).

    The dashboard may be manually started or managed by the optional
    ``hermes-dashboard.service`` systemd unit.  Managed units are restarted
    through their owning systemd scope; only manually-started processes use
    the kill path because we can't know their original launch args.

    *exclude_pids* is an optional set of PIDs that must never be returned.
    This is used by the Hermes Desktop Electron app to protect its own
    backend child process: when the desktop spawns ``hermes serve`` as
    a backend and triggers an auto-update, the update must not kill the
    backend that the desktop itself manages.  The desktop sets the
    environment variable ``HERMES_DESKTOP_CHILD_PID`` on the spawned
    backend process; ``_kill_stale_dashboard_processes`` reads it and
    passes it here.  (#37532)

    Returns an empty list on any scan error (missing ps/wmic, timeout, etc.).
    """
    patterns = [
        "hermes dashboard",
        "hermes_cli.main dashboard",
        "hermes_cli/main.py dashboard",
        # The headless backend (`hermes serve`) is the same long-lived server
        # under a different command name — the desktop app spawns it. Reap it
        # on update for the same frontend/backend-mismatch reason.
        "hermes serve",
        "hermes_cli.main serve",
        "hermes_cli/main.py serve",
    ]
    self_pid = os.getpid()
    dashboard_processes: list[tuple[int, str]] = []

    try:
        if sys.platform == "win32":
            # wmic may emit text in the system code page (for example cp936
            # on zh-CN systems), not UTF-8. In text mode, subprocess output
            # decoding depends on Python's configuration (locale-dependent
            # by default, or UTF-8 in UTF-8 mode). The important protection
            # here is errors="ignore": it prevents a reader-thread
            # UnicodeDecodeError from leaving result.stdout=None and turning
            # the later .split() into an AttributeError (#17049).
            # CREATE_NO_WINDOW hides the conhost flash: this scan can run from
            # the windowless pythonw.exe desktop/gateway backend during an
            # update, where a bare wmic spawn would pop a console window.
            from hermes_cli._subprocess_compat import windows_hide_flags

            result = subprocess.run(
                ["wmic", "process", "get", "ProcessId,CommandLine", "/FORMAT:LIST"],
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
                errors="ignore",
                creationflags=windows_hide_flags(),
            )
            if result.returncode != 0 or result.stdout is None:
                return []
            current_cmd = ""
            for line in result.stdout.split("\n"):
                line = line.strip()
                if line.startswith("CommandLine="):
                    current_cmd = line[len("CommandLine=") :]
                elif line.startswith("ProcessId="):
                    pid_str = line[len("ProcessId=") :]
                    if (
                        any(p in current_cmd for p in patterns)
                        and int(pid_str) != self_pid
                    ):
                        try:
                            dashboard_processes.append((int(pid_str), current_cmd))
                        except ValueError:
                            pass
        else:
            # Linux / macOS: scan the process table via ps and match against
            # the same explicit patterns list used on Windows.  Using ps
            # (rather than `pgrep -f "hermes.*dashboard"`) keeps us consistent
            # with `hermes_cli.gateway._scan_gateway_pids` and avoids the
            # greedy regex matching unrelated cmdlines that merely contain
            # both words (e.g. a chat session discussing "dashboard").
            result = subprocess.run(
                ["ps", "-A", "-o", "pid=,command="],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=10,
            )
            if result.returncode == 0:
                for line in getattr(result, "stdout", "").split("\n"):
                    stripped = line.strip()
                    if not stripped or "grep" in stripped:
                        continue
                    parts = stripped.split(None, 1)
                    if len(parts) != 2:
                        continue
                    try:
                        pid = int(parts[0])
                    except ValueError:
                        continue
                    command = parts[1]
                    if any(p in command for p in patterns) and pid != self_pid:
                        dashboard_processes.append((pid, command))
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []

    if exclude_pids:
        dashboard_processes = [
            proc for proc in dashboard_processes if proc[0] not in exclude_pids
        ]
    return dashboard_processes

def _kill_stale_dashboard_processes(
    reason: str = "the running backend no longer matches the updated frontend",
    *,
    restart_managed: bool = False,
) -> dict[str, list]:
    """Kill running ``hermes dashboard`` / ``hermes serve`` processes.

    Called at the end of ``hermes update`` (default ``reason``) and also
    from ``hermes dashboard --stop`` (which overrides ``reason``).  The
    dashboard has no service manager, so after a code update the running
    process is guaranteed to be serving stale Python against a
    freshly-updated JS bundle.  Leaving it alive produces silent
    frontend/backend mismatches (new auth headers the old backend doesn't
    recognise → every API call 401s).

    POSIX: SIGTERM, wait up to ~3s for graceful exit, SIGKILL any survivors.
    Windows: ``taskkill /PID <pid> /F`` since there's no clean SIGTERM
    equivalent for background console apps.

    Manually-started dashboards are not auto-restarted because we don't know
    the original launch args (--host, --port, --insecure, --tui, --no-open).
    When ``restart_managed`` is true (the ``hermes update`` path), a detected
    ``hermes-dashboard.service`` is restarted through systemd; any OTHER
    killed PID that was supervised by a systemd unit (custom unit names —
    e.g. a remote backend's ``hermes-serve.service``) has its owning unit
    restarted after the kill, because systemd treats our SIGTERM as a clean
    stop and ``Restart=on-failure`` would never fire (#68934).
    """
    if restart_managed and _m()._restart_managed_dashboard_service(reason):
        return {"matched": [], "killed": [], "failed": []}

    # When the Hermes Desktop Electron app spawns this dashboard as a
    # backend child, it sets HERMES_DESKTOP_CHILD_PID so that the update
    # path can skip killing the desktop-managed process.  (#37532)
    exclude: set[int] | None = None
    raw_pid = os.environ.get("HERMES_DESKTOP_CHILD_PID")
    if raw_pid:
        # The desktop may manage several backends (one per active profile) and
        # passes them comma-separated; a lone int still parses for back-compat.
        parsed: set[int] = set()
        for part in raw_pid.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                parsed.add(int(part))
            except (ValueError, TypeError):
                pass
        if parsed:
            exclude = parsed

    pids = _m()._find_stale_dashboard_pids(exclude_pids=exclude)
    if not pids:
        return {"matched": [], "killed": [], "failed": []}

    print()
    print(f"⟲ Stopping {len(pids)} dashboard process(es) ({reason})")

    # Before killing, snapshot systemd cgroup info for each PID so we can
    # restart supervised services after the kill (the cgroup disappears
    # along with the process).  Only meaningful on Linux, and only when the
    # caller asked for restarts (the `hermes update` path) — `--stop` must
    # stay a stop, not a restart.
    pid_cgroup: dict[int, str | None] = {}
    pid_service: dict[int, str | None] = {}
    pid_cmdline: dict[int, list[str]] = {}
    if restart_managed and sys.platform != "win32":
        for pid in pids:
            cg_path = _m()._get_pid_cgroup_path(pid)
            pid_cgroup[pid] = cg_path
            pid_service[pid] = _m()._get_systemd_service_for_pid(pid)
            if not pid_service[pid]:
                # Manually-started process: preserve its exact argv so we
                # can respawn it after the update (#40449, #68934).
                cmdline = _m()._dashboard_cmdline_for_pid(pid)
                if cmdline:
                    pid_cmdline[pid] = cmdline

    killed: list[int] = []
    failed: list[tuple[int, str]] = []

    if sys.platform == "win32":
        for pid in pids:
            try:
                result = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                    timeout=10,
                )
                if result.returncode == 0:
                    killed.append(pid)
                else:
                    failed.append((pid, (result.stderr or result.stdout or "").strip()))
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
                failed.append((pid, str(e)))
    else:
        import signal as _signal
        import time as _time

        # SIGTERM first — give each process a chance to shut down cleanly
        # (uvicorn closes its socket, flushes logs, etc.).
        for pid in pids:
            try:
                os.kill(pid, _signal.SIGTERM)
            except ProcessLookupError:
                # Already gone — count as killed.
                killed.append(pid)
            except (PermissionError, OSError) as e:
                failed.append((pid, str(e)))

        # Poll for exit up to ~3s total.
        deadline = _time.monotonic() + 3.0
        pending = [
            p for p in pids if p not in killed and p not in {f[0] for f in failed}
        ]
        while pending and _time.monotonic() < deadline:
            _time.sleep(0.1)
            still_pending = []
            # On Windows, os.kill(pid, 0) is NOT a no-op. Route through
            # the cross-platform existence check.
            from gateway.status import _pid_exists
            for pid in pending:
                if _pid_exists(pid):
                    still_pending.append(pid)
                else:
                    killed.append(pid)
            pending = still_pending

        # SIGKILL any survivors.
        for pid in pending:
            try:
                os.kill(pid, _signal.SIGKILL)
                killed.append(pid)
            except ProcessLookupError:
                killed.append(pid)
            except (PermissionError, OSError) as e:
                failed.append((pid, str(e)))

    for pid in killed:
        print(f"    ✓ stopped PID {pid}")
    for pid, err_msg in failed:
        print(f"    ✗ failed to stop PID {pid}: {err_msg}")

    # Restart what we just killed (update path only).  Two categories:
    #  - systemd-supervised PIDs: restart the owning unit.  Without this, a
    #    remote backend (hermes serve) under Restart=on-failure never comes
    #    back after our clean SIGTERM, and the Desktop can't reconnect (#68934).
    #  - manually-started PIDs: respawn the argv captured before the kill
    #    (#40449) — detached, headless, logged to logs/dashboard-restart.log.
    restarted_services: list[str] = []
    unrecovered: list[int] = []
    if killed and restart_managed:
        failed_restarts: list[tuple[str, str]] = []
        seen_services: set[str] = set()
        respawn_cmds: list[list[str]] = []
        for pid in killed:
            svc_name = pid_service.get(pid)
            if svc_name:
                if svc_name in seen_services:
                    continue
                seen_services.add(svc_name)
                if _m()._try_restart_systemd_service(svc_name, pid_cgroup.get(pid)):
                    restarted_services.append(svc_name)
                else:
                    failed_restarts.append((svc_name, "systemctl restart returned non-zero"))
                    unrecovered.append(pid)
            elif pid in pid_cmdline:
                respawn_cmds.append(pid_cmdline[pid])
            else:
                unrecovered.append(pid)

        for svc in restarted_services:
            print(f"    ✓ restarted systemd service {svc}")
        for svc, err in failed_restarts:
            print(f"    ⚠ {svc}: {err}")

        if respawn_cmds:
            failed_cmds = _m()._respawn_dashboard_processes(respawn_cmds)
            if failed_cmds:
                unrecovered.extend(p for p in killed if pid_cmdline.get(p) in failed_cmds)

        if failed_restarts or unrecovered:
            print("  Restart anything not auto-restarted when you're ready:")
            print("    hermes dashboard --port <port>")
    elif killed:
        unrecovered = list(killed)
        print("  Restart the dashboard when you're ready:")
        print("    hermes dashboard --port <port>")

    return {
        "matched": list(pids),
        "killed": list(killed),
        "failed": list(failed),
        "unrecovered": list(unrecovered),
    }

def _detect_concurrent_hermes_instances(
    scripts_dir: Path, *, exclude_pid: int | None = None
) -> list[tuple[int, str]]:
    """Find other live processes whose .exe is one of our entry-point shims.

    Windows blocks DELETE/REPLACE on a running .exe — and even RENAME on the
    same .exe when another process opened it without ``FILE_SHARE_DELETE``.
    The Hermes Desktop Electron app spawns ``hermes.EXE`` as a backend child,
    so during ``hermes update`` the user-invoked process and the desktop's
    child both hold the same file. The quarantine rename then fails with
    ``[WinError 32]`` and uv inherits the lock.

    This helper enumerates processes whose ``exe`` matches one of the venv's
    shims (``hermes.exe`` / ``hermes-gateway.exe``) and returns ``(pid,
    process_name)`` pairs. The caller's own PID and its entire ancestor
    chain are excluded so the running ``hermes update`` invocation never
    reports itself — this matters on Windows where the setuptools .exe
    launcher (``hermes.exe``) is a separate process from the Python
    interpreter it loads (``python.exe``).

    Returns an empty list off-Windows, on missing psutil, or when no other
    instances exist. Never raises — process enumeration is best-effort.
    """
    if not _m()._is_windows():
        return []

    try:
        import psutil
    except Exception:
        return []

    # Resolve every shim path to its canonical form once for cheap comparison.
    shim_paths: set[str] = set()
    for shim in _m()._hermes_exe_shims(scripts_dir):
        try:
            shim_paths.add(str(shim.resolve()).lower())
        except OSError:
            shim_paths.add(str(shim).lower())
    if not shim_paths:
        return []

    # Build a set of PIDs to exclude: the Python process itself plus every
    # ancestor whose executable is one of our shims. On Windows the
    # setuptools-generated hermes.exe launcher is a separate native process
    # that spawns python.exe (the interpreter that runs our code).
    # os.getpid() returns the Python PID, but the launcher (which holds the
    # file lock) is the parent. Without excluding it, every ``hermes update``
    # reports its own launcher as a concurrent instance — a false positive
    # (issues #29341, #34795).
    #
    # Two robustness points learned from the field:
    #   1. Use ``proc.parents()`` — it returns the WHOLE ancestor list in one
    #      call. The earlier per-hop ``current.parent()`` loop bailed on the
    #      first psutil error (AccessDenied/NoSuchProcess is common on Windows
    #      across session/elevation boundaries), leaving the launcher shim in
    #      the candidate set and re-triggering the false positive.
    #   2. Only exclude ancestors whose exe is itself a shim. A genuine second
    #      hermes.exe sitting *under* a non-Hermes parent (e.g. a Hermes
    #      Desktop backend child) must still be flagged, so we don't blanket-
    #      exclude unrelated ancestors like the shell or terminal.
    # Broad ``except Exception`` guards against partially-stubbed psutil in
    # unit tests; this helper is documented as "never raises".
    if exclude_pid is not None:
        exclude_pids: set[int] = {int(exclude_pid)}
    else:
        exclude_pids = {os.getpid()}
    try:
        seed = next(iter(exclude_pids))
        try:
            ancestors = psutil.Process(seed).parents()
        except Exception:
            ancestors = []
        for ancestor in ancestors:
            try:
                anc_exe = ancestor.exe()
            except Exception:
                continue
            if not anc_exe:
                continue
            try:
                anc_norm = str(Path(anc_exe).resolve()).lower()
            except (OSError, ValueError):
                anc_norm = str(anc_exe).lower()
            if anc_norm in shim_paths:
                try:
                    exclude_pids.add(int(ancestor.pid))
                except Exception:
                    continue
    except Exception:
        pass

    matches: list[tuple[int, str]] = []
    try:
        proc_iter = psutil.process_iter(["pid", "exe", "name"])
    except Exception:
        return []

    for proc in proc_iter:
        try:
            info = proc.info
        except Exception:
            continue
        pid = info.get("pid")
        exe = info.get("exe")
        if not exe or pid is None or pid in exclude_pids:
            continue
        try:
            exe_norm = str(Path(exe).resolve()).lower()
        except (OSError, ValueError):
            exe_norm = str(exe).lower()
        if exe_norm in shim_paths:
            name = info.get("name") or Path(exe).name
            matches.append((int(pid), str(name)))

    return matches


def _is_desktop_local_serve_cmdline(command: str) -> bool:
    """True for the Desktop-local serve spawn shape (loopback + ephemeral port).

    Desktop primary/pool backends launch as::

        hermes serve --host 127.0.0.1 --port 0
        hermes serve --isolated --host 127.0.0.1 --port 0 ...

    Intentional long-lived headless serves (e.g. ``--host <tailscale-ip>
    --port 9119``) must never match — those are operator-managed remote
    backends and may legitimately run with ppid 1 under launchd/nohup.
    """
    cmd = command.lower()
    if "serve" not in cmd:
        return False
    if "hermes" not in cmd and "hermes_cli" not in cmd:
        return False
    # Ephemeral desktop bind: host loopback + port 0 (exact tokens).
    has_loopback = (
        "--host 127.0.0.1" in cmd
        or "--host=127.0.0.1" in cmd
        or "--host localhost" in cmd
        or "--host=localhost" in cmd
    )
    has_ephemeral = "--port 0" in cmd or "--port=0" in cmd
    if not (has_loopback and has_ephemeral):
        return False
    # Spare anything with a concrete non-zero port flag first (defensive).
    # (port 0 already required above.)
    return True


def _process_ppid(pid: int) -> int | None:
    """Best-effort parent pid lookup. None on failure."""
    try:
        if sys.platform == "win32":
            return None  # Windows orphan reap is handled by desktop tree-kill.
        result = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        return int(result.stdout.strip().split()[0])
    except (ValueError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _exclude_pids_from_env() -> set[int]:
    """PIDs Desktop marks as live backends (HERMES_DESKTOP_CHILD_PID)."""
    raw = os.environ.get("HERMES_DESKTOP_CHILD_PID", "")
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


# --- SSH remote-backend lock ownership -------------------------------------
#
# ``backend.lock.json`` is the ownership record the Desktop SSH runtime writes
# on the *remote* host for every ``hermes serve`` backend it spawns over SSH
# (see apps/desktop/electron/remote-lifecycle.ts). A backend started from
# another client/machine — e.g. a MacBook driving a ``hermes serve`` on a Mac
# Mini over SSH — is a *legitimate, lock-owned* backend even though it has no
# parent on this host (sshd has long since exited, reparenting it to pid 1).
#
# The orphan reap must NEVER kill a PID that a valid ``backend.lock.json``
# claims as its owner. Doing so murdered a real production SSH remote backend
# on a Mac Mini the first time the local Desktop app rebooted. The lock file is
# the source of truth for "is this serve legitimately owned by some client",
# regardless of which machine started it.

# Mirror the schema constants in remote-lifecycle.ts (the writer). Bumping one
# side without the other makes the lock unreadable on purpose, which is the
# safe failure mode for reuse — but for the reap we only ever *spare*, so a
# mismatched-schema record is simply ignored (never used to kill).
_LOCKFILE_SCHEMA_VERSION = 2
_PROTOCOL_VERSION = 1
_REMOTE_LOCK_SUBDIR = "desktop-ssh"
_HEX32 = set("0123456789abcdef")
_HEX16 = _HEX32


def _hermes_home_dir() -> Path:
    """Resolved Hermes home (HERMES_HOME override or ~/.hermes)."""
    override = os.environ.get("HERMES_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".hermes"


def _valid_lockfile_payload(parsed: object, ownership_id: str) -> bool:
    """Validate a parsed ``backend.lock.json`` body, mirroring readLockfile().

    Returns True only when every structural field the SSH runtime writes is
    present and well-formed. A lock that fails validation is ignored (treated
    as "no ownership claim"), which never causes a kill — the reap only ever
    *adds* lock-owned PIDs to its spare-set.
    """
    if not isinstance(parsed, dict):
        return False
    if parsed.get("schemaVersion") != _LOCKFILE_SCHEMA_VERSION:
        return False
    if parsed.get("protocolVersion") != _PROTOCOL_VERSION:
        return False
    if parsed.get("ownershipId") != ownership_id:
        return False
    spawn_nonce = parsed.get("spawnNonce")
    if not isinstance(spawn_nonce, str) or len(spawn_nonce) != 16:
        return False
    if set(spawn_nonce) - _HEX16:
        return False
    token_fp = parsed.get("tokenFingerprint")
    if not isinstance(token_fp, str) or len(token_fp) != 32 or set(token_fp) - _HEX32:
        return False
    pid = parsed.get("pid")
    if not isinstance(pid, int) or pid <= 0 or pid > 4194304:
        return False
    port = parsed.get("port")
    if not isinstance(port, int) or port < 0 or port > 65535:
        return False
    # String fields must be present and bounded (the writer enforces <=1024).
    for field in ("profile", "hermesPath", "hermesHome", "logPath", "startedAt"):
        value = parsed.get(field)
        if not isinstance(value, str) or len(value) > 1024:
            return False
    # logPath is written as ``{lock_root}/{ownershipId}/{spawnNonce}.log``. We
    # only check the suffix so a relocated HERMES_HOME (different leading path)
    # doesn't falsely reject a legitimate remote-owned backend — a false reject
    # here would re-introduce the exact kill we're fixing.
    log_path = parsed["logPath"]
    if not log_path.endswith(f"/{ownership_id}/{spawn_nonce}.log"):
        return False
    return True


def _lock_owned_serve_pids(base_dir: Path | None = None) -> set[int]:
    """PIDs claimed as owners by valid ``backend.lock.json`` records on this host.

    Scans ``{hermes_home}/desktop-ssh/<ownershipId>/backend.lock.json`` (the
    same directory the Desktop SSH runtime writes to). Any PID a valid lock
    names is a legitimately-owned backend — including backends another client
    or machine started over SSH — and must be spared by the orphan reap.

    Best-effort: any read/parse/IO error for a single record is swallowed and
    that record contributes no PID. Never raises.
    """
    import json

    root = base_dir if base_dir is not None else (
        _hermes_home_dir() / _REMOTE_LOCK_SUBDIR
    )
    owned: set[int] = set()
    if not root.is_dir():
        return owned
    try:
        entries = list(root.iterdir())
    except OSError:
        return owned
    for entry in entries:
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        ownership_id = entry.name
        # Mirror validateOwnershipId(): exactly 32 lowercase hex chars.
        if len(ownership_id) != 32 or set(ownership_id) - _HEX32:
            continue
        lock_path = entry / "backend.lock.json"
        try:
            if not lock_path.is_file():
                continue
            with open(lock_path, "rb") as handle:
                data = handle.read()
        except OSError:
            continue
        if len(data) > 65536:
            continue
        try:
            parsed = json.loads(data)
        except (UnicodeDecodeError, ValueError):
            continue
        if _valid_lockfile_payload(parsed, ownership_id):
            try:
                owned.add(int(parsed["pid"]))
            except (TypeError, ValueError):
                continue
    return owned


def _reap_orphaned_desktop_local_serves(
    *,
    reason: str = "orphaned desktop-local hermes serve",
    signal_term=None,
    signal_kill=None,
    sleep_fn=None,
    lock_owned_pids_fn=None,
) -> dict[str, list]:
    """Kill leftover Desktop-local ``hermes serve`` backends with no parent.

    When Electron dies uncleanly (crash / SIGKILL / update handoff), local
    ``serve --host 127.0.0.1 --port 0`` children can be reparented to pid 1 and
    keep their full MCP trees alive. The next Desktop boot then stacks a fresh
    backend on top of the corpses until the machine hits EMFILE and the UI
    loses tabs/sidebar.

    The parent-death watchdog prevents *future* orphans once a backend is
    running under HERMES_PARENT_PID; this helper clears *already* orphaned
    corpses at the start of a new Desktop backend.

    Safety:
    - only the Desktop-local spawn shape (loopback + ``--port 0``)
    - only processes whose current ppid is 1 (or 0 on some supervisors)
    - never self / never HERMES_DESKTOP_CHILD_PID entries
    - never a PID a valid ``backend.lock.json`` claims as its owner — that is
      a legitimately lock-owned backend, *including SSH remote backends started
      by another client/machine* which legitimately sit at ppid 1 after sshd
      exits. Killing those is a production incident, not cleanup.
    - never fixed-port remote serves (e.g. ``--port 9119``)
    - best-effort; failures never raise to the caller
    """
    import signal as _signal
    import time as _time

    if signal_term is None:
        signal_term = _signal.SIGTERM
    if signal_kill is None:
        signal_kill = getattr(_signal, "SIGKILL", _signal.SIGTERM)
    if sleep_fn is None:
        sleep_fn = _time.sleep
    if lock_owned_pids_fn is None:
        lock_owned_pids_fn = _lock_owned_serve_pids

    if sys.platform == "win32":
        # Windows desktop uses taskkill tree teardown; orphan scan here is POSIX.
        return {"matched": [], "killed": [], "failed": []}

    exclude = _exclude_pids_from_env()
    exclude.add(os.getpid())
    # Also spare our direct parent (the desktop / sshd wrapper).
    try:
        exclude.add(os.getppid())
    except Exception:
        pass
    # Spare every PID a valid backend.lock.json owns — SSH remote backends
    # started by other clients/machines are legitimate, lock-owned owners even
    # though they are orphaned (ppid 1) on this host. (#78872 regression)
    try:
        exclude |= set(lock_owned_pids_fn())
    except Exception:
        # Best-effort: never let lock scanning block or widen the reap.
        pass

    try:
        scanned = _scan_dashboard_processes(exclude_pids=exclude)
    except Exception:
        return {"matched": [], "killed": [], "failed": []}

    # Re-read lock ownership defensively: the scan above already filtered
    # exclude PIDs, but a lock file may have been written between the scan and
    # now. Defense in depth — never kill a freshly-claimed owner.
    try:
        owned_now = set(lock_owned_pids_fn())
    except Exception:
        owned_now = set()

    targets: list[tuple[int, str]] = []
    for pid, cmd in scanned:
        if not _is_desktop_local_serve_cmdline(cmd):
            continue
        if pid in owned_now:
            continue
        ppid = _process_ppid(pid)
        if ppid is None:
            continue
        # Orphaned under init/launchd.
        if ppid not in (0, 1):
            continue
        targets.append((pid, cmd))

    if not targets:
        return {"matched": [], "killed": [], "failed": []}

    matched = [pid for pid, _ in targets]
    killed: list[int] = []
    failed: list[int] = []

    for pid, _cmd in targets:
        try:
            os.kill(pid, signal_term)
        except ProcessLookupError:
            continue
        except PermissionError:
            failed.append(pid)
            continue
        except OSError:
            failed.append(pid)
            continue

    # Brief grace, then SIGKILL survivors.
    sleep_fn(1.5)
    # psutil.pid_exists for the liveness probe: os.kill(pid, 0) is a
    # Windows footgun (sends CTRL_C_EVENT, bpo-14484). This path is
    # POSIX-only (win32 early-returns above), but the linter blocks the
    # pattern everywhere and psutil is a core dependency anyway.
    import psutil

    for pid, _cmd in targets:
        if pid in failed:
            continue
        if not psutil.pid_exists(pid):
            killed.append(pid)
            continue
        try:
            os.kill(pid, signal_kill)
            killed.append(pid)
        except ProcessLookupError:
            killed.append(pid)
        except OSError:
            failed.append(pid)

    if matched:
        try:
            print(
                f"⟲ Reaped {len(killed)} orphaned desktop-local serve "
                f"backend(s) ({reason}): {killed or matched}"
            )
        except Exception:
            pass

    return {"matched": matched, "killed": killed, "failed": failed}


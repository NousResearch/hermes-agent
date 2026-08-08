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


def _hermes_home_for_pid(pid: int) -> str | None:
    """Best-effort ``HERMES_HOME`` from *pid*'s environment."""
    try:
        import psutil

        home = psutil.Process(pid).environ().get("HERMES_HOME")
        if home:
            return home
    except Exception:
        pass
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except (OSError, PermissionError):
        return None
    for part in raw.split(b"\x00"):
        if part.startswith(b"HERMES_HOME="):
            return part.split(b"=", 1)[1].decode("utf-8", errors="replace") or None
    return None


def _is_ephemeral_port_zero_backend(argv: list[str]) -> bool:
    """True for Desktop-style ``serve|dashboard --port 0`` backends (#78821).

    Ephemeral-port backends are owned by Hermes Desktop (or become PPID-1
    orphans after a prior update respawn).  Replaying them after
    ``hermes update`` multiplies listening backends because ``--port 0``
    always binds a fresh free port.  Covers both ``serve`` and the legacy
    ``dashboard --no-open`` fallback older Desktop runtimes use.
    """
    if _dashboard_subcommand_index(argv) is None:
        return False
    for i, tok in enumerate(argv):
        if tok == "--port" and i + 1 < len(argv) and str(argv[i + 1]) == "0":
            return True
        if tok.startswith("--port=") and tok.split("=", 1)[1].strip() == "0":
            return True
    return False


def _dashboard_subcommand_index(argv: list[str]) -> int | None:
    for i, tok in enumerate(argv):
        if tok in ("serve", "dashboard"):
            return i
    return None


def _normalize_dashboard_cmdline(argv: list[str]) -> tuple[str, ...]:
    """Collapse argv to profile flags + serve/dashboard tail for dedupe."""
    idx = _dashboard_subcommand_index(argv)
    if idx is None:
        return tuple(argv)
    prefix: list[str] = []
    i = 0
    while i < idx:
        tok = argv[i]
        if tok in ("--profile", "-p") and i + 1 < idx:
            prefix.extend([tok, argv[i + 1]])
            i += 2
            continue
        if tok.startswith("--profile="):
            prefix.append(tok)
        i += 1
    return tuple(prefix + list(argv[idx:]))


def _profile_key_for_respawn(
    argv: list[str], hermes_home: str | None = None
) -> str:
    """Stable owner key: ``HERMES_HOME`` when known, else ``--profile`` / ``-p``.

    ``HERMES_HOME`` ending in ``profiles/<name>`` is normalized to
    ``profile:<name>`` so it shares a cap with an explicit ``--profile``
    flag for the same profile (#78821).  Non-profile homes (including
    distinct ``…/.hermes`` roots) keep a resolved ``home:`` key so
    unrelated installs do not collapse together.
    """
    profile_name: str | None = None
    for i, tok in enumerate(argv):
        if tok in ("--profile", "-p") and i + 1 < len(argv):
            profile_name = argv[i + 1]
            break
        if tok.startswith("--profile="):
            profile_name = tok.split("=", 1)[1]
            break

    if hermes_home:
        try:
            home_path = Path(hermes_home).resolve()
        except (OSError, RuntimeError, ValueError):
            home_path = Path(hermes_home)
        parts = home_path.parts
        if len(parts) >= 2 and parts[-2] == "profiles" and parts[-1]:
            return f"profile:{parts[-1]}"
        try:
            return f"home:{os.path.normcase(str(home_path))}"
        except (OSError, RuntimeError, ValueError):
            return f"home:{os.path.normcase(hermes_home)}"

    if profile_name:
        return f"profile:{profile_name}"
    return "profile:default"


def _filter_dashboard_respawn_candidates(
    candidates: list[tuple[int, list[str], str | None]],
) -> list[list[str]]:
    """Select which killed manual backends to respawn after ``hermes update``.

    Each candidate is ``(pid, argv, hermes_home)``.

    Rules (#78821):
    1. Never resurrect Desktop ephemeral ``serve|dashboard --port 0``
       backends — Desktop (``HERMES_DESKTOP_CHILD_PID``) owns their
       lifecycle.  These are also the PPID-1 orphans that previously
       multiplied across updates because ``--port 0`` always binds a
       fresh free port.
    2. Dedupe by normalized cmdline (identical argv → one respawn).
    3. Cap at most one managed backend per profile / ``HERMES_HOME``.

    Intentionally does **not** blanket-skip every PPID-1 process: a prior
    ``hermes update`` respawn detaches with ``start_new_session=True``, so
    fixed-port manual backends are reparented to init and must still be
    eligible for the next update's #40449 restart.
    """
    selected: list[list[str]] = []
    seen_cmdlines: set[tuple[str, ...]] = set()
    seen_profiles: set[str] = set()

    for _pid, argv, hermes_home in candidates:
        if not argv:
            continue
        if _is_ephemeral_port_zero_backend(argv):
            continue
        norm = _normalize_dashboard_cmdline(argv)
        if norm in seen_cmdlines:
            continue
        profile_key = _profile_key_for_respawn(argv, hermes_home)
        if profile_key in seen_profiles:
            continue
        seen_cmdlines.add(norm)
        seen_profiles.add(profile_key)
        selected.append(list(argv))

    return selected


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
    pid_home: dict[int, str | None] = {}
    if restart_managed and sys.platform != "win32":
        for pid in pids:
            cg_path = _m()._get_pid_cgroup_path(pid)
            pid_cgroup[pid] = cg_path
            pid_service[pid] = _m()._get_systemd_service_for_pid(pid)
            if not pid_service[pid]:
                # Manually-started process: preserve its exact argv so we
                # can respawn it after the update (#40449, #68934).
                # Snapshot HERMES_HOME before the kill so per-profile caps
                # still work after the process is gone (#78821).
                cmdline = _m()._dashboard_cmdline_for_pid(pid)
                if cmdline:
                    pid_cmdline[pid] = cmdline
                    pid_home[pid] = _hermes_home_for_pid(pid)

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
    #    Filtered so Desktop ``serve|dashboard --port 0`` backends are not
    #    resurrected and duplicates collapse to one per profile (#78821).
    restarted_services: list[str] = []
    unrecovered: list[int] = []
    if killed and restart_managed:
        failed_restarts: list[tuple[str, str]] = []
        seen_services: set[str] = set()
        respawn_candidates: list[tuple[int, list[str], str | None]] = []
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
                respawn_candidates.append(
                    (pid, pid_cmdline[pid], pid_home.get(pid))
                )
            else:
                unrecovered.append(pid)

        for svc in restarted_services:
            print(f"    ✓ restarted systemd service {svc}")
        for svc, err in failed_restarts:
            print(f"    ⚠ {svc}: {err}")

        respawn_cmds = _filter_dashboard_respawn_candidates(respawn_candidates)
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

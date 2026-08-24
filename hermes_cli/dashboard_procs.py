"""Dashboard process-hygiene helpers — extracted from ``hermes_cli/main.py``.

Helpers defined in ``hermes_cli.main_dashboard`` / ``hermes_cli.main_install_repair`` are imported at
call time so imports stay one-way (both of those modules import this one lazily).
"""

import contextlib
import os
import subprocess
import sys
from pathlib import Path

# Cmdline substrings identifying the long-lived server (``serve`` = the headless name Desktop
# spawns; reaped on update for the same reason).
_DASHBOARD_PATTERNS = tuple(
    f"{launcher} {cmd}"
    for cmd in ("dashboard", "serve")
    for launcher in ("hermes", "hermes_cli.main", "hermes_cli/main.py"))
_PS_RUN_KWARGS = dict(capture_output=True, text=True, encoding="utf-8", errors="replace")


def _empty_result() -> dict[str, list]:
    return {"matched": [], "killed": [], "failed": []}


def _append_row(rows: list[tuple[int, str]], pid_text: str, command: str) -> None:
    try:
        rows.append((int(pid_text), command))
    except ValueError:
        pass


def _iter_process_table() -> list[tuple[int, str]]:
    """``(pid, cmdline)`` for every process, via wmic (Windows) or ps. Raises on scan failure."""
    rows: list[tuple[int, str]] = []
    if sys.platform == "win32":
        # errors="ignore": wmic may emit the system code page. bounded_probe_run, not run():
        # run()'s post-timeout cleanup joins pipe readers unbounded and a conhost descendant
        # holding duplicated handles wedges it forever.
        # In text mode, subprocess output decoding depends on Python's configuration (locale-dependent by
        # default, or UTF-8 in UTF-8 mode). The important protection here is errors="ignore": it prevents a
        # reader-thread UnicodeDecodeError from leaving result.stdout=None and turning the later .split()
        # into an AttributeError (#17049). bounded_probe_run (rather than subprocess.run with a timeout)
        # keeps a slow scan from wedging the caller forever: run()'s post-timeout cleanup joins the pipe
        # reader threads unbounded, and a conhost.exe descendant holding duplicated pipe handles blocks that
        # join indefinitely (#87134). It also passes CREATE_NO_WINDOW: this scan can run from the windowless
        # pythonw.exe desktop/gateway backend during an update, where a bare wmic spawn would pop a console
        # window.
        from hermes_cli._subprocess_compat import bounded_probe_run
        result = bounded_probe_run(
            ["wmic", "process", "get", "ProcessId,CommandLine", "/FORMAT:LIST"],
            timeout=10, errors="ignore")
        if result is None or result.returncode != 0 or result.stdout is None:
            return rows
        current_cmd = ""
        for line in result.stdout.split("\n"):
            line = line.strip()
            if line.startswith("CommandLine="):
                current_cmd = line[len("CommandLine=") :]
            elif line.startswith("ProcessId="):
                _append_row(rows, line[len("ProcessId=") :], current_cmd)
        return rows
    # ps, not `pgrep -f "hermes.*dashboard"` (greedy regex; consistent with gateway pid scan).
    result = subprocess.run(["ps", "-A", "-o", "pid=,command="], timeout=10, **_PS_RUN_KWARGS)
    if result.returncode == 0:
        for line in getattr(result, "stdout", "").split("\n"):
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and "grep" not in line:
                _append_row(rows, parts[0], parts[1])
    return rows


def _scan_dashboard_processes(*, exclude_pids: set[int] | None = None) -> list[tuple[int, str]]:
    """``(pid, cmdline)`` of running ``dashboard``/``serve`` processes; empty on any scan error.

    A forgotten dashboard keeps the old Python backend against the new JS bundle after
    ``hermes update`` (every API call 401s). *exclude_pids* (Desktop's HERMES_DESKTOP_CHILD_PID
    backends) are never returned.

    *exclude_pids* is an optional set of PIDs that must never be returned. This is used by the Hermes
    Desktop Electron app to protect its own backend child process: when the desktop spawns ``hermes serve``
    as a backend and triggers an auto-update, the update must not kill the backend that the desktop itself
    manages. The desktop sets the environment variable ``HERMES_DESKTOP_CHILD_PID`` on the spawned backend
    process; ``_kill_stale_dashboard_processes`` reads it and passes it here. (#37532)
    """
    skip = {os.getpid(), *(exclude_pids or ())}
    try:
        found = [(pid, cmd) for pid, cmd in _iter_process_table()
                 if pid not in skip and any(p in cmd for p in _DASHBOARD_PATTERNS)]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    # Spawn-ledger augmentation: substring patterns miss profiled launches (`hermes --profile p
    # serve`); the ledger holds live-verified pids. Unavailable ledger → scan-only.
    with contextlib.suppress(Exception):
        # Every serve/ dashboard registers itself in the machine spawn ledger at startup with live-verified
        # (pid, create_time), so ledger rows are positive identity, not argv guessing. Add any live ledger
        # serve/dashboard the scan missed; prefer the ledger's recorded argv (full launch args) over the
        # scan's truncated view. See #81564.
        from hermes_cli.process_identity import ledger_entries
        seen = {pid for pid, _ in found} | skip
        for entry in ledger_entries():
            pid = entry.get("pid")
            if (entry.get("purpose") in ("serve", "dashboard") and isinstance(pid, int)
                    and pid not in seen):
                found.append((pid, str(entry.get("argv") or "")))
    return found


def _hermes_home_for_pid(pid: int) -> str | None:
    """Best-effort ``HERMES_HOME`` from *pid*'s environment (psutil, then /proc)."""
    with contextlib.suppress(Exception):
        import psutil
        if home := psutil.Process(pid).environ().get("HERMES_HOME"):
            return home
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    for part in raw.split(b"\x00"):
        if part.startswith(b"HERMES_HOME="):
            return part.split(b"=", 1)[1].decode("utf-8", errors="replace") or None
    return None


def _dashboard_subcommand_index(argv: list[str]) -> int | None:
    return next((i for i, tok in enumerate(argv) if tok in ("serve", "dashboard")), None)


def _profile_flag_value(argv: list[str]) -> str | None:
    """Value of the first ``--profile X`` / ``-p X`` / ``--profile=X`` in *argv*."""
    for i, tok in enumerate(argv):
        if tok in ("--profile", "-p") and i + 1 < len(argv):
            return str(argv[i + 1])
        if tok.startswith("--profile="):
            return tok.split("=", 1)[1]
    return None


def _is_ephemeral_port_zero_backend(argv: list[str]) -> bool:
    """True for Desktop-style ``serve|dashboard --port 0`` backends — replaying them after
    ``hermes update`` multiplies listening backends because ``--port 0`` binds a fresh port.

    See #78821.
    """
    if _dashboard_subcommand_index(argv) is None:
        return False
    return any((tok == "--port" and i + 1 < len(argv) and str(argv[i + 1]) == "0")
               or (tok.startswith("--port=") and tok.split("=", 1)[1].strip() == "0")
               for i, tok in enumerate(argv))


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


def _resolved_home(home: str) -> Path:
    try:
        return Path(home).resolve()
    except (OSError, RuntimeError, ValueError):
        return Path(home)


def _normalized_home_for_compare(home: str) -> str:
    """Install-identity key for *home*: symlinked / differently-spelled roots compare equal.

    See #94030.
    """
    return os.path.normcase(str(_resolved_home(home)))


def _profile_key_for_respawn(argv: list[str], hermes_home: str | None = None) -> str:
    """Stable owner key: ``HERMES_HOME`` when known, else ``--profile`` / ``-p``.

    A home ending in ``profiles/<name>`` → ``profile:<name>`` (shares a cap with an explicit
    ``--profile``); other homes keep a ``home:`` key so unrelated installs never collapse.

    See #78821.
    """
    if hermes_home:
        parts = _resolved_home(hermes_home).parts
        if len(parts) >= 2 and parts[-2] == "profiles" and parts[-1]:
            return f"profile:{parts[-1]}"
        return f"home:{_normalized_home_for_compare(hermes_home)}"
    return f"profile:{_profile_flag_value(argv) or 'default'}"


def _filter_dashboard_respawn_candidates(
    candidates: list[tuple[int, list[str], str | None]], *, own_home: str | None = None
) -> list[list[str]]:
    """Select which killed manual backends ``(pid, argv, hermes_home)`` to respawn after update.

    Rules: never resurrect Desktop ``--port 0`` backends; never replay a backend from a
    **foreign** ``HERMES_HOME`` (the argv-only respawn would come back on this install's home
    and steal the foreign install's fixed port → EADDRINUSE crash-loop; unreadable ``None``
    stays eligible); dedupe by normalized cmdline; one backend per profile / home. PPID-1 is
    NOT skipped: a prior respawn detaches, so fixed-port manual backends sit under init.

    1. Never resurrect Desktop ephemeral ``serve|dashboard --port 0`` backends — Desktop
    (``HERMES_DESKTOP_CHILD_PID``) owns their lifecycle. These are also the PPID-1 orphans that previously
    multiplied across updates because ``--port 0`` always binds a fresh free port. 2. A foreign install's
    backend is owned by that install's supervisor/user. 3. 4. See #78821, #94030.
    Intentionally does **not** blanket-skip every PPID-1 process: a prior ``hermes update`` respawn detaches
    with ``start_new_session=True``, so fixed-port manual backends are reparented to init and must still be
    eligible for the next update's #40449 restart.
    """
    if own_home is None:
        try:
            from hermes_constants import get_hermes_home
            own_home = str(get_hermes_home())
        except Exception:
            own_home = ""
    own_key = _normalized_home_for_compare(own_home) if own_home else ""
    selected: list[list[str]] = []
    seen_cmdlines: set[tuple[str, ...]] = set()
    seen_profiles: set[str] = set()
    for _pid, argv, hermes_home in candidates:
        if not argv or _is_ephemeral_port_zero_backend(argv):
            continue
        if own_key and hermes_home and _normalized_home_for_compare(hermes_home) != own_key:
            continue
        norm = _normalize_dashboard_cmdline(argv)
        profile_key = _profile_key_for_respawn(argv, hermes_home)
        if norm in seen_cmdlines or profile_key in seen_profiles:
            continue
        seen_cmdlines.add(norm)
        seen_profiles.add(profile_key)
        selected.append(list(argv))
    return selected


def _exclude_pids_from_env() -> set[int]:
    """PIDs Desktop marks as live backends (``HERMES_DESKTOP_CHILD_PID``, comma-separated)."""
    out: set[int] = set()
    for part in os.environ.get("HERMES_DESKTOP_CHILD_PID", "").split(","):
        with contextlib.suppress(ValueError):
            out.add(int(part))
    return out


def _kill_pids_windows(pids: list[int], killed: list[int], failed: list[tuple[int, str]]) -> None:
    """``taskkill /F`` each PID after re-verifying its identity."""
    from gateway.status import get_process_start_time
    from hermes_cli._subprocess_compat import pid_is_hermes, windows_hide_flags
    # Identity captured right after discovery: a PID reused before the kill fails the check.
    pid_start_times = {pid: get_process_start_time(pid) for pid in pids}
    for pid in pids:
        try:
            expected_start_time = pid_start_times.get(pid)
            if expected_start_time is None:
                failed.append((pid, "could not verify process identity"))
            elif not pid_is_hermes(pid, expected_start_time=expected_start_time):
                failed.append((pid, "not hermes-owned or process identity changed"))
            else:
                result = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"], stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                    errors="replace", timeout=10, creationflags=windows_hide_flags())
                if result.returncode == 0:
                    killed.append(pid)
                else:
                    failed.append((pid, (result.stderr or result.stdout or "").strip()))
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            failed.append((pid, str(e)))


def _kill_pids_posix(pids: list[int], killed: list[int], failed: list[tuple[int, str]]) -> None:
    """SIGTERM, wait up to ~3s for graceful exit, SIGKILL survivors."""
    import signal as _signal
    import time as _time

    from gateway.status import _pid_exists

    def _send(pid: int, sig) -> None:
        try:
            os.kill(pid, sig)
            if sig == _signal.SIGKILL:
                killed.append(pid)
        except ProcessLookupError:
            killed.append(pid)  # already gone — count as killed
        except (PermissionError, OSError) as e:
            failed.append((pid, str(e)))

    for pid in pids:
        _send(pid, _signal.SIGTERM)
    deadline = _time.monotonic() + 3.0
    pending = [p for p in pids if p not in killed and p not in {f[0] for f in failed}]
    while pending and _time.monotonic() < deadline:
        _time.sleep(0.1)
        alive = [p for p in pending if _pid_exists(p)]  # os.kill(pid, 0) breaks on Windows
        killed.extend(p for p in pending if p not in alive)
        pending = alive
    for pid in pending:
        _send(pid, _signal.SIGKILL)


def _kill_stale_dashboard_processes(
    reason: str = "the running backend no longer matches the updated frontend", *,
    restart_managed: bool = False, already_restarted_units: "set[str] | None" = None,
) -> dict[str, list]:
    """Kill running ``hermes dashboard`` / ``hermes serve`` processes (update end, ``--stop``).

    With ``restart_managed`` (update only) systemd-owned PIDs get their unit restarted after the
    kill (systemd treats our SIGTERM as a clean stop, so ``Restart=on-failure`` never fires) and
    manual PIDs are respawned from captured argv. PIDs owned by *already_restarted_units* (no
    ``.service`` suffix) are left untouched, not killed twice.

    Manually-started dashboards are not auto-restarted because we don't know the original launch args
    (--host, --port, --insecure, --tui, --no-open). See #68934.
    *already_restarted_units* names units (no ``.service`` suffix) the caller already restarted directly —
    e.g. ``hermes update``'s systemd fleet-restart loop, which restarts ``hermes-serve*`` units before this
    function runs. Without excluding them, a Serve-only install's freshly restarted process is found again
    here and restarted a second time for no benefit (review on #83595).
    """
    from hermes_cli import main_dashboard as _dash

    if restart_managed and _dash._restart_managed_dashboard_service(reason):
        # The dashboard unit is handled but other backends (e.g. hermes-serve.service) are not:
        # mark the unit handled so the filter below drops its PIDs, and keep going.
        _dash_unit = getattr(_dash, "_DASHBOARD_SYSTEMD_UNIT", "hermes-dashboard.service")
        already_restarted_units = set(already_restarted_units or ()) | {
            str(_dash_unit).removesuffix(".service")}
    exclude = _exclude_pids_from_env()
    if restart_managed:
        # An SSH-owned backend belongs to an attached Desktop client; killing it strands that
        # client's fixed SSH port-forward. Same ownership records as the reaper.
        exclude |= _lock_owned_serve_pids()
    pids = _dash._find_stale_dashboard_pids(exclude_pids=exclude or None)
    if not pids:
        return _empty_result()
    # Snapshot systemd unit/cgroup and argv BEFORE killing (the cgroup dies with the process).
    pid_cgroup: dict[int, str | None] = {}
    pid_service: dict[int, str | None] = {}
    pid_cmdline: dict[int, list[str]] = {}
    pid_home: dict[int, str | None] = {}
    if restart_managed and sys.platform != "win32":
        for pid in pids:
            pid_cgroup[pid] = _dash._get_pid_cgroup_path(pid)
            pid_service[pid] = _dash._get_systemd_service_for_pid(pid)
            if not pid_service[pid] and (cmdline := _dash._dashboard_cmdline_for_pid(pid)):
                # Manual process: exact argv + HERMES_HOME for the respawn and its profile cap.
                # Manually-started process: preserve its exact argv so we can respawn it after the update
                # (#40449, #68934). Snapshot HERMES_HOME before the kill so per-profile caps still work
                # after the process is gone (#78821).
                pid_cmdline[pid] = cmdline
                pid_home[pid] = _hermes_home_for_pid(pid)
        if already_restarted_units:
            pids = [pid for pid in pids if (pid_service.get(pid) or "").removesuffix(".service")
                    not in already_restarted_units]
            if not pids:
                return _empty_result()
    print(f"\n⟲ Stopping {len(pids)} dashboard process(es) ({reason})")
    killed: list[int] = []
    failed: list[tuple[int, str]] = []
    (_kill_pids_windows if sys.platform == "win32" else _kill_pids_posix)(pids, killed, failed)
    for pid in killed:
        print(f"    ✓ stopped PID {pid}")
    for pid, err_msg in failed:
        print(f"    ✗ failed to stop PID {pid}: {err_msg}")
    if killed and restart_managed:
        unrecovered = _restart_killed_backends(killed, pid_service, pid_cgroup, pid_cmdline, pid_home)
    else:
        unrecovered = list(killed)
        if killed:
            print("  Restart the dashboard when you're ready:\n    hermes dashboard --port <port>")
    return {"matched": list(pids), "killed": list(killed), "failed": list(failed),
            "unrecovered": list(unrecovered)}


def _restart_killed_backends(
    killed: list[int], pid_service: dict[int, str | None], pid_cgroup: dict[int, str | None],
    pid_cmdline: dict[int, list[str]], pid_home: dict[int, str | None]) -> list[int]:
    """Update path: restart systemd units, respawn manual argv (detached, headless, logged to
    logs/dashboard-restart.log; one per profile, no ``--port 0``). Returns PIDs not brought back."""
    # Two categories: Without this, a remote backend (hermes serve) under Restart=on-failure never comes
    # back after our clean SIGTERM, and the Desktop can't reconnect (#68934). Filtered so Desktop
    # ``serve|dashboard --port 0`` backends are not resurrected and duplicates collapse to one per profile
    # (#78821).
    from hermes_cli import main_dashboard as _dash
    unrecovered: list[int] = []
    failed_restarts: list[tuple[str, str]] = []
    seen_services: set[str] = set()
    respawn_candidates: list[tuple[int, list[str], str | None]] = []
    for pid in killed:
        svc_name = pid_service.get(pid)
        if svc_name:
            if svc_name in seen_services:
                continue
            seen_services.add(svc_name)
            if _dash._try_restart_systemd_service(svc_name, pid_cgroup.get(pid)):
                print(f"    ✓ restarted systemd service {svc_name}")
            else:
                failed_restarts.append((svc_name, "systemctl restart returned non-zero"))
                unrecovered.append(pid)
        elif pid in pid_cmdline:
            respawn_candidates.append((pid, pid_cmdline[pid], pid_home.get(pid)))
        else:
            unrecovered.append(pid)
    for svc, err in failed_restarts:
        print(f"    ⚠ {svc}: {err}")
    respawn_cmds = _filter_dashboard_respawn_candidates(respawn_candidates)
    failed_cmds = _dash._respawn_dashboard_processes(respawn_cmds) if respawn_cmds else None
    if failed_cmds:
        unrecovered.extend(p for p in killed if pid_cmdline.get(p) in failed_cmds)
    if failed_restarts or unrecovered:
        print("  Restart anything not auto-restarted when you're ready:\n    hermes dashboard --port <port>")
    return unrecovered


def _norm_exe(path) -> str:
    """Canonical lower-cased executable path for comparison."""
    try:
        return str(Path(path).resolve()).lower()
    except (OSError, ValueError):
        return str(path).lower()


def _detect_concurrent_hermes_instances(
    scripts_dir: Path, *, exclude_pid: int | None = None) -> list[tuple[int, str]]:
    """``(pid, name)`` of other live processes whose .exe is one of our entry-point shims.

    Windows blocks DELETE/REPLACE on a running .exe, so a Desktop-spawned ``hermes.EXE`` makes
    the update's quarantine rename fail with ``[WinError 32]``. Excludes our PID and every
    *shim* ancestor (the setuptools launcher is a separate native process from its
    ``python.exe``); ``proc.parents()`` at once because a per-hop loop bailed on the first
    AccessDenied. Empty off-Windows / without psutil. Never raises.
    """
    from hermes_cli.main_install_repair import _hermes_exe_shims, _is_windows

    if not _is_windows():
        return []
    try:
        import psutil
    except Exception:
        return []
    shim_paths = {_norm_exe(shim) for shim in _hermes_exe_shims(scripts_dir)}
    if not shim_paths:
        return []
    seed = int(exclude_pid) if exclude_pid is not None else os.getpid()
    exclude_pids: set[int] = {seed}
    # Broad ``except Exception`` guards against partially-stubbed psutil in unit tests; this helper is
    # documented as "never raises". Only the per-ancestor exe()/pid reads skip that ancestor; anything
    # else aborts the whole walk (BASE semantics).
    try:
        for ancestor in psutil.Process(seed).parents():
            try:
                anc_exe = ancestor.exe()
            except Exception:
                continue
            if not anc_exe:
                continue
            if _norm_exe(anc_exe) in shim_paths:
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
        pid, exe = info.get("pid"), info.get("exe")
        if exe and pid is not None and pid not in exclude_pids and _norm_exe(exe) in shim_paths:
            matches.append((int(pid), str(info.get("name") or Path(exe).name)))
    return matches


def _is_desktop_local_serve_cmdline(command: str) -> bool:
    """True for the Desktop-local shape ``hermes serve [--isolated] --host 127.0.0.1 --port 0``.

    Long-lived headless serves (``--host <tailscale-ip> --port 9119``) must never match —
    those are operator-managed remote backends that legitimately run with ppid 1.
    """
    cmd = command.lower()
    if "serve" not in cmd or ("hermes" not in cmd and "hermes_cli" not in cmd):
        return False
    has_loopback = any(tok in cmd for tok in (
        "--host 127.0.0.1", "--host=127.0.0.1", "--host localhost", "--host=localhost"))
    return has_loopback and ("--port 0" in cmd or "--port=0" in cmd)


def _process_ppid(pid: int) -> int | None:
    """Best-effort parent pid; None on failure (always None on Windows: desktop tree-kill reaps)."""
    try:
        if sys.platform == "win32":
            return None
        result = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], timeout=5, **_PS_RUN_KWARGS)
        if result.returncode != 0 or not result.stdout:
            return None
        return int(result.stdout.strip().split()[0])
    except (ValueError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


# SSH remote-backend lock ownership: ``backend.lock.json`` is written by the Desktop SSH runtime
# (apps/desktop/electron/remote-lifecycle.ts) for every ``hermes serve`` it spawns. Such a backend
# is legitimate even at ppid 1 (sshd exited); the reap must NEVER kill a PID a valid lock claims
# — that once killed a production backend. Schema mirrors the writer; mismatches are ignored.
_LOCKFILE_SCHEMA_VERSION = 2
_PROTOCOL_VERSION = 1
_REMOTE_LOCK_SUBDIR = "desktop-ssh"
_HEX32 = set("0123456789abcdef")


def _hermes_home_dir() -> Path:
    """Resolved Hermes home (HERMES_HOME override or ~/.hermes)."""
    override = os.environ.get("HERMES_HOME", "").strip()
    return Path(override).expanduser() if override else Path.home() / ".hermes"


def _is_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and not (set(value) - _HEX32)


def _valid_lockfile_payload(parsed: object, ownership_id: str) -> bool:
    """Validate a parsed ``backend.lock.json`` body, mirroring readLockfile()."""
    if (
        not isinstance(parsed, dict)
        or parsed.get("schemaVersion") != _LOCKFILE_SCHEMA_VERSION
        or parsed.get("protocolVersion") != _PROTOCOL_VERSION
        or parsed.get("ownershipId") != ownership_id
        or not _is_hex(parsed.get("spawnNonce"), 16)
        or not _is_hex(parsed.get("tokenFingerprint"), 32)):
        return False
    pid, port = parsed.get("pid"), parsed.get("port")
    if not (isinstance(pid, int) and 0 < pid <= 4194304 and isinstance(port, int)
            and 0 <= port <= 65535):
        return False
    # String fields must be present and bounded (the writer enforces <=1024).
    if any(not isinstance(parsed.get(f), str) or len(parsed[f]) > 1024
           for f in ("profile", "hermesPath", "hermesHome", "logPath", "startedAt")):
        return False
    # Suffix-only check of logPath so a relocated HERMES_HOME can't reject a legitimate backend.
    return parsed["logPath"].endswith(f"/{ownership_id}/{parsed['spawnNonce']}.log")


def _lock_owned_serve_pids(base_dir: Path | None = None) -> set[int]:
    """PIDs claimed by valid ``{hermes_home}/desktop-ssh/<ownershipId>/backend.lock.json`` records
    (best-effort: a bad record contributes no PID; never raises)."""
    import json
    root = base_dir if base_dir is not None else _hermes_home_dir() / _REMOTE_LOCK_SUBDIR
    owned: set[int] = set()
    try:
        entries = list(root.iterdir()) if root.is_dir() else []
    except OSError:
        return owned
    for entry in entries:
        ownership_id = entry.name
        lock_path = entry / "backend.lock.json"
        try:  # validateOwnershipId(): exactly 32 lowercase hex chars
            if not entry.is_dir() or not _is_hex(ownership_id, 32) or not lock_path.is_file():
                continue
            data = lock_path.read_bytes()
            if len(data) > 65536:
                continue
            parsed = json.loads(data)
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if _valid_lockfile_payload(parsed, ownership_id):
            owned.add(parsed["pid"])  # validated as int above
    return owned


# Covers the gap between process start and the Desktop client writing backend.lock.json.
_REAP_MIN_AGE_SECONDS = 180.0


def _process_age_seconds(pid: int) -> float:
    """Process age from psutil's cross-platform start timestamp."""
    import time as _time

    import psutil as _psutil
    return max(0.0, _time.time() - _psutil.Process(pid).create_time())


def _reap_orphaned_desktop_local_serves(
    *, reason: str = "orphaned desktop-local hermes serve", signal_term=None, signal_kill=None,
    sleep_fn=None, lock_owned_pids_fn=None, process_age_seconds_fn=None) -> dict[str, list]:
    """Kill leftover Desktop-local ``hermes serve`` backends with no parent. Never raises.

    When Electron dies uncleanly its ``serve --host 127.0.0.1 --port 0`` children are
    reparented to pid 1 with their MCP trees alive; each Desktop boot then stacks a fresh
    backend on the corpses until EMFILE. Reaped only if ALL hold: Desktop-local shape; ppid
    0/1; not self / parent / HERMES_DESKTOP_CHILD_PID; not claimed by a valid
    ``backend.lock.json`` (SSH backends other clients started legitimately sit at ppid 1);
    older than ``_REAP_MIN_AGE_SECONDS`` with a determinable age (Desktop writes the lock only
    after HERMES_BACKEND_READY, so a live sibling mid-startup is briefly unowned).
    """
    import signal as _signal
    import time as _time
    signal_term = _signal.SIGTERM if signal_term is None else signal_term
    signal_kill = getattr(_signal, "SIGKILL", _signal.SIGTERM) if signal_kill is None else signal_kill
    sleep_fn = sleep_fn or _time.sleep
    lock_owned_pids_fn = lock_owned_pids_fn or _lock_owned_serve_pids
    process_age_seconds_fn = process_age_seconds_fn or _process_age_seconds
    if sys.platform == "win32":  # Windows desktop uses taskkill tree teardown
        return _empty_result()

    def _owned_pids() -> set[int]:
        try:
            return set(lock_owned_pids_fn())
        except Exception:
            return set()  # never let lock scanning block or widen the reap

    def _is_stale_orphan(pid: int) -> bool:
        try:  # never let a liveness probe failure widen the reap
            return process_age_seconds_fn(pid) >= _REAP_MIN_AGE_SECONDS
        except Exception:
            return False

    exclude = _exclude_pids_from_env() | {os.getpid()} | _owned_pids()
    with contextlib.suppress(Exception):
        exclude.add(os.getppid())  # the desktop / sshd wrapper
    try:
        scanned = _scan_dashboard_processes(exclude_pids=exclude)
    except Exception:
        return _empty_result()
    owned_now = _owned_pids()  # re-read: a lock may have been written since the scan
    matched = [pid for pid, cmd in scanned
               if _is_desktop_local_serve_cmdline(cmd) and pid not in owned_now
               and _process_ppid(pid) in (0, 1) and _is_stale_orphan(pid)]
    if not matched:
        return _empty_result()
    killed: list[int] = []
    failed: list[int] = []
    for pid in matched:
        try:
            os.kill(pid, signal_term)
        except ProcessLookupError:
            continue
        except OSError:
            failed.append(pid)
    # Brief grace, then SIGKILL survivors (psutil.pid_exists: os.kill(pid, 0) is a Windows footgun).
    sleep_fn(1.5)
    import psutil
    for pid in matched:
        if pid in failed:
            continue
        try:
            if psutil.pid_exists(pid):
                os.kill(pid, signal_kill)
            killed.append(pid)
        except ProcessLookupError:
            killed.append(pid)
        except OSError:
            failed.append(pid)
    with contextlib.suppress(Exception):
        print(f"⟲ Reaped {len(killed)} orphaned desktop-local serve backend(s) ({reason}): {killed or matched}")
    return {"matched": matched, "killed": killed, "failed": failed}

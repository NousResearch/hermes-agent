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

            result = bounded_probe_run(
                ["wmic", "process", "get", "ProcessId,CommandLine", "/FORMAT:LIST"],
                timeout=10,
                errors="ignore",
            )
            if result is None or result.returncode != 0 or result.stdout is None:
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

    # Spawn-ledger augmentation (#63206/#81564): the substring patterns above
    # miss profiled launches — `hermes --profile p serve --host <ip>` contains
    # neither "hermes serve" nor "hermes_cli.main serve". Every serve/
    # dashboard registers itself in the machine spawn ledger at startup with
    # live-verified (pid, create_time), so ledger rows are positive identity,
    # not argv guessing. Add any live ledger serve/dashboard the scan missed;
    # prefer the ledger's recorded argv (full launch args) over the scan's
    # truncated view.
    try:
        from hermes_cli.process_identity import ledger_entries

        seen = {pid for pid, _ in dashboard_processes}
        for entry in ledger_entries():
            if entry.get("purpose") not in ("serve", "dashboard"):
                continue
            pid = entry.get("pid")
            if not isinstance(pid, int) or pid == self_pid or pid in seen:
                continue
            if exclude_pids and pid in exclude_pids:
                continue
            dashboard_processes.append((pid, str(entry.get("argv") or "")))
    except Exception:
        pass  # ledger unavailable → scan-only behavior, exactly as before

    return dashboard_processes


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


def _profile_key_for_respawn(
    argv: list[str], hermes_home: str | None = None
) -> str:
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


def _normalized_home_for_compare(home: str) -> str:
    """Resolve *home* for install-identity comparison (#94030).

    Same normalization ``_profile_key_for_respawn`` applies to ``home:``
    keys, so symlinked / differently-spelled roots compare equal.
    """
    try:
        return os.path.normcase(str(Path(home).resolve()))
    except (OSError, RuntimeError, ValueError):
        return os.path.normcase(home)


def _filter_dashboard_respawn_candidates(
    candidates: list[tuple[int, list[str], str | None]],
    *,
    own_home: str | None = None,
) -> list[list[str]]:
    """Select which killed manual backends ``(pid, argv, hermes_home)`` to respawn after update.

    Each candidate is ``(pid, argv, hermes_home)``.  *own_home* is the
    updating install's home; it defaults to this process's
    ``get_hermes_home()`` and exists as a parameter so tests can pin it.

    Rules (#78821, #94030):
    1. Never resurrect Desktop ephemeral ``serve|dashboard --port 0``
       backends — Desktop (``HERMES_DESKTOP_CHILD_PID``) owns their
       lifecycle.  These are also the PPID-1 orphans that previously
       multiplied across updates because ``--port 0`` always binds a
       fresh free port.
    2. Never replay a backend from a **foreign** ``HERMES_HOME``.  The
       respawn below is argv-only (no ``env=`` replay), so a foreign
       backend would come back running on the *updating* install's home
       and steal the foreign install's fixed port, leaving its own
       supervisor (launchd/systemd/...) to crash-loop on ``EADDRINUSE``
       (#94030).  A foreign install's backend is owned by that install's
       supervisor/user.  An unreadable home (``None``) stays eligible —
       keep the pre-#94030 behaviour when we cannot tell.
    3. Dedupe by normalized cmdline (identical argv → one respawn).
    4. Cap at most one managed backend per profile / ``HERMES_HOME``.

    Intentionally does **not** blanket-skip every PPID-1 process: a prior
    ``hermes update`` respawn detaches with ``start_new_session=True``, so
    fixed-port manual backends are reparented to init and must still be
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
    if restart_managed and _m()._restart_managed_dashboard_service(reason):
        # The dashboard unit is handled; every OTHER backend is not (#92145).
        # This used to return here, which meant a host running BOTH
        # ``hermes-dashboard.service`` and ``hermes-serve.service`` -- the
        # exact unit set in the report -- restarted only the dashboard and
        # never even scanned for the serve backend that hosts
        # ``tui_gateway``. That backend then kept its pre-update
        # ``sys.modules`` while the checkout moved on. Record the unit as
        # already handled (the filter below drops PIDs it owns, including
        # the one systemd just replaced) and keep scanning.
        _dash_unit = getattr(
            _m(), "_DASHBOARD_SYSTEMD_UNIT", "hermes-dashboard.service"
        )
        already_restarted_units = set(already_restarted_units or ()) | {
            str(_dash_unit).removesuffix(".service")
        }

    # When the Hermes Desktop Electron app spawns this dashboard as a
    # backend child, it sets HERMES_DESKTOP_CHILD_PID so that the update
    # path can skip killing the desktop-managed process.  (#37532)
    exclude: set[int] = set()
    raw_pid = os.environ.get("HERMES_DESKTOP_CHILD_PID")
    if raw_pid:
        # The desktop may manage several backends (one per active profile) and
        # passes them comma-separated; a lone int still parses for back-compat.
        for part in raw_pid.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                exclude.add(int(part))
            except (ValueError, TypeError):
                pass

    if restart_managed:
        # An SSH-owned backend belongs to an attached Desktop client even when
        # the updater runs from an unrelated remote shell with no Desktop child
        # PID. Honor the same validated ownership records as the orphan reaper;
        # killing one permanently strands that client's fixed SSH port-forward.
        exclude |= _lock_owned_serve_pids()

    pids = _m()._find_stale_dashboard_pids(exclude_pids=exclude or None)
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
    from hermes_cli.update_cmd_windows import _hermes_holder_subcommand
    # Canonical token matcher, never argv substrings: ``kanban --preserve-cache`` contains "serve" and
    # ``vim notes about hermes serve`` contains both markers — this predicate decides a kill.
    if _hermes_holder_subcommand(command) != "serve":
        return False
    tokens = command.lower().split()
    host = _flag_value(tokens, "--host")
    return host in ("127.0.0.1", "localhost") and _flag_value(tokens, "--port") == "0"


def _flag_value(tokens: list[str], flag: str) -> str | None:
    """``--flag value`` / ``--flag=value`` from split argv, or None."""
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith(flag + "="):
            return tok.partition("=")[2]
    return None


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
    """The process's Hermes home: remote-backend locks are a process-level asset, so a request scoped
    to another profile must still see the same lock dir."""
    from hermes_constants import get_process_hermes_home
    return get_process_hermes_home()


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

    root = base_dir if base_dir is not None else (
        _hermes_home_dir() / _REMOTE_LOCK_SUBDIR
    )
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


# Grace window before an orphaned-looking backend may be reaped. Covers the
# gap between process start and the Desktop client writing backend.lock.json.
_REAP_MIN_AGE_SECONDS = 180.0


def _process_age_seconds(pid: int) -> float:
    """Return a process age using psutil's cross-platform start timestamp."""
    import time as _time

    import psutil as _psutil

    return max(0.0, _time.time() - _psutil.Process(pid).create_time())


def _reap_orphaned_desktop_local_serves(
    *,
    reason: str = "orphaned desktop-local hermes serve",
    signal_term=None,
    signal_kill=None,
    sleep_fn=None,
    lock_owned_pids_fn=None,
    process_age_seconds_fn=None,
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
    - never a candidate younger than ``_REAP_MIN_AGE_SECONDS`` (or whose age
      cannot be determined). The Desktop client writes ``backend.lock.json``
      only after the backend reports HERMES_BACKEND_READY, so during
      concurrent multi-profile startup a live sibling is briefly unowned and
      otherwise indistinguishable from a corpse; sparing young processes
      closes that mutual-reap window. A genuine corpse merely waits for a
      later scan.
    - best-effort; failures never raise to the caller
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

    if signal_term is None:
        signal_term = _signal.SIGTERM
    if signal_kill is None:
        signal_kill = getattr(_signal, "SIGKILL", _signal.SIGTERM)
    if sleep_fn is None:
        sleep_fn = _time.sleep
    if lock_owned_pids_fn is None:
        lock_owned_pids_fn = _lock_owned_serve_pids
    if process_age_seconds_fn is None:
        process_age_seconds_fn = _process_age_seconds

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
        # Spare backends that are still starting up. backend.lock.json is
        # written by the *Desktop client* only after the backend reports
        # HERMES_BACKEND_READY, so a sibling spawned seconds ago is not yet
        # lock-owned and is invisible to the owned_now guard above. When
        # Desktop opens several profiles at once (each its own SSH spawn),
        # every new backend reaped its concurrently-starting siblings, whose
        # clients then reconnected and reaped the next batch -- a mutual-reap
        # storm. A genuine corpse from a previous Desktop session is always
        # older than this grace window; anything younger is a live sibling.
        try:
            if process_age_seconds_fn(pid) < _REAP_MIN_AGE_SECONDS:
                continue
        except Exception:
            # Never let a liveness probe failure widen the reap.
            continue
        targets.append((pid, cmd))

    if not targets:
        return {"matched": [], "killed": [], "failed": []}

    matched = [pid for pid, _ in targets]
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


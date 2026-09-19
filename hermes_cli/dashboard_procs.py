"""Dashboard process-hygiene helpers — extracted from ``hermes_cli/main.py``.

Helpers defined in ``hermes_cli.main_dashboard`` / ``hermes_cli.main_install_repair`` are imported at
call time so imports stay one-way (both of those modules import this one lazily).
"""

import contextlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

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


def _is_dashboard_lifecycle_probe(command: str) -> bool:
    """True for short-lived ``--stop`` / ``--status`` web-server commands."""
    # Shell/process wrappers can carry the complete Hermes command as one
    # quoted argv token. The exact flag boundary handles that shape before the
    # structured parser below handles ordinary console-script argv.
    if re.search(r"(?<!\S)--(?:status|stop)(?=\s|$|['\";])", command):
        return True
    try:
        argv = shlex.split(command, posix=sys.platform != "win32")
    except ValueError:
        return False
    index = _dashboard_subcommand_index(argv)
    if index is None:
        return False
    return any(token in {"--status", "--stop"} for token in argv[index + 1 :])


def _is_hermes_web_server_command(command: str) -> bool:
    """True only when argv structurally invokes a Hermes web-server command."""
    try:
        from hermes_cli.update_cmd_windows import _hermes_holder_subcommand

        return _hermes_holder_subcommand(command) in {"dashboard", "serve", "webapp"}
    except Exception:
        return False


def _ledger_web_server_processes() -> dict[int, str]:
    """Positively identified live web servers for this Hermes install."""
    try:
        from hermes_cli.process_identity import ledger_entries

        entries = ledger_entries()
    except Exception:
        return {}

    processes: dict[int, str] = {}
    for entry in entries:
        if entry.get("purpose") not in {"dashboard", "serve", "webapp"}:
            continue
        pid = entry.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            continue
        command = str(entry.get("argv") or "")
        # register_self intentionally records only a bounded argv prefix. When
        # the process table is unreadable, treat the positive identity as an
        # ephemeral port so status reports the live process without guessing a
        # fixed listener address.
        if "--port" not in command.split():
            command = f"{command} --port 0".strip()
        processes[pid] = command
    return processes


def _scan_dashboard_processes(*, exclude_pids: set[int] | None = None) -> list[tuple[int, str]]:
    """Find live web servers using ledger identity and structural argv matching.

    Registered servers remain visible when the process table is unreadable. Lifecycle
    probes and Desktop-owned excluded PIDs must never enter the reaper's result.
    """
    skip = {os.getpid(), *(exclude_pids or ())}
    positive = _ledger_web_server_processes()
    try:
        found = [(pid, cmd) for pid, cmd in _iter_process_table()
                 if pid not in skip and (pid in positive or _is_hermes_web_server_command(cmd))
                 and not _is_dashboard_lifecycle_probe(cmd)]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        found = []
    seen = {pid for pid, _ in found} | skip
    found.extend((pid, cmd) for pid, cmd in positive.items() if pid not in seen)
    return found


def _pid_environ(pid: int) -> dict[str, str] | None:
    """Exec-time environment of *pid* (psutil, then /proc); ``None`` when unreadable."""
    with contextlib.suppress(Exception):
        import psutil
        return dict(psutil.Process(pid).environ())
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    env: dict[str, str] = {}
    for part in raw.split(b"\x00"):
        key, sep, value = part.partition(b"=")
        if sep:
            env[key.decode("utf-8", errors="replace")] = value.decode("utf-8", errors="replace")
    return env


def _hermes_home_for_pid(pid: int) -> str | None:
    """The Hermes home *pid* runs on, tri-state: ``None`` ONLY when its environment is unreadable
    (another user, hardened ``/proc``) — callers spare those, never guess.

    A readable environment always resolves, replaying ``_apply_profile_override`` on the target's
    exec-time env + argv (``hermes -p X serve`` rewrites ``HERMES_HOME`` in ``os.environ`` AFTER
    startup, which ``/proc/<pid>/environ`` never reflects): a profile-shaped ``HERMES_HOME``
    without a flag is the home; otherwise the root is ``HERMES_HOME`` (its grandparent when
    profile-shaped) or the platform default of the process's own ``HOME`` / ``LOCALAPPDATA``, and
    the profile is the ``--profile``/``-p`` flag, else the root's sticky ``active_profile`` unless
    the process has a fixed identity (supervised child, post-swap updater, Desktop SSH backend).
    """
    env = _pid_environ(pid)
    if env is None:
        return None
    from hermes_cli.main_dashboard import _dashboard_cmdline_for_pid
    from hermes_cli.profiles import get_active_profile, normalize_profile_name, profile_root_for_env_home
    argv = _dashboard_cmdline_for_pid(pid) or []
    env_home = env.get("HERMES_HOME", "").strip()
    profile = _profile_flag_value(argv)
    if profile is None and env_home and (
        Path(env_home).parent.name == "profiles" or env.get("HERMES_UPDATE_POST_SWAP") == "1"
    ):
        return env_home
    if sys.platform == "win32":
        local_appdata = env.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path(env.get("USERPROFILE") or Path.home()) / "AppData" / "Local"
        default_home = base / "hermes"
    else:
        default_home = Path(env.get("HOME") or Path.home()) / ".hermes"
    root = profile_root_for_env_home(env_home, default_home)
    fixed_identity = any(env.get(k) for k in ("HERMES_SUPERVISED_CHILD", "HERMES_S6_SUPERVISED_CHILD",
                                               "HERMES_GATEWAY_EXTERNAL_SUPERVISOR")) or "--ssh-session-token-file" in argv
    if profile is None and not fixed_identity:
        profile = get_active_profile(root)
    canon = normalize_profile_name(profile) if profile else "default"
    return str(root) if canon == "default" else str(root / "profiles" / canon)


def _dashboard_subcommand_index(argv: list[str]) -> int | None:
    return next((i for i, tok in enumerate(argv) if tok in ("serve", "dashboard", "webapp")), None)


def _profile_flag_value(argv: list[str]) -> str | None:
    """Value of the first ``--profile X`` / ``-p X`` / ``--profile=X`` in *argv*."""
    for i, tok in enumerate(argv):
        if tok in ("--profile", "-p") and i + 1 < len(argv):
            return str(argv[i + 1])
        if tok.startswith("--profile="):
            return tok.split("=", 1)[1]
    return None


def _is_ephemeral_port_zero_backend(argv: list[str]) -> bool:
    """True for Desktop-style ``serve|dashboard|webapp --port 0`` backends — replaying them after
    ``hermes update`` multiplies listening backends because ``--port 0`` binds a fresh port.

    See #78821.
    """
    if _dashboard_subcommand_index(argv) is None:
        return False
    return any((tok == "--port" and i + 1 < len(argv) and str(argv[i + 1]) == "0")
               or (tok.startswith("--port=") and tok.split("=", 1)[1].strip() == "0")
               for i, tok in enumerate(argv))


def _normalize_dashboard_cmdline(argv: list[str]) -> tuple[str, ...]:
    """Collapse argv to profile flags + web-server subcommand tail for dedupe."""
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


def _pids_owned_by_hermes_home(pids: list[int], home: str) -> list[int]:
    """Return only *pids* whose resolved Hermes home (``_hermes_home_for_pid``) is ``home``.

    Dashboard argv is discovery-only: it is not an ownership proof because
    several Hermes installs and profiles can run the same command on one
    machine.  An unreadable process environment is deliberately not treated
    as a match, so a stop request fails closed rather than taking down an
    unrelated backend.
    """
    target = _normalized_home_for_compare(home)
    return [
        pid for pid in pids
        if (pid_home := _hermes_home_for_pid(pid))
        and _normalized_home_for_compare(pid_home) == target
    ]


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


#: Executables that only *carry* a hermes command line. A process headed by one of these
#: never serves traffic itself; when its argv matches the dashboard patterns it is a
#: wrapper around the command (``bash -c 'hermes dashboard --stop'``), not a backend.
_WRAPPER_HEAD_COMMANDS = frozenset({
    "ash", "bash", "csh", "dash", "fish", "ksh", "sh", "tcsh", "zsh",
    "env", "nohup", "nice", "stdbuf", "timeout", "watch", "xargs",
    "screen", "tmux", "sudo",
})


def _caller_ancestor_pids() -> set[int]:
    """PIDs of THIS process's ancestors (self excluded), best-effort; empty on any failure.

    ``--stop`` and the update sweep must never kill the process tree they were invoked
    from. psutil is primary; the ``/proc`` walk keeps the answer when psutil is unusable.
    """
    try:
        import psutil

        return {p.pid for p in psutil.Process().parents()}
    except Exception:
        pass
    ancestors: set[int] = set()
    cur = os.getpid()
    for _ in range(2048):  # cycle / corrupt-PPid guard
        try:
            status_text = Path(f"/proc/{cur}/status").read_text(
                encoding="utf-8", errors="replace")
            for line in status_text.splitlines():
                if line.startswith("PPid:"):
                    cur = int(line.split()[1])
                    break
            else:
                return ancestors
        except (OSError, ValueError, IndexError):
            return ancestors
        if cur <= 1:
            return ancestors
        ancestors.add(cur)
    return ancestors


def _argv_head_command(pid: int) -> str | None:
    """Basename of *pid*'s first argv token, best-effort; ``None`` when unreadable."""
    try:
        import psutil

        argv = psutil.Process(pid).cmdline()
        if argv:
            return os.path.basename(str(argv[0]))
    except Exception:
        pass
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    head = raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
    return head.rsplit("/", 1)[-1] or None


def _is_caller_wrapper_shell(pid: int, ancestors: set[int]) -> bool:
    """True when *pid* is a caller ancestor headed by a wrapper executable.

    Root selection is a substring match, so the shell a ``--stop`` was typed into (or a
    ``bash -c 'hermes dashboard --stop'`` wrapper) matches on its own argv. Ancestor alone
    is not a spare: the backend hosting a shell-escaped TUI is also the caller's ancestor
    and must stay stoppable — only a wrapper-headed ancestor is spared.
    """
    if pid not in ancestors:
        return False
    return (_argv_head_command(pid) or "") in _WRAPPER_HEAD_COMMANDS


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


# SIGTERM → SIGKILL grace for the dashboard/serve backend. Must outlast the lifespan teardown in
# hermes_cli/web_server.py::_lifespan: stop_hosted_room_service(timeout=5.0) + the startup-thread
# join(1.0) + PTY_REGISTRY.close_all() (≤1.5s per attached Chat PTY, serial). A SIGKILL inside
# that window skips close_all(), so the ui-tui / tui_gateway.entry children outlive the backend
# and keep the deleted state.db-wal inode open — the next hermes start refuses with a FATAL
# DeletedWalGenerationError (#111912). The orphan reaper's 1.5s (`_reap_orphaned_desktop_local_serves`)
# is deliberately shorter: it runs on the Desktop boot path under a 10s ready-probe.
_POSIX_TERM_GRACE_SECONDS = 10.0
# Grace for a descendant that outlived the backend's own teardown. It already got the backend's
# SIGTERM forwarded (or SIGHUP from its PTY master closing); anything still up is wedged, and a
# wedged ui-tui keeps the deleted state.db-wal inode open until the next start refuses with
# DeletedWalGenerationError (#112631) — no finite root grace can cover an unbounded teardown.
_POSIX_DESCENDANT_GRACE_SECONDS = 2.0
_NO_TTY = ("?", "??", "-")  # Linux / macOS / BSD spellings of "no controlling terminal"


def _is_detached_session_leader(pid: int, tty: str) -> bool:
    """True for a process the dashboard launched with ``start_new_session`` (own session, no tty).

    Messaging-gateway bots and profile actions started from ``/api/gateway/*`` are such processes:
    they are the user's, not the dashboard's, and must survive a dashboard stop. A hosted
    ``hermes --tui`` child is a session leader too (``pty.fork``) but owns the pts whose master the
    dashboard held, so its tty column is set and it stays in the sweep.

    Known gap: the turn-isolation ``tui_gateway.compute_host`` and ``slash_worker`` children are
    launched the same way (``start_new_session=True``, no tty; ``compute_host`` also holds
    ``state.db``), so a wedged WAL holder of that class is spared here too. Their exit relies on
    their own ppid watchdogs (``compute_host._parent_guard_loop``,
    ``slash_worker._start_parent_death_watchdog``), not on this sweep. Discriminating by WAL-holder
    identity (``iter_deleted_sqlite_sidecar_holders``) was deliberately not done in this change.
    """
    if tty not in _NO_TTY:
        return False
    try:
        return os.getsid(pid) == pid
    except OSError:
        return False


def _posix_descendants(roots: list[int]) -> dict[int, tuple[int, int | None]]:
    """``{pid: (root, start_time)}`` of every dashboard-owned descendant of *roots*, snapshotted
    BEFORE the kill: once the root dies its children are reparented and the PPID link is gone.
    Detached session leaders (see ``_is_detached_session_leader``) are pruned together with their own
    subtrees. So is the calling process with its subtree and its ancestor chain: ``hermes dashboard
    --stop`` / ``hermes update`` run from a shell escape inside the hosted Chat TUI are same-session
    descendants of the backend, and sweeping them would SIGTERM the caller mid-run (POSIX twin of the
    Windows #98814 hazard). The start-time fingerprint is the PID-reuse guard (same one
    ``_kill_pids_windows`` uses). Empty on scan failure → root-only kill, the historical behaviour.
    """
    from gateway.status import get_process_start_time
    try:
        result = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,tty="], timeout=10, **_PS_RUN_KWARGS)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {}
    children: dict[int, list[tuple[int, str]]] = {}
    parent: dict[int, int] = {}
    for line in (result.stdout or "").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append((int(parts[0]), parts[2]))
            parent[int(parts[0])] = int(parts[1])
    me = os.getpid()
    ancestors: set[int] = set()
    cur = me
    while (cur := parent.get(cur, 0)) > 1 and cur not in ancestors:
        ancestors.add(cur)
    found: dict[int, tuple[int, int | None]] = {}
    pending = [(root, root) for root in roots]
    while pending:
        root, cur = pending.pop()
        for pid, tty in children.get(cur, ()):
            if pid in found or pid in roots or pid == me or _is_detached_session_leader(pid, tty):
                continue
            if pid not in ancestors:  # an ancestor of the caller is spared, but its other children are not
                found[pid] = (root, get_process_start_time(pid))
            pending.append((root, pid))
    return found


def _wait_gone(pids: list[int], seconds: float) -> list[int]:
    """Poll up to *seconds*; return the PIDs still alive (zombies count as gone)."""
    import time as _time

    from gateway.status import _pid_exists

    deadline = _time.monotonic() + seconds
    alive = list(pids)
    while alive and _time.monotonic() < deadline:
        _time.sleep(0.1)
        alive = [p for p in alive if _pid_exists(p)]  # os.kill(pid, 0) breaks on Windows
    return alive


def _kill_pids_posix(pids: list[int], killed: list[int], failed: list[tuple[int, str]]) -> None:
    """SIGTERM, wait up to ``_POSIX_TERM_GRACE_SECONDS`` for graceful exit, SIGKILL survivors, then
    sweep the dashboard-owned descendants that outlived the root and wait for the tree to be gone.

    *killed* reports the roots only; swept descendants are the roots' own teardown debt. A descendant
    still alive after its SIGKILL grace is appended to *failed*: the stop must not be declared
    complete while a wedged ui-tui still holds the deleted state.db-wal inode (#112631).
    """
    import signal as _signal

    from gateway.status import get_process_start_time

    descendants = _posix_descendants(pids)

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
    pending = [p for p in pids if p not in killed and p not in {f[0] for f in failed}]
    alive = _wait_gone(pending, _POSIX_TERM_GRACE_SECONDS)
    killed.extend(p for p in pending if p not in alive)
    for pid in alive:
        _send(pid, _signal.SIGKILL)

    # Snapshot identity must still match: a PID recycled during the grace is not ours to signal.
    survivors = [p for p, (_root, start) in descendants.items()
                 if start is not None and get_process_start_time(p) == start]
    for sig in (_signal.SIGTERM, _signal.SIGKILL):
        for pid in survivors:
            with contextlib.suppress(OSError):
                os.kill(pid, sig)
        survivors = _wait_gone(survivors, _POSIX_DESCENDANT_GRACE_SECONDS)
    failed.extend((pid, "descendant of the stopped backend still alive after SIGKILL")
                  for pid in survivors)


def _kill_stale_dashboard_processes(
    reason: str = "the running backend no longer matches the updated frontend", *,
    restart_managed: bool = False, already_restarted_units: "set[str] | None" = None,
    include_pids: set[int] | None = None,
    scope_home: str | None = None,
) -> dict[str, list]:
    """Kill running ``hermes dashboard`` / ``hermes serve`` processes (update end, ``--stop``).

    With ``restart_managed`` (update only) systemd-owned PIDs get their unit restarted after the
    kill (systemd treats our SIGTERM as a clean stop, so ``Restart=on-failure`` never fires) and
    manual PIDs are respawned from captured argv. PIDs owned by *already_restarted_units* (no
    ``.service`` suffix) are left untouched, not killed twice.

    When *scope_home* is supplied, only processes with that exact live
    ``HERMES_HOME`` are candidates; unknown ownership fails closed. This is
    used by ``dashboard --stop`` and the per-profile update cleanup.

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
    pids = _dash._find_stale_dashboard_pids(exclude_pids=exclude or None, scope_home=scope_home)
    if include_pids is not None:
        pids = [pid for pid in pids if pid in include_pids]
    if not pids:
        return _empty_result()
    # Snapshot systemd unit/cgroup and argv BEFORE killing (the cgroup dies with the process).
    pid_cgroup: dict[int, str | None] = {}
    pid_service: dict[int, str | None] = {}
    pid_launchd: dict[int, tuple[str, str, int | None]] = {}
    pid_cmdline: dict[int, list[str]] = {}
    pid_home: dict[int, str | None] = {}
    # macOS: a backend supervised by a launchd job (LaunchAgent / LaunchDaemon) must come back
    # through launchd, never as a detached argv respawn — the respawn holds the job's port, the
    # job then fails every KeepAlive restart with "port already in use", and the running backend
    # is left unsupervised. Snapshot the loaded jobs once, before the kill; ``--stop`` reads them
    # too, so it can say that a KeepAlive job will undo the stop.
    launchd_jobs = _dash._loaded_launchd_backend_jobs() if sys.platform != "win32" else []

    def _launchd_owner(pid: int, cmdline: list[str] | None):
        return _dash._launchd_job_owning_backend(pid, cmdline, launchd_jobs, ancestors=_process_ancestors(pid))

    if restart_managed and sys.platform != "win32":
        for pid in pids:
            pid_cgroup[pid] = _dash._get_pid_cgroup_path(pid)
            pid_service[pid] = _dash._get_systemd_service_for_pid(pid)
            if pid_service[pid]:
                continue
            cmdline = _dash._dashboard_cmdline_for_pid(pid)
            if launchd_jobs and (job := _launchd_owner(pid, cmdline)):
                pid_launchd[pid] = job
            elif cmdline:
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
    elif launchd_jobs:
        for pid in pids:
            if job := _launchd_owner(pid, _dash._dashboard_cmdline_for_pid(pid)):
                pid_launchd[pid] = job
    print(f"\n⟲ Stopping {len(pids)} Hermes web server process(es) ({reason})")
    killed: list[int] = []
    failed: list[tuple[int, str]] = []
    (_kill_pids_windows if sys.platform == "win32" else _kill_pids_posix)(pids, killed, failed)
    for pid in killed:
        print(f"    ✓ stopped PID {pid}")
    for pid, err_msg in failed:
        print(f"    ✗ failed to stop PID {pid}: {err_msg}")
    if killed and restart_managed:
        unrecovered = _restart_killed_backends(
            killed, pid_service, pid_cgroup, pid_cmdline, pid_home, pid_launchd=pid_launchd)
    else:
        unrecovered = list(killed)
        # A stopped launchd job with KeepAlive restarts itself: say so instead of a misleading
        # "restart it yourself" hint, and give the command that actually keeps it down.
        for target in sorted({f"{d}/{l}" for p in killed if (j := pid_launchd.get(p)) for d, l, _ in (j,)}):
            print(f"  ⚠ PID(s) supervised by launchd job {target}: a KeepAlive job restarts itself.\n"
                  f"    To keep it down: launchctl bootout {target}")
        if any(p not in pid_launchd for p in killed):
            print("  Restart the browser surface or backend when you're ready:\n    hermes dashboard|webapp|serve --port <port>")
    return {"matched": list(pids), "killed": list(killed), "failed": list(failed),
            "unrecovered": list(unrecovered)}


def _restart_killed_backends(
    killed: list[int], pid_service: dict[int, str | None], pid_cgroup: dict[int, str | None],
    pid_cmdline: dict[int, list[str]], pid_home: dict[int, str | None], *,
    pid_launchd: dict[int, tuple[str, str, int | None]] | None = None) -> list[int]:
    """Update path: restart systemd units, kickstart launchd jobs (macOS), respawn manual argv
    (detached, headless, logged to logs/dashboard-restart.log; one per profile, no ``--port 0``).
    Returns PIDs not brought back."""
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
        launchd_job = (pid_launchd or {}).get(pid)
        if svc_name:
            if svc_name in seen_services:
                continue
            seen_services.add(svc_name)
            if _dash._try_restart_systemd_service(svc_name, pid_cgroup.get(pid)):
                print(f"    ✓ restarted systemd service {svc_name}")
            else:
                failed_restarts.append((svc_name, "systemctl restart returned non-zero"))
                unrecovered.append(pid)
        elif launchd_job:
            # launchd owns the backend: the job brings it back (KeepAlive, or the kickstart below),
            # and success means launchd reports a fresh supervised PID — an argv respawn would sit
            # on the job's port and leave it failing forever.
            domain, label, old_pid = launchd_job
            target = f"{domain}/{label}"
            if target in seen_services:
                continue
            seen_services.add(target)
            if _dash._restart_launchd_job(domain, label, old_pid):
                print(f"    ✓ restarted launchd job {target}")
            else:
                # A LaunchDaemon (system domain) can only be kickstarted by root; the hint must
                # be the command that works from the shell the operator is actually in.
                sudo = "sudo " if domain.startswith("system") and os.geteuid() != 0 else ""  # windows-footgun: ok — launchd jobs exist only on macOS
                failed_restarts.append(
                    (target, f"launchd is not supervising a fresh process; run: {sudo}launchctl kickstart -k {target}"))
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


def _process_ancestors(pid: int, *, max_depth: int = 8) -> list[int]:
    """Parent chain of *pid* (nearest first), stopping at init / a lookup failure / *max_depth*."""
    chain: list[int] = []
    current = pid
    while len(chain) < max_depth:
        parent = _process_ppid(current)
        if parent is None or parent <= 1 or parent == pid or parent in chain:
            break
        chain.append(parent)
        current = parent
    return chain


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


def _remote_lock_roots(base_dir: Path | None) -> list[Path]:
    """Every dir the Desktop may have written ``desktop-ssh/<ownershipId>/backend.lock.json`` under.

    The Desktop writes SSH locks beneath the ROOT home (``~/.hermes/desktop-ssh``), but a profile
    backend (``hermes --profile X serve``) runs with ``HERMES_HOME=<root>/profiles/X`` — scanning only
    the process home found no lock there and its reaper killed the sibling profile's live SSH
    backend on every profile switch (#89811)."""
    if base_dir is not None:
        return [base_dir]
    from hermes_constants import get_default_hermes_root
    roots: list[Path] = []
    for home in (_hermes_home_dir(), get_default_hermes_root()):
        root = home / _REMOTE_LOCK_SUBDIR
        if root not in roots:
            roots.append(root)
    return roots


def _lock_owned_serve_pids(base_dir: Path | None = None) -> set[int]:
    """PIDs claimed by valid ``{hermes_home}/desktop-ssh/<ownershipId>/backend.lock.json`` records
    (best-effort: a bad record contributes no PID; never raises)."""
    import json
    owned: set[int] = set()
    entries: list[Path] = []
    for root in _remote_lock_roots(base_dir):
        try:
            entries.extend(root.iterdir() if root.is_dir() else [])
        except OSError:
            continue
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
    descendants = _posix_descendants(matched)  # before the kill: the root's death reparents them
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
    # A SIGKILLed backend never ran PTY_REGISTRY.close_all(): its hosted ui-tui / MCP trees would
    # keep the deleted state.db-wal inode open (#112631). The boot-path budget leaves no second
    # grace, and these trees already lost their Electron and their backend.
    # A root whose own kill raised (EPERM: not ours) keeps its subtree — do not orphan it half-way.
    from gateway.status import get_process_start_time
    for pid, (root, start) in descendants.items():
        if root not in failed and start is not None and get_process_start_time(pid) == start:
            with contextlib.suppress(OSError):
                os.kill(pid, signal_kill)
    with contextlib.suppress(Exception):
        print(f"⟲ Reaped {len(killed)} orphaned desktop-local serve backend(s) ({reason}): {killed or matched}")
    return {"matched": matched, "killed": killed, "failed": failed}

"""Windows gateway lifecycle for ``hermes update``: pause/resume/cold-start the service, sweep venv holders, reap orphaned backends.

Split out of ``update_cmd.py``; names are re-imported there so ``hermes_cli.update_cmd.<name>`` still resolves/monkeypatches.
Origin helpers are imported lazily per function (no cycle; test patches on the origin stay effective).
"""

import logging
from contextlib import contextmanager, suppress
import os
import re
import shlex
import subprocess
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

from hermes_cli.update_cmd_common import _best_effort

logger = logging.getLogger("hermes_cli.update_cmd")  # log-record parity with the origin module

_BACKEND_PURPOSES = ("serve", "dashboard")


def _try_call(fn, log_message: str, *log_args, default=None):
    """``fn()``, or *default* after logging the exception at debug (``log_message`` gets ``*log_args, exc``)."""
    try:
        return fn()
    except Exception as exc:
        logger.debug(log_message, *log_args, exc)
        return default


@contextmanager
def _abort_on_error(prefix: str):
    """Re-raise any failure of a mandatory step as ``RuntimeError(f"{prefix}: {exc}")`` (chained)."""
    try:
        yield
    except Exception as exc:
        raise RuntimeError(f"{prefix}: {exc}") from exc


def _write_update_planned_stop_marker(profile_path: Path, pid: int) -> bool:
    """Write a planned-stop marker into a specific profile home."""
    try:
        from gateway.status import _get_process_start_time
        from utils import atomic_json_write
        atomic_json_write(
            Path(profile_path) / ".gateway-planned-stop.json",
            {"target_pid": pid, "target_start_time": _get_process_start_time(pid), "stopper_pid": os.getpid(),
             "written_at": datetime.now(timezone.utc).isoformat()},
            indent=None, separators=(",", ":"),
        )
        return True
    except (OSError, PermissionError):
        return False


def _wait_for_windows_update_gateway_exit(pids: list[int], *, timeout: float) -> set[int]:
    """Wait for the given gateway PIDs to exit, returning survivors."""
    if not pids:
        return set()
    from gateway.status import _pid_exists

    def _alive(pid: int) -> bool:
        try:
            return bool(_pid_exists(pid))
        except Exception:
            return False

    remaining = set(pids)

    def _all_gone() -> bool:
        nonlocal remaining
        remaining = {pid for pid in remaining if _alive(pid)}
        return not remaining

    _poll_until(_all_gone, max(timeout, 0.0), 0.25)
    return {pid for pid in remaining if _alive(pid)}


def _self_and_non_gateway_ancestor_pids(psutil) -> set[int]:
    """PIDs a venv-holder scan must never nominate: this process and its non-gateway ancestry.

    Do NOT blanket-exclude ancestors: under ``/update`` the updater is a CHILD of the gateway, and hiding it
    dead-ends the update on ``venv-blocked``. Gateway ancestors stay visible (the pause path stops them
    gracefully; a detached child survives on Windows); interactive ancestry is never a blocker."""
    _is_gw = None
    with suppress(Exception):
        # Never return ourselves or our own ancestry: a CLI ``hermes update`` runs from the venv python and
        # would otherwise nominate itself. Same #87594 carve-out as _detect_venv_python_processes: a GATEWAY
        # ancestor is not "our own ancestry" in the interactive sense — it is the process the pause
        # machinery must see (the /update-from-gateway topology makes the updater the gateway's child).
        from gateway.status import looks_like_gateway_command_line as _is_gw
    skip: set[int] = {os.getpid()}
    with suppress(Exception):
        for anc in psutil.Process().parents():
            anc_cmdline = _cmdline_or_empty(anc)
            if not (_is_gw is not None and anc_cmdline and _is_gw(anc_cmdline)):
                skip.add(int(anc.pid))
    return skip


def _cmdline_or_empty(proc) -> str:
    """Joined argv of a psutil process, ``""`` when it can't be read."""
    try:
        return " ".join(proc.cmdline() or [])
    except Exception:
        return ""


def _lower_dir_prefix(path: Path) -> str:
    """``str(path)`` lower-cased with one trailing separator, resolved when possible (prefix matching)."""
    try:
        raw = str(path.resolve())
    except OSError:
        raw = str(path)
    return raw.lower().rstrip(os.sep) + os.sep


def _psutil():
    """The ``psutil`` module, or ``None`` when it can't be imported (callers degrade, never raise)."""
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    return psutil


def _parent_is_live(proc) -> bool:
    """True when *proc* has a running parent that is not a recycled PID (a "parent" created after its child)."""
    parent = proc.parent()
    return parent is not None and parent.is_running() and parent.create_time() <= proc.create_time()


def _detect_venv_python_processes(*, exclude_pids: set[int] | None = None) -> list[tuple[int, str, str]]:
    """Live processes running from the project venv's interpreter as ``(pid, name, cmdline)``; never raises.

    The hermes.exe shim guard misses the Desktop backend and anything off ``venv\\Scripts\\python(w).exe``;
    they keep ``.pyd`` files mapped so a mid-update dependency sync dies half-way. Empty off-Windows / without
    psutil; self + non-gateway ancestors excluded. cmdline/cwd are expensive per process on Windows (500+
    procs can blow the Desktop preflight watchdog), so they are fetched lazily for plausible candidates only.
    The FULL cmdline is kept: callers parse it (the pausable-gateway exemption looks for ``gateway run``).
    """
    from hermes_cli.update_cmd import _m
    psutil = _psutil()
    if not _m()._is_windows() or psutil is None:
        return []
    venv_prefix = _lower_dir_prefix(_m().PROJECT_ROOT / "venv")
    root_prefix = _lower_dir_prefix(_m().PROJECT_ROOT)
    skip = set(exclude_pids or set()) | _self_and_non_gateway_ancestor_pids(psutil)
    matches: list[tuple[int, str, str]] = []
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
        if not exe or pid is None or int(pid) in skip:
            continue
        try:
            exe_norm = str(Path(exe).resolve()).lower()
        except (OSError, ValueError):
            exe_norm = str(exe).lower()
        # Primary match: exe lives under this venv (desktop backend / gateway case).
        in_venv = exe_norm.startswith(venv_prefix)
        name = str(info.get("name") or Path(exe).name)
        name_low = name.lower()
        if not (in_venv or name_low.startswith(("python", "pypy")) or name_low in {"uv.exe", "uvx.exe", "hermes.exe"}):
            continue
        cmdline_raw = _cmdline_or_empty(proc)
        cmdline_low = cmdline_raw.lower()
        # Fallback: uv/base-interpreter trampolines have an exe OUTSIDE the venv yet hold
        # its .pyd files — match cmdline (venv path, or `-m hermes_cli.main` + root/cwd).
        if in_venv or venv_prefix in cmdline_low or (
            "hermes_cli.main" in cmdline_low and (root_prefix in cmdline_low or _cwd_prefix(proc).startswith(root_prefix))
        ):
            matches.append((int(pid), name, cmdline_raw))
    return matches


def _cwd_prefix(proc) -> str:
    """Lower-cased cwd of *proc* with one trailing separator; bare ``os.sep`` when unreadable."""
    try:
        return str(proc.cwd() or "").lower().rstrip(os.sep) + os.sep
    except Exception:
        return os.sep


_HOLDER_VALUE_FLAGS_FALLBACK = frozenset({
    "--profile", "-p", "--config", "--model", "-m", "--provider", "--reasoning", "--toolsets", "-t",
    "--skills", "-s", "--continue", "-c", "--resume", "-r", "--oneshot", "-z", "--in", "--usage-file",
})
_holder_value_flags_cache: frozenset | None = None


def _holder_value_flags() -> frozenset:
    """Top-level CLI flags that consume a value, from the REAL parser (nargs != 0); cached per process.

    Derived so the holder classifier can't drift from argparse (a handwritten subset misparsed ``--reasoning high
    serve``). Pre-argparse profile selectors are added explicitly (stripped before argparse sees argv). Falls back
    to a static snapshot when the parser can't import — the updater must classify holders even on a broken tree.

    Introspects ``build_top_level_parser()`` (every option with nargs != 0) so the holder classifier can
    never drift from the argparse surface (#91869 review: a handwritten subset misparsed ``--reasoning high
    serve`` as subcommand ``high`` and ``-m dashboard serve`` as ``dashboard`` — recreating the wrong-hint
    class).
    """
    global _holder_value_flags_cache
    if _holder_value_flags_cache is not None:
        return _holder_value_flags_cache
    flags: set[str] = {"--profile", "-p", "--config"}
    try:
        from hermes_cli._parser import build_top_level_parser
        for action in build_top_level_parser()[0]._actions:
            if action.option_strings and action.nargs != 0:
                flags.update(action.option_strings)
        _holder_value_flags_cache = frozenset(flags)
    except Exception:
        _holder_value_flags_cache = _HOLDER_VALUE_FLAGS_FALLBACK
    return _holder_value_flags_cache


def _hermes_holder_subcommand(cmdline: str) -> str | None:
    """The actual Hermes SUBCOMMAND a venv-holder argv runs, or None (callers must NOT guess a label).

    Token-based, never substring (``kanban --preserve-cache`` contains "serve"): find the ``hermes_cli.main`` /
    ``hermes(.exe)`` entry token, return the first following token that isn't a flag or a flag's value.

    Profile selectors (``--profile X``, ``-p X``) are skipped like the canonical gateway matcher does. See
    #90778.
    """
    try:
        tokens = shlex.split(cmdline, posix=False)
    except Exception:
        tokens = cmdline.split()

    def _is_entry(i: int, token: str) -> bool:
        low = token.lower().strip('"')
        return (low.endswith("hermes_cli.main") and i > 0 and tokens[i - 1] == "-m") or (
            low.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] in ("hermes", "hermes.exe"))

    entry_idx = next((i for i, token in enumerate(tokens) if _is_entry(i, token)), None)
    if entry_idx is None:
        return None
    value_flags = _holder_value_flags()
    i = entry_idx + 1
    while i < len(tokens):
        token = tokens[i]
        if token in value_flags or token.split("=", 1)[0] in value_flags:
            i += 1 if "=" in token else 2  # --flag value consumes two tokens; --flag=value one.
        elif token.startswith("-"):
            i += 1
        else:
            return token.lower()
    return None


def _format_venv_python_holders_message(matches: list[tuple[int, str, str]]) -> str:
    """Explain which venv processes block the update and how to clear them.

    Labels come from the parsed SUBCOMMAND, never substring: a standalone ``hermes dashboard`` must not be
    called the Desktop backend, ``--preserve-cache`` must not match "serve". Unknown argv gets no hint.

    See #90778.
    """
    hint_by_subcommand = {
        "serve": "  ← Hermes backend (if the Desktop app is open, close it)",
        "dashboard": "  ← hermes dashboard (stop it: hermes dashboard stop, or close that terminal)",
        "gateway": "  ← gateway",
    }
    lines = ["✗ Other Hermes processes are running from this install's venv:"]
    for pid, name, cmdline in matches[:6]:
        hint = hint_by_subcommand.get(_hermes_holder_subcommand(cmdline) or "", "")
        lines.append(f"  PID {pid}  {name}  {cmdline[:120]}{hint}")
    if len(matches) > 6:
        lines.append(f"  ... and {len(matches) - 6} more")
    lines.append(
        "\n  On Windows these keep native extension files (.pyd) locked, so the\n"
        "  dependency update would fail partway and leave a broken install.\n"
        "  Close the Hermes desktop app / other Hermes terminals, then re-run:\n    hermes update\n"
        "  (or use `hermes update --force-venv` to proceed anyway at your own risk)"
    )
    return "\n".join(lines)


def _venv_launcher_ancestors(pids: list[int]) -> list[int]:
    """Venv-interpreter parents of *pids* that hold the install open; never raises.

    A shim-started gateway is a chain: ``venv\\Scripts\\python.exe`` launcher (keeps ``.pyd`` mapped) -> uv
    CPython worker (writes the PID file). The pause set sees the worker, the venv scan sees the launcher, so a
    paused gateway still tripped the guard. One hop up only, venv-prefixed only (bounds blast radius)."""
    from hermes_cli.update_cmd import _m
    psutil = _psutil()
    if not _m()._is_windows() or not pids or psutil is None:
        return []
    venv_prefix = _lower_dir_prefix(_m().PROJECT_ROOT / "venv")
    skip = _self_and_non_gateway_ancestor_pids(psutil) | set(pids)
    found: list[int] = []
    for pid in pids:
        with suppress(Exception):
            parent = psutil.Process(int(pid)).parent()
            if parent is None:
                continue
            ppid = int(parent.pid)
            if ppid not in skip and ppid not in found and (parent.exe() or "").lower().startswith(venv_prefix):
                found.append(ppid)
    return found


def _leftover_pausable_gateway_pids(matches: list[tuple[int, str, str]]) -> list[int] | None:
    """PIDs from *matches* when EVERY remaining venv holder is a pausable gateway, else ``None`` (keep refusing).

    A gateway respawned inside the pause->guard window (or via an unmapped spawn path) still holds ``.pyd`` files.
    Uses the Desktop preflight's ``_is_pausable_gateway`` so exemption and tolerance cannot drift; live argv is
    re-read via psutil when possible since the scan may hold only a cmdline prefix."""
    from hermes_cli._scan_venv_blockers import _is_pausable_gateway
    psutil = _psutil()
    pids: list[int] = []
    for pid, _name, cmdline in matches:
        argv = cmdline
        if psutil is not None:
            with suppress(Exception):
                argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
        if not _is_pausable_gateway(argv):
            return None
        pids.append(int(pid))
    return pids


def _refuse_gateway_ancestor_tree_kill(pids: list[int], *, gateway_mode: bool) -> bool:
    """Refuse a plain Windows update that would tree-kill its own ancestry (a chat agent's ``hermes update`` is
    a gateway child; ``taskkill /T /F`` kills the updater first). ``--gateway`` is exempt (detached delivery).
    Refuse only when a nominated gateway is positively an ancestor; unknown ancestry keeps existing recovery.

    The leftover holder recovery below uses ``taskkill /T /F`` on Windows, so force-stopping that gateway
    also kills the updater before it can mutate the checkout (#98814).
    """
    if gateway_mode or not pids:
        return False
    def _ancestors():
        from hermes_cli.gateway import _is_pid_ancestor_of_current_process
        return [int(pid) for pid in pids if _is_pid_ancestor_of_current_process(int(pid))]

    ancestors = _try_call(_ancestors, "Could not inspect gateway ancestry before tree-kill: %s")
    if not ancestors:
        return False
    print(
        "✗ Refusing to stop the gateway process tree because this updater "
        f"is running inside it (gateway PID(s): {', '.join(str(pid) for pid in ancestors)}).\n"
        "  On Windows, taskkill /T would terminate the updater before the update can run.\n"
        "  From a chat platform, use `/update` instead.\n  Otherwise, run `hermes update` from a separate terminal."
    )
    return True


def _ledger_manual_serve_holders(matches: list[tuple[int, str, str]]) -> list[dict]:
    """Full ledger entries for venv holders that are MANUAL serve/dashboard backends.

    Positive identity only: self-registered purpose serve/dashboard, live (pid, create_time), recorded spawner
    NOT alive (a Desktop-owned backend keeps its live Electron spawner and must keep the refusal — the app would
    respawn what we kill). Full entries let the relauncher rebuild from host/port/profile, not argv."""
    try:
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead
    except Exception:
        return []
    holder_pids = {int(pid) for pid, _name, _cmd in matches}
    return [
        entry for entry in ledger_entries()
        if entry.get("purpose") in _BACKEND_PURPOSES and isinstance(entry.get("pid"), int) and entry["pid"] in holder_pids
        and spawner_is_dead(entry) is not False  # False = live Desktop supervisor owns it; keep refusing
    ]


def _serve_relaunch_commands(entries: list[dict]) -> list[list[str]]:
    """Rebuild launch commands for stopped serves from ledger host/port/profile — never argv parsing
    (joined argv cannot round-trip Windows paths with spaces). Entries without a port are skipped."""
    from hermes_cli.update_cmd import _m
    hermes = "hermes"
    with suppress(Exception):
        scripts_dir = _m()._venv_scripts_dir()
        if scripts_dir is not None:
            hermes = next((str(scripts_dir / n) for n in ("hermes.exe", "hermes") if (scripts_dir / n).is_file()), hermes)
    commands: list[list[str]] = []
    for entry in entries:
        port = entry.get("port")
        if not isinstance(port, int) or port <= 0:
            continue
        profile, host = str(entry.get("profile") or ""), str(entry.get("host") or "")
        commands.append(
            [hermes] + (["--profile", profile] if profile and profile != "default" else [])
            + [str(entry.get("purpose"))] + (["--host", host] if host else []) + ["--port", str(port)]
        )
    return commands


def _relaunch_stopped_serves(token: dict) -> None:
    """Idempotent atexit relaunch of manual serves stopped by the venv guard.

    `pending` flips False on first invocation so explicit call + atexit registration cannot double-spawn."""
    from hermes_cli.update_cmd import _m, _record_update_step
    if not token.get("pending"):
        return
    token["pending"] = False
    entries = token.get("entries") or []
    if not entries:
        return
    commands = _serve_relaunch_commands(entries)
    skipped = len(entries) - len(commands)
    failed: list = []
    if commands:
        print("  ⟲ Relaunching stopped serve/dashboard backend(s)")
        failed = _m()._respawn_dashboard_processes(commands)
    if skipped or failed:
        print("  ⚠ Some stopped backends could not be relaunched automatically; restart them manually (hermes serve --host <ip> --port <port>).")
    _record_update_step(
        "serve_relaunch", not failed and not skipped,
        f"relaunched={len(commands) - len(failed)} failed={len(failed)} skipped={skipped}",
    )


def _is_backend_argv(argv_low: str) -> bool:
    """Whether a lower-cased argv is a Desktop backend (``hermes_cli.main`` running ``serve``/``dashboard``)."""
    return "hermes_cli.main" in argv_low and (" serve" in argv_low or " dashboard" in argv_low)


def _live_argv_low(psutil, pid, cmdline: str) -> str | None:
    """Current lower-cased argv of *pid* (falls back to the scanned *cmdline*); ``None`` if it exited."""
    argv = cmdline
    try:
        argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
    except psutil.NoSuchProcess:
        return None
    except Exception:
        pass
    return argv.lower()


def _orphaned_desktop_backend_pids(matches: list[tuple[int, str, str]]) -> list[tuple[int, int]] | None:
    """``(pid, start_time)`` roots from *matches* when every remaining holder is an ORPHANED backend, else ``None``.

    Killing a Desktop-owned ``serve`` is futile (the app respawns it), but a straggler whose Desktop is gone
    would dead-end the update with "Hermes is still running" and zero open windows. Qualifies only if cmdline
    is a Hermes backend AND the parent is demonstrably gone (PID missing or reused). Tree-aware: holders inside
    an accepted root's tree fold into it; only roots are returned (``taskkill /T`` reaps descendants). Any
    live-parent backend, unjustified non-backend, unprovable case, or no psutil -> ``None``. Never raises.

    The venv-holder guard refuses on the Desktop app's ``serve`` backend by design: while the Desktop is
    open, killing its backend is futile (the app supervises and respawns it within seconds), so the user
    must close the app. But in the GUI-updater handoff path the Desktop has *already exited* — by contract
    it tree-kills its backends and waits for the venv shim before spawning hermes-setup, and the
    update-in-progress marker parks any relaunched Desktop from spawning a fresh backend (#50238). A
    ``serve`` backend still holding the venv at that point is a straggler whose supervisor is gone: SIGTERM
    raced its spawn, or it belongs to a crashed window. Nothing will respawn it, and refusing on it
    dead-ends the update with "Hermes is still running" while the user stares at zero open windows (ryanc's
    2026-08-09 01:59/02:17 failures).
    """
    psutil = _psutil()
    if psutil is None:
        return None
    # Pass 1: find orphaned backend ROOTS among the holders.
    roots: list[tuple[int, int]] = []
    remaining: list[int] = []  # holders still to justify
    for pid, _name, cmdline in matches:
        low = _live_argv_low(psutil, pid, cmdline)
        if low is None:
            continue  # exited between scan and classification — nothing to reap
        if not _is_backend_argv(low):
            remaining.append(int(pid))
            continue
        try:
            proc = psutil.Process(int(pid))
            # Fingerprint from the SAME psutil handle, centisecond-quantized like
            # gateway.status.get_process_start_time so pid_is_hermes round-trips at kill time.
            process_start_time = int(round(proc.create_time() * 100))
        except psutil.NoSuchProcess:
            continue  # exited during classification — nothing to reap
        except Exception:
            return None
        try:
            ppid = proc.ppid()
            parent = psutil.Process(ppid) if ppid else None
            # PID-reuse check: a "parent" created after its child is a recycled PID. A live parent is
            # not a root but may be an orphan root's descendant (the venv trampoline re-execs uv
            # python with the SAME argv) — defer to pass 2.
            if parent is not None and parent.is_running() and parent.create_time() <= proc.create_time():
                remaining.append(int(pid))
                continue
        except psutil.NoSuchProcess:
            pass  # parent gone → orphan
        except Exception:
            return None
        roots.append((int(pid), process_start_time))
    # Pass 2: every non-backend holder must descend from an accepted orphan root
    # (dies with the tree reap); anything else keeps the refusal.
    root_set = {pid for pid, _start_time in roots}
    for pid in remaining:
        try:
            if not root_set or not root_set & {int(a.pid) for a in psutil.Process(pid).parents()}:
                return None
        except psutil.NoSuchProcess:
            continue  # exited already
        except Exception:
            return None
    return roots


def _ledger_reapable_backend_pids(matches: list[tuple[int, str, str]]) -> list[int]:
    """PIDs the spawn ledger positively identifies as orphaned backends; never raises.

    Strongest rung (no PPID/cmdline inference): qualifies when ``(pid, create_time)`` matches a live ledger entry
    (PID reuse can't forge it), purpose is a REAPABLE kind (never interactive), and the recorded SPAWNER is
    provably dead. Safe in ANY context. Unlisted holders fall to later rungs and never disqualify identified ones."""
    try:
        from hermes_cli.process_identity import REAPABLE_PURPOSES, ledger_entries, spawner_is_dead
        entries = ledger_entries()
    except Exception:
        return []
    by_pid = {e.get("pid"): e for e in entries if isinstance(e.get("pid"), int)}
    return [
        int(pid) for pid, _name, _cmdline in matches
        if (entry := by_pid.get(int(pid))) and entry.get("purpose") in REAPABLE_PURPOSES and spawner_is_dead(entry) is True
    ]


def _handoff_reapable_backend_pids(matches: list[tuple[int, str, str]]) -> list[int] | None:
    """Backend PIDs safe to tree-reap during a GUI-updater hand-off, INCLUDING ones with a live parent; never raises.

    The orphan-only rung bails on ANY live parent (mid-teardown Electron, launcher->worker chain) and hung a
    hand-off. Inside the hand-off gate (marker + ``--gateway`` + no live ``hermes.exe`` shim) nothing legitimate
    supervises a ``serve`` from this venv, so survivors are leaks. Any non-backend holder or no psutil ->
    ``None``. The CALLER must have confirmed the gate; outside it the stricter orphan-only path stands.

    Any ``serve`` backend still holding the venv here is therefore a leak, live parent or not, and reaping
    its tree is correct rather than a race. See #50238.
    """
    psutil = _psutil()
    if psutil is None:
        return None
    roots: list[int] = []
    for pid, _name, cmdline in matches:
        low = _live_argv_low(psutil, pid, cmdline)
        if low is None:
            continue  # exited — nothing to reap
        if not _is_backend_argv(low):
            return None  # unexpected non-backend holder: refuse the whole set
        roots.append(int(pid))
    return roots or None


def _stop_process_trees(pids: list[int] | list[tuple[int, int]]) -> None:
    """Force-stop each PID with its full child tree (Windows); best effort, never raises.

    ``taskkill /T /F``: stopping only the parent can leave a ``.hermes-runtime`` child holding the install open.

    See #70026.
    """
    from gateway.status import get_process_start_time
    from hermes_cli._subprocess_compat import pid_is_hermes, windows_hide_flags
    for entry in pids:
        pid, expected_start_time = entry if isinstance(entry, tuple) else (int(entry), get_process_start_time(int(entry)))
        try:
            if expected_start_time is None:
                logger.debug("Skipping taskkill of PID %s: process identity unavailable", pid)
                continue
            if not pid_is_hermes(pid, expected_start_time=expected_start_time):
                logger.debug("Skipping taskkill of non-Hermes or changed PID %s", pid)
                continue
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"], check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                creationflags=windows_hide_flags(),
            )
        except Exception as exc:
            logger.debug("Could not stop process tree %s: %s", pid, exc)


def _looks_like_desktop_control_plane(cmdline: str) -> bool:
    """True for this-install ``hermes serve`` / ``hermes dashboard`` argv (Desktop control plane).

    Not the messaging gateway — don't feed into ``looks_like_gateway_command_line``. Token-based via the
    parser-derived classifier, never substring (``kanban --preserve-cache``, ``-m dashboard chat``).
    Undeterminable subcommand is NOT a control plane.

    See #92091.
    A cmdline whose subcommand cannot be determined is NOT a control plane — callers must not guess
    ownership. See #90778, #91869.
    """
    return "hermes_cli.main" in (cmdline or "").lower() and _hermes_holder_subcommand(cmdline) in _BACKEND_PURPOSES


def _desktop_owns_gateway_lifecycle() -> bool:
    """True when Desktop currently supervises this install's control plane (updater must not steal gateway start).

    Not proof messaging is served: serve is the control plane, the gateway a detached sibling. Prefer the spawn
    ledger; fall back to the venv-holder scan. An orphaned control plane (supervisor gone) does not count;
    without psutil orphanhood is unprovable and a live control plane suffices.

    See #76129, #92091.
    """
    from hermes_cli.update_cmd import _m
    with _best_effort('Desktop-lifecycle ledger probe failed: %s'):
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead
        if any(e.get("purpose") in _BACKEND_PURPOSES and spawner_is_dead(e) is False for e in ledger_entries()):
            return True
    psutil = _psutil()
    for pid, _name, cmdline in _try_call(_m()._detect_venv_python_processes, "Desktop-lifecycle holder scan failed: %s") or []:
        if not _looks_like_desktop_control_plane(cmdline):
            continue
        if psutil is None:
            return True
        with suppress(Exception):
            if _parent_is_live(psutil.Process(int(pid))):
                return True
    return False


def _win_service(name: str):
    """``(psutil, service)`` for the named SCM service (psutil imported here so tests can stub the module)."""
    import psutil  # noqa: PLC0415
    return psutil, psutil.win_service_get(name)


def _sc_exe(verb: str, name: str, service, settled_status: str) -> None:
    """``sc.exe <verb> <name>``; a non-zero exit is only an error when SCM doesn't already report *settled_status*."""
    result = subprocess.run(
        ["sc.exe", verb, name], capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=10, check=False,
    )
    if result.returncode != 0 and service.status() != settled_status:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f"sc.exe {verb} failed with {result.returncode}")


def _original_process_is_alive(psutil, pid: int, create_time: float) -> bool:
    """Is the process with this exact ``(pid, create_time)`` identity still alive? AccessDenied/unknown reads
    True (fail closed: the venv may still be locked)."""
    try:
        current = float(psutil.Process(pid).create_time())
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except Exception:
        return True
    return abs(current - create_time) <= 0.001


def _poll_until(condition, timeout: float, interval: float = 0.2) -> bool:
    """Poll *condition* every *interval* seconds until true (True) or *timeout* elapses (False)."""
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if condition():
            return True
        _time.sleep(interval)
    return False


def _process_create_time(psutil, pid: int, label: str) -> float:
    """``create_time`` of *pid*; unreadable identity aborts the stop (RuntimeError)."""
    try:
        return float(psutil.Process(int(pid)).create_time())
    except Exception as exc:
        raise RuntimeError(f"Windows {label} process identity is unavailable before stop") from exc


def _verify_service_identities(psutil, name: str, service, expected_service_identity, expected_gateway_identity) -> None:
    """Refuse to stop *name* unless SCM state, service/gateway process identities and ancestry all still match."""
    if expected_service_identity is not None:
        try:
            current_status, current_service_pid = str(service.status()), int(service.pid() or 0)
        except Exception as exc:
            raise RuntimeError(f"Windows service {name} SCM identity is unavailable before stop") from exc
        if current_status != "running":
            raise RuntimeError(f"Windows service {name} is not stably running before stop: {current_status}")
        if current_service_pid != int(expected_service_identity[0]):
            raise RuntimeError(f"Windows service {name} SCM process identity changed before stop")
    for label, identity in (("service", expected_service_identity), ("gateway", expected_gateway_identity)):
        if identity is not None and abs(_process_create_time(psutil, identity[0], label) - float(identity[1])) > 0.001:
            raise RuntimeError(f"Windows {label} process identity changed before stop")
    if expected_service_identity is not None and expected_gateway_identity is not None:
        try:
            ancestor_pids = {int(parent.pid) for parent in psutil.Process(int(expected_gateway_identity[0])).parents()}
        except Exception as exc:
            raise RuntimeError("Windows gateway ancestry is unavailable before service stop") from exc
        if int(expected_service_identity[0]) not in ancestor_pids:
            raise RuntimeError(f"Windows gateway is no longer owned by service {name}")


def _stop_windows_gateway_service(
    name: str, *, expected_processes: tuple[tuple[int, float], ...] = (), expected_service_identity: tuple[int, float] | None = None,
    expected_gateway_identity: tuple[int, float] | None = None, timeout: float = 30.0,
) -> None:
    """Stop one verified Windows service and wait until SCM reports it down.

    Lingering matching-identity processes after SCM says stopped make venv mutation unsafe — fail closed."""
    psutil, service = _win_service(name)
    _verify_service_identities(psutil, name, service, expected_service_identity, expected_gateway_identity)
    _sc_exe("stop", name, service, "stopped")

    def _alive() -> list[int]:
        return [pid for pid, create_time in expected_processes if _original_process_is_alive(psutil, pid, create_time)]

    if _poll_until(lambda: service.status() == "stopped" and not _alive(), timeout):
        return
    if service.status() != "stopped":
        raise RuntimeError(f"Windows service {name} did not stop within {timeout:.0f}s; venv mutation unsafe.")
    if alive_after_stop := _alive():
        raise RuntimeError(f"Windows service {name} stopped but its process tree is still alive: {alive_after_stop}")


def _start_windows_gateway_service(name: str, *, timeout: float = 30.0) -> None:
    """Start one previously paused Windows service and verify it is running."""
    service = _win_service(name)[1]
    _sc_exe("start", name, service, "running")
    if not _poll_until(lambda: service.status() == "running", timeout):
        raise RuntimeError(f"Windows service {name} did not start within {timeout:.0f}s")


def _restore_windows_gateway_service(name: str, *, timeout: float = 60.0) -> None:
    """Restore a service after an uncertain stop, including STOP_PENDING."""
    from hermes_cli.update_cmd import _start_windows_gateway_service
    service = _win_service(name)[1]

    def _settled() -> bool:
        status = service.status()
        if status == "stopped":
            _start_windows_gateway_service(name)
        return status in ("running", "stopped")

    if not _poll_until(_settled, timeout):
        raise RuntimeError(f"Windows service {name} did not reach a restorable state within {timeout:.0f}s")


def _windows_cold_start_plan() -> dict | None:
    """Pause token for the no-running-gateway case: cold-start after update when an autostart entry exists.

    An installed autostart entry is an explicit "I want a gateway" signal; a gateway that died between
    updates would otherwise stay down until next login (resume only relaunches what was running).
    Desktop-owned lifecycle -> ``None`` (spawning ``gateway run`` beside Desktop races ports/state);
    the skip is ownership, not liveness."""
    from hermes_cli.update_cmd import _desktop_owns_gateway_lifecycle
    with _best_effort('Could not check Desktop gateway-lifecycle ownership before update: %s'):
        if _desktop_owns_gateway_lifecycle():
            logger.debug("Skipping Windows gateway cold-start plan: Desktop owns gateway lifecycle")
            return None
    with _best_effort('Could not check Windows gateway autostart state before update: %s'):
        from hermes_cli import gateway_windows
        if gateway_windows.is_installed():
            return {"resume_needed": True, "profiles": {}, "unmapped_pids": [], "unmapped": [], "cold_start_if_installed": True}
    return None


def _pause_windows_gateway_services(service_gateways, token: dict, profiles: dict, unmapped: list) -> dict:
    """Stop each SCM gateway service, recording them on *token*; roll everything back on failure.

    Runs after every fallible ordinary-gateway step so a failure here restores the attempted
    services AND the already-paused ordinary gateways before re-raising."""
    from hermes_cli.update_cmd import _restore_windows_gateway_service, _stop_windows_gateway_service
    paused_services = []
    current_service_name = None
    try:
        for service in service_gateways:
            current_service_name = str(service.name)
            _stop_windows_gateway_service(
                current_service_name, expected_processes=tuple(getattr(service, "descendant_identities", ())),
                expected_service_identity=(int(service.service_pid), float(service.service_create_time)),
                expected_gateway_identity=(int(service.gateway_pid), float(service.gateway_create_time)),
            )
            paused_services.append(current_service_name)
            current_service_name = None
        if paused_services:
            token.update(services=paused_services, expected_services=list(paused_services), restarted_services=[])
            token["service_profiles"] = {str(s.name): str(s.profile) for s in service_gateways if str(s.name) in paused_services}
            print("  ✓ Paused Windows gateway service(s): " + ", ".join(paused_services))
        return token
    except Exception as exc:
        restore_names = ([current_service_name] if current_service_name else []) + list(reversed(paused_services))
        rollback: list[tuple[str, object]] = [(n, lambda n=n: _restore_windows_gateway_service(n)) for n in dict.fromkeys(restore_names)]
        if profiles or unmapped:
            rollback.append(("ordinary gateways", lambda: _resume_windows_gateways_after_update(token)))
        rollback_failures = []
        for label, restore in rollback:
            try:
                restore()
            except Exception as restore_exc:
                rollback_failures.append(f"{label}: {restore_exc}")
        detail = f"Could not stop Windows gateway service {current_service_name or 'unknown'}: {exc}"
        if rollback_failures:
            detail += "; rollback failures: " + "; ".join(rollback_failures)
        raise RuntimeError(detail) from exc


def _discover_windows_gateways():
    """``(profile_processes, service_gateways, service_gateway_pids, running_pids)`` for the pause; any indeterminate probe aborts."""
    from hermes_cli.gateway import find_gateway_pids, find_profile_gateway_processes, find_windows_gateway_services
    with _abort_on_error("Could not map Windows gateway PIDs to profiles"):
        profile_process_list = find_profile_gateway_processes(strict=True)
        profile_processes = {proc.pid: proc for proc in profile_process_list}
    with _abort_on_error("Could not determine Windows gateway service ownership"):
        service_gateways = find_windows_gateway_services(profile_processes=profile_process_list)
    service_gateway_pids = {int(service.gateway_pid) for service in service_gateways}
    with _abort_on_error("Could not discover Windows gateway PIDs before update"):
        running_pids = list(dict.fromkeys(
            [*find_gateway_pids(all_profiles=True), *sorted(profile_processes), *sorted(service_gateway_pids)]
        ))
    return profile_processes, service_gateways, service_gateway_pids, running_pids


def _request_socket_pauses(running_pids, profile_processes, service_gateway_pids):
    """Marker + socket-first pause for every profile-mapped gateway; ``(profiles, mapped_pids, socket_acks)``.

    Socket ACK = the gateway drains and exits by its own graceful path. No answer (older
    gateway) -> the marker poll / force-kill ladder in the caller."""
    profiles: dict[str, int] = {}
    mapped_pids = []
    socket_acks: list[dict] = []
    for pid in running_pids:
        proc = None if pid in service_gateway_pids else profile_processes.get(pid)
        if proc is None:
            continue
        profiles[str(proc.profile)] = int(pid)
        mapped_pids.append(int(pid))
        _write_update_planned_stop_marker(Path(proc.path), int(pid))
        try:
            # Socket-first pause (#92091 step 2): ask the gateway to drain and exit itself instead of
            # relying on the marker poll + force-kill ladder. A positive ACK means the gateway is running
            # its own graceful restart path (same drain as SIGUSR1/service restarts) and will release its
            # venv handles on the way out. No answer (older gateway, no socket) → the marker watcher /
            # force-kill fallback below behaves exactly as before this verb existed.
            from gateway.control_socket import pause_gateway_for_update
            ack = pause_gateway_for_update(Path(proc.path))
            if ack and (ack.get("pausing") or ack.get("already_stopping")):
                socket_acks.append(ack)
        except Exception as exc:
            logger.debug("Socket pause unavailable for gateway %s: %s", pid, exc)
    return profiles, mapped_pids, socket_acks


def _gateway_drain_timeout(socket_acks: list[dict]) -> float:
    """Drain budget: configured restart drain (>= 1s), raised to a socket-paused gateway's declared
    ACTIVE-TURN budget + teardown grace so it isn't force-killed mid-turn."""
    from hermes_cli.gateway import _get_restart_drain_timeout
    try:
        drain_timeout = max(float(_get_restart_drain_timeout()), 1.0)
    except Exception:
        drain_timeout = 10.0
    if socket_acks:
        with suppress(Exception):
            declared = max(float(a.get("drain_timeout") or 0.0) for a in socket_acks)
            drain_timeout = max(drain_timeout, declared + 10.0)
        print(f"  → {len(socket_acks)} gateway(s) ACKed socket pause; waiting up to {int(drain_timeout)}s for graceful exit")
    return drain_timeout


def _pause_windows_gateways_for_update() -> dict | None:
    """Stop running Windows gateways before mutating the checkout or venv.

    Scheduled/startup gateways run via pythonw.exe, invisible to the hermes.exe instance guard, yet keep files
    locked during ``git``/``uv``. Stop only PIDs the gateway discovery code identifies."""
    from hermes_cli.update_cmd import _m
    if not _m()._is_windows():
        return None
    with _abort_on_error("Could not prepare Windows gateway pause for update"):
        from gateway.status import get_process_start_time, terminate_pid
        from hermes_cli.gateway import _capture_gateway_argv
    profile_processes, service_gateways, service_gateway_pids, running_pids = _discover_windows_gateways()
    if not running_pids:
        return _windows_cold_start_plan()
    profiles, mapped_pids, socket_acks = _request_socket_pauses(running_pids, profile_processes, service_gateway_pids)
    # Resolve venv-side launchers BEFORE draining: a dead worker's parent cannot be recovered (NoSuchProcess).
    # The launcher keeps ``.pyd`` mapped and would trip the venv-holder guard; it is killed with the survivors.
    launcher_pids = _m()._venv_launcher_ancestors(mapped_pids)
    print("→ Stopping Windows gateway process(es) before updating Hermes...")
    drain_timeout = _gateway_drain_timeout(socket_acks)
    survivors = _m()._wait_for_windows_update_gateway_exit(mapped_pids, timeout=drain_timeout)
    unmapped_pids = [pid for pid in running_pids if pid not in profile_processes and pid not in service_gateway_pids]
    # Snapshot unmapped gateways' argv *before* force-killing so resume can replay it.
    # Unmapped = no profile->PID-file mapping (e.g. Scheduled Task ``pythonw.exe -m ...``).
    unmapped = [
        {"pid": int(pid), "argv": _try_call(lambda p=int(pid): _capture_gateway_argv(p),
                                            "Could not capture argv for unmapped gateway %s: %s", int(pid))}
        for pid in unmapped_pids
    ]
    # Tree-kill survivors, unmapped gateways, and pre-drain launchers; a launcher
    # already gone with its worker raises ProcessLookupError and is skipped.
    force_killed = []
    for pid in sorted(set(survivors).union(unmapped_pids).union(launcher_pids)):
        with suppress(ProcessLookupError, PermissionError, OSError):
            terminate_pid(int(pid), force=True, expected_start_time=get_process_start_time(int(pid)))
            force_killed.append(int(pid))
    if profiles:
        print(f"  ✓ Paused gateway profile(s): {', '.join(sorted(profiles))}")
    if force_killed:
        print(f"  → Force-stopped {len(force_killed)} gateway process(es)")
    if unmapped_pids:
        print(f"  → Stopped {len(unmapped_pids)} gateway process(es) without profile mapping")
        if any(not u.get("argv") for u in unmapped):  # no recoverable cmdline (psutil missing, denied, gone)
            print("    Restart manually after update: hermes gateway run")
    token = {"resume_needed": True, "profiles": profiles, "unmapped_pids": unmapped_pids, "unmapped": unmapped}
    return _pause_windows_gateway_services(service_gateways, token, profiles, unmapped)


def _cold_start_windows_gateway_after_update() -> bool:
    """Direct-spawn a detached gateway after update for the ``cold_start_if_installed`` case (installed but down).

    Idempotent: re-checks nothing is running so a concurrent autostart can't duplicate. A successful Popen
    doesn't prove survival (a job object denying breakaway kills it), so success is gated on the liveness poll.
    Vouched PIDs are attested so a death AFTER updater exit is reported by the next CLI invocation.

    A successful ``Popen`` only proves the process was created, not that it survived (e.g. a Windows job
    object denying breakaway kills it before it logs anything — #84185). So the success line is gated on the
    same post-spawn liveness poll every other ``_spawn_detached`` caller uses
    (``gateway_windows._report_gateway_start``), instead of being printed unconditionally from the returned
    PID.
    """
    from hermes_cli.update_cmd import _desktop_owns_gateway_lifecycle, _m
    if not _m()._is_windows():
        return True
    with _abort_on_error("Could not load Windows gateway cold-start helpers"):
        from hermes_cli import gateway_windows
        from hermes_cli.gateway import find_gateway_pids
    with _abort_on_error("Could not re-check gateway liveness before cold-start"):
        if list(find_gateway_pids(all_profiles=True)):
            return True
    with _abort_on_error("Could not re-check Desktop gateway-lifecycle ownership before cold-start"):
        if _desktop_owns_gateway_lifecycle():
            logger.debug("Skipping Windows gateway cold-start: Desktop owns gateway lifecycle")
            return True
    with _abort_on_error("Could not cold-start Windows gateway after update"):
        pid = gateway_windows._spawn_detached()
    if not pid:
        raise RuntimeError("Windows gateway cold-start did not return a process ID")
    ready_pids = gateway_windows._wait_for_gateway_ready()
    if not ready_pids:
        raise RuntimeError(f"Windows gateway cold-start PID {pid} did not become ready")
    print(f"\n✓ Gateway started via cold-start after update (PID: {', '.join(map(str, ready_pids))})")
    with suppress(Exception):
        gateway_windows._write_start_attestation(ready_pids, "cold-start after update")
    return True


def _refresh_windows_gateway_launchers() -> None:
    """Regenerate installed Windows gateway launcher scripts after update; best-effort, never fails the update.

    Launchers are written once at install, so old installs kept launching via ``pythonw.exe`` (``sys.stderr is
    None`` death). The task's /TR points at a stable path, so rewriting in place retargets it without UAC.

    The Scheduled Task / Startup-folder launchers (``gateway.cmd`` + ``gateway.vbs``) are persistence
    artifacts written once at install time — ``hermes update`` never touched them, so installs created
    before the hidden-console rework (aa2ae36c3f) kept launching the gateway through ``pythonw.exe``
    forever: every descendant spawn flashed a conhost (#54220/#56747) and, since #70344, the console-less
    gateway died at startup with ``RuntimeError: sys.stderr is None`` (#71671).
    """
    from hermes_cli.update_cmd import _m
    if not _m()._is_windows():
        return
    with _best_effort('Could not refresh Windows gateway launchers after update: %s'):
        from hermes_cli import gateway_windows
        if gateway_windows.is_installed():
            gateway_windows._write_task_script()
            print("  ✓ Refreshed Windows gateway launcher scripts")


def _refresh_bootstrap_cache_scripts(branch: str = "main") -> None:
    """Overwrite ``$HERMES_HOME/bootstrap-cache/install-<ref>.{ps1,sh}`` for *branch* from the fresh checkout.

    Old ``hermes-setup.exe`` builds NEVER re-download a cached branch-ref script, so a stale one runs
    months-old code forever. Guards mirror ``install_script.rs``: only the sanitized *branch* key is rewritten;
    commit-SHA pins (7-40 hex) are immutable and skipped. Best-effort: never fails the update.

    Installer binaries built before the #67193 cache-refresh fix (June 2026 and earlier) NEVER re-download a
    cached branch-ref script — ``install-main.ps1`` cached at install time is reused forever, executing
    months-stale code with long-fixed bugs (the 2026-08-09 incident: a June 4 cached script's venv stage
    lacked the 81327 process-tree sweep and died on ``Access denied``). The binary has no self-update path,
    so the poisoned cache outlives every ``hermes update``.
    Overwriting the cached script for *branch* with the freshly pulled ``scripts/install.ps1`` /
    ``scripts/install.sh`` on every update turns the stale binary's unconditional reuse into a feature: it
    "reuses" a file this function keeps permanently current. Post-#67193 installers re-download on each run
    anyway, so for them this is a harmless pre-seed of the same bytes.
    The .ps1 copy gets a UTF-8 BOM to match the installer's cache format (#67193 encoding fix).
    """
    from hermes_cli.update_cmd import _m
    with _best_effort('Could not refresh bootstrap-cache scripts after update: %s'):
        cache_dir = Path(_m().get_hermes_home()) / "bootstrap-cache"
        if not cache_dir.is_dir():
            return
        safe_ref = re.sub(r"[^A-Za-z0-9._-]", "_", str(branch or "main"))  # install_script.rs::sanitize_ref()
        if re.fullmatch(r"[0-9a-fA-F]{7,40}", safe_ref):  # install_script.rs::is_valid_commit(): immutable pin
            return
        refreshed = []
        for kind, src_name in (("ps1", "install.ps1"), ("sh", "install.sh")):
            src = _m().PROJECT_ROOT / "scripts" / src_name
            cached = cache_dir / f"install-{safe_ref}.{kind}"
            if not src.is_file() or not cached.is_file():
                continue  # this ref was never bootstrap-cached — nothing to heal
            data = src.read_bytes()
            if kind == "ps1" and not data.startswith(b"\xef\xbb\xbf"):
                data = b"\xef\xbb\xbf" + data  # PowerShell needs the BOM or localized/em-dash text mis-decodes.
            # See #67193.
            if cached.read_bytes() == data:
                continue
            tmp = cached.with_suffix(cached.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, cached)
            refreshed.append(cached.name)
        if refreshed:
            print("  ✓ Refreshed installer bootstrap-cache script(s): " + ", ".join(sorted(refreshed)))


def _resume_windows_services(token: dict) -> None:
    """Restart the SCM services recorded on *token*; failed ones stay on the token so a retry sees them."""
    from hermes_cli.update_cmd import _start_windows_gateway_service
    services = list(token.get("services") or [])
    token.setdefault("expected_services", list(services))
    verified_restarts = list(token.get("restarted_services") or [])
    restarted_services = []
    failed_services = []
    for service_name in map(str, services):
        try:
            _start_windows_gateway_service(service_name)
            restarted_services.append(service_name)
            if service_name not in verified_restarts:
                verified_restarts.append(service_name)
        except Exception as exc:
            logger.warning("Could not restart Windows gateway service %s after update: %s", service_name, exc)
            print(f"  ⚠ Could not restart Windows gateway service: {service_name}")
            failed_services.append(service_name)
    token["restarted_services"] = verified_restarts
    token["services"] = failed_services
    if failed_services:
        raise RuntimeError("Could not restart Windows gateway service(s): " + ", ".join(failed_services))
    if restarted_services:
        print("\n  ✓ Restarted Windows gateway service(s): " + ", ".join(restarted_services))


def _relaunch_paused_gateways(token: dict, profiles: dict, unmapped: list) -> tuple[list[str], int]:
    """Relaunch profile gateways and replay unmapped argv; ``(relaunched_profiles, unmapped_count)``.

    Failed relaunches stay on the token (and off ``relaunched_profiles``) so plan-vs-execution
    reconciliation still surfaces them — Windows has no watcher to recover them."""
    with _abort_on_error("Could not load Windows gateway restart helper"):
        from hermes_cli.gateway import launch_detached_gateway_restart_by_cmdline, launch_detached_profile_gateway_restart

    # An exception from a launch (incl. bad pid/argv coercion) logs at debug and reads as a failed relaunch.
    relaunched = []
    failed_profiles = {}
    for profile, old_pid in sorted(profiles.items()):
        if _try_call(lambda p=profile, o=old_pid: launch_detached_profile_gateway_restart(str(p), int(o)),
                     "Could not restart Windows gateway profile %s after update: %s", profile):
            relaunched.append(str(profile))
        else:
            failed_profiles[str(profile)] = int(old_pid)
    # Surface the outcome on the token (#91277 Phase 2 plan-vs-execution reconciliation): the git-based
    # update path's fleet reconciliation cross-checks every planned runtime against restarted_services /
    # relaunched_profiles / externally_supervised_profiles / killed_pids — bookkeeping this Windows-specific
    # pause/resume never fed, so a correctly-paused-and-relaunched Windows gateway was reported
    # "unaccounted" (loud warning + exit 1) even though the restart succeeded. The caller merges this into
    # the shared relaunched_profiles list before reconciliation runs. A profile whose relaunch genuinely
    # failed is deliberately left off this list — it must still surface as unaccounted so the user is told
    # to restart it manually (Windows has no watcher to recover a failed relaunch).
    token["relaunched_profiles"] = relaunched
    unmapped_relaunched = 0
    failed_unmapped = []
    for entry in unmapped:
        argv, old_pid = entry.get("argv"), entry.get("pid")
        if argv and old_pid and _try_call(lambda o=old_pid, a=argv: launch_detached_gateway_restart_by_cmdline(int(o), list(a)),
                                          "Could not restart unmapped Windows gateway (pid %s) after update: %s", old_pid):
            unmapped_relaunched += 1
        else:
            failed_unmapped.append(entry)
    token["profiles"] = failed_profiles
    token["unmapped"] = failed_unmapped
    if failed_profiles or failed_unmapped:
        raise RuntimeError("Could not restart every paused Windows gateway")
    return relaunched, unmapped_relaunched


def _verify_relaunched_gateways_alive(token: dict, profiles: dict, unmapped: list) -> None:
    """Gate success on the shared liveness poll: a truthy launch only proves the watcher was created.

    A parent Job Object denying CREATE_BREAKAWAY_FROM_JOB can kill the gateway on updater teardown;
    ``all_profiles=True`` covers the fleet. Vouched PIDs are persisted so a death AFTER updater exit
    is reported by the next CLI invocation (best-effort)."""
    with _abort_on_error("Could not load Windows gateway liveness helpers"):
        from hermes_cli import gateway_windows
    ready_pids = gateway_windows._wait_for_gateway_ready(timeout_s=30.0, all_profiles=True)
    if not ready_pids:
        token["profiles"] = dict(profiles)
        token["unmapped"] = list(unmapped)
        print(
            "\n  ⚠ Windows gateway restart could not be verified — no stable gateway process appeared after relaunch.\n"
            "    (The respawned gateway may have been killed by a parent Job Object during updater teardown, #48820.)\n"
            "    Recover with: hermes gateway restart"
        )
        raise RuntimeError("Windows gateway relaunch after update was not verified alive")
    with suppress(Exception):
        gateway_windows._write_start_attestation(ready_pids, "post-update relaunch")


def _resume_windows_gateways_after_update(token: dict | None) -> None:
    """Restart Windows profile gateways previously paused for update."""
    from hermes_cli.update_cmd import _m
    if not token or not token.get("resume_needed"):
        return
    if not _m()._is_windows():
        token["resume_needed"] = False
        return
    # Regenerate launcher scripts before respawning so a legacy pythonw-era
    # autostart entry comes back on the current design at next login too.
    _m()._refresh_windows_gateway_launchers()
    _resume_windows_services(token)
    profiles = token.get("profiles") or {}
    unmapped = token.get("unmapped") or []
    if not profiles and not any(u.get("argv") for u in unmapped):
        if token.get("cold_start_if_installed"):
            if not _m()._cold_start_windows_gateway_after_update():
                raise RuntimeError("Windows gateway cold-start was not verified")
            token["cold_start_if_installed"] = False
        token["resume_needed"] = False
        return
    relaunched, unmapped_relaunched = _relaunch_paused_gateways(token, profiles, unmapped)
    if relaunched or unmapped_relaunched:
        _verify_relaunched_gateways_alive(token, profiles, unmapped)
    token["resume_needed"] = False
    if relaunched:
        print(f"\n  ✓ Restarting Windows gateway profile(s): {', '.join(relaunched)}")
    if unmapped_relaunched:
        lead = "" if relaunched else "\n"
        print(f"{lead}  ✓ Restarting {unmapped_relaunched} unmapped Windows gateway process(es)")


def _resume_windows_gateways_and_merge_outcome(outcome, _windows_gateway_resume, gateway_mode: bool):
    """Resume gateways paused for a Windows update and fold the token into ``outcome``'s systemd/launchd-style
    bookkeeping so reconciliation never reports a healthy gateway as unaccounted. Must never abort the update."""
    from hermes_cli.update_cmd import _m, _write_gateway_update_exit_code
    try:
        _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
    except Exception as _windows_resume_exc:
        outcome.incomplete = True
        outcome.phase_errors.append(str(_windows_resume_exc))
        print(f"  ⚠ Windows gateway service restart incomplete: {_windows_resume_exc}")
        if gateway_mode:
            _write_gateway_update_exit_code(False)
    if not isinstance(_windows_gateway_resume, dict):
        return

    def _extend_unique(target: list, items) -> None:
        target.extend(item for item in dict.fromkeys(items) if item not in target)

    token = _windows_gateway_resume
    # Failed relaunches are absent from the token so they still surface. Best-effort.
    with _best_effort('Could not merge Windows relaunch outcome into fleet reconciliation bookkeeping: %s'):
        _extend_unique(outcome.relaunched_profiles, token.get("relaunched_profiles") or [])
    windows_restarted = list(token.get("restarted_services") or [])
    service_profiles = token.get("service_profiles") or {}
    _extend_unique(outcome.restarted_services, windows_restarted)
    _extend_unique(outcome.relaunched_profiles, (p for p in (service_profiles.get(n) for n in windows_restarted) if p))
    _extend_unique(outcome.failed_or_stale_units, (str(service_profiles.get(n) or n) for n in (token.get("services") or [])))
    with suppress(Exception):
        from hermes_cli.update_receipt import record_gateway_restart
        record_gateway_restart(
            restarted_services=outcome.restarted_services, relaunched_profiles=outcome.relaunched_profiles,
            externally_supervised_profiles=outcome.externally_supervised_profiles, killed_pids=sorted(outcome.killed_pids),
            failed_units=outcome.failed_or_stale_units, incomplete=outcome.incomplete or bool(outcome.failed_or_stale_units),
            phase_error="; ".join(outcome.phase_errors) or None,
        )


def _reap_and_rescan(message: str, pids, stop=None) -> list[tuple[int, str, str]]:
    """Announce *message*, stop *pids* (tree-kill unless *stop* given), settle 1s, re-scan venv holders."""
    from hermes_cli.update_cmd import _m
    print(message)
    (stop or _m()._stop_process_trees)(pids)
    _time.sleep(1.0)
    return _m()._detect_venv_python_processes()


def _terminate_leftover_gateways(pids) -> None:
    """Force-stop leftover gateways one by one; a failure is logged, never raised."""
    from gateway.status import get_process_start_time, terminate_pid
    for _pid in pids:
        _try_call(lambda p=int(_pid): terminate_pid(p, force=True, expected_start_time=get_process_start_time(p)),
                  "Could not stop leftover gateway %s: %s", _pid)


def _in_handoff_without_live_shim(args) -> bool:
    """GUI hand-off gate: ``--gateway`` + update-incomplete marker AND no live ``hermes.exe`` shim.

    Fail closed: unverifiable marker or shim state reads as "not a hand-off" / "live shim"."""
    from hermes_cli.update_cmd import _m
    try:
        if not (bool(getattr(args, "gateway", False)) and _m()._update_marker_path().exists()):
            return False
        scripts_dir = _m()._venv_scripts_dir()
        return scripts_dir is not None and not _m()._detect_concurrent_hermes_instances(scripts_dir)
    except Exception:
        return False


def _clear_windows_venv_holders_or_exit(args, gateway_mode: bool, _windows_gateway_resume):
    """Windows: stop every venv-python holder we can positively identify, else resume paused gateways and exit 2.

    Rungs in order: leftover pausable gateways -> ledger orphaned backends -> orphaned Desktop backends ->
    ledger manual serve (relaunched at exit on the same bind) -> GUI hand-off leaks. Remaining holders are
    refused (the sync would corrupt against a locked .pyd)."""
    from hermes_cli.update_cmd import _m, _record_update_step, _refuse_gateway_ancestor_tree_kill

    def _resume_and_exit():
        _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
        sys.exit(2)

    holders = _m()._detect_venv_python_processes()
    # Gateways the pause machinery owns (respawned in the pause->guard window or unmapped
    # spawn path): stop and re-check; post-update resume brings them back.
    if holders and (gateway_holders := _m()._leftover_pausable_gateway_pids(holders)) is not None:
        if _refuse_gateway_ancestor_tree_kill(gateway_holders, gateway_mode=gateway_mode):
            _resume_and_exit()
        holders = _reap_and_rescan(
            f"  ⚠ {len(gateway_holders)} gateway process(es) still hold the venv after the pause; stopping them",
            gateway_holders, stop=_terminate_leftover_gateways,
        )
    # Tree-reap rungs. Ledger rung = positive identity in any context (self-registered backend, spawner
    # provably dead; no PPID archaeology). Orphan rung = Desktop `serve` whose app is GONE (nothing
    # respawns an orphan); live-Desktop backends return None and keep the refusal.
    for classifier, message in (
        (_m()._ledger_reapable_backend_pids, "ledger-identified orphaned Hermes backend process(es) hold the venv"),
        (_m()._orphaned_desktop_backend_pids, "orphaned Desktop backend process(es) still hold the venv"),
    ):
        if holders and (backends := classifier(holders)):
            holders = _reap_and_rescan(f"  ⚠ {len(backends)} {message}; stopping their trees", backends)
    # Manual serve/dashboard rung (e.g. `hermes serve --host <ip>` for a REMOTE Desktop): ledger identity
    # only (spawner dead; Desktop-owned keep the refusal). Stop and register an idempotent atexit relaunch
    # on the SAME host/port/profile — success or failure.
    if holders and (serve_entries := _m()._ledger_manual_serve_holders(holders)):
        def _stop_and_park(pids):
            _m()._stop_process_trees(pids)
            _record_update_step("serve_pause", True, f"stopped={len(serve_entries)}")
            import atexit as _serve_atexit
            _serve_atexit.register(_m()._relaunch_stopped_serves, {"pending": True, "entries": serve_entries})

        holders = _reap_and_rescan(
            f"  ⚠ {len(serve_entries)} manual serve/dashboard backend(s) hold the venv; stopping them for "
            "the update (they will be relaunched on their recorded endpoints)",
            [int(e["pid"]) for e in serve_entries], stop=_stop_and_park,
        )
    # Final rung: in a GUI hand-off the Desktop is contractually gone; surviving `serve` backends are leaks
    # even with a live parent (which made the orphan-only rung bail and hang) — reap by cmdline.
    if holders and _in_handoff_without_live_shim(args) and (handoff_backends := _m()._handoff_reapable_backend_pids(holders)):
        holders = _reap_and_rescan(
            f"  ⚠ {len(handoff_backends)} Hermes backend process(es) "
            "still hold the venv after the Desktop hand-off; stopping their trees", handoff_backends,
        )
    if holders:
        print(_format_venv_python_holders_message(holders))
        _resume_and_exit()

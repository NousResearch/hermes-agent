"""Process identity, venv holders, and update admission guards.

Extracted mechanically from :mod:`hermes_cli.update_cmd`.  Runtime
references to the historical module surface resolve through the
compatibility facade so imports and monkeypatches remain effective.
"""

from pathlib import Path
from typing import Optional

from hermes_cli._update_compat import facade as _u


def _format_concurrent_instances_message(
    matches: list[tuple[int, str]], scripts_dir: Path
) -> str:
    """Build a human-readable explanation + remediation hint for the user."""
    shim = scripts_dir / "hermes.exe"
    lines = ["✗ Another hermes.exe is running:"]
    for pid, name in matches:
        lines.append(f"    PID {pid}  {name}")
    lines.append("")
    lines.append(f"  Updating now would fail to overwrite {shim} because")
    lines.append("  Windows blocks REPLACE on a running executable.")
    lines.append("")
    lines.append("  Close Hermes Desktop, exit any open `hermes` REPLs, and")
    lines.append("  stop the gateway (`hermes gateway stop`) before retrying.")
    lines.append("")
    if matches:
        pid_args = " ".join(f"/PID {pid}" for pid, _ in matches)
        lines.append("  If you've already closed everything and these PIDs are")
        lines.append("  stale, terminate them directly, then retry the update:")
        lines.append(f"      taskkill {pid_args} /F")
        lines.append("")
    lines.append("  Override with `hermes update --force` if you've already")
    lines.append("  confirmed those processes will not write to the venv.")
    return "\n".join(lines)


def _classify_concurrent_instance(pid: int) -> str:
    """Return ``"gateway"`` when ``pid``'s command line is a gateway runtime.

    Delegates to ``_is_pausable_gateway`` — the same canonical
    ``gateway run`` matcher (``gateway.status.looks_like_gateway_command_line``,
    shlex-tokenized, profile-selector aware) used by the Desktop preflight
    exemption and the venv-holder guard fallback — so a PID classified as
    ``"gateway"`` here is exactly the set the pause/kill+restart machinery
    downstream will stop. That symmetry is what lets the pre-update
    concurrent gate skip the abort for gateway-only matches: the gateway is
    going to be stopped by ``_pause_windows_gateways_for_update()`` moments
    later anyway, so refusing the update just to make the user kill it
    manually is friction without benefit.

    Returns ``"non-gateway"`` when the cmdline doesn't match, and
    ``"unknown"`` when psutil can't read it (process gone, access denied,
    psutil missing). The gate treats ``"unknown"`` as non-gateway — we'd
    rather block an update we could have completed than proceed against a
    process we couldn't positively identify as a gateway.
    """
    try:
        import psutil  # noqa: PLC0415
    except Exception:
        return "unknown"

    try:
        proc = psutil.Process(int(pid))
        cmdline_list = proc.cmdline()
    except Exception:
        return "unknown"

    from hermes_cli._scan_venv_blockers import _is_pausable_gateway  # noqa: PLC0415

    cmdline = " ".join(cmdline_list or [])
    if _is_pausable_gateway(cmdline):
        return "gateway"
    return "non-gateway"


def _filter_non_gateway_concurrent_instances(
    matches: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Return only the concurrent-instance matches that are NOT the gateway.

    Used by the pre-update concurrent gate to decide whether to abort
    ``hermes update``. If every concurrent instance is a gateway, the pause
    machinery (``_pause_windows_gateways_for_update``) and the post-update
    kill+restart block handle it — the update proceeds. If anything else (a
    TUI shell, a Hermes Desktop backend child, an unrelated ``hermes`` REPL)
    is in the list, the gate still aborts with the existing message, since
    those have no pause machinery downstream.
    """
    non_gateway: list[tuple[int, str]] = []
    for pid, name in matches:
        if _classify_concurrent_instance(pid) != "gateway":
            non_gateway.append((pid, name))
    return non_gateway


def _write_update_planned_stop_marker(profile_path: Path, pid: int) -> bool:
    """Write a planned-stop marker into a specific profile home."""
    try:
        from datetime import timezone

        from gateway.status import _get_process_start_time
        from utils import atomic_json_write

        record = {
            "target_pid": pid,
            "target_start_time": _get_process_start_time(pid),
            "stopper_pid": os.getpid(),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json_write(
            Path(profile_path) / ".gateway-planned-stop.json",
            record,
            indent=None,
            separators=(",", ":"),
        )
        return True
    except (OSError, PermissionError):
        return False


def _wait_for_windows_update_gateway_exit(
    pids: list[int], *, timeout: float
) -> set[int]:
    """Wait for the given gateway PIDs to exit, returning survivors."""
    if not pids:
        return set()

    from gateway.status import _pid_exists

    remaining = set(pids)
    deadline = _time.monotonic() + max(timeout, 0.0)
    while remaining and _time.monotonic() < deadline:
        for pid in list(remaining):
            try:
                if not _pid_exists(pid):
                    remaining.discard(pid)
            except Exception:
                remaining.discard(pid)
        if remaining:
            _time.sleep(0.25)

    survivors: set[int] = set()
    for pid in remaining:
        try:
            if _pid_exists(pid):
                survivors.add(pid)
        except Exception:
            pass
    return survivors


def _venv_core_imports_healthy() -> tuple[bool, str]:
    """Probe the project venv for the core imports the backend needs to boot.

    Runs a tiny import check inside the venv interpreter (NOT this process —
    ``hermes update`` may be driven by a different Python). Catches the
    half-updated-venv state: git checkout current but a dependency sync that
    failed or was killed partway (e.g. Windows access-denied on a loaded
    .pyd), leaving imports like ``fastapi``'s new transitive deps missing.
    Without this probe, ``hermes update`` on a current checkout prints
    "Already up to date!" and returns without ever re-syncing dependencies —
    the user's install stays broken no matter how many times they update
    (ryanc's incident, July 2026).

    Returns ``(healthy, detail)``. Never raises; unknown states report
    healthy so a probe failure can't force needless reinstalls.
    """
    venv_dir = _m().PROJECT_ROOT / "venv"
    venv_python = venv_python_path(venv_dir, windows=_m()._is_windows())
    if not venv_python.exists():
        # No venv interpreter at all. In a dev checkout that's normal (the
        # dev may run hermes from any interpreter), so report healthy to
        # avoid forcing reinstalls. But on a MANAGED install (the Windows
        # installer / desktop bootstrap stamps `.hermes-bootstrap-complete`,
        # and an interrupted update leaves `.update-incomplete`), the venv
        # IS the install — its absence means a repair got interrupted after
        # the old venv was moved aside, and "Already up to date!" would
        # gaslight the user while nothing can run.
        managed_markers = (
            _m().PROJECT_ROOT / ".hermes-bootstrap-complete",
            _m()._update_marker_path(),
        )
        if any(m.exists() for m in managed_markers):
            return False, f"venv python missing ({venv_python})"
        return True, ""

    # Core web/serve imports plus their newest transitive deps. Import (not
    # just metadata) — a package can have intact dist-info but a missing
    # module after an interrupted uninstall/install cycle.
    check = (
        "import importlib\n"
        "mods = ['fastapi', 'uvicorn', 'pydantic', 'openai', 'yaml']\n"
        "missing = []\n"
        "for m in mods:\n"
        "    try: importlib.import_module(m)\n"
        "    except Exception as e: missing.append(f'{m}: {e}')\n"
        "print('\\n'.join(missing))\n"
    )
    try:
        result = subprocess.run(
            [str(venv_python), "-c", check],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=60,
            cwd=_m().PROJECT_ROOT,
        )
    except Exception as exc:
        logger.debug("venv health probe failed to run: %s", exc)
        return True, ""

    missing = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if result.returncode != 0 and not missing:
        # Interpreter itself is broken (e.g. deleted stdlib) — that IS unhealthy.
        detail = (result.stderr or "").strip().splitlines()
        return False, detail[0] if detail else "venv python failed to run"
    if missing:
        return False, "; ".join(missing[:4])
    return True, ""


def _detect_venv_python_processes(
    *, exclude_pids: set[int] | None = None
) -> list[tuple[int, str, str]]:
    """Find live processes running from the project venv's interpreter.

    The hermes.exe shim guard misses the biggest lock-holder class on
    Windows: the Desktop app's backend (``python.exe -m hermes_cli.main
    serve``) and anything else running straight off ``venv\\Scripts\\python
    (w).exe``. Those processes keep native ``.pyd`` extensions mapped, so a
    dependency sync mid-update dies with access-denied and strands the venv
    half-updated (ryanc's brotlicffi/_sodium.pyd incidents, July 2026).

    Killing them from here is pointless — the Desktop app supervises its
    backend and respawns it within seconds — so the caller should refuse and
    tell the user to close the app instead. Returns ``(pid, name, cmdline)``
    tuples; empty off-Windows / without psutil / when nothing matches. The
    calling process and its ancestors are always excluded (a CLI ``hermes
    update`` itself runs from the venv python). Never raises.
    """
    if not _m()._is_windows():
        return []
    try:
        import psutil
    except Exception:
        return []

    venv_dir = _m().PROJECT_ROOT / "venv"
    try:
        venv_prefix = str(venv_dir.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        venv_prefix = str(venv_dir).lower().rstrip(os.sep) + os.sep
    try:
        root_prefix = str(_m().PROJECT_ROOT.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        root_prefix = str(_m().PROJECT_ROOT).lower().rstrip(os.sep) + os.sep

    skip: set[int] = set(exclude_pids or set())
    skip.add(os.getpid())
    try:
        from gateway.status import looks_like_gateway_command_line as _is_gw
    except Exception:
        _is_gw = None
    try:
        for anc in psutil.Process().parents():
            # #87594: do NOT blanket-exclude ancestors. When `/update` runs
            # from a messaging platform the updater is a CHILD of the gateway
            # — excluding all ancestors hides the gateway from the scan, so
            # the pause machinery downstream never sees the one process it
            # exists to stop, and the update dead-ends on `venv-blocked`.
            # A GATEWAY ancestor stays visible (the pause path stops it
            # gracefully; a detached child updater survives its parent's
            # stop on Windows). Every other ancestor (shells, terminals,
            # this CLI's own venv python chain) stays excluded — an updater
            # must never nominate its own interactive ancestry as blockers.
            try:
                anc_cmdline = " ".join(anc.cmdline() or [])
            except Exception:
                anc_cmdline = ""
            if _is_gw is not None and anc_cmdline and _is_gw(anc_cmdline):
                continue
            skip.add(int(anc.pid))
    except Exception:
        pass

    matches: list[tuple[int, str, str]] = []
    try:
        # On Windows, prefetching cmdline and cwd performs two expensive
        # per-process queries. A busy workstation can have 500+ processes, so
        # querying those fields for every unrelated process can exceed the
        # Desktop preflight watchdog. First collect only cheap identity fields;
        # fetch cmdline/cwd lazily for plausible Python/uv/Hermes candidates.
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
        if not exe or pid is None or int(pid) in skip:
            continue
        try:
            exe_norm = str(Path(exe).resolve()).lower()
        except (OSError, ValueError):
            exe_norm = str(exe).lower()
        # Primary match: the executable itself lives under this venv
        # (venv\Scripts\python(w).exe — the desktop backend / gateway case).
        is_holder = exe_norm.startswith(venv_prefix)
        name = str(info.get("name") or Path(exe).name)
        name_low = name.lower()

        if not is_holder and not (
            name_low.startswith(("python", "pypy"))
            or name_low in {"uv.exe", "uvx.exe", "hermes.exe"}
        ):
            continue

        try:
            cmdline_raw = " ".join(proc.cmdline() or [])
        except Exception:
            cmdline_raw = ""
        cmdline_low = cmdline_raw.lower()
        # Fallback: uv/base-interpreter trampolines run a python whose exe is
        # OUTSIDE the venv but which still imports from it and holds its .pyd
        # files. Catch those by what they're running: a cmdline that references
        # this venv's path, or a `-m hermes_cli.main ...` invocation tied to
        # this install (install root in the cmdline or as the working dir).
        if not is_holder and venv_prefix in cmdline_low:
            is_holder = True
        if not is_holder and "hermes_cli.main" in cmdline_low:
            try:
                cwd_low = str(proc.cwd() or "").lower().rstrip(os.sep) + os.sep
            except Exception:
                cwd_low = os.sep
            if root_prefix in cmdline_low or cwd_low.startswith(root_prefix):
                is_holder = True
        if not is_holder:
            continue
        name = info.get("name") or Path(exe).name
        # Return the FULL cmdline: callers match against it (the Desktop
        # preflight's pausable-gateway exemption parses for `gateway run`).
        # Truncating here cut long managed-runtime interpreter paths before
        # the `-m hermes_cli.main gateway run` argv, so autostarted gateways
        # were misreported as blockers and the update dead-ended. Truncate
        # only at display time.
        matches.append((int(pid), str(name), cmdline_raw))
    return matches


_SELF_LOCKING_NATIVE_MODULES: dict[str, tuple[str, str]] = {
    "cryptography.hazmat.bindings._rust": ("cryptography (_rust.pyd)", "cryptography"),
    "yaml._yaml": ("PyYAML (_yaml.pyd)", "pyyaml"),
}


def _dependency_sync_would_rewrite(dist_name: str) -> bool | None:
    """Whether ``uv pip install -e .[all]`` would replace *dist_name*'s files.

    Compares the installed distribution version against every applicable
    requirement for it in the on-disk ``pyproject.toml`` (base dependencies
    plus all optional extras).  Returns:

    - ``False`` — installed version satisfies every pin: the resolver will
      leave the wheel alone, so a mapped extension is NOT at risk.
    - ``True``  — some pin is not satisfied (or the distribution is
      missing): the sync will rewrite it.
    - ``None``  — could not determine (parse failure, unparseable pins).

    Never raises.  Callers treat ``None`` as fail-OPEN (no deferral): a
    module in the registry can be loaded by every process (PyYAML), so
    deferring on uncertainty would recreate the #86735 always-firing loop.
    """
    try:
        from importlib import metadata as _ilmd

        installed = _ilmd.version(dist_name)
    except Exception:
        return True  # not installed → the sync will definitely install it
    try:
        import tomllib

        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        from packaging.version import Version

        pyproject = _m().PROJECT_ROOT / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = data.get("project") or {}
        req_strings: list[str] = list(project.get("dependencies") or [])
        for extra_reqs in (project.get("optional-dependencies") or {}).values():
            req_strings.extend(extra_reqs or [])

        target = canonicalize_name(dist_name)
        installed_v = Version(installed)
        saw_pin = False
        for req_str in req_strings:
            try:
                req = Requirement(req_str)
            except Exception:
                continue
            if canonicalize_name(req.name) != target:
                continue
            if req.marker is not None and not req.marker.evaluate():
                continue
            saw_pin = True
            if installed_v not in req.specifier:
                return True
        if saw_pin:
            return False
        # Not pinned anywhere in pyproject: the resolver may still move it
        # as a transitive — we cannot cheaply predict that, so stay honest
        # about the uncertainty.
        return None
    except Exception:
        return None


def _detect_self_loaded_native_modules() -> list[str]:
    """Native venv extensions loaded into THIS process that the sync would rewrite.

    Returns display names (empty off Windows — POSIX lets a running process
    keep using an unlinked inode, so self-locking is a Windows-only hazard).
    A loaded module whose installed version already satisfies the on-disk
    pyproject pins is NOT reported: the dependency sync will not touch its
    files, so there is no swap at risk (#86735 — the always-firing variant
    of this preflight bricked every Windows update).  Never raises.
    """
    if not _m()._is_windows():
        return []
    found = []
    for prefix, (display, dist) in _SELF_LOCKING_NATIVE_MODULES.items():
        if prefix not in sys.modules:
            continue
        # Defer ONLY on a CONFIRMED pending rewrite. An "unknown" result
        # (unreadable/unparseable pyproject, no pin found) must fail OPEN:
        # PyYAML is loaded in every CLI process, so treating unknown as
        # at-risk would re-create the exact always-firing loop this guard's
        # first version caused (#86735). The downside of a missed deferral
        # is the pre-existing failure mode — a mid-sync os error 5 that the
        # marker recovery already handles — which is strictly less harmful
        # than an update that can never run.
        if _m()._dependency_sync_would_rewrite(dist) is not True:
            continue
        found.append(display)
    return sorted(set(found))


def _abort_dependency_sync_if_self_locked(gateway_resume=None) -> None:
    """Defer the venv rewrite when THIS process holds something it must replace.

    Runs at the last moment before the venv rewrite — after the code swap —
    so the on-disk pyproject reflects the update target and a deferral
    leaves the user on NEW code with only the dependency install pending.
    No-op when nothing at-risk is held.

    Two hazards, both "this process holds a file the sync must replace", and
    they end differently because their recoveries differ:

    - A mapped native extension (``.pyd``).  Exit 2 and let the next launch's
      marker recovery finish the install: that launch runs the install before
      importing anything heavy, so it maps nothing and the swap succeeds.

    - The ``hermes.exe`` console shim we were launched from (#88838, #89599).
      The marker cannot help here — every future ``hermes`` launch is also the
      shim, so deferring to the next launch defers forever.  Hand the install
      to a child under the venv interpreter and exit, releasing the shim.
    """
    locked = _m()._detect_self_loaded_native_modules()
    if locked:
        _m()._defer_update_for_self_lock(locked)
        if gateway_resume is not None:
            _m()._resume_windows_gateways_after_update(gateway_resume)
        sys.exit(2)

    if _m()._reexec_dependency_sync_off_windows_shim():
        if gateway_resume is not None:
            _m()._resume_windows_gateways_after_update(gateway_resume)
        sys.exit(0)


def _defer_update_for_self_lock(loaded: list[str]) -> None:
    """Bail out before the dependency sync when the updater holds a lock.

    The install cannot win this race from inside the locked process — even
    killing threads would not unmap the image — so defer it: drop the
    update-incomplete marker (next launch's fresh process completes the
    install before importing anything heavy), explain, and exit 2 like the
    other preflight refusals.
    """
    print("✗ This updater process has already loaded native venv modules that")
    print("  the dependency sync must replace:")
    for name in loaded:
        print(f"    {name}")
    print()
    print("  On Windows a mapped extension cannot be replaced by the process")
    print("  holding it. The code update has been applied; only the dependency")
    print("  sync has been deferred: the next `hermes` launch will complete it")
    print("  in a fresh process before anything imports these modules.")
    _m()._write_update_incomplete_marker()


_HOLDER_VALUE_FLAGS_FALLBACK = frozenset(
    {
        "--profile", "-p", "--config",
        "--model", "-m", "--provider", "--reasoning",
        "--toolsets", "-t", "--skills", "-s",
        "--continue", "-c", "--resume", "-r",
        "--oneshot", "-z", "--in", "--usage-file",
    }
)


def _holder_value_flags() -> frozenset:
    """Top-level CLI flags that consume a value — derived from the REAL parser.

    Introspects ``build_top_level_parser()`` (every option with nargs != 0)
    so the holder classifier can never drift from the argparse surface
    (#91869 review: a handwritten subset misparsed ``--reasoning high
    serve`` as subcommand ``high`` and ``-m dashboard serve`` as
    ``dashboard`` — recreating the wrong-hint class). The pre-argparse
    profile selectors (``--profile``/``-p``, ``--config``) are added
    explicitly since they are stripped before argparse sees argv. Falls
    back to a static snapshot when the parser cannot be imported (the
    updater must classify holders even mid-upgrade on a broken tree).
    Cached per process.
    """
    if _u()._holder_value_flags_cache is not None:
        return _u()._holder_value_flags_cache
    flags: set[str] = {"--profile", "-p", "--config"}
    try:
        from hermes_cli._parser import build_top_level_parser

        parser = build_top_level_parser()[0]
        for action in parser._actions:
            if action.option_strings and action.nargs != 0:
                flags.update(action.option_strings)
        _u()._holder_value_flags_cache = frozenset(flags)
    except Exception:
        _u()._holder_value_flags_cache = _HOLDER_VALUE_FLAGS_FALLBACK
    return _u()._holder_value_flags_cache


def _hermes_holder_subcommand(cmdline: str) -> str | None:
    """The actual Hermes SUBCOMMAND a venv-holder argv runs, or None.

    Token-based, never substring (#90778: ``kanban --preserve-cache``
    contained \"serve\" and got labeled as the Desktop backend). Finds the
    ``hermes_cli.main`` / ``hermes(.exe)`` entry token, then returns the
    first following token that is not a flag or a flag's value. Profile
    selectors (``--profile X``, ``-p X``) are skipped like the canonical
    gateway matcher does. Returns None when no subcommand can be
    determined — callers must NOT guess a label in that case.
    """
    try:
        import shlex

        tokens = shlex.split(cmdline, posix=False)
    except Exception:
        tokens = cmdline.split()

    entry_idx: int | None = None
    for i, token in enumerate(tokens):
        low = token.lower().strip('"')
        if low.endswith("hermes_cli.main") and i > 0 and tokens[i - 1] == "-m":
            entry_idx = i
            break
        base = low.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        if base in ("hermes", "hermes.exe"):
            entry_idx = i
            break
    if entry_idx is None:
        return None

    value_flags = _holder_value_flags()
    i = entry_idx + 1
    while i < len(tokens):
        token = tokens[i]
        if token in value_flags or token.split("=", 1)[0] in value_flags:
            # --flag value consumes two tokens; --flag=value consumes one.
            i += 1 if "=" in token else 2
            continue
        if token.startswith("-"):
            i += 1
            continue
        return token.lower()
    return None


def _format_venv_python_holders_message(matches: list[tuple[int, str, str]]) -> str:
    """Explain which venv processes block the update and how to clear them.

    Holder labels come from the parsed SUBCOMMAND, never substring matching
    (#90778): a standalone ``hermes dashboard`` must not be labeled as the
    Desktop backend (advice to close an app that isn't running), and flags
    like ``--preserve-cache`` must not match \"serve\". Unknown argv gets no
    hint rather than a wrong one.
    """
    lines = [
        "✗ Other Hermes processes are running from this install's venv:",
    ]
    hint_by_subcommand = {
        "serve": "  ← Hermes backend (if the Desktop app is open, close it)",
        "dashboard": "  ← hermes dashboard (stop it: hermes dashboard stop, or close that terminal)",
        "gateway": "  ← gateway",
    }
    for pid, name, cmdline in matches[:6]:
        sub = _hermes_holder_subcommand(cmdline)
        hint = hint_by_subcommand.get(sub or "", "")
        lines.append(f"  PID {pid}  {name}  {cmdline[:120]}{hint}")
    if len(matches) > 6:
        lines.append(f"  ... and {len(matches) - 6} more")
    lines.append("")
    lines.append(
        "  On Windows these keep native extension files (.pyd) locked, so the"
    )
    lines.append(
        "  dependency update would fail partway and leave a broken install."
    )
    lines.append(
        "  Close the Hermes desktop app / other Hermes terminals, then re-run:"
    )
    lines.append("    hermes update")
    lines.append("  (or use `hermes update --force-venv` to proceed anyway at your own risk)")
    return "\n".join(lines)


def _venv_launcher_ancestors(pids: list[int]) -> list[int]:
    """Return venv-interpreter ancestors of *pids* that hold the install open.

    On Windows a gateway started through the venv shim is a **two-process
    chain**: ``venv\\Scripts\\python.exe`` (the launcher, which keeps native
    ``.pyd`` files from the venv mapped) spawns the actual interpreter from
    uv's managed CPython directory (``AppData\\Roaming\\uv\\python\\...``).
    The gateway writes its PID file from the *child*, so
    ``find_gateway_pids()`` — and therefore this module's pause set — only
    ever sees the uv-side worker.

    ``_detect_venv_python_processes()`` matches on the venv path prefix, so
    the guard downstream of the pause sees the *launcher* instead. The two
    sets are disjoint, which meant a paused gateway still tripped the
    venv-holder guard and aborted the update every time (the Desktop
    "venv-blocked: N process(es) hold the install" dead-end, where the
    reported holder is a gateway the updater believes it already stopped).

    Walking one hop up from each mapped gateway PID and keeping ancestors
    that live under the project venv closes the gap. Only the venv-side
    parent is returned — unrelated ancestors (the Scheduled Task's
    ``cmd.exe``, an operator's shell) are ignored so we never widen the
    blast radius beyond the gateway's own launcher. Never raises.
    """
    if not _m()._is_windows() or not pids:
        return []
    try:
        import psutil
    except Exception:
        return []

    venv_dir = _m().PROJECT_ROOT / "venv"
    try:
        venv_prefix = str(venv_dir.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        venv_prefix = str(venv_dir).lower().rstrip(os.sep) + os.sep

    # Never return ourselves or our own ancestry: a CLI ``hermes update``
    # runs from the venv python and would otherwise nominate itself.
    # Same #87594 carve-out as _detect_venv_python_processes: a GATEWAY
    # ancestor is not "our own ancestry" in the interactive sense — it is
    # the process the pause machinery must see (the /update-from-gateway
    # topology makes the updater the gateway's child).
    try:
        from gateway.status import looks_like_gateway_command_line as _is_gw
    except Exception:
        _is_gw = None
    skip: set[int] = {os.getpid()}
    try:
        for anc in psutil.Process().parents():
            try:
                anc_cmdline = " ".join(anc.cmdline() or [])
            except Exception:
                anc_cmdline = ""
            if _is_gw is not None and anc_cmdline and _is_gw(anc_cmdline):
                continue
            skip.add(int(anc.pid))
    except Exception:
        pass

    found: list[int] = []
    for pid in pids:
        try:
            parent = psutil.Process(int(pid)).parent()
        except Exception:
            continue
        if parent is None:
            continue
        ppid = int(parent.pid)
        if ppid in skip or ppid in found or ppid in set(pids):
            continue
        try:
            exe = (parent.exe() or "").lower()
        except Exception:
            continue
        if exe.startswith(venv_prefix):
            found.append(ppid)
    return found


def _leftover_pausable_gateway_pids(
    matches: list[tuple[int, str, str]],
) -> list[int] | None:
    """PIDs from *matches* when every remaining venv holder is a pausable gateway.

    ``_pause_windows_gateways_for_update()`` stops every gateway its discovery
    finds, but the venv-holder guard downstream sees the process table as it
    is *now*: a gateway respawned by its supervisor (Scheduled Task, login
    watchdog) inside the pause→guard window, or one started through a spawn
    path the discovery does not map, still holds venv ``.pyd`` files and
    would dead-end the update — an abort pointed at exactly the kind of
    process the pause machinery exists to stop.

    Holders are classified with the same matcher the Desktop preflight uses
    to exempt them (``_is_pausable_gateway``), so the preflight's exemption
    and this guard's tolerance cannot drift apart — matcher drift between
    two views of the same process table is what produced the launcher/worker
    dead-end fixed above. The scan captures only a 120-char cmdline prefix,
    so the live argv is re-read where psutil allows; an unreadable argv
    falls back to the captured prefix.

    Returns ``None`` when any holder is not a pausable gateway — an operator
    REPL, a stray script, or the Desktop backend has no pause machinery
    downstream, and the guard must keep refusing exactly as before.
    """
    from hermes_cli._scan_venv_blockers import _is_pausable_gateway

    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None

    pids: list[int] = []
    for pid, _name, cmdline in matches:
        argv = cmdline
        if psutil is not None:
            try:
                argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
            except Exception:
                pass
        if not _is_pausable_gateway(argv):
            return None
        pids.append(int(pid))
    return pids


def _ledger_manual_serve_holders(
    matches: list[tuple[int, str, str]],
) -> list[dict]:
    """Ledger entries for venv holders that are MANUAL serve/dashboard backends.

    Positive identity only (#63206): the process self-registered in the spawn
    ledger with purpose serve/dashboard, its (pid, create_time) still matches
    a live process, and its recorded spawner is NOT alive (a Desktop-owned
    backend keeps its live Electron spawner and must keep the refusal — the
    app would respawn what we kill; a PowerShell-launched serve has no live
    Hermes spawner). Returns the full ledger entries so the relauncher can
    rebuild the launch command from structured host/port/profile instead of
    parsing argv.
    """
    try:
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead
    except Exception:
        return []
    holder_pids = {int(pid) for pid, _name, _cmd in matches}
    out: list[dict] = []
    for entry in ledger_entries():
        if entry.get("purpose") not in ("serve", "dashboard"):
            continue
        pid = entry.get("pid")
        if not isinstance(pid, int) or pid not in holder_pids:
            continue
        if spawner_is_dead(entry) is False:
            continue  # live Desktop supervisor owns it — keep refusing
        out.append(entry)
    return out


def _serve_relaunch_commands(entries: list[dict]) -> list[list[str]]:
    """Rebuild launch commands for stopped serves from structured identity.

    Uses the ledger's host/port/profile fields — never argv parsing (a
    joined argv string cannot round-trip Windows paths with spaces). Entries
    without a recorded port are skipped; the caller prints the manual hint
    for those.
    """
    commands: list[list[str]] = []
    hermes = None
    try:
        scripts_dir = _m()._venv_scripts_dir()
        if scripts_dir is not None:
            for name in ("hermes.exe", "hermes"):
                candidate = scripts_dir / name
                if candidate.is_file():
                    hermes = str(candidate)
                    break
    except Exception:
        hermes = None
    if hermes is None:
        hermes = "hermes"
    for entry in entries:
        port = entry.get("port")
        if not isinstance(port, int) or port <= 0:
            continue
        cmd = [hermes]
        profile = str(entry.get("profile") or "")
        if profile and profile != "default":
            cmd += ["--profile", profile]
        cmd.append(str(entry.get("purpose")))
        host = str(entry.get("host") or "")
        if host:
            cmd += ["--host", host]
        cmd += ["--port", str(port)]
        commands.append(cmd)
    return commands


def _relaunch_stopped_serves(token: dict) -> None:
    """Idempotent atexit relaunch of manual serves stopped by the venv guard.

    Mirrors the gateway resume token contract: `pending` flips False on the
    first invocation so the explicit call and the atexit registration cannot
    double-spawn (#63206).
    """
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
        print(
            "  ⚠ Some stopped backends could not be relaunched automatically; "
            "restart them manually (hermes serve --host <ip> --port <port>)."
        )
    try:
        from hermes_cli.update_receipt import record_step

        record_step(
            "serve_relaunch",
            not failed and not skipped,
            f"relaunched={len(commands) - len(failed)} failed={len(failed)} skipped={skipped}",
        )
    except Exception:
        pass


def _orphaned_desktop_backend_pids(
    matches: list[tuple[int, str, str]],
) -> list[tuple[int, int]] | None:
    """PIDs from *matches* when every remaining holder is an ORPHANED backend.

    The venv-holder guard refuses on the Desktop app's ``serve`` backend by
    design: while the Desktop is open, killing its backend is futile (the app
    supervises and respawns it within seconds), so the user must close the
    app. But in the GUI-updater handoff path the Desktop has *already
    exited* — by contract it tree-kills its backends and waits for the venv
    shim before spawning hermes-setup, and the update-in-progress marker
    parks any relaunched Desktop from spawning a fresh backend (#50238). A
    ``serve`` backend still holding the venv at that point is a straggler
    whose supervisor is gone: SIGTERM raced its spawn, or it belongs to a
    crashed window. Nothing will respawn it, and refusing on it dead-ends
    the update with "Hermes is still running" while the user stares at zero
    open windows (ryanc's 2026-08-09 01:59/02:17 failures).

    A holder qualifies only when BOTH hold:

    - its cmdline is a Hermes backend (``hermes_cli.main`` + ``serve`` /
      ``dashboard``), and
    - its supervising parent is demonstrably gone: the parent PID no longer
      exists, or the PID was reused (parent created *after* the child).

    Tree-aware: the scanner can return an orphaned backend AND one of its
    managed-runtime descendants (the ``.hermes-runtime`` interpreter child)
    in the same holder set. That descendant has a live parent — the orphaned
    backend itself — and isn't a ``serve`` cmdline, so per-process rules
    would refuse a set that is entirely safe to reap. Holders that sit
    inside an accepted orphan root's tree are therefore folded into that
    root (only roots are returned; ``taskkill /T`` reaps the descendants).

    Any other live-parent backend (the Desktop is still open), non-backend
    holder outside an orphan tree, or unprovable case disqualifies the whole
    set — the guard must keep refusing exactly as before. Returns ``None``
    in that case, or when psutil is unavailable (can't prove orphanhood →
    refuse). Never raises.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None

    def _is_backend(argv_low: str) -> bool:
        return "hermes_cli.main" in argv_low and (
            " serve" in argv_low or " dashboard" in argv_low
        )

    # Pass 1: find orphaned backend ROOTS among the holders.
    roots: list[tuple[int, int]] = []
    remaining: list[tuple[int, str]] = []  # (pid, argv_low) still to justify
    for pid, _name, cmdline in matches:
        argv = cmdline
        try:
            argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
        except psutil.NoSuchProcess:
            # Holder exited between scan and classification — nothing to
            # reap, nothing blocking. Skip it.
            continue
        except Exception:
            pass
        low = argv.lower()
        if not _is_backend(low):
            remaining.append((int(pid), low))
            continue
        try:
            proc = psutil.Process(int(pid))
            # Fingerprint from the SAME psutil handle used for classification
            # below, quantized to centiseconds — the exact scheme
            # gateway.status.get_process_start_time uses on Windows, so the
            # value round-trips through pid_is_hermes at kill time. (Reading
            # /proc/<pid>/stat here instead would consult the HOST process
            # table and use different units.)
            process_start_time = int(round(proc.create_time() * 100))
        except psutil.NoSuchProcess:
            # The candidate itself exited during classification; there is
            # nothing left to reap and no identity to pass to taskkill.
            continue
        except Exception:
            return None

        try:
            ppid = proc.ppid()
            parent = psutil.Process(ppid) if ppid else None
            if parent is not None and parent.is_running():
                # PID-reuse check: a "parent" created after its child is a
                # recycled PID, not the real (dead) supervisor.
                if parent.create_time() <= proc.create_time():
                    # Live parent — NOT a root. But it may still be a
                    # descendant of an orphan root: the venv python.exe is
                    # a trampoline that re-execs the uv-managed interpreter
                    # with the SAME backend argv, so the worker half of the
                    # two-process chain lands here. Defer to pass 2 instead
                    # of refusing outright.
                    remaining.append((int(pid), low))
                    continue
        except psutil.NoSuchProcess:
            pass  # parent gone → orphan
        except Exception:
            return None
        roots.append((int(pid), process_start_time))

    # Pass 2: every non-backend holder must be a descendant of an accepted
    # orphan root — then it dies with the root's tree reap. Anything else
    # (operator REPL, stray script) keeps the refusal.
    root_set = {pid for pid, _start_time in roots}
    for pid, _low in remaining:
        if not root_set:
            return None
        try:
            ancestors = {int(a.pid) for a in psutil.Process(pid).parents()}
        except psutil.NoSuchProcess:
            continue  # exited already
        except Exception:
            return None
        if not (ancestors & root_set):
            return None
    return roots


def _ledger_reapable_backend_pids(
    matches: list[tuple[int, str, str]],
) -> list[int]:
    """PIDs positively identified by the spawn ledger as orphaned backends.

    The strongest rung: instead of inferring lineage from PPIDs or cmdline
    shape, look each venv holder up in the machine spawn ledger
    (``hermes_cli.process_identity``). A holder qualifies when ALL of:

    - its ``(pid, create_time)`` matches a live ledger entry (PID reuse
      cannot forge this pair);
    - the entry's purpose is a reapable backend kind (serve/dashboard/
      gateway — never interactive processes);
    - the entry's recorded SPAWNER is provably dead (``spawner_is_dead``).

    Unlike the heuristic rungs, this is safe in ANY update context — no
    hand-off contract needed — because the ownership claim is explicit: the
    process itself declared who supervises it, and that supervisor is gone.
    Holders not in the ledger are simply not returned (they fall through to
    the later rungs); they never disqualify the identified ones. Never raises.
    """
    try:
        from hermes_cli.process_identity import (
            REAPABLE_PURPOSES,
            ledger_entries,
            spawner_is_dead,
        )

        entries = ledger_entries()
    except Exception:
        return []
    by_pid = {e.get("pid"): e for e in entries if isinstance(e.get("pid"), int)}
    roots: list[int] = []
    for pid, _name, _cmdline in matches:
        entry = by_pid.get(int(pid))
        if not entry:
            continue
        if entry.get("purpose") not in REAPABLE_PURPOSES:
            continue
        if spawner_is_dead(entry) is True:
            roots.append(int(pid))
    return roots


def _handoff_reapable_backend_pids(
    matches: list[tuple[int, str, str]],
) -> list[int] | None:
    """PIDs of Hermes ``serve``/``dashboard`` backends safe to reap during a
    GUI-updater hand-off, INCLUDING ones with a still-live parent.

    Complements ``_orphaned_desktop_backend_pids``, which only reaps backends
    whose supervisor is provably dead. That check returns ``None`` (keep
    refusing) the moment ANY holder still has a live parent — which is exactly
    the case that produced the field incident this fixes: a Windows Desktop
    update hand-off (``update --yes --gateway --force``) left a *swarm* of
    per-profile ``serve`` backends (mr-tester, probe-inherit, turqoise, …)
    holding ``cryptography\\_rust.pyd``. Several still had a lingering
    parent (the tearing-down Electron process, or the two-hop venv
    launcher→worker chain mid-exit), so the orphan check disqualified the
    WHOLE set and the update dead-ended — the user saw a 12-minute hang, then
    force-closed, and the half-done state stranded bot sessions.

    The hand-off is the safe signal: when the update-incomplete marker is
    present (the GUI updater claimed it) AND this is a ``--gateway`` hand-off
    run AND no live Desktop shim (``hermes.exe``) is open, NOTHING legitimate
    is supervising or respawning a ``serve`` backend from this venv — by the
    hand-off contract the Desktop tree-kills its backends and parks any
    relaunch behind the marker (#50238). Any ``serve`` backend still holding
    the venv here is therefore a leak, live parent or not, and reaping its
    tree is correct rather than a race.

    Guarded conservatively:

    - Only Hermes backends (``hermes_cli.main`` + ``serve``/``dashboard``)
      from THIS install's venv qualify; a non-backend holder (operator REPL,
      stray script) disqualifies the whole set → ``None`` (keep refusing), so
      we never widen the blast radius during a hand-off.
    - Only runs when the CALLER has confirmed the hand-off context
      (``args.gateway`` AND a claimed update-incomplete marker AND no live
      ``hermes.exe`` shim) — outside that gate this function is never called
      and the stricter orphan-only path stands.
    - psutil unavailable → ``None`` (can't re-read argv to classify → refuse).

    Returns the backend root PIDs to tree-reap, or ``None`` to leave the
    decision to the caller's existing rungs. Never raises.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None

    def _is_backend(argv_low: str) -> bool:
        return "hermes_cli.main" in argv_low and (
            " serve" in argv_low or " dashboard" in argv_low
        )

    roots: list[int] = []
    for pid, _name, cmdline in matches:
        argv = cmdline
        try:
            argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
        except psutil.NoSuchProcess:
            # Exited between scan and classification — nothing to reap.
            continue
        except Exception:
            pass
        if not _is_backend(argv.lower()):
            # A non-backend holder during a hand-off is unexpected; refuse the
            # whole set rather than reap something we cannot justify.
            return None
        roots.append(int(pid))

    return roots or None


def _stop_process_trees(
    pids: list[int] | list[tuple[int, int]],
) -> None:
    """Force-stop each PID with its full child tree (Windows).

    ``taskkill /T /F`` mirrors the Desktop's ``forceKillProcessTree`` and
    install.ps1's venv sweep: stopping only the parent can leave a managed
    ``.hermes-runtime`` interpreter child alive and holding the install open
    (#70026). Best effort; never raises.
    """
    from gateway.status import get_process_start_time
    from hermes_cli._subprocess_compat import pid_is_hermes, windows_hide_flags

    for entry in pids:
        if isinstance(entry, tuple):
            pid, expected_start_time = entry
        else:
            pid = int(entry)
            expected_start_time = get_process_start_time(pid)
        try:
            if expected_start_time is None:
                logger.debug(
                    "Skipping taskkill of PID %s: process identity unavailable",
                    pid,
                )
                continue
            if not pid_is_hermes(
                pid,
                expected_start_time=expected_start_time,
            ):
                logger.debug(
                    "Skipping taskkill of non-Hermes or changed PID %s",
                    pid,
                )
                continue
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=windows_hide_flags(),
            )
        except Exception as exc:
            logger.debug("Could not stop process tree %s: %s", pid, exc)


def _looks_like_desktop_control_plane(cmdline: str) -> bool:
    """True for this-install ``hermes serve`` / ``hermes dashboard`` argv.

    That is the Desktop control plane, not the messaging gateway. Serve and
    dashboard do not host platform adapters (#92091); do not feed this into
    ``looks_like_gateway_command_line``.

    Token-based via the parser-derived subcommand classifier — never
    substring (#90778/#91869: ``kanban --preserve-cache`` contains "serve",
    ``-m dashboard chat`` contains " dashboard"; both are NOT control
    planes). A cmdline whose subcommand cannot be determined is NOT a
    control plane — callers must not guess ownership.
    """
    if "hermes_cli.main" not in (cmdline or "").lower():
        return False
    return _hermes_holder_subcommand(cmdline) in ("serve", "dashboard")


def _desktop_owns_gateway_lifecycle() -> bool:
    """True when Desktop currently supervises this install's control plane.

    The updater must not steal gateway start in that case: Desktop owns
    start/stop via ``/api/gateway/*``. This is *not* proof messaging is
    already served — a live serve process is the control plane, and the
    gateway is a detached sibling (#76129 / #92091).

    Prefer the spawn ledger (owned identity). Fall back to the install-scoped
    venv-holder scan already used by the lock guard; an orphaned control-plane
    process (supervisor gone) does not count.
    """
    try:
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead

        for entry in ledger_entries():
            if entry.get("purpose") not in ("serve", "dashboard"):
                continue
            if spawner_is_dead(entry) is False:
                return True
    except Exception as exc:
        logger.debug("Desktop-lifecycle ledger probe failed: %s", exc)

    try:
        import psutil
    except Exception:
        psutil = None

    try:
        holders = _m()._detect_venv_python_processes()
    except Exception as exc:
        logger.debug("Desktop-lifecycle holder scan failed: %s", exc)
        return False

    for pid, _name, cmdline in holders:
        if not _looks_like_desktop_control_plane(cmdline):
            continue
        if psutil is None:
            # Cannot prove orphanhood; a live this-install control plane is
            # enough to refuse stealing gateway start.
            return True
        try:
            proc = psutil.Process(int(pid))
            parent = proc.parent()
            if parent is None or not parent.is_running():
                continue
            if parent.create_time() > proc.create_time():
                continue
            return True
        except Exception:
            continue
    return False


def _refuse_gateway_ancestor_tree_kill(
    pids: list[int], *, gateway_mode: bool
) -> bool:
    """Refuse a plain Windows update that would kill its own process tree.

    A chat agent can launch plain ``hermes update`` through its terminal tool.
    In that topology the updater is a child of the gateway.  The leftover
    holder recovery below uses ``taskkill /T /F`` on Windows, so force-stopping
    that gateway also kills the updater before it can mutate the checkout
    (#98814).

    ``/update`` uses the supported ``--gateway`` hand-off and is deliberately
    exempt: it detaches the updater and provides file-based progress/result
    delivery.  For every other invocation, refuse only when a nominated
    gateway is positively identified as this process's ancestor.  If ancestry
    cannot be established, preserve the existing holder recovery behavior.
    """
    if gateway_mode or not pids:
        return False

    try:
        from hermes_cli.gateway import _is_pid_ancestor_of_current_process

        ancestors = [
            int(pid)
            for pid in pids
            if _is_pid_ancestor_of_current_process(int(pid))
        ]
    except Exception as exc:
        logger.debug("Could not inspect gateway ancestry before tree-kill: %s", exc)
        return False

    if not ancestors:
        return False

    rendered = ", ".join(str(pid) for pid in ancestors)
    print(
        "✗ Refusing to stop the gateway process tree because this updater "
        f"is running inside it (gateway PID(s): {rendered})."
    )
    print(
        "  On Windows, taskkill /T would terminate the updater before the "
        "update can run."
    )
    print("  From a chat platform, use `/update` instead.")
    print("  Otherwise, run `hermes update` from a separate terminal.")
    return True


def _path_uid(path) -> Optional[int]:
    """Owner uid of ``path`` via ``os.stat`` — ``None`` when unreadable.

    Separate seam so tests can simulate root-owned files without chown
    (which needs root). Never raises.
    """
    try:
        return os.stat(path, follow_symlinks=False).st_uid
    except OSError:
        return None


def _venv_foreign_owned_paths(venv_root, limit: int = 5) -> list:
    """Bounded scan for venv entries not owned by the current user (#83529).

    A venv that was ever touched by ``sudo pip`` / ``sudo hermes`` contains
    root-owned files (classically ``*.dist-info/INSTALLER``). A later normal
    ``hermes update`` then dies mid-mutation inside ``uv pip install -e .``
    ("Permission denied (os error 13)") with ``venv/bin/hermes`` already
    deleted — the CLI is bricked. Same philosophy as the contended-venv gate
    (#87331): a venv we cannot safely mutate is never mutated at all.

    Checks a deliberately BOUNDED set (no full recursion — must never be
    slow): the venv root, each direct entry of ``venv/bin``, the top-level
    entries of the first ``lib/python*/site-packages`` found, and the direct
    children of each ``*.dist-info`` there. Caps stat calls at ~2000 and
    returned paths at ``limit``. POSIX-only: returns ``[]`` on Windows
    (no ``os.geteuid``) and when running as root. Swallows every per-entry
    ``OSError`` and returns ``[]`` on any structural surprise — this helper
    must NEVER raise and must never add noticeable latency to update.

    Returns a list of ``(path_str, uid)`` tuples, at most ``limit`` long.
    """
    try:
        if not hasattr(os, "geteuid"):
            return []  # windows-footgun: ok — POSIX ownership concept only
        euid = os.geteuid()  # windows-footgun: ok — guarded by hasattr above
        if euid == 0:
            return []  # root can rewrite anything; nothing to refuse

        venv_root = Path(venv_root)
        budget = 2000  # max stat() calls — hard bound on preflight cost
        foreign: list = []

        def _check(p) -> bool:
            """stat one path; True while scan should continue."""
            nonlocal budget
            if budget <= 0 or len(foreign) >= limit:
                return False
            budget -= 1
            uid = _path_uid(p)
            if uid is not None and uid != euid:
                foreign.append((str(p), uid))
            return budget > 0 and len(foreign) < limit

        def _scan_dir(d, recurse_dist_info: bool = False) -> None:
            try:
                entries = list(os.scandir(d))
            except OSError:
                return
            for entry in entries:
                if not _check(entry.path):
                    return
                if recurse_dist_info and entry.name.endswith(".dist-info"):
                    try:
                        children = list(os.scandir(entry.path))
                    except OSError:
                        continue
                    for child in children:
                        if not _check(child.path):
                            return

        if not _check(venv_root):
            return foreign[:limit]
        _scan_dir(venv_root / "bin")

        # First lib/python*/site-packages (POSIX venv layout).
        site_packages = next(
            iter(sorted(venv_root.glob("lib/python*/site-packages"))), None
        )
        if site_packages is not None:
            _scan_dir(site_packages, recurse_dist_info=True)

        return foreign[:limit]
    except Exception:
        # Preflight is advisory: any structural surprise means "no verdict",
        # never a crashed or blocked update.
        return []


def _refuse_update_if_venv_foreign_owned(project_root) -> None:
    """Refuse-before-mutate ownership gate for the dependency install (#83529).

    Runs after the code pull (pulling code is safe) and immediately before
    the first venv mutation. If the venv contains files owned by another
    uid, the ``uv pip install -e .`` below would die mid-mutation and brick
    the install — so refuse up front, with the exact recovery command,
    while the venv is still fully intact. No subprocess calls here: update
    tests mock ``subprocess.run`` with sequenced side effects.
    """
    foreign = _venv_foreign_owned_paths(Path(project_root) / "venv")
    if not foreign:
        return
    print("\n✗ Update stopped: this install's venv contains files owned by another user.")
    print("  Updating now would fail midway (Permission denied) and leave Hermes broken.")
    print("  This usually happens after running hermes or pip with sudo. Offending paths:")
    for p, uid in foreign:
        print(f"    - {p} (owner uid {uid})")
    print("\n  Fix ownership, then re-run the update:")
    print(f"    sudo chown -R $(id -un): {project_root}")
    print("    hermes update")
    print("\n  Nothing in the venv was modified.")
    sys.exit(1)

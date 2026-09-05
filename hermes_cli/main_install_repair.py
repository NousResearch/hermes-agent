"""Install/update recovery: interrupted-install markers, lazy-refresh repair, Windows shim quarantine, dependency verification.

Split out of ``hermes_cli/main.py``. Names that still live in main (``PROJECT_ROOT``, ...)
are imported lazily inside the functions that use them (avoids an import cycle).
"""

import contextlib
import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time as _time

from pathlib import Path
from hermes_cli import _early_recovery as _early_recovery_mod

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.main")


def _pyproject_project(debug_fmt: str | None = None) -> dict | None:
    """``[project]`` table of pyproject.toml, or ``None`` when absent/unreadable."""
    from hermes_cli.main import PROJECT_ROOT
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        import tomllib
        with pyproject.open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
    except Exception as exc:
        if debug_fmt:
            logger.debug(debug_fmt, exc)
        return None
    return project if isinstance(project, dict) else None


def _naive_requirement(spec: str) -> tuple[str, str]:
    """``(name, head)`` of a ``name OP version ; marker`` spec without ``packaging``."""
    head = spec.split(";", 1)[0].strip()
    bare = head
    for op in ("==", ">=", "<=", "~=", ">", "<", "!="):
        if op in bare:
            bare = bare.split(op, 1)[0]
            break
    return bare.strip().split("[", 1)[0].strip(), head


def _parse_requirements(raw_deps: list[str]) -> list[tuple[str, "object | None", str]]:
    """``(name, marker, head)`` per dep spec — ``packaging`` when importable, else a naive split."""
    parsed: list[tuple[str, "object | None", str]] = []
    try:
        from packaging.requirements import Requirement  # type: ignore
        for spec in raw_deps:
            try:
                req = Requirement(spec)
            except Exception:
                continue
            parsed.append((req.name, req.marker, spec.split(";", 1)[0].strip()))
    except Exception:
        for spec in raw_deps:
            name, head = _naive_requirement(spec)
            if name:
                parsed.append((name, None, head))
    return parsed


def _load_installable_optional_extras(group: str = "all") -> list[str]:
    """Return optional extras referenced by a dependency group (``all`` or ``termux-all``)."""
    optional_deps = (_pyproject_project() or {}).get("optional-dependencies", {})
    if not isinstance(optional_deps, dict):
        return []
    referenced: list[str] = []
    for ref in optional_deps.get(group, []):
        if "[" in ref and "]" in ref:
            name = ref.split("[", 1)[1].split("]", 1)[0]
            if name in optional_deps:
                referenced.append(name)
    return referenced


# Install-scoped breadcrumbs live next to the venv (not under $HERMES_HOME)
# because the venv is shared across profiles.
#   ``.update-incomplete``       — generic core ``.[all]`` install was interrupted;
#     cleared only after a confirmed full dependency reinstall/recovery.
#   ``.lazy-refresh-incomplete`` — lazy-backend refresh may have corrupted packages;
#     cleared only after import-probe repair confirms healthy (never on indeterminate).
# Narrow lazy probes must NEVER clear the generic core marker.
# See #58004.
def _update_marker_path() -> Path:
    from hermes_cli.main import PROJECT_ROOT
    return PROJECT_ROOT / ".update-incomplete"


def _lazy_refresh_marker_path() -> Path:
    from hermes_cli.main import PROJECT_ROOT
    return PROJECT_ROOT / ".lazy-refresh-incomplete"


def _pytest_owns_live_checkout(root: Path) -> bool:
    """True under pytest when ``root`` is this checkout: unsandboxed update/recovery tests must
    neither litter the live repo root with breadcrumbs (false-arming the developer's next launch)
    nor run a real reinstall against the executing venv (cf. ``managed_scope._under_pytest``)."""
    return "PYTEST_CURRENT_TEST" in os.environ and root == Path(__file__).resolve().parent.parent


def _clear_marker_file(path: Path, *, label: str) -> None:
    """Remove an update-recovery breadcrumb. Never raises."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("Could not clear %s marker: %s", label, exc)


def _clear_update_incomplete_marker() -> None:
    """Remove the interrupted core-install breadcrumb. Never raises."""
    _clear_marker_file(_update_marker_path(), label="update-incomplete")


def _clear_lazy_refresh_incomplete_marker() -> None:
    """Remove the interrupted lazy-refresh breadcrumb. Never raises."""
    _clear_marker_file(_lazy_refresh_marker_path(), label="lazy-refresh-incomplete")


def _claim_recovery_lock(lock_path: Path) -> bool:
    """Atomically claim the single-flight recovery lock; False when another process holds it.

    A crashed holder's stale lock is broken after an hour (well past any realistic install).
    Failing to CREATE the lock (read-only fs, perms) proceeds unlocked — the install itself
    will surface the real problem."""
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
    except FileExistsError:
        try:
            if _time.time() - lock_path.stat().st_mtime > 3600:
                lock_path.unlink()
        except OSError:
            pass
        return False
    except OSError as exc:
        logger.debug("Could not create install-recovery lock: %s", exc)
    return True


@contextlib.contextmanager
def _stdout_to_stderr():
    """Route Python prints AND the fd 1 that pip/uv inherit to stderr: launches whose stdout is
    a protocol stream (``hermes acp`` speaks JSON-RPC on stdout) must never get install noise."""
    saved_stdout_fd = None
    saved_sys_stdout = sys.stdout
    try:
        saved_stdout_fd = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        saved_stdout_fd = None
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = saved_sys_stdout
        if saved_stdout_fd is not None:
            try:
                os.dup2(saved_stdout_fd, 1)
                os.close(saved_stdout_fd)
            except OSError:
                pass


def _recover_from_interrupted_install() -> None:
    """Finish update work left half-done by a prior ``hermes update``.

    ``.update-incomplete`` recovers via full quarantined reinstall; ``.lazy-refresh-incomplete``
    via package-only import probes (cleared only when probes confirm healthy/repaired). Never
    raises: on failure it prints the manual command and leaves the marker for the next launch.
    Concurrent launches race on the shared venv, so an ``O_EXCL`` lockfile lets one process
    recover while the others skip.
    """
    from hermes_cli.main import PROJECT_ROOT
    if _pytest_owns_live_checkout(PROJECT_ROOT):
        return
    lazy_marker = _lazy_refresh_marker_path().exists()
    if not lazy_marker and not _update_marker_path().exists():
        return
    # Managed/Docker installs and git-less PyPI installs never run the source-tree
    # update path, so a stray marker is not ours to act on. Just clear it.
    if not (PROJECT_ROOT / "pyproject.toml").is_file():
        _clear_update_incomplete_marker()
        _clear_lazy_refresh_incomplete_marker()
        return
    lock_path = PROJECT_ROOT / ".update-incomplete.lock"
    if not _claim_recovery_lock(lock_path):
        return
    try:
        with _stdout_to_stderr():
            if lazy_marker:
                _recover_lazy_refresh_marker_locked()
            if _update_marker_path().exists():
                _recover_core_update_marker_locked()
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass


def _recover_lazy_refresh_marker_locked() -> None:
    """Heal ``.lazy-refresh-incomplete`` via confirmed import-probe repair."""
    print(
        "⚠ A previous lazy-backend refresh may have left the venv unhealthy — "
        "running import-based package repair...")
    install_prefix, install_env = _default_venv_install_target()
    status = _repair_venv_via_import_probes(install_prefix, env=install_env)
    if status in ("healthy", "repaired"):
        _clear_lazy_refresh_incomplete_marker()
        print("✓ Lazy-refresh venv recovery confirmed — install is healthy again.")
        return
    indeterminate = status == "indeterminate"
    problem = (
        "Import probes unavailable — cannot confirm venv health." if indeterminate
        else "Lazy-refresh package repair incomplete.")
    print(f"  ⚠ {problem} Leaving `.lazy-refresh-incomplete` for the next launch.")
    if indeterminate:
        return
    print("  Recover manually with:")
    all_specs = _lazy_refresh_repair_specs(sorted(set(_LAZY_REFRESH_REPAIR_PACKAGES.values())))
    print(
        f"    {' '.join(install_prefix)} install --force-reinstall "
        + " ".join(shlex.quote(s) for s in all_specs))


def _recover_core_update_marker_locked() -> None:
    """Heal ``.update-incomplete`` via full ``.[all]`` reinstall only.

    Narrow lazy-refresh import probes are not proof that a generic interrupted core
    install finished — a missing dep outside that probe set would look healthy and
    clear the breadcrumb too early.
    """
    from hermes_cli.main import PROJECT_ROOT
    print(
        "⚠ A previous `hermes update` was interrupted mid-install — "
        "finishing dependency installation now...")

    # Windows: a ``hermes.exe`` launch has the launcher as an ancestor; the quarantined full
    # reinstall can still replace it. Package-only repair is first aid and NEVER clears the marker.
    # Full editable reinstall uses quarantine so the live shim can still be replaced. Package-only import
    # repair may help as first aid but must NEVER clear this core marker on its own (#58004 review).
    self_locked = _windows_running_hermes_launcher_locked()
    if self_locked:
        install_prefix, install_env = _default_venv_install_target()
        print(
            "  → Running from hermes.exe; applying package-only first aid, "
            "then quarantined full reinstall (core marker stays until that "
            "succeeds)...")
        _repair_venv_via_import_probes(install_prefix, env=install_env)
    try:
        from hermes_cli import _install_repair as _ir

        # ensure_uv bootstraps uv itself when missing (the early pass's stdlib-only lookup
        # cannot), so a venv whose uv vanished mid-update still heals.
        from hermes_cli.managed_uv import ensure_uv
        ensure_uv()
        # Shared stdlib executor: late path and pre-import early pass run exactly the same
        # reinstall. Its own stdout→stderr redirect nests harmlessly inside ours.
        _ir.run_core_install(PROJECT_ROOT)
        _clear_update_incomplete_marker()
        print("✓ Dependency installation recovered — your install is healthy again.")
    except Exception as exc:
        # Leave the marker so the next launch retries; give the exact manual command.
        logger.debug("Interrupted-install recovery failed: %s", exc)
        print("✗ Could not auto-recover the interrupted install.")
        manual = (
            "  Hermes is still running from the launcher that needs "
            "replacing. Close other Hermes windows, restart from a "
            "different terminal, then run:",
            f'    cd /d "{PROJECT_ROOT}"',
            f'    "{sys.executable}" -m pip install -e ".[all]"',
        ) if self_locked else (
            "  Recover manually with:",
            f"    cd {PROJECT_ROOT}",
            f"    {sys.executable} -m ensurepip --upgrade",
            f"    {sys.executable} -m pip install -e '.[all]'",
        )
        for line in manual:
            print(line)


def _norm_exe_path(path) -> str:
    """Case-folded resolved path, for comparing executables on Windows."""
    try:
        return str(Path(path).resolve()).lower()
    except OSError:
        return str(path).lower()


def _windows_shim_in_process_chain() -> Path | None:
    """The venv console shim this process runs from or under, if any.

    ``venv\\Scripts\\hermes.exe`` holds itself open (no ``FILE_SHARE_DELETE``) for the whole
    process lifetime, so an editable install run from one can never rewrite it. Two probes, since
    either can come up empty: own launch paths (argv[0], ``__main__`` file/spec origin — runpy/
    zipapp puts ``<shim>\\__main__.py`` there) and psutil ancestry. Candidates are intersected
    with the project venv's own shims so a foreign ``hermes.exe`` never matches.

    See #88838, #89599.
    """
    if not _is_windows():
        return None
    scripts_dir = _venv_scripts_dir()
    if scripts_dir is None:
        return None
    shims = {_norm_exe_path(shim): shim for shim in _hermes_exe_shims(scripts_dir)}
    if not shims:
        return None

    def _match(candidate) -> Path | None:
        path = Path(candidate)
        if path.name.lower() == "__main__.py":
            path = path.parent
        return shims.get(_norm_exe_path(path))

    main_mod = sys.modules.get("__main__")
    candidates = [*sys.argv[:1], *filter(None, (
        getattr(main_mod, "__file__", None),
        getattr(getattr(main_mod, "__spec__", None), "origin", None)))]
    for candidate in candidates:
        matched = _match(candidate)
        if matched is not None:
            return matched

    with contextlib.suppress(Exception):
        import psutil
        me = psutil.Process()
        for proc in [me] + list(me.parents()):
            try:
                matched = _match(proc.exe())
            except Exception:
                continue
            if matched is not None:
                return matched
    return None


def _windows_running_hermes_launcher_locked() -> bool:
    """True when a venv ``hermes*.exe`` shim is this process or an ancestor (best-effort)."""
    return _windows_shim_in_process_chain() is not None


# Set on the re-exec'd child so it can never spawn another one.
_UPDATE_REEXEC_ENV = "HERMES_UPDATE_REEXEC"


def _reexec_dependency_sync_off_windows_shim() -> bool:
    """Hand the dependency sync to the venv interpreter, off the console shim.

    Returns True when a child was spawned and the caller must exit at once (releasing the
    shim before the child reaches ``pip install -e .``); False to continue in-process.

    Called at the dependency-sync boundary, NOT at the top of the command: by then the code swap
    is done and every interactive question has been answered; only the venv rewrite — the one
    step that cannot run inside the shim — remains. Earlier would detach every run (even the
    ``Already up to date!`` no-op) and take the prompts along. Waiting on the child deadlocks
    (we hold the handle it needs) and Windows has no exec, so the shell returns; the child keeps
    the console, prints its own result, and ``--gateway`` writes the true exit code to
    ``.update_exit_code``. The child re-runs ``hermes update`` so the sync and its tail happen
    exactly once; ``_UPDATE_REEXEC_ENV`` stops it spawning again and stops the "already up to
    date" early return from swallowing the sync. ``.update-incomplete`` is already written, so
    a child that dies mid-install is finished by the next launch's recovery.

    Called at the dependency-sync boundary, NOT at the top of the command — the same placement rule as the
    native-module deferral beside it, and for the same reason (#86735): a hand-off that fires before the
    fetch detaches every run, including the ``Already up to date!`` no-op that never touches the venv at
    all, and it takes the interactive prompts with it. By the time we reach here the code swap is done and
    every question — stash, branch switch, config migration — has already been asked and answered in the
    user's own console.
    ``venv\\Scripts\\hermes.exe`` is a launcher that runs the interpreter with the shim as its script and
    holds it open without ``FILE_SHARE_DELETE`` for the whole command, so the quarantine rename is refused
    and uv fails to replace it with os error 32 (#88838, #89599).
    """
    if os.environ.get(_UPDATE_REEXEC_ENV) == "1":
        return False
    shim = _windows_shim_in_process_chain()
    if shim is None:
        return False
    from hermes_constants import venv_python_path
    python_exe = venv_python_path(shim.parent.parent, windows=True)
    cmd = [str(python_exe), "-m", "hermes_cli.main", *sys.argv[1:]]
    if python_exe.is_file():
        try:
            subprocess.Popen(
                cmd, env={**os.environ, _UPDATE_REEXEC_ENV: "1"}, stdin=subprocess.DEVNULL)
            print(
                f"→ Windows: {shim.name} cannot replace itself while it runs; "
                "finishing the dependency install under the venv Python.")
            print(
                "  The code update is already applied. The install continues "
                "below and this shell returns right away.")
            return True
        except OSError as exc:
            logger.debug("Dependency-sync hand-off via %s failed: %s", python_exe, exc)
        print(f"  ⚠ Could not hand the dependency install off {shim.name}.")
        print("    Continuing in-process; if it cannot replace the shim, run:")
        print(f"    {subprocess.list2cmdline(cmd)}")
    return False


def _default_venv_install_target() -> tuple[list[str], dict[str, str] | None]:
    """Return ``(install_cmd_prefix, env)`` for the project venv when possible."""
    from hermes_cli.main import PROJECT_ROOT
    try:
        from hermes_cli.managed_uv import ensure_uv
        uv_bin = ensure_uv()
    except Exception:
        uv_bin = None
    if uv_bin:
        from hermes_constants import project_venv_dir
        venv_dir = project_venv_dir(PROJECT_ROOT) or PROJECT_ROOT / "venv"
        env = {**os.environ, "VIRTUAL_ENV": str(venv_dir)}
        if _is_termux_env(env):
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
        return [uv_bin, "pip"], env
    return [sys.executable, "-m", "pip"], None


def _run_install_with_heartbeat(
    cmd: list[str], *, env: dict[str, str] | None = None, heartbeat_interval_seconds: int = 30
) -> None:
    """Run a dependency install, printing an elapsed-time heartbeat while pip/uv is silent.

    Resolvers/build backends compiling Rust/C extensions can stay quiet for minutes.
    """
    from hermes_cli.main import PROJECT_ROOT
    done = threading.Event()
    start = _time.time()

    def _heartbeat() -> None:
        # Wait first, then print, so short installs don't emit noise.
        while not done.wait(heartbeat_interval_seconds):
            elapsed = int(_time.time() - start)
            print(
                f"  … still installing dependencies ({elapsed}s elapsed)"
                " — compiling Rust/C extensions can take several minutes",
                flush=True)

    t = threading.Thread(target=_heartbeat, daemon=True)
    t.start()
    try:
        if _is_termux_env(env):
            # Termux: prefer bionic PATH + arch-aware uv --python-platform so
            # Android wheel-tag rejects and glibc-first PATH breakages don't
            # strand updates (bare "linux" is x86_64 on uv 0.12+).
            from hermes_cli._early_recovery import (
                prefer_termux_bionic_path,
                with_uv_termux_python_platform,
            )

            env = prefer_termux_bionic_path(env)
            cmd = with_uv_termux_python_platform(cmd, env)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)
    finally:
        done.set()
        t.join(timeout=0.2)


def _run_repair_step(run, cmd: list[str], *, log_msg: str, fail_msg: str | None, **kwargs) -> bool:
    """``run(cmd, **kwargs)``; on ``CalledProcessError`` log + print the failure and return False."""
    try:
        run(cmd, **kwargs)
    except subprocess.CalledProcessError as e:
        logger.warning(log_msg, e)
        if fail_msg is not None:
            print(fail_msg)
        return False
    return True


def _report_still_missing(missing: list[str], hint: str, *, ok: str) -> None:
    if missing:
        print(f"  ⚠ Still missing after repair: {', '.join(missing)}. {hint}")
    else:
        print(ok)


def _is_windows() -> bool:
    return sys.platform == "win32"


def _venv_scripts_dir() -> Path | None:
    """Return the venv Scripts directory if we're running inside the project venv."""
    from hermes_cli.main import PROJECT_ROOT
    from hermes_constants import project_venv_dir, venv_bin_dir
    venv_dir = project_venv_dir(PROJECT_ROOT)
    if venv_dir is None:
        return None
    scripts = venv_bin_dir(venv_dir, windows=_is_windows())
    return scripts if scripts.is_dir() else None


def _hermes_exe_shims(scripts_dir: Path) -> list[Path]:
    """Entry-point shims uv may rewrite during ``pip install -e .`` — Windows .exe launchers
    only; POSIX shims are plain scripts replaced atomically."""
    if not _is_windows():
        return []
    names = set(_load_console_script_names()) or {"hermes", "hermes-agent", "hermes-acp"}
    # Not a [project.scripts] entry point, but older update/install paths still
    # rewrite and quarantine it.
    names.add("hermes-gateway")
    return [scripts_dir / f"{name}.exe" for name in sorted(names)]


_QUARANTINE_BACKOFF_MS = (0, 100, 250, 500, 1000)


def _rename_with_backoff(source: Path, target: Path, attempts: int) -> OSError | None:
    """Rename with the quarantine backoff ladder; returns the last ``OSError`` or ``None``."""
    for delay_ms in _QUARANTINE_BACKOFF_MS[:attempts]:
        if delay_ms:
            _time.sleep(delay_ms / 1000.0)
        try:
            source.rename(target)
            return None
        except OSError as e:
            last_exc = e
    return last_exc


def _quarantine_running_hermes_exe(
    scripts_dir: Path, *, max_attempts: int = 4, failed_out: list[str] | None = None
) -> list[tuple[Path, Path]]:
    """Pre-empt the Windows file lock on the running ``hermes.exe``.

    Windows allows RENAMING a running executable but blocks DELETE/REPLACE (uv fails with
    ``Access is denied. (os error 5)``), so live shims are renamed to ``<shim>.old.<unix-ms>``
    first; ``_cleanup_quarantined_exes`` sweeps the ``.old`` files next invocation. Rename can
    still fail when another process holds the .exe without ``FILE_SHARE_DELETE`` (AV scanner:
    transient; Hermes Desktop backend child: until closed) — retry with backoff, then warn
    naming the likely culprit. Returns ``(original, quarantined)`` pairs for rollback;
    ``failed_out`` collects shims whose rename failed every attempt so the update dependency
    sync can refuse instead of stranding a half-broken venv.

    See #87331.
    """
    moved: list[tuple[Path, Path]] = []
    if not _is_windows():
        return moved
    stamp = int(_time.time() * 1000)
    # First attempt immediate; 100/250/500ms covers the typical AV re-scan window.
    attempts = max(1, min(max_attempts, len(_QUARANTINE_BACKOFF_MS)))
    for shim in _hermes_exe_shims(scripts_dir):
        if not shim.exists():
            continue
        target = shim.with_suffix(shim.suffix + f".old.{stamp}")
        last_exc = _rename_with_backoff(shim, target, attempts)
        if last_exc is None:
            moved.append((shim, target))
            continue

        # Every rename failed. MOVEFILE_DELAY_UNTIL_REBOOT is no fallback (needs elevation, frees
        # nothing now, moves a later repaired shim aside at boot). Report; let uv try its luck.
        print(
            f"  ⚠ Could not quarantine {shim.name} ({last_exc.__class__.__name__}: "
            f"another process is holding it open).")
        print(
            "    Close Hermes Desktop, exit other `hermes` REPLs, stop the "
            "gateway, or pause AV scanning, then re-run `hermes update`.")
        if failed_out is not None:
            failed_out.append(shim.name)

    return moved


_PENDING_RENAME_KEY = r"SYSTEM\CurrentControlSet\Control\Session Manager"
_PENDING_RENAME_VALUE = "PendingFileRenameOperations"


def _filter_pending_shim_renames(entries: list[str], shims: list[Path]) -> tuple[list[str], int]:
    """Drop our ``<shim>`` -> ``<shim>.old.<stamp>`` pairs from a PendingFileRenameOperations
    value (a flat REG_MULTI_SZ of (source, target) pairs shared with other installers).
    Returns the entries to keep and how many pairs were dropped."""
    import ntpath

    def _norm(value: str) -> str:
        path = str(value).lstrip("!")
        if path.startswith("\\??\\"):
            path = path[4:]
        return ntpath.normcase(ntpath.normpath(path))

    shim_paths = {_norm(str(shim)) for shim in shims}
    kept: list[str] = []
    removed = 0
    for index in range(0, len(entries) - 1, 2):
        source, target = entries[index], entries[index + 1]
        source_norm = _norm(source)
        if source_norm in shim_paths and _norm(target).startswith(f"{source_norm}.old."):
            removed += 1
        else:
            kept.extend((source, target))
    if len(entries) % 2:
        kept.append(entries[-1])
    return kept, removed


def _cleanup_pending_shim_renames(scripts_dir: Path) -> int:
    """Drop reboot renames older Hermes versions queued for our shims: ``MOVEFILE_DELAY_UNTIL_REBOOT``
    fallbacks outlive the update that queued them and move away whatever sits at the shim path
    at next boot — even a shim a later repair just wrote. Needs elevation; a no-op otherwise."""
    if not _is_windows():
        return 0
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, _PENDING_RENAME_KEY, 0,
            winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE,
        ) as key:
            entries, value_type = winreg.QueryValueEx(key, _PENDING_RENAME_VALUE)
            if value_type != winreg.REG_MULTI_SZ or not isinstance(entries, list):
                return 0
            kept, removed = _filter_pending_shim_renames(entries, _hermes_exe_shims(scripts_dir))
            if not removed:
                return 0
            if kept:
                winreg.SetValueEx(key, _PENDING_RENAME_VALUE, 0, winreg.REG_MULTI_SZ, kept)
            else:
                winreg.DeleteValue(key, _PENDING_RENAME_VALUE)
            return removed
    except (OSError, ValueError):
        return 0


def _restore_quarantined_exes(moved: list[tuple[Path, Path]]) -> None:
    """Roll back ``_quarantine_running_hermes_exe`` if uv didn't write replacements. Safety-
    critical: a failed quarantine only aborts an update; a failed restore leaves no ``hermes``
    on PATH. Delegates to the stdlib-only retrying helper shared with ``_install_repair``.

    The outbound rename already retries a lock, so this one must too rather than swallow the first
    ``OSError`` in silence. See #75584.
    """
    _early_recovery_mod.restore_quarantined_shims(moved)


class ShimQuarantineError(RuntimeError):
    """A live ``hermes*.exe`` shim could not be renamed aside. Raised by
    :func:`_run_quarantined_install` in ``strict_quarantine`` mode BEFORE the install runs: a
    process holds the venv hard enough that the sync would die partway — refuse, don't warn.

    See #87331.
    """

    def __init__(self, failed_shims: list[str]):
        self.failed_shims = list(failed_shims)
        super().__init__("could not quarantine live shim(s): " + ", ".join(self.failed_shims))


def _run_quarantined_install(
    cmd: list[str], *, env: dict[str, str] | None = None, scripts_dir: Path | None = None,
    strict_quarantine: bool = False,
) -> None:
    """Run an editable install, quarantining the running ``hermes.exe`` first.

    Every editable install rewrites the entry-point shims; on Windows the live ``hermes.exe``
    can be neither deleted nor overwritten, so without quarantine ``hermes`` drops off PATH.
    ``strict_quarantine=True`` (the update dependency sync): a shim whose rename failed every
    retry proves a hard venv hold — the install WILL hit the same lock on .pyd files — so roll
    back and raise :class:`ShimQuarantineError` without installing. Non-strict callers already
    mutated the venv, so refusing buys nothing. ``scripts_dir is None`` is a pass-through.

    See #87331.
    """
    moved: list[tuple[Path, Path]] = []
    failed: list[str] = []
    if scripts_dir is not None:
        moved = _quarantine_running_hermes_exe(scripts_dir, failed_out=failed)
    if strict_quarantine and failed:
        _restore_quarantined_exes(moved)
        raise ShimQuarantineError(failed)
    try:
        _run_install_with_heartbeat(cmd, env=env)
    finally:
        # Restore on FAILURE and SUCCESS: an already-satisfied editable install is a uv no-op
        # that rewrites no entry points. Skips shims the installer replaced; finally re-raises.
        if scripts_dir is not None:
            _restore_quarantined_exes(moved)


# A quarantine file younger than this may belong to an update running RIGHT NOW in
# another process, whose restore step still needs it — the only copy of that shim.
# Restore shims when the installer didn't write replacements — on FAILURE (install died before the
# entry-points step) and on SUCCESS too: uv audits an already-satisfied editable install as a no-op and
# rewrites no entry points, which would otherwise leave the shims quarantined aside and `hermes` missing
# from PATH after a green install (#75584). _restore_quarantined_exes skips any shim the installer actually
# replaced, so this never clobbers fresh output. Errors are not swallowed — the finally re-raises whatever
# escaped.
_QUARANTINE_GRACE_SECONDS = 15 * 60


def _quarantine_stamp_ms(stale: Path) -> int | None:
    """The ``.old.<unix-ms>`` stamp in a quarantine filename; ``None`` (not ours — neither rescued
    nor deleted) otherwise. Parsed from the NAME, not ``st_mtime``: ``rename`` preserves the
    shim's mtime (when uv wrote it), not when it was quarantined."""
    try:
        return int(stale.name.rsplit(".old.", 1)[1])
    except (IndexError, ValueError):
        return None


def _cleanup_quarantined_exes(scripts_dir: Path | None = None) -> None:
    """Sweep — and where necessary RESCUE — ``hermes.exe.old.*`` from updates.

    Called early on every invocation. Two cases an unconditional ``unlink()`` gets wrong:
    (1) orphan rescue — ``hermes.exe`` missing while ``hermes.exe.old.*`` exists means the .old
    file is the ONLY surviving copy (update died between rename and uv's write); put it back via
    the same retry-and-report helper the update-time restore uses. (2) concurrency — a fresh
    quarantine file may belong to an update in flight in another process; leave anything inside
    the grace window alone. Silent no-op on non-Windows, nothing to do, or locked/permission errors.

    Deleting it converts a one-rename recovery into a full reinstall. See #75584.
    """
    if not _is_windows():
        return
    scripts_dir = scripts_dir if scripts_dir is not None else _venv_scripts_dir()
    if scripts_dir is None:
        return
    _cleanup_pending_shim_renames(scripts_dir)
    now = _time.time()
    try:
        candidates = [
            (stamp, stale) for stale in scripts_dir.glob("*.exe.old.*")
            if (stamp := _quarantine_stamp_ms(stale)) is not None]
    except OSError:
        return
    # Newest first by PARSED stamp: lexicographic order breaks when a stray ``.old.999`` exists.
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    for stamp, stale in candidates:
        try:
            original = stale.with_name(stale.name.rsplit(".old.", 1)[0])
            if not original.exists():
                # Orphan rescue: last copy of the shim — retry ladder + recovery message.
                _early_recovery_mod.restore_quarantined_shims([(original, stale)])
                continue
            if now - stamp / 1000.0 < _QUARANTINE_GRACE_SECONDS:
                continue  # may be a live quarantine from a concurrent update
            stale.unlink()
        except OSError:
            pass  # still locked or in use — try again next run


# Import probes for venv corruption after a failed lazy ``uv pip install`` (metadata can
# look fine while ``.py`` files were removed mid-install). Canonical tables live in the
# stdlib-only ``_early_recovery`` module so the early and full recovery layers never drift.
# See #57828.
_LAZY_REFRESH_IMPORT_PROBES: tuple[tuple[str, str], ...] = (
    _early_recovery_mod.LAZY_REFRESH_IMPORT_PROBES)
_LAZY_REFRESH_REPAIR_PACKAGES: dict[str, str] = _early_recovery_mod.LAZY_REFRESH_REPAIR_PACKAGES


def _run_package_only_install(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    """Package-only pip/uv install — no shim quarantine: ``--force-reinstall <pkg>`` never rewrites
    ``hermes.exe``, and the quarantine path would rename shims uv then never recreates.

    See #57828.
    """
    _run_install_with_heartbeat(cmd, env=env)


def _lazy_refresh_repair_specs(packages: list[str]) -> list[str]:
    """Map repair package names to their declared pin specs in pyproject.toml."""
    project = _pyproject_project("lazy refresh repair spec lookup failed: %s")
    if project is None:
        return packages
    name_to_spec = {
        name.lower(): head
        for name, _, head in _parse_requirements(project.get("dependencies", []) or [])}
    return [name_to_spec.get(pkg.lower(), pkg) for pkg in packages]


def _venv_probe(venv_python: Path, script: str, *args: str, env: dict[str, str] | None):
    """Run ``script`` in the target venv's interpreter, capturing UTF-8 stdout."""
    return subprocess.run(
        [str(venv_python), "-c", script, *args],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        check=False,
        env=env)


def _nonblank_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _detect_broken_lazy_refresh_imports(
    install_cmd_prefix: list[str], *, env: dict[str, str] | None = None) -> list[str] | None:
    """Probe lazy-refresh packages via real imports: ``[]`` all clean, ``[dist, ...]`` failures,
    ``None`` when the probe could not run (no venv Python, subprocess failure, non-zero exit)
    — *indeterminate*, not healthy."""
    venv_python = _resolve_install_target_python(install_cmd_prefix, env)
    if venv_python is None:
        return None
    probe_lines = "\n".join(
        f"    ({mod!r}, {attr!r})," for mod, attr in _LAZY_REFRESH_IMPORT_PROBES)
    check_script = (
        "import os\n"
        "import sys\n"
        "probes = [\n"
        f"{probe_lines}\n"
        "]\n"
        "broken = []\n"
        "for mod, attr in probes:\n"
        "    try:\n"
        "        imported = __import__(mod)\n"
        "        if not hasattr(imported, attr):\n"
        "            broken.append(mod)\n"
        "        elif mod == 'certifi':\n"
        "            # The module can import cleanly while cacert.pem is\n"
        "            # missing/corrupt (brew Python upgrade, interrupted venv\n"
        "            # rebuild) - every TLS call then fails (#29866).\n"
        "            bundle = imported.where()\n"
        "            if not os.path.isfile(bundle) or os.path.getsize(bundle) < 1024:\n"
        "                broken.append(mod)\n"
        "    except Exception:\n"
        "        broken.append(mod)\n"
        "print('\\n'.join(broken))\n")
    try:
        result = _venv_probe(venv_python, check_script, env=env)
    except Exception as exc:
        logger.debug("lazy refresh import probe failed: %s", exc)
        return None
    if result.returncode != 0:
        logger.debug("lazy refresh import probe exited %s: %s",
                     result.returncode, (result.stderr or "")[:200])
        return None
    packages: list[str] = []
    for mod in _nonblank_lines(result.stdout):
        pkg = _LAZY_REFRESH_REPAIR_PACKAGES.get(mod)
        if pkg and pkg not in packages:
            packages.append(pkg)
    return packages


def _repair_broken_lazy_refresh_imports(
    install_cmd_prefix: list[str], packages: list[str], *, env: dict[str, str] | None = None
) -> bool:
    """Force-reinstall ``packages`` and re-probe imports. Never raises."""
    if not packages:
        return True
    specs = _lazy_refresh_repair_specs(packages)
    if not _run_repair_step(
        _run_package_only_install, install_cmd_prefix + ["install", "--force-reinstall", *specs],
        env=env, log_msg="lazy refresh venv repair failed: %s", fail_msg=None):
        return False
    # Indeterminate re-probe is not confirmed success.
    return _detect_broken_lazy_refresh_imports(install_cmd_prefix, env=env) == []


def _repair_venv_via_import_probes(
    install_cmd_prefix: list[str], *, env: dict[str, str] | None = None) -> str:
    """Probe imports and force-reinstall any broken lazy-refresh packages.

    Real ``import`` checks (not distribution metadata) catch a venv where METADATA remains
    but ``.py`` files were wiped mid-install. Package-only reinstall — never rewrites
    ``hermes.exe``. Never raises. Returns ``"healthy"``, ``"repaired"``, ``"failed"``
    (repair did not confirm clean) or ``"indeterminate"`` (probes could not run; NOT healthy).

    See #57828.
    """
    broken = _detect_broken_lazy_refresh_imports(install_cmd_prefix, env=env)
    if broken is None:
        print("  ⚠ Import probes unavailable — cannot confirm venv package health.")
        return "indeterminate"
    if not broken:
        return "healthy"
    print(
        "  → Detected corrupted venv packages via import probes: "
        f"{', '.join(broken)}; repairing...")
    if _repair_broken_lazy_refresh_imports(install_cmd_prefix, broken, env=env):
        print("  ✓ Venv repair succeeded")
        return "repaired"
    manual = " ".join(shlex.quote(s) for s in _lazy_refresh_repair_specs(broken))
    print("  ⚠ Venv repair incomplete. Run manually, then `hermes update`:")
    print(f"    {' '.join(install_cmd_prefix)} install --force-reinstall {manual}")
    return "failed"


def _is_uv_command(install_cmd_prefix: list[str]) -> bool:
    """True for a uv/uvx binary (bare or path) or ``python -m uv`` / ``python -m uvx``."""
    if not install_cmd_prefix:
        return False
    first = str(install_cmd_prefix[0]).lower()
    if "uv" in Path(first).name:
        return True
    return (
        len(install_cmd_prefix) >= 3
        and first.endswith(("python", "python.exe"))
        and install_cmd_prefix[1] == "-m"
        and install_cmd_prefix[2] in ("uv", "uvx"))


def _insert_python_pin(args: list[str]) -> list[str]:
    """Insert ``--python <sys.executable>`` into a uv command line; an explicit caller ``--python`` wins."""
    if "--python" in args:
        return args
    return [args[0], "--python", str(sys.executable), *args[1:]]


def _interpreter_scripts_dir() -> Path | None:
    """Scripts/bin dir of ``sys.executable``: on a site-packages install ``PROJECT_ROOT/venv``
    does not exist and the shims uv rewrites live next to the interpreter. Layout via the
    canonical ``venv_bin_dir`` (hand-rolling Scripts/bin is lint-tested against).

    See #76105.
    """
    from hermes_constants import venv_bin_dir
    exe = Path(sys.executable)
    # sys.executable lives IN the bin/Scripts dir; parent.parent is the env root.
    cand = venv_bin_dir(exe.parent.parent, windows=_is_windows())
    if cand.is_dir():
        return cand
    return exe.parent if exe.parent.is_dir() else None


def _install_python_dependencies_with_optional_fallback(
    install_cmd_prefix: list[str], *, env: dict[str, str] | None = None, group: str = "all"
) -> None:
    """Install base deps plus as many optional extras as the environment supports.

    Targets ``.[all]`` by default; Termux callers pass ``group='termux-all'``. On Windows every
    attempt quarantines the live ``hermes*.exe`` shims first. When ``env`` carries a
    ``VIRTUAL_ENV`` that does not exist (pip / site-packages install), ``uv pip`` fails with
    ``Failed to inspect Python interpreter from active virtual environment`` before doing any
    work — pin the install at the running interpreter instead.

    Pin the install at the running interpreter instead so the update/recovery path succeeds on those
    installs (#71510 fixed the ZIP path, #83335 fixed lazy-deps; this closes the shared helper for the
    remaining callers).
    """
    scripts_dir = _venv_scripts_dir() if _is_windows() else None

    # Only uv needs the explicit pin; pip resolves the target from sys.executable itself.
    pin_python = bool(
        env and env.get("VIRTUAL_ENV") and not Path(env["VIRTUAL_ENV"]).is_dir()
        and install_cmd_prefix and _is_uv_command(install_cmd_prefix))
    if pin_python:
        env = {**env}
        env.pop("VIRTUAL_ENV", None)
        # Pinned to sys.executable, the shims uv rewrites live in THAT interpreter's Scripts
        # dir, not PROJECT_ROOT/venv; quarantining the wrong dir leaves hermes.exe locked.
        if scripts_dir is None and _is_windows():
            scripts_dir = _interpreter_scripts_dir()

    def _install(args: list[str]) -> None:
        if pin_python:
            args = _insert_python_pin(args)
        # strict_quarantine: this is the UPDATE dependency sync; ShimQuarantineError propagates
        # to the sync boundary, which defers via the update-incomplete marker instead.
        # A shim that cannot be renamed aside proves a hard venv hold; running uv anyway is how installs
        # strand half-updated (#87331).
        _run_quarantined_install(
            install_cmd_prefix + args, env=env, scripts_dir=scripts_dir, strict_quarantine=True)

    try:
        _install(["install", "-e", f".[{group}]"])
        _verify_console_scripts_installed(install_cmd_prefix, env=env)
        return
    except subprocess.CalledProcessError:
        print(
            "  ⚠ Optional extras failed, reinstalling base dependencies and retrying extras individually..."
        )

    _install(["install", "-e", "."])
    failed_extras: list[str] = []
    installed_extras: list[str] = []
    for extra in _load_installable_optional_extras(group=group):
        try:
            _install(["install", "-e", f".[{extra}]"])
            installed_extras.append(extra)
        except subprocess.CalledProcessError:
            failed_extras.append(extra)
    if installed_extras:
        print(f"  ✓ Reinstalled optional extras individually: {', '.join(installed_extras)}")
    if failed_extras:
        print(f"  ⚠ Skipped optional extras that still failed: {', '.join(failed_extras)}")
    # uv's incremental resolver has left newly added base deps silently missing on a half-stale
    # venv, surfacing hours later as a downstream ModuleNotFoundError. Verify here instead.
    _verify_core_dependencies_installed(install_cmd_prefix, env=env, group=group)
    _verify_console_scripts_installed(install_cmd_prefix, env=env)


def _load_console_script_names() -> list[str]:
    """Return ``[project.scripts]`` entry-point names from pyproject.toml."""
    project = _pyproject_project("console script verification: failed to read pyproject.toml: %s")
    scripts = (project or {}).get("scripts", {}) or {}
    return [str(name) for name in scripts if name]


def _verify_console_scripts_installed(
    install_cmd_prefix: list[str], *, env: dict[str, str] | None = None) -> None:
    """Ensure every declared console_script shim exists on disk after install.

    On Windows ``uv pip install -e .`` can register ``hermes.exe`` in the wheel RECORD while the
    file never lands (live shim locked, launcher write skipped), so ``hermes`` drops off PATH
    after a "successful" install. Missing shims get ``--reinstall -e .`` under quarantine.

    The symptom is ``hermes-agent.exe`` and ``hermes-acp.exe`` present but ``hermes.exe`` missing, so
    ``hermes`` drops off PATH even though the install reported success (issue #52931).
    """
    if not _is_windows():
        return
    scripts_dir = _venv_scripts_dir()
    names = _load_console_script_names() if scripts_dir is not None else []
    if not names:
        return

    def _missing() -> list[str]:
        return [name for name in names if not (scripts_dir / f"{name}.exe").is_file()]

    missing = _missing()
    if not missing:
        return
    print(
        f"  ⚠ Verification: {len(missing)} console script(s) missing on disk: "
        f"{', '.join(missing)}")
    print("  → Reinstalling entry points with --reinstall...")
    if not _run_repair_step(
        _run_quarantined_install, install_cmd_prefix + ["install", "--reinstall", "-e", "."],
        env=env, scripts_dir=scripts_dir,
        log_msg="console script verification: repair install failed: %s",
        fail_msg=(
            "  ⚠ Entry point repair failed; try `hermes update --force` after "
            "closing other hermes processes.")):
        return
    _report_still_missing(
        _missing(), "Workaround: python -m hermes_cli.main <command>",
        ok="  ✓ All console entry points restored")


def _applicable_dependency_names(raw_deps: list[str]) -> list[str]:
    """Declared dep names whose ``;`` markers apply here (else ``ptyprocess ; sys_platform !=
    'win32'`` would false-positive on Windows). An unevaluable marker counts as applicable."""
    applicable: list[str] = []
    for name, marker, _ in _parse_requirements(raw_deps):
        try:
            if marker is None or marker.evaluate():  # type: ignore[union-attr]
                applicable.append(name)
        except Exception:
            applicable.append(name)
    return applicable


_MISSING_DEPS_SCRIPT = (
    "import importlib.metadata as md, sys\n"
    "missing=[]\n"
    "for name in sys.argv[1:]:\n"
    "    try: md.version(name)\n"
    "    except md.PackageNotFoundError: missing.append(name)\n"
    "print('\\n'.join(missing))\n")


def _verify_core_dependencies_installed(
    install_cmd_prefix: list[str], *, env: dict[str, str] | None = None, group: str = "all"
) -> None:
    """Check that every base dep from pyproject.toml is installed in the target venv; if not, retry.

    Reads ``pyproject.toml`` directly (not the venv's stale metadata), drops deps whose ``;``
    markers don't apply here, and probes ``importlib.metadata.version()`` in the venv
    interpreter. Missing deps trigger a base-group ``--reinstall``, then a per-package force
    install. The final state is a warning, not a hard failure, so one broken-on-PyPI dep can't
    block an otherwise-successful update — but the partial install is visible where it happened.
    """
    project = _pyproject_project("dep verification: failed to read pyproject.toml: %s")
    if project is None:
        return
    raw_deps = project.get("dependencies", []) or []
    applicable = _applicable_dependency_names(raw_deps)
    if not applicable:
        return
    # Probe inside the venv Python — sys.executable may be the outer Python that drove
    # ``hermes update``; the install prefix/env encode which environment we targeted.
    venv_python = _resolve_install_target_python(install_cmd_prefix, env)
    if venv_python is None:
        return

    def _missing_deps() -> list[str]:
        try:
            result = _venv_probe(venv_python, _MISSING_DEPS_SCRIPT, *applicable, env=env)
        except Exception as e:
            logger.debug("dep verification: subprocess failed: %s", e)
            return []
        return _nonblank_lines(result.stdout)

    missing = _missing_deps()
    if not missing:
        return
    print(
        f"  ⚠ Verification: {len(missing)} declared dep(s) missing after install: "
        f"{', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")
    print("  → Reinstalling base group with --reinstall to repair...")
    # Base group only, not ``[{group}]``: the missing dep is a *base* dep and the all-extras
    # install costs minutes. Quarantine first: ``--reinstall -e .`` rewrites the shims.
    scripts_dir = _venv_scripts_dir() if _is_windows() else None
    if not _run_repair_step(
        _run_quarantined_install, install_cmd_prefix + ["install", "--reinstall", "-e", "."],
        env=env, scripts_dir=scripts_dir,
        log_msg="dep verification: repair install failed: %s",
        fail_msg="  ⚠ Repair install failed; check `hermes update` output above."):
        return
    still_missing = _missing_deps()
    if not still_missing:
        print("  ✓ All declared core dependencies now installed")
        return
    # Last-ditch: install each remaining missing dep with its pin directly — uv's
    # resolver can think the env is satisfied while on-disk metadata disagrees.
    name_to_spec = dict(_naive_requirement(spec) for spec in raw_deps)
    specs = [name_to_spec.get(n, n) for n in still_missing]
    print(f"  → Force-installing remaining missing dep(s): {', '.join(specs)}")
    if not _run_repair_step(
        _run_install_with_heartbeat, install_cmd_prefix + ["install", "--reinstall", *specs],
        env=env,
        log_msg="dep verification: per-package repair failed: %s",
        fail_msg=(
            f"  ⚠ Could not install: {', '.join(still_missing)}. "
            "Run `hermes update --force` after closing other hermes processes.")):
        return
    _report_still_missing(
        _missing_deps(), "Run `hermes update --force` after closing other hermes processes.",
        ok="  ✓ All declared core dependencies now installed")


def _resolve_install_target_python(
    install_cmd_prefix: list[str], env: dict[str, str] | None) -> Path | None:
    """Python interpreter the install targeted: ``VIRTUAL_ENV`` from ``env`` for the
    ``[uv, pip]`` shape, else ``install_cmd_prefix[0]`` for ``[sys.executable, -m, pip]``."""
    if env and "VIRTUAL_ENV" in env:
        from hermes_constants import venv_python_path
        candidate = venv_python_path(Path(env["VIRTUAL_ENV"]), windows=_is_windows())
        if candidate.exists():
            return candidate
    if install_cmd_prefix:
        first = Path(install_cmd_prefix[0])
        if first.exists() and "uv" not in first.name.lower():
            return first
    return None


def _is_termux_env(env: dict[str, str] | None = None) -> bool:
    from hermes_cli.main import _is_termux_startup_environment
    return _is_termux_startup_environment(env)


def _is_windows_npm_path(npm_path: str) -> bool:
    """True if ``npm_path`` points at a Windows npm shim (WSL ``/mnt/c`` interop, ``.cmd``/``.exe``, UNC).

    Callers use this only on a POSIX host — on native Windows ``npm.cmd`` is correct.
    """
    low = npm_path.lower()
    return low.endswith((".exe", ".cmd", ".bat")) or low.startswith("/mnt/") or "\\" in npm_path


def _resolve_node_runtime_npm() -> str | None:
    """Resolve an npm executable that belongs to the host's Node runtime.

    On WSL, PATH interop can hand back a Windows npm that fails with EISDIR / symlink errors over
    ``\\\\wsl.localhost\\...`` UNC paths. Refuse it on a POSIX host and re-scan PATH minus the
    ``/mnt/*`` drive mounts. ``None`` when no suitable npm is reachable.

    On WSL/Linux ``shutil.which("npm")`` may resolve a Windows npm exposed through PATH interop. See #30271.
    """
    from hermes_constants import find_node_executable
    npm = find_node_executable("npm")
    if _is_windows():
        return npm
    if not npm:
        return None
    if not _is_windows_npm_path(npm):
        return npm
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory or directory.lower().startswith("/mnt/"):
            continue
        candidate = shutil.which("npm", path=directory)
        if candidate and not _is_windows_npm_path(candidate):
            return candidate
    return None


def _resolve_update_branch(args) -> str:
    """Normalize ``args.branch`` to a non-empty name (default ``main``; blank/whitespace = default)."""
    return (getattr(args, "branch", None) or "main").strip() or "main"

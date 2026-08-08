"""Update-recovery breadcrumb helpers — extracted from ``hermes_cli/main.py``.

Mechanical move (main.py decomposition, wave-1 shard s4 cluster c3): the
update-recovery marker lifecycle helpers (``_update_marker_path``,
``_lazy_refresh_marker_path``, ``_pytest_owns_live_checkout``,
``_clear_marker_file``, ``_clear_update_incomplete_marker``,
``_clear_lazy_refresh_incomplete_marker``, ``_recover_from_interrupted_install``,
``_recover_lazy_refresh_marker_locked``, ``_recover_core_update_marker_locked``,
``_windows_running_hermes_launcher_locked``) are lifted verbatim. Function
bodies are byte-identical except that references to helpers/constants that
STAY in ``hermes_cli.main`` (and moved-but-test-patched siblings) are routed
through ``_m()`` — a lazy ``hermes_cli.main`` reference — so existing call
sites and test monkeypatches that target ``hermes_cli.main.<name>``
(``PROJECT_ROOT``, ``_is_windows``, ``_default_venv_install_target``,
``_repair_venv_via_import_probes``, ``_lazy_refresh_repair_specs``,
``_LAZY_REFRESH_REPAIR_PACKAGES``, ``_is_termux_env``,
``_install_python_dependencies_with_optional_fallback``, ...) keep working
unchanged. ``main.py`` re-imports every moved name (``# noqa: F401``) so
callers and test patches on ``hermes_cli.main`` resolve unchanged.

Imports are one-way: ``hermes_cli.main`` imports this module, never the
reverse at import time (``_m()`` resolves lazily at call time, when main.py
is fully loaded, so there is no import cycle).
"""

import logging
import os
import shlex
import subprocess
import sys
import time as _time
from pathlib import Path

logger = logging.getLogger(__name__)


def _m():
    """Lazy ``hermes_cli.main`` reference.

    Lets callers keep patching ``hermes_cli.main.<helper>`` (the historical
    test surface) and have those patches reach this code path, and defers the
    import so ``hermes_cli.main`` -> ``hermes_cli.update_recovery`` stays
    one-way at import time.
    """
    from hermes_cli import main

    return main


def _update_marker_path() -> Path:
    return _m().PROJECT_ROOT / ".update-incomplete"


def _lazy_refresh_marker_path() -> Path:
    return _m().PROJECT_ROOT / ".lazy-refresh-incomplete"


def _pytest_owns_live_checkout(root: Path) -> bool:
    """True when running under pytest AND ``root`` is this checkout itself.

    Tests that drive update/recovery without sandboxing ``PROJECT_ROOT``
    must neither litter the live repo root with recovery breadcrumbs
    (a leftover ``.lazy-refresh-incomplete`` / ``.update-incomplete``
    false-arms recovery on the developer's next real launch) nor run a real
    reinstall against the executing venv. Sandboxed tests point at a
    tmp_path and are unaffected (same posture as
    ``managed_scope._under_pytest``)."""
    return (
        "PYTEST_CURRENT_TEST" in os.environ
        and root == Path(__file__).resolve().parent.parent
    )


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


def _recover_from_interrupted_install() -> None:
    """Finish update work left half-done by a prior ``hermes update``.

    Handles two independent breadcrumbs:

    - ``.update-incomplete`` — core ``.[all]`` install interrupted. Recovers
      via full quarantined reinstall. Never cleared by the narrow lazy-refresh
      import probes alone.
    - ``.lazy-refresh-incomplete`` — lazy-backend refresh may have corrupted
      packages. Recovers via package-only import probes; cleared only when
      probes confirm healthy/repaired (indeterminate keeps the marker).

    Never raises: a recovery failure must not block launch.  If it can't
    self-heal it prints the manual command and leaves the relevant marker so
    the next launch tries again.

    Concurrency: markers live next to the shared venv, so a gateway start
    plus a CLI launch (or two profiles starting at once) can both see them.
    An ``O_EXCL`` lockfile ensures only one process runs recovery; the
    others skip and let the winner clear markers.

    Output: everything — our status lines AND the streamed pip/uv install
    (which inherits fd 1) — is routed to stderr.  Launches whose stdout is a
    protocol stream (``hermes acp`` speaks JSON-RPC on stdout) must never get
    install noise on stdout.
    """
    if _pytest_owns_live_checkout(_m().PROJECT_ROOT):
        return
    core_marker = _update_marker_path().exists()
    lazy_marker = _lazy_refresh_marker_path().exists()
    if not core_marker and not lazy_marker:
        return

    # Skip in managed/Docker installs and on PyPI installs with no git checkout:
    # those don't run the source-tree update path, so a stray marker is not ours
    # to act on. Just clear it.
    if not (_m().PROJECT_ROOT / "pyproject.toml").is_file():
        _clear_update_incomplete_marker()
        _clear_lazy_refresh_incomplete_marker()
        return

    # Single-flight guard: atomically claim the recovery lock. If another
    # process holds it, skip — it is running the same reinstall into the same
    # shared venv right now. A crashed holder leaves a stale lock; break it
    # after an hour (well past any realistic install) so recovery can't be
    # wedged forever.
    lock_path = _m().PROJECT_ROOT / ".update-incomplete.lock"
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
        return
    except OSError as exc:
        # Couldn't create the lock (read-only fs, perms). Proceed unlocked —
        # the install itself will surface the real problem.
        logger.debug("Could not create install-recovery lock: %s", exc)

    saved_stdout_fd = None
    saved_sys_stdout = sys.stdout
    try:
        # Route Python-level prints AND subprocess-inherited fd 1 to stderr
        # for the duration of recovery (see docstring: ACP stdout safety).
        try:
            saved_stdout_fd = os.dup(1)
            os.dup2(2, 1)
        except OSError:
            saved_stdout_fd = None
        sys.stdout = sys.stderr

        if lazy_marker:
            _recover_lazy_refresh_marker_locked()

        if _update_marker_path().exists():
            _recover_core_update_marker_locked()
    finally:
        sys.stdout = saved_sys_stdout
        if saved_stdout_fd is not None:
            try:
                os.dup2(saved_stdout_fd, 1)
                os.close(saved_stdout_fd)
            except OSError:
                pass
        try:
            lock_path.unlink()
        except OSError:
            pass


def _recover_lazy_refresh_marker_locked() -> None:
    """Heal ``.lazy-refresh-incomplete`` via confirmed import-probe repair."""
    print(
        "⚠ A previous lazy-backend refresh may have left the venv unhealthy — "
        "running import-based package repair..."
    )
    install_prefix, install_env = _m()._default_venv_install_target()
    status = _m()._repair_venv_via_import_probes(install_prefix, env=install_env)
    if status in ("healthy", "repaired"):
        _clear_lazy_refresh_incomplete_marker()
        print("✓ Lazy-refresh venv recovery confirmed — install is healthy again.")
        return
    if status == "indeterminate":
        print(
            "  ⚠ Import probes unavailable — cannot confirm venv health. "
            "Leaving `.lazy-refresh-incomplete` for the next launch."
        )
    else:
        print(
            "  ⚠ Lazy-refresh package repair incomplete. "
            "Leaving `.lazy-refresh-incomplete` for the next launch."
        )
        print("  Recover manually with:")
        all_specs = _m()._lazy_refresh_repair_specs(
            sorted(set(_m()._LAZY_REFRESH_REPAIR_PACKAGES.values()))
        )
        print(
            f"    {' '.join(install_prefix)} install --force-reinstall "
            + " ".join(shlex.quote(s) for s in all_specs)
        )


def _recover_core_update_marker_locked() -> None:
    """Heal ``.update-incomplete`` via full ``.[all]`` reinstall only.

    Narrow lazy-refresh import probes are not sufficient proof that a generic
    interrupted core install finished — a missing dep outside that probe set
    would otherwise look healthy and clear the breadcrumb too early.
    """
    print(
        "⚠ A previous `hermes update` was interrupted mid-install — "
        "finishing dependency installation now..."
    )

    # Windows: a normal ``hermes.exe`` launch always has the launcher as an
    # ancestor. Full editable reinstall uses quarantine so the live shim can
    # still be replaced. Package-only import repair may help as first aid but
    # must NEVER clear this core marker on its own (#58004 review).
    self_locked = _m()._windows_running_hermes_launcher_locked()
    if self_locked:
        install_prefix, install_env = _m()._default_venv_install_target()
        print(
            "  → Running from hermes.exe; applying package-only first aid, "
            "then quarantined full reinstall (core marker stays until that "
            "succeeds)..."
        )
        _m()._repair_venv_via_import_probes(install_prefix, env=install_env)

    try:
        from hermes_cli.managed_uv import ensure_uv

        # Always bootstrap pip first: a killed install can leave the venv with
        # no pip module at all, and uv may also be gone. ensurepip restores a
        # known-good pip so at least the plain-pip path below can proceed.
        try:
            subprocess.run(
                [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                cwd=_m().PROJECT_ROOT,
                capture_output=True,
            )
        except Exception as exc:
            logger.debug("ensurepip during install recovery failed: %s", exc)

        uv_bin = ensure_uv()
        if uv_bin:
            uv_env = {**os.environ, "VIRTUAL_ENV": str(_m().PROJECT_ROOT / "venv")}
            if _m()._is_termux_env(uv_env):
                uv_env.pop("PYTHONPATH", None)
                uv_env.pop("PYTHONHOME", None)
            _m()._install_python_dependencies_with_optional_fallback(
                [uv_bin, "pip"],
                env=uv_env,
                group="termux-all" if _m()._is_termux_env(uv_env) else "all",
            )
        else:
            _m()._install_python_dependencies_with_optional_fallback(
                [sys.executable, "-m", "pip"],
                group="termux-all" if _m()._is_termux_env() else "all",
            )

        _clear_update_incomplete_marker()
        print("✓ Dependency installation recovered — your install is healthy again.")
    except Exception as exc:
        # Leave the marker in place so the next launch retries. Give the user
        # the exact manual recovery command in the meantime.
        logger.debug("Interrupted-install recovery failed: %s", exc)
        print("✗ Could not auto-recover the interrupted install.")
        if self_locked:
            print(
                "  Hermes is still running from the launcher that needs "
                "replacing. Close other Hermes windows, restart from a "
                "different terminal, then run:"
            )
            print(f'    cd /d "{_m().PROJECT_ROOT}"')
            print(
                f'    "{sys.executable}" -m pip install -e ".[all]"'
            )
        else:
            print("  Recover manually with:")
            print(f"    cd {_m().PROJECT_ROOT}")
            print(f"    {sys.executable} -m ensurepip --upgrade")
            print(f"    {sys.executable} -m pip install -e '.[all]'")


def _windows_running_hermes_launcher_locked() -> bool:
    """True when a venv ``hermes*.exe`` shim is this process or an ancestor.

    Best-effort: returns False when psutil is unavailable or inspection fails.
    """
    if not _m()._is_windows():
        return False
    scripts_dir = _m()._venv_scripts_dir()
    if scripts_dir is None:
        return False
    shims = _m()._hermes_exe_shims(scripts_dir)
    if not shims:
        return False
    shim_set: set[str] = set()
    for shim in shims:
        try:
            shim_set.add(str(shim.resolve()).lower())
        except OSError:
            shim_set.add(str(shim).lower())
    try:
        import psutil

        me = psutil.Process()
        for proc in [me] + list(me.parents()):
            try:
                exe_norm = str(Path(proc.exe()).resolve()).lower()
            except Exception:
                continue
            if exe_norm in shim_set:
                return True
    except Exception:
        return False
    return False

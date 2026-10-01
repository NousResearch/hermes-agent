"""Dependency install execution shared between early recovery and full recovery.

Callers: ``_early_recovery.recover_if_needed`` (stdlib-only, runs BEFORE ``hermes_cli.main``'s
third-party imports so a pending update completes while no native extension is mapped yet) and
``hermes_cli.main_install_repair._recover_core_update_marker_locked`` (the post-import recovery path).
Deliberately **stdlib-only** so importing it can never fail in the corrupted-venv state it exists
to repair; ``managed_uv`` and friends belong to the late path only.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# _early_recovery owns the recovery-lock lifecycle and uv lookup; importing it is free (stdlib).
from hermes_cli import _early_recovery as _er


def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_termux_env(env: dict | None = None) -> bool:
    """Stdlib Termux probe (hermes_cli.main's version lives behind imports)."""
    env = env if env is not None else os.environ
    try:
        return bool(env.get("TERMUX_VERSION")) or "com.termux" in env.get("PREFIX", "")
    except Exception:
        return False


@contextlib.contextmanager
def _stdout_to_stderr():
    """Route fd 1 (and sys.stdout) to stderr for the duration of an install.

    ``hermes acp`` speaks JSON-RPC on stdout; an inherited-fd install child writing there would
    corrupt the protocol.
    """
    saved_sys_stdout = sys.stdout
    try:
        saved_fd = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        saved_fd = None
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = saved_sys_stdout
        if saved_fd is not None:
            with contextlib.suppress(OSError):
                os.dup2(saved_fd, 1)
            with contextlib.suppress(OSError):
                os.close(saved_fd)


def _resolve_install_target(root: Path) -> tuple[list[str], dict | None]:
    """(install_cmd_prefix, env) for the project venv — stdlib uv lookup.

    ``VIRTUAL_ENV`` steers ``uv pip`` at the project venv even when invoked from the base
    interpreter. Termux strips leaked interpreter-path env vars so uv resolves the venv correctly.
    """
    uv_bin = _er._find_uv_binary()
    if uv_bin:
        from hermes_constants import project_venv_dir

        env = {**os.environ, "VIRTUAL_ENV": str(project_venv_dir(root) or root / "venv")}
        if _is_termux_env(env):
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
        return [uv_bin, "pip"], env
    return [sys.executable, "-m", "pip"], None


def _venv_scripts_dir(root: Path) -> Path | None:
    """Project venv Scripts/bin dir, when present (hermes_constants is stdlib-only)."""
    # hermes_constants is stdlib-only, so the canonical layout helpers are safe to use from this
    # corrupted-venv repair path (#76105: never open-code the Scripts/bin split).
    from hermes_constants import project_venv_dir, venv_bin_dir

    venv_dir = project_venv_dir(root)
    if venv_dir is None:
        return None
    scripts = venv_bin_dir(venv_dir, windows=_is_windows())
    return scripts if scripts.is_dir() else None


#: Launcher names install.ps1's Set-PathVariable exposes from the managed binary dir (the default
#: Hermes root's ``bin``, next to uv.exe). Keep in lockstep with scripts/install.ps1.
_WINDOWS_BIN_LAUNCHERS = ("hermes", "hermes-acp")


def _launcher_present(target: Path, name: str) -> bool:
    return (target / f"{name}.exe").exists() or (target / f"{name}.cmd").exists()


def _launchers_missing(target: Path) -> bool:
    return any(not _launcher_present(target, name) for name in _WINDOWS_BIN_LAUNCHERS)


def _default_hermes_root() -> Path | None:
    """The DEFAULT Hermes root (not ``get_hermes_home()``, which under ``hermes -p <name>`` is
    ``profiles\\<name>`` and would fail the managed-clone gate for profile users); ``None`` when
    unresolvable."""
    from hermes_constants import get_default_hermes_root

    try:
        return Path(get_default_hermes_root())
    except Exception:
        return None


def _venv_is_relocatable(venv_dir: Path) -> bool:
    r"""True when the venv's pyvenv.cfg declares ``relocatable = true``.

    A relocatable venv's console-script trampolines embed a RELATIVE interpreter reference, so a
    copy placed outside ``venv\Scripts`` fails (``uv trampoline failed to canonicalize script
    path``); non-relocatable venvs survive copying. Decides which launcher form a PATH dir gets.
    """
    try:
        cfg = (Path(venv_dir) / "pyvenv.cfg").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(
        key.strip().lower() == "relocatable" and value.strip().lower() == "true"
        for key, _, value in (line.partition("=") for line in cfg.splitlines())
    )


def _normalize_windows_path(value) -> str:
    """Windows path equality key: backslashes, no trailing separator, lowered.

    ``.lower()`` rather than ``os.path.normcase`` (identity on POSIX) so the comparison behaves
    Windows-correct even when tests exercise the Windows branch from another host.
    """
    return str(value).replace("/", "\\").rstrip("\\").lower()


def _windows_user_path_entries() -> list[str]:
    """User PATH entries from the registry (what install.ps1 writes); process PATH fallback."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            raw, _kind = winreg.QueryValueEx(key, "Path")
        value = os.path.expandvars(str(raw))
    except (OSError, ImportError):
        value = os.environ.get("PATH", "")
    return [entry for entry in value.split(";") if entry.strip()]


def ensure_windows_bin_launchers(
    root, *, windows: bool | None = None, user_path_entries: list[str] | None = None,
) -> list[str]:
    r"""Re-stage the Windows ``hermes`` launchers when they vanish.

    On Windows, ``hermes`` resolves through launchers derived from the venv console scripts — never
    ``venv\Scripts`` itself on PATH, which would shadow the user's ``python``. Targets: the
    canonical managed binary dir (only when *root* is the managed clone, so source checkouts
    elsewhere never gain launchers) and the legacy ``<root>\bin`` (only while the user PATH still
    points at it). Never raises.

    The canonical launcher home is the managed binary dir — the default Hermes root's ``bin``
    (``%LOCALAPPDATA%\\hermes\\bin``, next to the managed uv) — which lives OUTSIDE the git checkout so no
    git operation can ever touch it. It is a per-machine dir shared by every profile: ``get_hermes_home()``
    would point inside ``profiles\\<name>`` under ``hermes -p``, so the anchor here is
    :func:`hermes_constants.get_default_hermes_root`. See #83797.
    """
    if windows is None:
        windows = _is_windows()
    if not windows:
        return []
    root = Path(root)
    home = _default_hermes_root()
    if home is None:
        return []

    targets: list[Path] = []
    # Runs at every hermes_cli.main process start, so the healthy path must stay a few stat calls.
    if _normalize_windows_path(root.parent) == _normalize_windows_path(home):
        canonical = home / "bin"
        if _launchers_missing(canonical):
            targets.append(canonical)
    # Legacy target: compared as normalized literal strings — the installer wrote the long literal
    # path, and realpath'ing arbitrary PATH entries could hang on dead network shares. An entry
    # stored another way (8.3 short path, subst drive) misses the re-stage, which fails safe.
    legacy = root / "bin"
    if _launchers_missing(legacy):
        if user_path_entries is None:
            user_path_entries = _windows_user_path_entries()
        configured = {_normalize_windows_path(entry) for entry in user_path_entries}
        if _normalize_windows_path(legacy) in configured:
            targets.append(legacy)
    if not targets:
        return []

    from hermes_constants import project_venv_dir, venv_bin_dir

    venv_dir = project_venv_dir(root)
    if venv_dir is None:
        return []
    scripts_dir = venv_bin_dir(venv_dir, windows=windows)
    sources = [(name, scripts_dir / f"{name}.exe") for name in _WINDOWS_BIN_LAUNCHERS
               if (scripts_dir / f"{name}.exe").is_file()]
    if not sources:
        return []
    relocatable = _venv_is_relocatable(venv_dir)

    restored: list[str] = []
    for target in targets:
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        for name, source in sources:
            if _launcher_present(target, name):
                continue
            final = target / (f"{name}.cmd" if relocatable else f"{name}.exe")
            staging = target / f"{final.name}.heal.{os.getpid()}"
            try:
                if relocatable:
                    staging.write_text("@echo off\r\n" f'"{source}" %*\r\n', encoding="ascii")
                else:
                    shutil.copy2(source, staging)
                os.replace(staging, final)
                restored.append(str(final))
            except OSError:
                with contextlib.suppress(OSError):
                    staging.unlink()
    if restored:
        # A closed/broken stderr must not turn a successful heal into a crash.
        with contextlib.suppress(OSError, ValueError):
            print("  ✓ Restored hermes launcher(s): " + ", ".join(restored), file=sys.stderr)
    return restored


def _read_user_path_raw() -> tuple[list[str], int]:
    """Raw (unexpanded) user PATH entries + registry value type (a rewrite preserves ``%VARS%``)."""
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        try:
            raw, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            return [], winreg.REG_EXPAND_SZ
    return [entry for entry in str(raw).split(";") if entry], int(kind)


def _write_user_path_raw(entries: list[str], kind: int) -> None:
    """Write the user PATH back, preserving the registry value type."""
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                        winreg.KEY_READ | winreg.KEY_WRITE) as key:
        winreg.SetValueEx(key, "Path", 0, kind, ";".join(entries))


def migrate_windows_bin_path(
    root, *, windows: bool | None = None, read_user_path=None, write_user_path=None,
) -> bool:
    """One-time PATH migration to the ``HERMES_HOME\\bin`` launcher layout (``hermes update`` tail).

    1. stage launchers into the managed binary dir; 2. verify both are present — otherwise STOP,
    leaving the user PATH untouched (never strip a working entry before its replacement is proven);
    3. prepend the managed binary dir to the user PATH; 4. strip the legacy ``<root>\\bin`` and
    ``<root>\\venv\\Scripts`` entries. Legacy ``<root>\\bin`` FILES stay: configs that captured
    absolute launcher paths keep working and the dir is git-ignored. Registry writes preserve the
    stored value type and raw ``%VARS%``. Never raises; True when the canonical layout is in place.
    *read_user_path*/*write_user_path* are injectable for tests.

    See #83797.
    """
    if windows is None:
        windows = _is_windows()
    if not windows:
        return False
    root = Path(root)

    from hermes_constants import venv_bin_dir

    home = _default_hermes_root()
    if home is None:
        return False
    if _normalize_windows_path(root.parent) != _normalize_windows_path(home):
        return False  # not the managed clone — nothing to migrate

    ensure_windows_bin_launchers(root, windows=windows, user_path_entries=[])
    home_bin = home / "bin"
    if any(not ((home_bin / f"{name}.exe").is_file() or (home_bin / f"{name}.cmd").is_file())
           for name in _WINDOWS_BIN_LAUNCHERS):
        return False  # staging incomplete — leave the PATH alone

    if read_user_path is None:
        read_user_path = _read_user_path_raw
    if write_user_path is None:
        write_user_path = _write_user_path_raw
    try:
        entries, kind = read_user_path()
    except (OSError, ImportError):
        return False

    legacy_keys = {
        _normalize_windows_path(root / "bin"),
        # The old installer put the venv's Scripts dir itself on PATH, always at the literal
        # `venv` layout (never `.venv`) — match what it wrote then, not where the venv lives now.
        # See #83797.
        _normalize_windows_path(venv_bin_dir(root / "venv", windows=True)),
    }
    home_bin_key = _normalize_windows_path(home_bin)

    def _entry_key(entry: str) -> str:
        return _normalize_windows_path(os.path.expandvars(entry))

    kept = [e for e in entries if _entry_key(e) not in legacy_keys]
    if not any(_entry_key(e) == home_bin_key for e in kept):
        kept = [str(home_bin)] + kept
    if kept != entries:
        try:
            write_user_path(kept, kind)
        except (OSError, ImportError):
            return False
        with contextlib.suppress(OSError, ValueError):
            print(f"  ✓ hermes launchers now resolve from {home_bin} "
                  "(legacy PATH entries removed)", file=sys.stderr)
    return True


def _load_console_script_names(root: Path) -> list[str]:
    """``[project.scripts]`` names from pyproject.toml (tomllib, 3.11+)."""
    project = _er._load_pyproject_project(root)
    try:
        scripts = (project or {}).get("scripts", {}) or {}
        return [str(name) for name in scripts if name]
    except Exception:
        return []


class ShimQuarantineError(RuntimeError):
    """A live shim could not be renamed aside — the venv is contended.

    Raised BEFORE the install command runs. Callers catch it like any install failure: the
    update-incomplete marker survives and a later launch retries once the holder exits — the
    contended venv is never mutated.

    See #87331.
    """

    def __init__(self, failed_shims: list[str]):
        self.failed_shims = list(failed_shims)
        super().__init__("could not quarantine live shim(s): " + ", ".join(self.failed_shims))


def _quarantine_running_hermes_exe(
    scripts_dir: Path, *, failed_out: list[str] | None = None
) -> list[tuple[Path, Path]]:
    """Rename live hermes*.exe shims aside so the installer can rewrite them.

    Windows blocks REPLACE on a running .exe but allows RENAME. Best-effort: silently skips anything
    that cannot be renamed (names appended to *failed_out*). Returns (original, quarantined) pairs.
    The console-script set comes from pyproject ``[project.scripts]`` (fallback: well-known trio).

    ``failed_out``: when provided, names of shims that could not be renamed are appended so the caller can
    refuse instead of mutating a contended venv (#87331 fail-closed).
    """
    if not _is_windows():
        return []
    names = set(_load_console_script_names(scripts_dir.parent.parent)) or {
        "hermes", "hermes-agent", "hermes-acp"}
    names.add("hermes-gateway")
    moved: list[tuple[Path, Path]] = []
    for name in sorted(names):
        shim = scripts_dir / f"{name}.exe"
        if not shim.exists():
            continue
        quarantined = shim.with_name(f"{name}.exe.old.{int(time.time() * 1000)}")
        try:
            os.rename(shim, quarantined)
            moved.append((shim, quarantined))
        except OSError:
            if failed_out is not None:
                failed_out.append(shim.name)
    return moved


def _restore_quarantined_exes(moved: list[tuple[Path, Path]]) -> None:
    """Put quarantined shims back when the installer did not replace them (shared retry ladder).

    Delegates to the shared helper in the stdlib-only ``_early_recovery`` module: one retry ladder and one
    recovery message for every restore site, instead of the near-identical copies that had already drifted
    (#75584). Warnings land on stderr — this module runs in the early-recovery path and ``hermes acp``
    speaks JSON-RPC on stdout.
    """
    _er.restore_quarantined_shims(moved)


def _run_install_cmd(cmd: list[str], *, env: dict | None, root: Path) -> None:
    """Run an install command with quarantine protection for venv shims.

    Fail-closed: when any live shim cannot be renamed aside, the venv is contended and the
    installer would die partway on the same locks — raise :class:`ShimQuarantineError` WITHOUT
    running it. Raises CalledProcessError on install failure (callers implement the per-extra
    fallback ladder).

    The caller's marker-keeping failure handling turns that into "retry next launch". See #87331.
    """
    scripts_dir = _venv_scripts_dir(root) if _is_windows() else None
    failed: list[str] = []
    moved = _quarantine_running_hermes_exe(scripts_dir, failed_out=failed) if scripts_dir else []
    if failed:
        _restore_quarantined_exes(moved)
        raise ShimQuarantineError(failed)
    try:
        subprocess.run(cmd, cwd=root, check=True, env=env)
    finally:
        # Restore on success AND failure: a SUCCESSFUL install can skip the entry-points step
        # entirely (uv audits an already-satisfied editable install as a no-op), which would leave
        # the shims renamed aside and `hermes` gone from PATH. Restore only renames back when the
        # installer did NOT write a fresh shim, so this is safe in both cases.
        # See #75584.
        if scripts_dir is not None:
            _restore_quarantined_exes(moved)


def _load_installable_optional_extras(root: Path, group: str) -> list[str]:
    """Optional extras referenced by a dependency group (all / termux-all)."""
    project = _er._load_pyproject_project(root)
    if project is None:
        return []
    optional_deps = project.get("optional-dependencies", {})
    if not isinstance(optional_deps, dict):
        return []
    referenced: list[str] = []
    for ref in optional_deps.get(group, []):
        if "[" in ref and "]" in ref:
            name = ref.split("[", 1)[1].split("]", 1)[0]
            if name in optional_deps:
                referenced.append(name)
    return referenced


def run_core_install(root: Path) -> None:
    """Full core ``.[all]`` editable reinstall — the recovery install.

    ensurepip bootstrap; ``uv pip`` with VIRTUAL_ENV at the project venv (``python -m pip``
    fallback); ``.[all]`` (``.[termux-all]`` on Termux) with the per-extra fallback ladder when the
    combined resolve fails; live shims quarantined on Windows; ALL output routed to stderr.
    """
    prefix, env = _resolve_install_target(root)
    group = "termux-all" if _is_termux_env(env) else "all"

    def install(target: str) -> None:
        _run_install_cmd(prefix + ["install", "-e", target], env=env, root=root)

    with _stdout_to_stderr():
        _er._run_ensurepip(root)
        try:
            install(f".[{group}]")
            return
        except subprocess.CalledProcessError:
            print("  ⚠ Optional extras failed, reinstalling base dependencies "
                  "and retrying extras individually...")
        install(".")
        failed_extras: list[str] = []
        installed_extras: list[str] = []
        for extra in _load_installable_optional_extras(root, group):
            try:
                install(f".[{extra}]")
                installed_extras.append(extra)
            except subprocess.CalledProcessError:
                failed_extras.append(extra)
        if installed_extras:
            print("  ✓ Reinstalled optional extras individually: " + ", ".join(installed_extras))
        if failed_extras:
            print("  ⚠ Skipped optional extras that still failed: " + ", ".join(failed_extras))


def bump_marker_attempts(marker_path: Path) -> int:
    """Increment the attempts counter stored inside the marker file's JSON body.

    The marker's existence is the signal; the body carries the retry count so a persistently
    failing install can back off. Corrupt/missing bodies restart at 1. Never raises.
    """
    attempts = _er._read_marker_attempts(marker_path) + 1
    with contextlib.suppress(OSError):
        marker_path.write_text(json.dumps({"attempts": attempts}), encoding="utf-8")
    return attempts

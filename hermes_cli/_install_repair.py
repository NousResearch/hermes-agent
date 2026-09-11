"""Repair Windows launchers and their registered PATH entries."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path


def _sync_windows_cli_launchers(root: Path) -> list[Path]:
    # Shim to stop the old updater doing work until relaunch. Copy no launchers.
    return []


def _is_windows() -> bool:
    return sys.platform == "win32"


#: Launcher command names install.ps1's Set-PathVariable exposes from the
#: managed binary dir (the default Hermes root's ``bin``, next to uv.exe)
#: on the user PATH. Keep in lockstep with WINDOWS_BIN_LAUNCHERS in
#: hermes_cli/_launchers.py and scripts/install.ps1.
_WINDOWS_BIN_LAUNCHERS = ("hermes", "hermes-acp")


def _normalize_windows_path(value) -> str:
    """Windows path equality key: backslashes, no trailing separator, lowered.

    Lowercase via ``.lower()`` (what ``ntpath.normcase`` does) rather than
    ``os.path.normcase`` — that is an identity function on POSIX, and this
    comparison must behave Windows-correct even when tests exercise the
    Windows branch from another host (same rationale as
    ``venv_bin_dir(windows=...)``).
    """
    return str(value).replace("/", "\\").rstrip("\\").lower()


def _windows_user_path_entries() -> list[str]:
    """User PATH entries from the registry — the value install.ps1 writes.

    Falls back to the process PATH when the registry is unreadable. Only
    called on Windows.
    """
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            raw, _kind = winreg.QueryValueEx(key, "Path")
        value = os.path.expandvars(str(raw))
    except (OSError, ImportError):
        value = os.environ.get("PATH", "")
    return [entry for entry in value.split(";") if entry.strip()]


def ensure_windows_bin_launchers(
    root,
    *,
    windows: bool | None = None,
    user_path_entries: list[str] | None = None,
) -> list[str]:
    """Re-stage the Windows ``hermes`` launchers when they vanish or when
    they still boot through the venv.

    On Windows, ``hermes`` resolves through staged launchers — never
    ``venv\\Scripts`` itself on PATH, which would shadow the user's
    ``python`` (#83797) — and under pm the launchers boot the pm STORE
    python with ``PYTHONPATH=<repo>;<venv>/site-packages``, never the venv
    interpreter (no-boot-through-venv; ``pyvenv.cfg`` is inert dead
    config). The canonical launcher home is
    the managed binary dir — the default Hermes root's ``bin``
    (``%LOCALAPPDATA%\\hermes\\bin``, next to the managed uv) — which lives
    OUTSIDE the git checkout so no git operation can ever touch it. It is
    a per-machine dir shared by every profile: ``get_hermes_home()`` would
    point inside ``profiles\\<name>`` under ``hermes -p``, so the anchor
    here is :func:`hermes_constants.get_default_hermes_root`.

    Earlier installer versions staged them at ``<checkout>\\bin`` instead —
    inside the git working tree — where ``hermes update``'s pre-update
    autostash (``git stash push --include-untracked``) swept them off disk;
    once the desktop updater stopped re-applying stashes (``--keep-stash``)
    nothing restored them and ``hermes`` stopped resolving in every new
    terminal. That legacy location is re-staged too, during the transition,
    for installs whose user PATH still resolves through it.

    A name counts as present when an exe exists that does NOT boot the
    venv interpreter — legacy copied-venv trampolines (detected by their
    embedded interpreter path) and placeholder .cmd delegators are replaced
    with a store-python launcher as soon as one can be minted.

    Two targets, two gates, both failing toward inaction:

    - canonical managed binary dir: only when *root* is the managed clone
      (``root.parent == get_default_hermes_root()``), so source checkouts
      elsewhere never gain launchers;
    - legacy ``<root>\\bin``: only when that dir is on the user PATH
      (registry value, process PATH as fallback), i.e. the install opted
      into the old layout and still resolves through it.

    Writes go through a staging name + ``os.replace`` so concurrent process
    starts cannot tear a launcher. Never raises; returns the restored paths.

    *windows* and *user_path_entries* are injectable for tests, same pattern
    as ``hermes_constants.venv_bin_dir``.
    """
    if windows is None:
        windows = _is_windows()
    if not windows:
        return []

    root = Path(root)

    # Per-machine anchor: the DEFAULT Hermes root, not get_hermes_home() —
    # under ``hermes -p <name>`` that returns ``profiles\\<name>``, which
    # would fail the managed-clone gate below and silently skip the heal
    # for profile users. The launcher dir serves the whole machine.
    from hermes_constants import get_default_hermes_root

    try:
        home = Path(get_default_hermes_root())
    except Exception:
        return []

    def _launcher_present(target: Path, name: str) -> bool:
        return (target / f"{name}.exe").exists() or (target / f"{name}.cmd").exists()

    # Launchers boot the pm STORE python with PYTHONPATH=repo;site-packages
    # — never the venv interpreter (no boot through the venv; pyvenv.cfg is
    # inert dead config). mint_launcher prefers a distlib exe trampoline
    # bound to the store python; a runtime-resolving .cmd is written when
    # the store has not materialized a python yet, and the repair upgrades
    # it (and any legacy copied-venv trampoline) to the exe once the store
    # python exists. See hermes_cli/_launchers.py.
    from hermes_cli._launchers import (
        exe_is_venv_bound,
        mint_launcher,
        resolve_store_python,
        stage_launcher,
    )

    from hermes_constants import project_venv_dir

    venv_dir = project_venv_dir(root)
    from hermes_cli.runtime_paths import site_packages as dependency_site

    site_packages = dependency_site(venv_dir) if venv_dir else None

    store_python = resolve_store_python(root)

    def _needs_attention(target: Path, name: str) -> bool:
        """Missing, a placeholder .cmd, or a launcher that still boots the
        venv interpreter — anything the store-python launcher should replace."""
        exe = target / f"{name}.exe"
        if not exe.exists():
            return True
        return exe_is_venv_bound(exe, venv_dir)

    targets: list[Path] = []

    # Canonical target — gate on the managed-clone shape. This runs at
    # every hermes_cli.main process start (right after the profile
    # override), so the healthy path must stay at a couple of stat calls.
    if _normalize_windows_path(root.parent) == _normalize_windows_path(home):
        canonical = home / "bin"
        if any(
            _needs_attention(canonical, name) for name in _WINDOWS_BIN_LAUNCHERS
        ):
            targets.append(canonical)

    # Legacy transition target — the pre-migration in-checkout dir. Only
    # re-staged while the user PATH still points at it (consent), compared
    # as normalized literal strings: the installer wrote the long literal
    # path, and realpath'ing arbitrary PATH entries could hang on dead
    # network shares. An entry stored some other way (8.3 short path,
    # subst drive) misses the re-stage, which fails safe: no-op.
    legacy = root / "bin"
    if any(_needs_attention(legacy, name) for name in _WINDOWS_BIN_LAUNCHERS):
        if user_path_entries is None:
            user_path_entries = _windows_user_path_entries()
        configured = {_normalize_windows_path(entry) for entry in user_path_entries}
        if _normalize_windows_path(legacy) in configured:
            targets.append(legacy)

    if not targets:
        return []

    restored: list[str] = []
    for target in targets:
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        for name in _WINDOWS_BIN_LAUNCHERS:
            if not _needs_attention(target, name):
                # Already a store-python launcher (or a form this heal does
                # not understand but that does not boot the venv): leave it.
                continue
            if store_python is not None:
                final = mint_launcher(name, root, target, store_python, site_packages)
            else:
                final = stage_launcher(name, root, target)
            if final is not None:
                # Windows resolves .exe before .cmd. A surviving venv-bound
                # launcher would shadow the successfully staged fallback.
                obsolete = target / f"{name}.exe"
                if final.suffix == ".cmd" and exe_is_venv_bound(obsolete, venv_dir):
                    obsolete.unlink()
                restored.append(str(final))
    if restored:
        # Guarded like everything else in this never-raises helper: a
        # closed/broken stderr must not turn a successful heal into a crash.
        with contextlib.suppress(OSError, ValueError):
            print(
                "  ✓ Restored hermes launcher(s): " + ", ".join(restored),
                file=sys.stderr,
            )
    return restored


def _read_user_path_raw() -> tuple[list[str], int]:
    """Raw (unexpanded) user PATH entries + registry value type.

    Raw so a rewrite preserves ``%VARS%`` exactly as the user stored them
    (same discipline as ``hermes_cli.uninstall``). Only called on Windows.
    """
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

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE
    ) as key:
        winreg.SetValueEx(key, "Path", 0, kind, ";".join(entries))


def migrate_windows_bin_path(
    root,
    *,
    windows: bool | None = None,
    read_user_path=None,
    write_user_path=None,
) -> bool:
    """One-time PATH migration to the ``HERMES_HOME\\bin`` launcher layout.

    Runs from the ``hermes update`` tail (and mirrors what install.ps1's
    Set-PathVariable does on fresh installs/repairs, which never reach
    existing installs — updates don't run install.ps1):

    1. stage the launcher copies into the managed binary dir (via
       :func:`ensure_windows_bin_launchers`);
    2. verify both launchers are present there — otherwise STOP, leaving
       the user PATH untouched (never strip a working entry before its
       replacement is proven);
    3. ensure the managed binary dir is on the user PATH (prepend);
    4. strip the legacy entries: ``<root>\\bin`` (in-checkout launcher dir
       the update autostash could sweep) and ``<root>\\venv\\Scripts``
       (shadowed the user's ``python``, #83797).

    The legacy ``<root>\\bin`` FILES are deliberately left in place: editor
    and ACP configs that captured absolute launcher paths keep working
    (the launchers run fine from there — only PATH resolution through a
    dir git could sweep was the bug), and the dir is git-ignored so it
    cannot dirty the tree.

    Registry writes preserve the stored value type and raw ``%VARS%``.
    Never raises; returns True when the canonical layout is in place.

    *read_user_path*/*write_user_path* are injectable for tests.
    """
    if windows is None:
        windows = _is_windows()
    if not windows:
        return False

    root = Path(root)

    # Same per-machine anchor as ensure_windows_bin_launchers (see there).
    from hermes_constants import get_default_hermes_root, venv_bin_dir

    try:
        home = Path(get_default_hermes_root())
    except Exception:
        return False
    if _normalize_windows_path(root.parent) != _normalize_windows_path(home):
        return False  # not the managed clone — nothing to migrate

    ensure_windows_bin_launchers(root, windows=windows, user_path_entries=[])

    home_bin = home / "bin"
    if any(
        not ((home_bin / f"{name}.exe").is_file() or (home_bin / f"{name}.cmd").is_file())
        for name in _WINDOWS_BIN_LAUNCHERS
    ):
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
        # The pre-#83797 installer put the venv's Scripts dir itself on PATH,
        # always at the literal `venv` layout (never `.venv`) — this strips
        # that stale entry, so it must match what the installer wrote then,
        # not where the venv lives now.
        _normalize_windows_path(venv_bin_dir(root / "venv", windows=True)),
    }
    home_bin_key = _normalize_windows_path(home_bin)

    def _entry_key(entry: str) -> str:
        return _normalize_windows_path(os.path.expandvars(entry))

    kept = [e for e in entries if _entry_key(e) not in legacy_keys]
    have_home_bin = any(_entry_key(e) == home_bin_key for e in kept)
    if not have_home_bin:
        kept = [str(home_bin)] + kept

    if kept != entries:
        try:
            write_user_path(kept, kind)
        except (OSError, ImportError):
            return False
        with contextlib.suppress(OSError, ValueError):
            print(
                f"  ✓ hermes launchers now resolve from {home_bin} "
                "(legacy PATH entries removed)",
                file=sys.stderr,
            )
    return True

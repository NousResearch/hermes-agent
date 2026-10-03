"""Hermes Desktop (Chat GUI) uninstaller: removes only GUI state — built Electron artifacts, the packaged
app, and the desktop's own ``userData`` — never agent source, venv, config, sessions or .env."""

import os
import shutil
import sys
from pathlib import Path

from hermes_constants import get_hermes_home

from hermes_cli.colors import Colors, color


def _logger(mark: str, col: str):
    return lambda msg: print(f"{color(mark, col)} {msg}")


log_info, log_success = _logger("→", Colors.CYAN), _logger("✓", Colors.GREEN)
log_warn = _logger("⚠", Colors.YELLOW)


def _env_dir(var: str, fallback: Path) -> Path:
    """``Path($var)`` when the env var is set, else *fallback*."""
    return Path(value) if (value := os.environ.get(var)) else fallback


def desktop_userdata_dir() -> Path:
    """Electron ``app.getPath('userData')`` for an app named "Hermes" on each platform (GUI-only state)."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Hermes"
    if sys.platform == "win32":
        return _env_dir("APPDATA", home / "AppData" / "Roaming") / "Hermes"
    return _env_dir("XDG_CONFIG_HOME", home / ".config") / "Hermes"


def _running_checkout_root() -> Path:
    """The checkout whose ``apps/desktop`` this code builds and uninstalls.

    ``hermes_cli/`` sits directly under the checkout root, so the running module's own location
    IS that root: ``$HERMES_HOME/hermes-agent`` on an install.sh install, or a custom install
    directory (``hermes --version`` prints it) when the git checkout was cloned elsewhere. A
    bundled or sealed install carries no ``apps/desktop`` under it, so that candidate is a no-op
    there.
    """
    return Path(__file__).resolve().parent.parent


def _same_path(left: Path, right: Path) -> bool:
    """Lexical identity, never ``resolve()``: either candidate may be a symlinked entry point."""
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def source_built_gui_artifacts(hermes_home: Path, project_root: "Path | None" = None) -> "list[Path]":
    """GUI build artifacts produced by ``hermes desktop`` inside the checkout.

    Two layouts own those artifacts and BOTH must be swept: the ``hermes-agent/`` directory install.sh
    writes under ``$HERMES_HOME``, and the running checkout itself, which a git install may place at
    any directory. The desktop build, the update tail and the rebuild decision all key off the running
    checkout, so reporting only the first let ``hermes uninstall --gui`` claim success on a
    custom-path checkout while the next ``hermes update`` rebuilt the desktop app from artifacts it
    had never removed.

    The workspace-root ``node_modules`` is shared with the TUI, dashboard and other workspaces, so only
    the desktop workspace's own dependencies belong to GUI removal.
    """
    roots = [hermes_home / "hermes-agent"]
    actual = _running_checkout_root() if project_root is None else Path(project_root)
    if not any(_same_path(actual, root) for root in roots):
        roots.append(actual)
    artifacts: "list[Path]" = []
    for root in roots:
        artifacts += [root / "apps" / "desktop" / "dist", root / "apps" / "desktop" / "release",
                      root / "apps" / "desktop" / "node_modules"]
    artifacts.append(hermes_home / "desktop-build-stamp.json")
    return artifacts


def desktop_install_record() -> Path:
    """Where ``hermes update`` records the installed ``Hermes.app`` copies it keeps current. The apps
    are machine-wide, so the record sits under the default root whichever profile runs; deleting it
    is what stops an uninstalled app from being put back by the next update."""
    from hermes_constants import get_default_hermes_root  # noqa: PLC0415
    return get_default_hermes_root() / "desktop-installed-apps.json"


def packaged_gui_app_paths() -> "list[Path]":
    """Standard install locations of the packaged desktop distributable for the current OS. Every candidate
    is returned; the caller filters to those that exist. Never globs system-wide — only the well-known
    electron-builder output locations for the "Hermes" product."""
    home = Path.home()
    if sys.platform == "darwin":
        return [Path("/Applications/Hermes.app"), home / "Applications" / "Hermes.app"]
    if sys.platform == "win32":
        local_base = _env_dir("LOCALAPPDATA", home / "AppData" / "Local")
        # NSIS per-user install (perMachine=false), an older/alternate layout, NSIS per-machine (needs admin).
        program_files = os.environ.get("ProgramFiles")
        return [local_base / "Programs" / "Hermes", local_base / "hermes-desktop"] + (
            [Path(program_files) / "Hermes"] if program_files else [])
    # Linux: an AppImage lives wherever the user put it and deb/rpm files belong to the package manager
    # (see the hint in ``uninstall_gui``), so only the desktop entry + hicolor icons are cleaned here.
    from hermes_cli.linux_desktop_entry import LEGACY_DESKTOP_ENTRY_NAME, desktop_entry_path
    data_base = _env_dir("XDG_DATA_HOME", home / ".local" / "share")
    icons = data_base / "icons" / "hicolor"
    # "scalable" plus every fixed-size dir the installer may have written (panel sizes + older native copies).
    # The legacy entry is a hidden alias of the app-id entry since #124492 — remove it with the real one.
    return [desktop_entry_path(),
            data_base / "applications" / LEGACY_DESKTOP_ENTRY_NAME,
            data_base / "applications" / "Hermes.desktop"] + [
        icons / size / "apps" / "hermes.png"
        for size in ("scalable", "24x24", "32x32", "48x48", "256x256", "512x512", "1024x1024")]


def agent_is_installed(hermes_home: Path) -> bool:
    """True when a usable Python agent install exists under HERMES_HOME (gates the desktop UI's options).
    Package source or a venv alone is enough — a source checkout without a venv is still "the agent is here"."""
    return any((hermes_home / "hermes-agent" / sub).is_dir() for sub in ("hermes_cli", "venv", ".venv"))


def gui_is_installed(hermes_home: Path, project_root: "Path | None" = None) -> bool:
    """Return True when any desktop GUI artifact or install record exists."""
    return any(p.exists() for p in (
        *source_built_gui_artifacts(hermes_home, project_root), *packaged_gui_app_paths(),
        desktop_userdata_dir(), desktop_install_record(),
    ))


def gui_install_summary(hermes_home: "Path | None" = None, project_root: "Path | None" = None) -> dict:
    """JSON-serializable snapshot of what's installed, for the desktop UI to render via IPC."""
    home: Path = hermes_home if hermes_home is not None else get_hermes_home()
    userdata = desktop_userdata_dir()
    # Steward facts, so the UI can gate its destructive options on the same
    # ladder the CLI uninstaller uses: only a git checkout may have its code
    # removed; sealed trees (nix / desktop-app / docker) get data-only.
    from hermes_cli import steward as steward_mod

    steward, code_removal_allowed = steward_mod.classify_install(home / "hermes-agent")

    return {
        "hermes_home": str(home),
        "agent_installed": agent_is_installed(home),
        "gui_installed": gui_is_installed(home, project_root),
        "source_built_artifacts": [str(p) for p in source_built_gui_artifacts(home, project_root) if p.exists()],
        "packaged_app_paths": [str(p) for p in packaged_gui_app_paths() if p.exists()],
        "userdata_dir": str(userdata),
        "userdata_exists": userdata.exists(),
        "platform": sys.platform,
        "steward": steward,
        "code_removal_allowed": code_removal_allowed,
    }


def _remove_path(path: Path) -> bool:
    """Remove a file or directory tree. Returns True when something was removed."""
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            return False
        return True
    except Exception as e:
        log_warn(f"Could not remove {path}: {e}")
        return False


def uninstall_gui(hermes_home: "Path | None" = None, *, remove_userdata: bool = True,
                  project_root: "Path | None" = None) -> "list[Path]":
    """Remove the desktop GUI's artifacts, leaving the agent + user data intact."""
    home: Path = hermes_home if hermes_home is not None else get_hermes_home()
    removed: list[Path] = []

    def _remove_existing(paths) -> bool:
        """Remove every existing path; True when at least one existed."""
        found = False
        for path in (p for p in paths if p.exists()):
            found = True
            if _remove_path(path):
                log_success(f"Removed {path}")
                removed.append(path)
        return found
    log_info("Removing built GUI artifacts (renderer, release, node_modules)...")
    _remove_existing([*source_built_gui_artifacts(home, project_root), desktop_install_record()])
    log_info("Removing installed desktop app...")
    if not _remove_existing(packaged_gui_app_paths()):
        log_info("No packaged desktop app found in standard locations")
    if remove_userdata and (userdata := desktop_userdata_dir()).exists():
        log_info("Removing desktop app data (Electron userData)...")
        _remove_existing([userdata])
    if not removed:
        log_info("No desktop GUI artifacts found to remove")
    if sys.platform.startswith("linux"):
        # The desktop entry was removed above but the menu caches still list it; reindex so Hermes
        # disappears from the launcher.
        try:
            from hermes_cli.linux_desktop_entry import desktop_entry_path, refresh_desktop_databases
            entry = desktop_entry_path()
            if entry in removed:
                for tool in refresh_desktop_databases(entry.parent):
                    log_success(f"Refreshed the application menu cache ({tool})")
        except Exception as e:
            log_warn(f"Could not refresh the application menu cache: {e}")
        log_info("If you installed the desktop via a .deb / .rpm package, remove it with your package manager "
                 "(e.g. 'sudo apt remove hermes' or 'sudo dnf remove hermes'). AppImage builds are a single "
                 "file you can delete from wherever you saved it.")
    return removed

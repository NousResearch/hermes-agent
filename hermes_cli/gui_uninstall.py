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


def source_built_gui_artifacts(hermes_home: Path) -> "list[Path]":
    """GUI build artifacts produced by ``hermes desktop`` inside the checkout (same ``hermes-agent/`` layout
    install.sh uses). The Python agent runs from source + venv and never needs the Electron build output or
    node_modules (the workspace-root node_modules only carries Electron, ~200MB)."""
    agent_root = hermes_home / "hermes-agent"
    desktop_dir = agent_root / "apps" / "desktop"
    return [desktop_dir / "dist", desktop_dir / "release", desktop_dir / "node_modules",
            agent_root / "node_modules", hermes_home / "desktop-build-stamp.json"]


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
        if program_files:
            # NSIS per-machine fallback (needs admin to remove).
            paths.append(Path(program_files) / "Hermes")
    else:
        # Linux: AppImage is a single file the user placed somewhere; we can
        # only reliably clean the desktop entry + icon we know the name of.
        # The AppImage itself lives wherever the user put it, so we surface a
        # hint rather than guessing. deb/rpm installs are owned by the system
        # package manager and must be removed via apt/dnf — see the message in
        # ``uninstall_gui``.
        from hermes_cli.linux_desktop_entry import desktop_entry_path

        data = os.environ.get("XDG_DATA_HOME")
        data_base = Path(data) if data else (home / ".local" / "share")
        paths += [
            # The launcher entry `hermes desktop` installs. Its icon is
            # also copied into the hicolor tree (see
            # linux_desktop_entry._install_icon_to_hicolor) — remove
            # every size dir the installer could have written.
            desktop_entry_path(),
            # Some packaged builds emit this casing.
            data_base / "applications" / "Hermes.desktop",
            data_base / "icons" / "hicolor" / "scalable" / "apps" / "hermes.png",
        ]
        # Fixed-size hicolor dirs the installer may have written (resized
        # panel sizes plus leftover native-size copies from older builds).
        for size in ("24x24", "32x32", "48x48", "256x256", "512x512", "1024x1024"):
            paths.append(data_base / "icons" / "hicolor" / size / "apps" / "hermes.png")
    return paths


def agent_is_installed(hermes_home: Path) -> bool:
    """True when a usable Python agent install exists under HERMES_HOME (gates the desktop UI's options).
    Package source or a venv alone is enough — a source checkout without a venv is still "the agent is here"."""
    return any((hermes_home / "hermes-agent" / sub).is_dir() for sub in ("hermes_cli", "venv", ".venv"))


def gui_is_installed(hermes_home: Path) -> bool:
    """Return True when any desktop GUI artifact exists (built or packaged)."""
    return any(p.exists() for p in (*source_built_gui_artifacts(hermes_home), *packaged_gui_app_paths(), desktop_userdata_dir()))


def gui_install_summary(hermes_home: "Path | None" = None) -> dict:
    """JSON-serializable snapshot of what's installed, for the desktop UI to render via IPC."""
    home: Path = hermes_home if hermes_home is not None else get_hermes_home()
    userdata = desktop_userdata_dir()
    return {"hermes_home": str(home), "agent_installed": agent_is_installed(home),
            "gui_installed": gui_is_installed(home),
            "source_built_artifacts": [str(p) for p in source_built_gui_artifacts(home) if p.exists()],
            "packaged_app_paths": [str(p) for p in packaged_gui_app_paths() if p.exists()],
            "userdata_dir": str(userdata), "userdata_exists": userdata.exists(), "platform": sys.platform}


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


def uninstall_gui(hermes_home: "Path | None" = None, *, remove_userdata: bool = True) -> "list[Path]":
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
    _remove_existing(source_built_gui_artifacts(home))
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

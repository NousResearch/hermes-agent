"""Copy each unified agent+desktop package's ``desktop/`` half into the app-level desktop-plugins folder.

A unified package keeps its agent half in ``<home>/plugins/<name>/`` (the default home and every
profile). Its desktop half is loaded from ``<root>/desktop-plugins/<name>/``, with a
``.hermes-package.json`` marker that pairs it back to the agent row. Native Desktop does this copy
in Electron (``reconcileUnifiedDesktopHalves`` in ``apps/desktop/electron/desktop-plugins-root.ts``),
so a host that Desktop only reaches over SSH never gets it. The Webapp serves that host's folder,
so its routes (``web_routers/desktop_plugins.py``) run this pass at the same points Electron does:
whenever the loader asks for the root, and on demand after an install.

Both writers share the folder on a host that runs native Desktop too, so this keeps Electron's
on-disk contract exactly: the marker's fields, the first-home-wins order, adoption of a byte-equal
marker-less copy, a byte compare for linked dev checkouts, staged publication, and a prune that
only fires when the source is really gone.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

PACKAGE_MARKER = ".hermes-package.json"
_GIT_ORIGIN = re.compile(r'\[remote "origin"\][^[]*?url\s*=\s*(\S+)')
#: Electron serializes every writer of the root in its main process; the server's threads do here.
_ROOT_WRITER = threading.Lock()


def _list_dirs(folder: Path, *, follow_links: bool = False) -> list[str]:
    """Folder names in *folder*. A package root follows links to folders (``ln -s ~/src/pkg
    ~/.hermes/plugins/`` is how a package is developed in place), the app root does not."""
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return []
    names = []
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False) or (follow_links and entry.is_symlink() and entry.is_dir()):
                names.append(entry.name)
        except OSError:
            continue
    return names


def _package_origin(package_dir: Path) -> dict:
    """Where the package came from (catalog sidecar first, then the git remote), for "Install here"."""
    try:
        sidecar = json.loads((package_dir / ".hermes-catalog.json").read_text(encoding="utf-8-sig"))
        if sidecar.get("repo"):
            origin = {"catalogName": sidecar.get("catalog_name"), "repo": sidecar["repo"], "sha": sidecar.get("sha")}
            return {key: value for key, value in origin.items() if value is not None}
    except (OSError, ValueError, AttributeError):
        pass
    try:
        match = _GIT_ORIGIN.search((package_dir / ".git" / "config").read_text(encoding="utf-8-sig"))
    except OSError:
        return {}
    return {"repo": match.group(1)} if match else {}


def _read_marker(folder: Path) -> dict | None:
    try:
        marker = json.loads((folder / PACKAGE_MARKER).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return marker if isinstance(marker, dict) and marker.get("package") and marker.get("source") else None


def _write_marker(folder: Path, marker: dict) -> None:
    (folder / PACKAGE_MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def _same_file(a: Path, b: Path) -> bool:
    """Same bytes on both paths. A comparison that cannot be made is never a match."""
    try:
        return a.read_bytes() == b.read_bytes()
    except OSError:
        return False


def _publish(source_dir: Path, target: Path, marker: dict) -> None:
    """Copy through a staging sibling and rename the finished tree into place, marker included, so a
    failure leaves the old folder or none: never a partial, marker-less one that a later pass would
    take for a hand install."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    staged = staging / target.name
    try:
        shutil.copytree(source_dir, staged, symlinks=True)
        _write_marker(staged, marker)
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)
        os.rename(staged, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def materialize_desktop_half(package_dir: Path, app_root: Path, name: str) -> Path | None:
    """Copy *package_dir*'s ``desktop/`` half to ``<app_root>/<name>/``. Returns the folder when it
    was copied or adopted, ``None`` when it is already current, absent, or a user's own plugin."""
    source_dir = package_dir / "desktop"
    entry = source_dir / "plugin.js"
    try:
        entry_stat = entry.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        logger.warning("desktop-plugins: cannot read %s: %s", name, exc)
        return None
    if not stat.S_ISREG(entry_stat.st_mode):
        return None

    target = app_root / name
    existing = _read_marker(target)
    linked = package_dir.is_symlink()
    marker = {
        "package": name,
        "source": str(source_dir),
        "sourceMtimeMs": entry_stat.st_mtime_ns / 1_000_000,
        **({"linked": True} if linked else {}),
        **_package_origin(package_dir),
    }

    if target.exists():
        if existing is None and (target / "plugin.js").exists():
            # Our own pre-marker copy still matches byte for byte: adopt it. Anything else is a
            # standalone plugin the user put there, and stays exactly as it is (#112450).
            if _same_file(target / "plugin.js", entry):
                _write_marker(target, marker)
                return target
            return None
        if (
            existing is not None
            and existing["source"] == str(source_dir)
            and bool(existing.get("linked")) == linked
            and (
                _same_file(target / "plugin.js", entry)
                if linked
                else existing.get("sourceMtimeMs", 0) >= marker["sourceMtimeMs"]
            )
        ):
            return None

    _publish(source_dir, target, marker)
    return target


def _source_gone(name: str, entry: Path) -> bool:
    """True only when the source ``plugin.js`` is really gone. A source we may not stat is not an
    uninstall: pruning its copy would silently drop the pane."""
    try:
        entry.stat()
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError as exc:
        logger.warning("desktop-plugins: keeping desktop half of unreadable package %s: %s", name, exc)
    return False


def reconcile_desktop_halves(hermes_root: Path, app_root: Path) -> list[str]:
    """Materialize every local home's desktop halves into *app_root* (first home wins for a name in
    several profiles, the default home first) and drop copies whose package is gone. Returns the
    folders it wrote or removed."""
    with _ROOT_WRITER:
        touched: list[str] = []
        seen: set[str] = set()
        homes = [hermes_root, *(hermes_root / "profiles" / name for name in _list_dirs(hermes_root / "profiles"))]
        for home in homes:
            plugins_root = home / "plugins"
            for name in _list_dirs(plugins_root, follow_links=True):
                if name in seen:
                    continue
                try:
                    result = materialize_desktop_half(plugins_root / name, app_root, name)
                except OSError as exc:
                    # One unreadable package must not stop every other half from loading.
                    logger.warning("desktop-plugins: skipping unreadable package %s: %s", name, exc)
                    continue
                if result is not None or (plugins_root / name / "desktop" / "plugin.js").exists():
                    seen.add(name)
                if result is not None:
                    touched.append(str(result))

        for name in _list_dirs(app_root):
            folder = app_root / name
            marker = _read_marker(folder)
            if marker and _source_gone(name, Path(marker["source"]) / "plugin.js"):
                shutil.rmtree(folder, ignore_errors=True)
                touched.append(str(folder))
        return touched

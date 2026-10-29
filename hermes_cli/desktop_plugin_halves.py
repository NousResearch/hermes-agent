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

from utils import rmtree_readonly

logger = logging.getLogger(__name__)

PACKAGE_MARKER = ".hermes-package.json"
#: The whole published folder; the loader reads at most a 16 MB ``plugin.js`` from it.
TREE_MAX_BYTES = 64 * 1024 * 1024
#: Folders a root-layout half never publishes (Electron's ``ROOT_LAYOUT_SKIP`` matches).
ROOT_LAYOUT_SKIP = (".git", "__pycache__")
_GIT_ORIGIN = re.compile(r'\[remote "origin"\][^[]*?url\s*=\s*(\S+)')
#: Electron serializes every writer of the root in its main process; the server's threads do here.
#: Public: ``desktop_plugin_install`` takes it too, so a clone-publish never interleaves with a
#: reconcile pass copying the same package's folder.
ROOT_WRITER = threading.Lock()


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


def mtime_ms(st: os.stat_result) -> float:
    """``Stats.mtimeMs`` exactly as Node computes it (seconds * 1e3 + nanoseconds / 1e6). Both
    writers compare this number against the marker the other may have written, so a value one ulp
    off would make each pass see the other's copy as stale and re-copy it."""
    seconds, nanoseconds = divmod(st.st_mtime_ns, 1_000_000_000)
    return seconds * 1e3 + nanoseconds / 1e6


def read_package_marker(folder: Path) -> dict | None:
    try:
        marker = json.loads((folder / PACKAGE_MARKER).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not (isinstance(marker, dict) and marker.get("package") and marker.get("source")):
        return None
    mtime = marker.get("sourceMtimeMs")
    if isinstance(mtime, bool) or not isinstance(mtime, (int, float)):
        # A hand-edited or truncated timestamp is stale metadata: the next pass re-copies the half
        # instead of raising mid-reconcile and taking every other plugin down with it.
        marker["sourceMtimeMs"] = -1
    return marker


def _write_marker(folder: Path, marker: dict) -> None:
    (folder / PACKAGE_MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def _same_file(a: Path, b: Path) -> bool:
    """Same bytes on both paths. A comparison that cannot be made is never a match."""
    try:
        return a.read_bytes() == b.read_bytes()
    except OSError:
        return False


def desktop_source(package_dir: Path) -> Path | None:
    """The folder that is *package_dir*'s desktop plugin, by the rule every installer and the install
    probe share (Electron's ``findDesktopEntry``, ``detect_components``): the package root itself when
    a ``plugin.js`` sits there, else its ``desktop/`` folder. ``None`` when neither holds one."""
    for source in (package_dir, package_dir / "desktop"):
        if (source / "plugin.js").is_file():
            return source
    return None


def root_layout_ignore(source_dir: Path, package_dir: Path):
    """What a root-layout half leaves behind when its package root is published: the clone's ``.git``,
    as ``desktop_plugin_install`` drops it (nothing reads it, and Git's read-only objects make a
    folder Windows cannot delete), and the agent half's ``__pycache__``. A ``desktop/`` half is
    copied whole."""
    return shutil.ignore_patterns(*ROOT_LAYOUT_SKIP) if source_dir == package_dir else None


def check_desktop_tree(source: Path, base: Path, *, linked_root: bool = False,
                       skip: tuple[str, ...] = ()) -> str | None:
    """Why *source* cannot be published as a desktop plugin, else ``None``. It must be a plain tree:
    a link could publish any folder of the host, and the copy is capped at :data:`TREE_MAX_BYTES`.
    *linked_root* accepts *source* itself being a link (a dev checkout linked into ``plugins/``);
    every entry inside it is still held to the rule. Folders named in *skip* are left out of the
    copy (:func:`root_layout_ignore`), so they are not walked."""
    if source.is_symlink() and not linked_root:
        return f"The desktop plugin contains a link ({source.relative_to(base)}); links are not installed."
    total = 0
    for folder, dirs, files in os.walk(source):
        dirs[:] = [n for n in dirs if n not in skip]
        for entry in (Path(folder) / n for n in dirs + files):
            if entry.is_symlink():
                return f"The desktop plugin contains a link ({entry.relative_to(base)}); links are not installed."
        total += sum((Path(folder) / n).stat().st_size for n in files)
        if total > TREE_MAX_BYTES:
            return f"The desktop plugin is larger than {TREE_MAX_BYTES // (1024 * 1024)} MB."
    return None


def publish_desktop_tree(source_dir: Path, target: Path, marker: dict | None, ignore=None) -> None:
    """Copy through a staging sibling and rename the finished tree into place, marker included, so a
    failure leaves the old folder or none: never a partial, marker-less one that a later pass would
    take for a hand install. A standalone plugin is published with no marker (``None``): one would
    claim another package's agent half."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    staged = staging / target.name
    try:
        shutil.copytree(source_dir, staged, symlinks=True, ignore=ignore)
        if marker is None:
            # Only an installer writes the marker: one shipped in the repository would claim
            # another package's agent half (Electron's ``installDesktopPluginFromGit`` drops it too).
            (staged / PACKAGE_MARKER).unlink(missing_ok=True)
        else:
            _write_marker(staged, marker)
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            rmtree_readonly(target)
        os.rename(staged, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _desktop_entry(package_dir: Path, name: str) -> tuple[Path, os.stat_result] | None:
    """``(source_dir, plugin.js stat)`` of *package_dir*'s desktop half under :func:`desktop_source`'s
    rule (root ``plugin.js`` first, then ``desktop/plugin.js``), or ``None``. A candidate we may not
    stat is logged and ends the search: falling through to the other layout would copy a half the
    installer would not have picked."""
    for source_dir in (package_dir, package_dir / "desktop"):
        try:
            entry_stat = (source_dir / "plugin.js").stat()
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as exc:
            logger.warning("desktop-plugins: cannot read %s: %s", name, exc)
            return None
        if stat.S_ISREG(entry_stat.st_mode):
            return source_dir, entry_stat
    return None


def materialize_desktop_half(package_dir: Path, app_root: Path, name: str) -> Path | None:
    """Copy *package_dir*'s desktop half to ``<app_root>/<name>/``: its ``desktop/`` folder, or the
    package root (minus :data:`ROOT_LAYOUT_SKIP`) when ``plugin.js`` sits at the root, as the
    installers publish a root-layout repository. Returns the folder when it was copied or adopted, ``None`` when it is
    already current, absent, refused (a link or an oversized tree), or a user's own plugin."""
    found = _desktop_entry(package_dir, name)
    if found is None:
        return None
    source_dir, entry_stat = found
    entry = source_dir / "plugin.js"

    target = app_root / name
    existing = read_package_marker(target)
    linked = package_dir.is_symlink()
    # A link at ``<app_root>/<name>`` is never a copy this pass may keep or stamp: a marker-less one is
    # the user's own linked plugin, and one carrying a marker is an earlier copy of a linked checkout
    # that kept the link instead of the files (Electron's ``cp`` did), its marker inside the checkout.
    target_is_link = target.is_symlink()
    if target_is_link and existing is None:
        return None
    marker = {
        "package": name,
        "source": str(source_dir),
        "sourceMtimeMs": mtime_ms(entry_stat),
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
            and not target_is_link
            and existing["source"] == str(source_dir)
            and bool(existing.get("linked")) == linked
            and (
                _same_file(target / "plugin.js", entry)
                if linked
                else existing.get("sourceMtimeMs", 0) >= marker["sourceMtimeMs"]
            )
        ):
            return None

    ignore = root_layout_ignore(source_dir, package_dir)
    if ignore is not None:
        # The whole package root is published, agent payload included: hold it to the installer's
        # plain-tree rule rather than copy a link to anywhere on the host into the served folder.
        refused = check_desktop_tree(source_dir, package_dir, linked_root=linked, skip=ROOT_LAYOUT_SKIP)
        if refused:
            logger.warning("desktop-plugins: not copying the desktop half of %s: %s", name, refused)
            return None
    publish_desktop_tree(source_dir, target, marker, ignore=ignore)
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
    with ROOT_WRITER:
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
                if result is not None or desktop_source(plugins_root / name) is not None:
                    seen.add(name)
                if result is not None:
                    touched.append(str(result))

        # Links count too: a marker read through one is an earlier copy of a linked checkout that kept
        # the link. Only the link goes, never the folder it points at.
        for name in _list_dirs(app_root, follow_links=True):
            folder = app_root / name
            marker = read_package_marker(folder)
            if marker and _source_gone(name, Path(marker["source"]) / "plugin.js"):
                if folder.is_symlink():
                    folder.unlink(missing_ok=True)
                else:
                    shutil.rmtree(folder, ignore_errors=True)
                touched.append(str(folder))
        return touched

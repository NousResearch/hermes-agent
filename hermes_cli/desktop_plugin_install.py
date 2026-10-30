"""Install a desktop plugin from Git onto the host that serves the Webapp.

Native Desktop clones one in Electron (``apps/desktop/electron/desktop-plugin-install.ts``,
``installDesktopPluginFromGit``). A browser-hosted tab has no disk of its own, so until now the install
dialog could only *read* a repository through the gateway (``plugins_cmd_inspect``) and then had
nothing to install with: a desktop-only package answered "nothing to install here", and the desktop
half of a mixed one was reported skipped. This is the missing half of the trip — the same inspection,
held to the commit it read, the same folder name and package-marker rules as Electron, published
atomically into the app-level root that ``/api/desktop-plugins`` serves.

The install is a clone, a staged copy and a rename: no code is imported or run here. The plugin loads
the way any other installed plugin does, on the next scan of ``desktop-plugins/``.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from hermes_cli import desktop_plugin_halves as halves
from hermes_constants import get_default_hermes_root
from utils import rmtree_readonly

DESKTOP_PLUGINS_DIR = "desktop-plugins"
_COMMIT = re.compile(r"[0-9a-f]{40}")


def desktop_plugins_root() -> Path:
    """The app-level root ``desktop-plugins/``, created on demand and resolved, exactly as Electron's
    ``desktopPluginsRoot`` and the Webapp's read routes resolve it: one folder whichever profile the
    caller names."""
    root = get_default_hermes_root() / DESKTOP_PLUGINS_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _pc():
    """The facade, read at call time: tests patch ``plugins_cmd.<name>`` and sibling calls must see it."""
    from hermes_cli import plugins_cmd
    return plugins_cmd


def _failure(error: str) -> dict[str, Any]:
    return {"ok": False, "error": error}


def _checked_name(name: Optional[str]) -> str:
    """*name* when it can be one folder under the root, else raise: the name comes from a repository
    (a subdirectory's last segment, a manifest's ``name``) and must never reach a path unresolved. A
    leading dot is refused too — the root's staging folders use that prefix."""
    candidate = (name or "").strip()
    if (not candidate or candidate in (".", "..") or candidate.startswith(".")
            or "/" in candidate or "\\" in candidate or "\0" in candidate
            or any(ord(char) < 32 for char in candidate) or len(candidate) > 128):
        raise _pc().PluginOperationError(
            f"'{name}' is not a desktop plugin folder name: it must be a plain folder name, not a path."
        )
    return candidate


def _resolve_source(identifier: str, catalog_name: Optional[str]) -> tuple[str, Optional[str], Optional[str]]:
    """``(git_url, subdir, catalog_sha)`` through the installer's own resolution and kill list, with the
    catalog's reviewed pin when *catalog_name* is given (its repository replaces *identifier*).

    Every scheme the CLI installs from is accepted (``git@``/``ssh://`` for private plugins, ``file://``,
    ``http://``): the Webapp opens only for the signed-in owner of this host, whose keys and disk these are."""
    pc = _pc()
    revision = None
    if catalog_name:
        from hermes_cli import plugins_cmd_catalog as catalog
        entry = catalog.get_live_catalog_entry(catalog_name)
        if entry is None:
            raise pc.PluginOperationError(f"'{catalog_name}' is not in the Hermes plugin catalog.")
        identifier, revision = entry.install_identifier, entry.sha or None
    try:
        git_url, subdir = pc._resolve_git_url(identifier)
    except ValueError as exc:
        raise pc.PluginOperationError(str(exc)) from exc
    from hermes_cli.plugins_cmd_catalog import raise_if_removed
    raise_if_removed(identifier, git_url, *((catalog_name,) if catalog_name else ()))
    return git_url, subdir, revision


def install_desktop_plugin(identifier: str, *, ref: Optional[str], catalog_name: Optional[str] = None,
                           force: bool = False) -> dict[str, Any]:
    """Clone a repository's desktop half into ``desktop-plugins/<name>`` on this host.

    *ref* is the commit the install dialog showed (``inspect``'s ``sha``): the clone is checked out at
    exactly that commit, so a repository that moved after review is refused rather than installed. The
    halves are read from that same checkout. Returns ``{ok, plugin_name, path, sha, paired_with_agent,
    warnings}``, or ``ok: False`` with ``error``. A package carrying both halves lands under its *agent*
    name with the package marker Electron writes, so the Plugins page pairs the two rows; a desktop-only
    repository keeps its Git-derived folder name and carries no marker.
    """
    from hermes_cli.plugins_cmd_inspect import _desktop_folder_name, detect_components

    pc = _pc()
    sha = (ref or "").strip().lower()
    try:
        if not _COMMIT.fullmatch(sha):
            raise pc.PluginOperationError("Installing a desktop plugin needs the commit the install dialog reviewed.")
        git_url, subdir, catalog_sha = _resolve_source(identifier, catalog_name)
        root = desktop_plugins_root()
        with tempfile.TemporaryDirectory(prefix="hermes-plugin-install-") as tmp:
            clone = Path(tmp) / "plugin"
            # A catalog entry is fetched at its own pin, as the agent installer fetches it: an
            # annotated-tag pin names the tag object, and only its peeled commit is what the dialog read.
            head = pc._clone_plugin_repo(clone, git_url, catalog_sha or sha, subdir)
            if head != sha:
                raise pc.PluginSourceMoved(
                    "The catalog's reviewed commit changed since the dialog read it; inspect it again.")
            # Nothing reads a plugin's Git metadata, and Git's read-only object files would make the
            # published folder one that Windows cannot delete.
            rmtree_readonly(clone / ".git")
            plugin_root = pc._resolve_subdir_within(clone, subdir) if subdir else clone
            fallback = subdir.rstrip("/").rsplit("/", 1)[-1] if subdir else pc._repo_name_from_url(git_url)
            found = detect_components(plugin_root, fallback)
            if not found["desktop"]:
                return _failure("This package ships no desktop plugin to install here.")
            package_name = _checked_name(found["agent_name"]) if found["agent"] else None
            folder = package_name or _checked_name(_desktop_folder_name(git_url, subdir))
            source_dir = _desktop_tree(plugin_root)
            target = root / folder
            # A package's half keeps the marker Electron writes (the Plugins page pairs it to the agent
            # row); a standalone plugin carries none.
            marker = {"package": package_name, "repo": git_url, "source": str(target),
                      "sourceMtimeMs": halves.mtime_ms((source_dir / "plugin.js").stat())} if package_name else None
            # The exists check and the publish must not interleave with a reconcile pass.
            with halves.ROOT_WRITER:
                if (target.is_symlink() or target.exists()) and not force:
                    return _failure(f"Desktop plugin '{folder}' already exists. Enable force reinstall to replace it.")
                halves.publish_desktop_tree(source_dir, target, marker)
    except pc.PluginOperationError as exc:
        return {**_failure(str(exc)), **({"source_moved": True} if isinstance(exc, pc.PluginSourceMoved) else {})}
    except OSError as exc:
        return _failure(f"Could not install the desktop plugin: {exc}")

    return {"ok": True, "plugin_name": folder, "path": str(target), "sha": sha,
            "paired_with_agent": bool(package_name), "warnings": []}


def _desktop_tree(plugin_root: Path) -> Path:
    """The folder that is the desktop plugin (:func:`halves.desktop_source`, the rule the host's
    reconcile applies to an installed package too). It must be a plain tree inside the clone."""
    source = halves.desktop_source(plugin_root) or plugin_root / "desktop"
    refused = halves.check_desktop_tree(source, plugin_root)
    if refused:
        raise _pc().PluginOperationError(refused)
    return source


def remove_desktop_plugin(name: str, *, force: bool = False) -> dict[str, Any]:
    """Delete one installed desktop plugin folder from this host.

    A folder carrying the package marker is the desktop half of an installed agent plugin: removing it
    alone would leave the Plugins page pairing a row with nothing behind it, and the next reconcile
    would copy it straight back, so it is refused unless *force* says the caller means it.
    """
    try:
        folder = _checked_name(name)
    except _pc().PluginOperationError as exc:
        return _failure(str(exc))
    target = desktop_plugins_root() / folder
    if target.is_symlink():
        return _failure(
            f"'{folder}' is linked from somewhere else on this host; remove that link where it lives."
        )
    if not target.is_dir():
        return _failure(f"No desktop plugin named '{folder}' is installed.")

    with halves.ROOT_WRITER:
        marker = halves.read_package_marker(target)
        if marker and not force:
            return _failure(
                f"'{folder}' is the desktop half of an agent plugin — uninstall that plugin instead."
            )
        try:
            rmtree_readonly(target)
        except OSError as exc:
            return _failure(f"Could not remove the desktop plugin: {exc}")

    return {"ok": True, "plugin_name": folder, "path": str(target), "unpaired_from": (marker or {}).get("package")}

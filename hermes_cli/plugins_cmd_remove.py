"""``hermes plugins remove``: tree + install-metadata removal kept consistent, config bookkeeping, and the
dashboard/TUI remove path.

Sibling of :mod:`hermes_cli.plugins_cmd` (the facade re-exports the names other modules use and is
imported late here, never at module level).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


def _pc():
    """The facade, read at call time: tests patch ``plugins_cmd.<name>`` and sibling calls must see it."""
    from hermes_cli import plugins_cmd
    return plugins_cmd


_PENDING_DELETE_DIR = "plugin-pending-delete"


def _pending_delete_root() -> Path:
    """``<HERMES_HOME>/cache/plugin-pending-delete``: trees a running process still held at removal."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / _PENDING_DELETE_DIR


def sweep_pending_plugin_deletes() -> None:
    """Delete what an earlier remove parked because another process held it; cheap when empty.
    A ``<name>.path`` note names a stripped tree that could not even be moved out of ``plugins/``."""
    root = _pending_delete_root()
    if not root.is_dir():
        return
    for entry in root.iterdir():
        if entry.suffix == ".path" and entry.is_file():
            held = Path(entry.read_text(encoding="utf-8-sig").strip())
            _pc().rmtree_readonly(held, ignore_errors=True)
            if not held.exists():
                entry.unlink(missing_ok=True)
        else:
            _pc().rmtree_readonly(entry, ignore_errors=True)


def _discard_tree(tree: Path) -> None:
    """Delete *tree*. On Windows a running gateway / plugin host that loaded the plugin keeps its
    native modules mapped and its open files locked (WinError 5/32), and deleting stops at the first
    one. The rest is then moved out of ``plugins/`` into the pending-delete area, so nothing loadable
    stays behind, and the sweep at the next start finishes the job."""
    try:
        _pc().rmtree_readonly(tree)
        return
    except PermissionError:
        if os.name != "nt" or not tree.exists():
            raise
    root = _pending_delete_root()
    root.mkdir(parents=True, exist_ok=True)
    parked = Path(tempfile.mkdtemp(prefix=f"{tree.name}-", dir=root)) / "plugin"
    try:
        os.replace(tree, parked)  # same volume (both under HERMES_HOME): a rename, allowed on mapped DLLs
    except OSError:
        # A held directory itself (a process's cwd, an open handle on it) cannot even be renamed. Strip
        # its manifests so the remnant no longer loads, and leave the files to the next start.
        for name in ("plugin.yaml", "plugin.yml", "plugin.json", "__init__.py"):
            (tree / name).unlink(missing_ok=True)
        (root / f"{tree.parent.name}-{tree.name}.path").write_text(str(tree), encoding="utf-8")


def _remove_plugin_core(target: Path) -> None:
    """Remove one plugin and its metadata without splitting their state."""
    if target.name not in _pc()._read_install_metadata():
        _discard_tree(target)
        return
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.remove-", dir=target.parent))
    backup = staging / "plugin"
    try:
        os.replace(target, backup)
    except PermissionError:
        _pc().rmtree_readonly(staging, ignore_errors=True)
        if os.name != "nt":
            raise
        # Windows refuses to rename a directory another process holds open; record removal first
        # (the plugin is uninstalled from this point), then discard the tree in place.
        _pc()._update_install_record(target.name, lambda _current: None)
        _discard_tree(target)
        return
    try:
        _pc()._update_install_record(target.name, lambda _current: None)
    except Exception:
        try:
            os.replace(backup, target)
        except OSError as restore_exc:
            raise _pc().PluginOperationError(
                f"Plugin metadata update failed and '{target.name}' could not be "
                f"restored automatically; recovery copy remains at {backup}."
            ) from restore_exc
        _pc().rmtree_readonly(staging, ignore_errors=True)
        raise
    _discard_tree(staging)


def cmd_remove(name: str) -> None:
    """Remove an installed plugin by name."""
    console = _pc()._console()
    plugins_dir = _pc()._plugins_dir()
    target = _pc()._require_installed_plugin(name, plugins_dir, console)
    try:
        result = _remove_user_plugin(plugins_dir, name, target)
    except (OSError, _pc().PluginOperationError) as exc:
        _pc()._fail(console, f"[red]Error:[/red] Could not remove plugin '{name}': {exc}")
    console.print()
    console.print(f"[red]✗[/red] Plugin [bold]{name}[/bold] removed from {plugins_dir}")
    if result.get("cleared_memory_provider"):
        console.print("[yellow]memory.provider pointed at this plugin and was reset; "
                      "run `hermes memory setup` to pick another.[/yellow]")
    console.print()


def _remove_user_plugin(plugins_dir: Path, name: str, target: Path) -> dict[str, Any]:
    """Shared ``remove`` tail for the CLI, the dashboard and the ``plugins.manage`` RPC.

    *target* is the resolved directory; when ``plugins_dir/name`` itself is a symlink only the link
    goes — the tree it points at may be another installed plugin (a dev alias to a sibling checkout),
    and following it deleted that plugin plus its install metadata while the alias stayed dangling.
    Config bookkeeping (aliases, toolset) is gathered before the tree disappears.
    """
    link = plugins_dir / name.strip("/")
    if link.is_symlink():
        link.unlink()
        return {"ok": True, "name": name, **_pc()._forget_plugin_config({link.name})}
    entry = next((e for e in _pc()._discover_all_plugins() if Path(str(e[4])) == target), None)
    key = entry[5] if entry else target.name
    aliases = _pc()._plugin_aliases(key) | {target.name}
    if _pc()._read_manifest(target).get("provides_tools"):
        _pc()._toggle_plugin_toolset(key, enable=False)
    _remove_plugin_core(target)
    return {"ok": True, "name": name, **_pc()._forget_plugin_config(aliases)}


def dashboard_remove_user_plugin(name: str) -> dict[str, Any]:
    """Delete a plugin tree under ``~/.hermes/plugins/`` only."""
    plugins_dir = _pc()._plugins_dir()
    if any(n == name and src == "bundled" for n, _ver, _d, src, _path, _key in _pc()._discover_all_plugins()):
        return {"ok": False, "error": "Bundled plugins cannot be removed from the dashboard."}
    target = _pc()._user_installed_plugin_dir(name)
    if target is None:
        return {"ok": False, "error": f"Plugin '{name}' was not found under {plugins_dir}."}
    try:
        return _remove_user_plugin(plugins_dir, name, target)
    except (OSError, _pc().PluginOperationError) as exc:
        return {"ok": False, "error": f"Could not remove plugin '{name}': {exc}"}

"""Plugin identity: which plugin a ``plugins.enabled`` / ``plugins.disabled`` entry names.

ONE rule for every reader of those lists: the loader's gate (``plugins_discovery.gate_manifest``), PM's
dependency selection (``pm.plugins_state``), the default-plugin installer (``default_plugins._settled``),
the left-core migration's notices and installers, and ``hermes plugins`` status all ask
:func:`plugin_listed` about a plugin's :func:`plugin_identity`.

A plugin's identity is its registry key (:func:`tree_key`: path-derived ``<category>/<dir>``, or the manifest
name for a flat tree), its manifest ``name:`` and, for a plugin that left core, the bundled key and manifest
name core shipped it under (``left_core_migration.legacy_plugin_keys``). An entry names the plugin exactly when
it is one of those strings: no case folding, no bare leaf of a nested key (``team/qqbot`` is not ``qqbot``),
and a blank entry names nothing. A reader that starts from a config entry resolves the installed tree first
(:func:`installed_trees`, discovery's walk) and asks about THAT tree's identity, so a fork or a nested copy is
judged by its own names, as the loader judges it.

Import-light on purpose: PM's isolated runtime reads selections through here without the agent's packages.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Iterator, Optional

# Per-harness manifest directories plugin repos ship for OTHER agent harnesses (e.g. obra/superpowers keeps one
# plugin.json per harness). Their plugin.json is not an Agent Plugins v1 manifest and can never validate, so
# parsing it on every discovery pass only spams warnings (#101962).
_FOREIGN_HARNESS_MANIFEST_DIRS = frozenset({
    ".claude-plugin", ".codex-plugin", ".cursor-plugin", ".devin-plugin", ".kimi-plugin",
})
# An installer's staging copy beside the plugins it replaces (``plugins_cmd_install``): read as a category, it
# would answer to the plugin's own name mid-install.
_STAGING_PREFIXES = (".install-", ".update-")


def is_skipped_plugin_dir(name: str) -> bool:
    """True for directory names discovery never treats as plugins. Cache/dunder dirs (__pycache__,
    __MACOSX__, …) can raise PermissionError when walked and take down every later tool call
    (#86996); foreign-harness dirs hold another tool's manifest convention; staging dirs are transient."""
    return ((name.startswith("__") and name.endswith("__")) or name in _FOREIGN_HARNESS_MANIFEST_DIRS
            or name.startswith(_STAGING_PREFIXES))


def tree_key(prefix: str, dir_name: str, manifest_name: str) -> str:
    """Registry key of a plugin tree: ``<prefix>/<dir>`` under a category dir, else its manifest name."""
    return f"{prefix}/{dir_name}" if prefix else manifest_name


def plugin_identity(key: str, name: str = "") -> frozenset[str]:
    """Every entry that names the plugin with registry *key* and manifest *name* (default: the key)."""
    from hermes_cli.left_core_migration import legacy_plugin_keys
    return frozenset(n for n in (key, name or key, *legacy_plugin_keys(key)) if isinstance(n, str) and n.strip())


def manifest_identity(manifest) -> frozenset[str]:
    """:func:`plugin_identity` of a discovered manifest."""
    return plugin_identity(manifest.key or manifest.name, manifest.name)


def plugin_listed(identity: frozenset[str], entries: Any) -> bool:
    """A ``plugins.enabled`` / ``plugins.disabled`` value (a list; anything else lists nothing) names the
    plugin with *identity*."""
    return isinstance(entries, (list, tuple, set, frozenset)) and any(
        isinstance(entry, str) and entry in identity for entry in entries)


def _manifest_name(child: Path) -> Optional[str]:
    """The ``name:`` discovery gives the tree at *child* (``plugin.yaml``, else portable ``plugin.json``), ``""``
    when it has neither manifest, ``None`` when its manifest is not a plugin discovery loads. Raises
    ``ValueError`` on an unreadable tree."""
    from pm.plugin_declarations import native_manifest_file, read_native_manifest
    manifest_file = native_manifest_file(child)
    if manifest_file is not None:
        try:
            return str(read_native_manifest(manifest_file).get("name", child.name))
        except ValueError as exc:
            if isinstance(exc.__cause__, OSError):
                raise
            return None  # malformed: discovery warns and skips it
    portable = child / "plugin.json"
    try:
        if not (portable.exists() or portable.is_symlink()):
            return ""
        data = json.loads(portable.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ValueError(f"could not read plugin manifest: {portable}") from exc
    except ValueError:
        return None
    name = data.get("name") if isinstance(data, dict) else None
    return name if isinstance(name, str) and name else None


def _walk(path: Path, prefix: str = "", depth: int = 0, *, skip_names: frozenset = frozenset(),
          unreadable: Optional[list] = None) -> Iterator[tuple[Path, str, str]]:
    """``(tree, key, manifest name)`` in discovery's order and nesting (``plugins_discovery.scan_directory``:
    a manifest-less dir at the top recurses once as a category). A folder discovery cannot read is skipped as
    discovery skips it, and noted in *unreadable*: only a reader that selects it may refuse
    (``pm.workspace.selected_plugin_dir``). Raises ``ValueError`` when *path* itself cannot be listed."""
    try:
        children = sorted(path.iterdir())
    except FileNotFoundError:
        return
    except OSError as exc:
        if depth:
            if unreadable is not None:
                unreadable.append(path)
            return
        raise ValueError(f"could not inspect plugin directory: {path}") from exc
    for child in children:
        if is_skipped_plugin_dir(child.name) or (depth == 0 and child.name in skip_names):
            continue
        try:
            if not stat.S_ISDIR(child.stat().st_mode):
                continue
            name = _manifest_name(child)
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            if unreadable is not None:
                unreadable.append(child)
            continue
        if name:
            yield child, tree_key(prefix, child.name, name), name
        elif name == "" and depth == 0:
            yield from _walk(child, tree_key(prefix, child.name, child.name), depth + 1, unreadable=unreadable)


# Bundled top-level dirs with their own discovery (memory providers, context engines, ...): never plugins of the
# general sweep (``plugins_discovery.collect_directory_manifests``).
BUNDLED_OWN_DISCOVERY = frozenset({
    "memory", "context_engine", "model-providers", "cron_providers", "computer_use", "config_backends",
})


def bundled_plugins_dir() -> Path:
    """Bundled ``plugins/`` dir: ``HERMES_BUNDLED_PLUGINS`` (Nix wrapper / packaged installs, read-only store
    paths) first, else the in-repo path. Stdlib only, so PM's isolated runtime resolves it too."""
    from hermes_cli._startup_fast import project_root_str
    return Path(os.getenv("HERMES_BUNDLED_PLUGINS") or Path(project_root_str()) / "plugins")


def _bundled_dirs() -> dict[str, str]:
    """Registry key -> directory name of every bundled plugin the general sweep reads."""
    try:
        found = _walk(bundled_plugins_dir(), skip_names=BUNDLED_OWN_DISCOVERY)
        return {key: tree.name for tree, key, _name in found}
    except ValueError:
        return {}  # discovery reads no bundled plugin from an unlistable dir either


def _winners(plugins_dir: Path, unreadable: Optional[list] = None) -> dict[str, tuple[Path, str]]:
    """Registry key -> ``(tree, manifest name)`` of every user tree discovery loads from *plugins_dir*, by
    ``resolve_manifest_winners``' rule: on a key collision the later tree wins, and a flat tree claiming a bundled
    plugin's key from a differently named directory is an impostor, skipped."""
    winners: dict[str, tuple[Path, str]] = {}
    bundled: Optional[dict[str, str]] = None
    for tree, key, name in _walk(Path(plugins_dir), unreadable=unreadable):
        if "/" not in key:  # a flat tree's key is its manifest name: it may claim a bundled one
            bundled = _bundled_dirs() if bundled is None else bundled
            if key in bundled and bundled[key] != tree.name:
                continue
        winners[key] = (tree, name)
    return winners


def installed_trees(plugins_dir: Path, unreadable: Optional[list] = None) -> dict[Path, frozenset[str]]:
    """Every plugin tree discovery loads from the user plugins dir *plugins_dir* (:func:`_winners`), with its
    identity. A folder it cannot read is left out, as discovery leaves it out, and noted in *unreadable*."""
    return {tree: plugin_identity(key, name) for key, (tree, name) in _winners(plugins_dir, unreadable).items()}


def registry_key(plugins_dir: Path, tree: Path) -> str:
    """The registry key discovery gives *tree* under *plugins_dir* (unique per loaded tree), ``""`` for a tree
    it does not load."""
    return next((key for key, (found, _name) in _winners(plugins_dir).items() if found == Path(tree)), "")


def tree_identity(plugins_dir: Path, tree: Path, *, default: str = "") -> frozenset[str]:
    """Identity of the installed *tree* under *plugins_dir*. A tree discovery does not load (no manifest yet,
    an unreadable dir) has no loader verdict to agree with: it answers to its dir name (the name its manifest
    would default to) and *default*, the catalog name it was installed under."""
    try:
        found = installed_trees(plugins_dir).get(Path(tree))
    except ValueError:
        found = None
    if found is not None:
        return found
    return plugin_identity(Path(tree).name) | (plugin_identity(default) if default else frozenset())

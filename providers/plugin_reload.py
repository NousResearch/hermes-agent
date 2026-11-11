"""Reload a ``$HERMES_HOME`` provider plugin whose files changed while the process runs (#153300).

A provider plugin is imported once per process per home. Without this, an update written to disk
never reached a running gateway, and a plugin that imports part of its package on first use (inside
``create_client``) read that one file from the new tree against siblings still loaded from the old
one. :func:`unload_changed` drops the whole package so the next import is the new version end to end.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Directories inside a plugin tree that never hold code the provider imports.
_SKIP_DIRS = frozenset({"__pycache__", ".git", "node_modules", "tests", "evals", ".venv"})
_CATALOG_SIDECAR = ".hermes-catalog.json"


def tree_identity(plugin_dir: Path) -> tuple:
    """What changes when the plugin is updated: the catalog pin when installed from the catalog (a re-pin
    rewrites it), else the newest mtime of the plugin's Python files and manifest."""
    try:
        return ("catalog", (plugin_dir / _CATALOG_SIDECAR).read_text(encoding="utf-8"))
    except OSError:
        pass
    newest = 0
    for root, dirs, files in os.walk(plugin_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            if name.endswith(".py") or name in ("plugin.yaml", "plugin.yml"):
                try:
                    newest = max(newest, os.stat(os.path.join(root, name)).st_mtime_ns)
                except OSError:
                    continue
    return ("mtime", newest)


def unload_changed(layer: Any, plugin_dir: Path, module_name: str) -> bool:
    """When *module_name* was imported into *layer* from an older tree of *plugin_dir*, drop it: the
    profiles it registered, its package and every submodule. True when it unloaded."""
    seen = layer.trees.get(module_name)
    if seen is None or seen == tree_identity(plugin_dir):
        return False
    for name in layer.owners.pop(module_name, ()):
        layer.registry.pop(name, None)
        for alias in [a for a, target in layer.aliases.items() if target == name]:
            del layer.aliases[alias]
    prefix = module_name + "."
    for name in [m for m in sys.modules if m == module_name or m.startswith(prefix)]:
        del sys.modules[name]
    layer.hosted.discard(module_name)
    del layer.trees[module_name]
    return True

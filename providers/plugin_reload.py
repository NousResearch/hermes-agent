"""Load ``$HERMES_HOME`` provider plugins into a home layer, and swap in an updated one whole (#153300).

A provider plugin used to be imported once per process. An update then never reached a running gateway, and a
plugin importing part of its package on first use (inside ``create_client``) read that one file from the new
tree against siblings loaded from the old one.

Each tree is imported under its own module name (the plugin's base name plus a fingerprint of its files) into a
staging layer, and only a complete import is published: the plugin's previous registrations stay live until
then, and stay live if the new tree fails to import. The previous package is left in ``sys.modules``, so
anything built from it (a cached agent's client) keeps resolving its own lazy imports there: every object runs
one version end to end, old or new, never a mix.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Directories inside a plugin tree that never hold code the provider imports.
_SKIP_DIRS = frozenset({"__pycache__", ".git", "node_modules", "tests", "evals", ".venv"})


def tree_fingerprint(plugin_dir: Path) -> str:
    """Digest of the plugin's Python files and manifest (path, size, mtime): it moves whenever an update
    rewrites them, however the plugin was installed."""
    entries = []
    for root, dirs, files in os.walk(plugin_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for name in sorted(files):
            if name.endswith(".py") or name in ("plugin.yaml", "plugin.yml"):
                path = os.path.join(root, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                entries.append(f"{os.path.relpath(path, plugin_dir)}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha1("\n".join(entries).encode("utf-8")).hexdigest()[:12]


def import_into_layer(plugin_dir: Path, source: str, base_name: str) -> None:
    """Import *plugin_dir* into the layer being scanned unless that exact tree is already live there; an
    updated tree replaces the live one only once it imported completely."""
    import providers as P

    layer = P._REGISTRATION_TARGET.get()
    fingerprint = tree_fingerprint(plugin_dir)
    live = layer.trees.get(base_name)
    if live is not None and live[0] == fingerprint:
        return
    module_name = f"{base_name}_{fingerprint}"
    staging = P._HomeLayer()
    token = P._REGISTRATION_TARGET.set(staging)
    try:
        P._import_user_module(plugin_dir, source, module_name)
    finally:
        P._REGISTRATION_TARGET.reset(token)
    if module_name in sys.modules or module_name in staging.hosted:
        _publish(layer, base_name, module_name, fingerprint, staging, replacing=live is not None)
        layer.errors.pop(plugin_dir.name, None)
        if live is not None:
            logger.info("Provider plugin %s changed on disk; new sessions use the updated version",
                        plugin_dir.name)
        return
    # A failed import leaves the previous version serving; drop what it half-loaded so a retry is clean.
    prefix = module_name + "."
    for name in [m for m in sys.modules if m.startswith(prefix)]:
        del sys.modules[name]
    if error := staging.errors.get(plugin_dir.name):
        layer.errors[plugin_dir.name] = error


def _publish(layer, base_name: str, module_name: str, fingerprint: str, staging, *, replacing: bool) -> None:
    """Swap *base_name*'s registrations in *layer* for *staging*'s in one step. A first import keeps the
    scan's last-writer-wins order; a reload never takes a name another plugin currently owns."""
    import providers as P

    with P._HOME_LAYERS_LOCK:
        dropped = {n for n, owner in layer.owner.items() if owner == base_name and n not in staging.registry}
        for name in dropped:
            layer.registry.pop(name, None)
            del layer.owner[name]
        for alias in [a for a, target in layer.aliases.items() if target in dropped]:
            del layer.aliases[alias]
        for name, profile in staging.registry.items():
            if replacing and layer.owner.get(name, base_name) != base_name:
                continue
            layer.registry[name] = profile
            layer.owner[name] = base_name
        for alias, target in staging.aliases.items():
            if layer.owner.get(target) == base_name:
                layer.aliases[alias] = target
        live = layer.trees.get(base_name)
        if live is not None:
            layer.hosted.discard(live[1])
        layer.hosted.update(staging.hosted)
        layer.trees[base_name] = (fingerprint, module_name)
        P._REGISTRY_GENERATION += 1

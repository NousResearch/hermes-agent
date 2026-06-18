"""Dynamic import of sibling Hermes plugins (in-tree)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_PLUGINS_ROOT = Path(__file__).resolve().parents[1]


def load_plugin_modules(
    plugin_dir: str,
    stems: tuple[str, ...],
    *,
    pkg_alias: str | None = None,
) -> str:
    """Import plugin modules under a synthetic package; returns package name."""
    plugin_path = _PLUGINS_ROOT / plugin_dir
    pkg_name = pkg_alias or f"hermes_{plugin_dir.replace('-', '_')}"
    if pkg_name in sys.modules and all(f"{pkg_name}.{s}" in sys.modules for s in stems):
        return pkg_name

    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(plugin_path)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg

    for stem in stems:
        mod_name = f"{pkg_name}.{stem}"
        if mod_name in sys.modules:
            continue
        file_path = plugin_path / f"{stem}.py"
        if not file_path.is_file():
            raise FileNotFoundError(f"{plugin_dir}/{stem}.py")
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {mod_name}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = pkg_name
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
    return pkg_name


def get_module(plugin_dir: str, stem: str, *, stems_chain: tuple[str, ...] | None = None) -> object:
    chain = stems_chain or (stem,)
    pkg = load_plugin_modules(plugin_dir, chain)
    return sys.modules[f"{pkg}.{stem}"]

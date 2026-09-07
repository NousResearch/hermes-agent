"""Shared directory-plugin loader for ``plugins/<kind>/<name>/`` discovery packages
(cron_providers, context_engine, memory): import ``__init__.py`` by path with siblings
pre-registered so relative imports work, then extract the provider via ``register(ctx)``
or an ABC-subclass fallback."""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

_PLUGINS_ROOT = Path(__file__).parent


def register_synthetic_package(name: str, search_locations: List[str]) -> None:
    """Register an empty package shell so ``<name>.<child>`` relative imports resolve."""
    if name in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = search_locations
    sys.modules[name] = importlib.util.module_from_spec(spec)


def user_plugins_dir() -> Optional[Path]:
    """Return ``$HERMES_HOME/plugins/`` or None if unavailable."""
    try:
        from hermes_constants import get_hermes_home
        d = get_hermes_home() / "plugins"
        return d if d.is_dir() else None
    except Exception:
        return None


def iter_plugin_dirs(root: Path) -> List[Path]:
    """Sorted child dirs of *root* that have an ``__init__.py`` (skips ``_``/``.`` names)."""
    if not root.is_dir():
        return []
    return [child for child in sorted(root.iterdir())
            if child.is_dir() and not child.name.startswith(("_", ".")) and (child / "__init__.py").exists()]


def read_plugin_description(plugin_dir: Path) -> str:
    """Return ``description`` from ``plugin.yaml`` (empty string if absent/unreadable)."""
    try:
        import yaml
        with open(plugin_dir / "plugin.yaml", encoding="utf-8-sig") as f:
            meta = yaml.safe_load(f) or {}
        return meta.get("description", "")
    except Exception:
        return ""


def _new_module(name: str, file: Path, search_locations: Optional[List[str]] = None) -> Optional[Any]:
    """spec -> module -> sys.modules[name] (NOT executed); None if no spec."""
    spec = importlib.util.spec_from_file_location(
        name, str(file), submodule_search_locations=search_locations)
    if not spec:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    return mod


def _exec(mod: Any, logger: Optional[logging.Logger] = None) -> bool:
    """Exec a ``_new_module`` module (None -> False); False + debug-log if it raised. The sys.modules
    entry stays on failure; callers needing a clean retry pop it themselves."""
    if mod is None:
        return False
    try:
        mod.__spec__.loader.exec_module(mod)
        return True
    except Exception as e:
        if logger:
            logger.debug("Failed to exec_module %s: %s", mod.__name__, e)
        return False


def load_plugin_module(module_name: str, plugin_dir: Path, *, parents: Tuple[str, ...],
                       logger: logging.Logger, synthetic_namespace: Optional[str] = None) -> Optional[Any]:
    """Import ``plugin_dir/__init__.py`` as *module_name* (reusing sys.modules when loaded).
    Order matters: parents first (relative imports need them), then siblings as ``module_name.<stem>``
    (so ``from ._x import Y`` resolves), then the module. Finally child is bound onto parent and
    siblings onto module — the shape normal imports produce, which monkeypatch relies on."""
    init_file = plugin_dir / "__init__.py"
    if not init_file.exists():
        return None
    # A synthetic package shell has no __file__; only reuse modules loaded from disk.
    cached = sys.modules.get(module_name)
    if cached is not None and getattr(cached, "__file__", None):
        return cached
    for parent in parents:
        parent_path = _PLUGINS_ROOT.joinpath(*parent.split(".")[1:])
        if parent not in sys.modules and (parent_path / "__init__.py").exists():
            _exec(_new_module(parent, parent_path / "__init__.py", [str(parent_path)]))
    if synthetic_namespace:
        register_synthetic_package(synthetic_namespace, [])
    # Reserve the name before siblings exec so their relative imports resolve.
    mod = _new_module(module_name, init_file, [str(plugin_dir)])
    if mod is None:
        return None
    loaded_submodules = []
    for sub_file in plugin_dir.glob("*.py"):
        full_sub_name = f"{module_name}.{sub_file.stem}"
        if sub_file.name == "__init__.py" or full_sub_name in sys.modules:
            continue
        sub_mod = _new_module(full_sub_name, sub_file)
        if _exec(sub_mod, logger):
            loaded_submodules.append((sub_file.stem, sub_mod))
    if not _exec(mod, logger):
        sys.modules.pop(module_name, None)
        return None
    parent_name, child_name = module_name.rsplit(".", 1)
    parent_mod = sys.modules.get(parent_name)
    if parent_mod is not None:
        setattr(parent_mod, child_name, mod)
    for sub_name, sub_mod in loaded_submodules:
        setattr(mod, sub_name, sub_mod)
    return mod


class NoopPluginContext:
    """Base for fake ``register(ctx)`` contexts: no-op registrations except the one a subclass overrides."""

    def _noop(self, *args, **kwargs):
        pass

    register_tool = register_hook = register_cli_command = register_memory_provider = _noop


def instance_from_module(mod: Any, *, collector: Any, collected_attr: str, base_cls: type, name: str,
                         logger: logging.Logger) -> Optional[Any]:
    """Extract the provider instance: ``register(ctx)`` first, then any ``base_cls`` subclass."""
    if hasattr(mod, "register"):
        try:
            mod.register(collector)
            instance = getattr(collector, collected_attr)
            if instance:
                return instance
        except Exception as e:
            logger.debug("register() failed for %s: %s", name, e)
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name, None)
        if isinstance(attr, type) and issubclass(attr, base_cls) and attr is not base_cls:
            with contextlib.suppress(Exception):
                return attr()
    return None


def load_named(name: str, plugin_dir: Path, load_from_dir: Callable[[Path], Optional[Any]], *, kind: str,
               noun: str, logger: logging.Logger) -> Optional[Any]:
    """Shared body of ``load_<kind>(name)``: load from *plugin_dir*, warn + None on failure."""
    try:
        instance = load_from_dir(plugin_dir)
    except Exception as e:
        logger.warning("Failed to load %s '%s': %s", kind.lower(), name, e)
        return None
    if not instance:
        logger.warning("%s '%s' loaded but no %s instance found", kind, name, noun)
    return instance or None


def probe_availability(load: Callable[[], Optional[Any]]) -> bool:
    """True iff *load()* returns an instance whose ``is_available()`` (if any) is truthy."""
    try:
        instance = load()
        return instance is not None and (instance.is_available() if hasattr(instance, "is_available") else True)
    except Exception:
        return False

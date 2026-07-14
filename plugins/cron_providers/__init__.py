"""Cron scheduler provider plugin discovery.

Scans two directories for cron scheduler provider plugins:

1. Bundled providers: ``plugins/cron_providers/<name>/`` (shipped with hermes-agent)
2. User-installed providers: ``$HERMES_HOME/plugins/<name>/``

Each subdirectory must contain ``__init__.py`` with a class implementing the
``CronScheduler`` ABC (``cron/scheduler_provider.py``). On name collisions,
bundled providers take precedence.

This is a near-verbatim clone of ``plugins/memory/__init__.py`` — the same
discovery/loader machinery, retargeted at ``CronScheduler``. The built-in
``InProcessCronScheduler`` is NOT discovered here: it is core (lives in
``cron/scheduler_provider.py``) so the fallback can never be accidentally
removed. Only NON-default providers (e.g. "chronos") live under this directory.

Only ONE provider can be active at a time, selected via ``cron.provider`` in
config.yaml (empty = built-in). See ``cron.scheduler_provider.resolve_cron_scheduler``.

Usage:
    from plugins.cron_providers import discover_cron_schedulers, load_cron_scheduler

    available = discover_cron_schedulers()   # [(name, desc, available), ...]
    provider = load_cron_scheduler("chronos")  # CronScheduler instance
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_CRON_PLUGINS_DIR = Path(__file__).parent

# Synthetic parent package for user-installed providers, so they don't
# collide with bundled providers in sys.modules.
_USER_NAMESPACE = "_hermes_user_cron"


def _register_synthetic_package(name: str, search_locations: List[str]) -> None:
    """Register an empty package shell in sys.modules.

    User-installed providers import as ``_hermes_user_cron.<name>``, a dotted
    name whose parents exist nowhere on disk. Unless those parents are present
    in ``sys.modules``, any relative import inside the plugin
    (``from . import config``) fails with
    ``ModuleNotFoundError: No module named '_hermes_user_cron'`` — the same
    reason the loader already registers ``plugins`` and ``plugins.cron_providers`` for
    bundled providers.
    """
    if name in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = search_locations
    sys.modules[name] = importlib.util.module_from_spec(spec)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _get_user_plugins_dir() -> Optional[Path]:
    """Return ``$HERMES_HOME/plugins/`` or None if unavailable."""
    try:
        from hermes_constants import get_hermes_home
        d = get_hermes_home() / "plugins"
        return d if d.is_dir() else None
    except Exception:
        return None


def _is_cron_provider_dir(path: Path) -> bool:
    """Heuristic: does *path* look like a cron scheduler provider plugin?

    Checks for ``register_cron_scheduler`` or ``CronScheduler`` in the
    ``__init__.py`` source. Cheap text scan — no import needed.
    """
    init_file = path / "__init__.py"
    if not init_file.exists():
        return False
    try:
        source = init_file.read_text(errors="replace")[:8192]
        return "register_cron_scheduler" in source or "CronScheduler" in source
    except Exception:
        return False


def _iter_provider_dirs() -> List[Tuple[str, Path]]:
    """Yield ``(name, path)`` for all discovered provider directories.

    Scans bundled first, then user-installed. Bundled takes precedence on
    name collisions (first-seen wins via ``seen`` set).
    """
    seen: set = set()
    dirs: List[Tuple[str, Path]] = []

    # 1. Bundled providers (plugins/cron_providers/<name>/)
    if _CRON_PLUGINS_DIR.is_dir():
        for child in sorted(_CRON_PLUGINS_DIR.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            if not (child / "__init__.py").exists():
                continue
            seen.add(child.name)
            dirs.append((child.name, child))

    # 2. User-installed providers ($HERMES_HOME/plugins/<name>/)
    user_dir = _get_user_plugins_dir()
    if user_dir:
        for child in sorted(user_dir.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            if child.name in seen:
                continue  # bundled takes precedence
            if not _is_cron_provider_dir(child):
                continue  # skip non-cron plugins
            dirs.append((child.name, child))

    return dirs


def find_provider_dir(name: str) -> Optional[Path]:
    """Resolve a provider name to its directory.

    Checks bundled first, then user-installed.
    """
    # Bundled
    bundled = _CRON_PLUGINS_DIR / name
    if bundled.is_dir() and (bundled / "__init__.py").exists():
        return bundled
    # User-installed
    user_dir = _get_user_plugins_dir()
    if user_dir:
        user = user_dir / name
        if user.is_dir() and _is_cron_provider_dir(user):
            return user
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_cron_schedulers() -> List[Tuple[str, str, bool]]:
    """Scan bundled and user-installed directories for available providers.

    Returns list of (name, description, is_available) tuples. May be empty —
    the built-in is core, not discovered here, so a fresh checkout with no
    bundled non-default provider returns []. Bundled providers take precedence
    on name collisions.
    """
    results = []

    for name, child in _iter_provider_dirs():
        # Read description from plugin.yaml if available
        desc = ""
        yaml_file = child / "plugin.yaml"
        if yaml_file.exists():
            try:
                import yaml
                with open(yaml_file, encoding="utf-8-sig") as f:
                    meta = yaml.safe_load(f) or {}
                desc = meta.get("description", "")
            except Exception:
                logger.debug("Suppressed exception", exc_info=True)

        # Quick availability check — try loading and calling is_available()
        available = True
        try:
            provider = _load_provider_from_dir(child)
            if provider:
                available = provider.is_available()
            else:
                available = False
        except Exception:
            available = False

        results.append((name, desc, available))

    return results


def load_cron_scheduler(name: str) -> Optional["CronScheduler"]:  # noqa: F821
    """Load and return a CronScheduler instance by name.

    Checks both bundled (``plugins/cron_providers/<name>/``) and user-installed
    (``$HERMES_HOME/plugins/<name>/``) directories. Bundled takes precedence
    on name collisions.

    Returns None if the provider is not found or fails to load.
    """
    provider_dir = find_provider_dir(name)
    if not provider_dir:
        logger.debug("Cron provider '%s' not found in bundled or user plugins", name)
        return None

    try:
        provider = _load_provider_from_dir(provider_dir)
        if provider:
            return provider
        logger.warning("Cron provider '%s' loaded but no provider instance found", name)
        return None
    except Exception as e:
        logger.warning("Failed to load cron provider '%s': %s", name, e)
        return None


def _load_provider_from_dir(provider_dir: Path) -> Optional["CronScheduler"]:  # noqa: F821
    """Import a provider module and extract the CronScheduler instance.

    The module must have either:
    - A register(ctx) function (plugin-style) — we simulate a ctx
    - A top-level class that extends CronScheduler — we instantiate it
    """
    name = provider_dir.name
    # Use a separate namespace for user-installed plugins so they don't
    # collide with bundled providers in sys.modules.
    _is_bundled = _CRON_PLUGINS_DIR in provider_dir.parents or provider_dir.parent == _CRON_PLUGINS_DIR
    module_name = f"plugins.cron_providers.{name}" if _is_bundled else f"{_USER_NAMESPACE}.{name}"
    init_file = provider_dir / "__init__.py"

    if not init_file.exists():
        return None

    # Check if already loaded. A synthetic package shell has no __file__;
    # only reuse modules that were actually loaded from disk.
    cached = sys.modules.get(module_name)
    if cached is not None and getattr(cached, "__file__", None):
        mod = cached
    else:
        # Ensure the parent packages are registered (for relative imports)
        for parent in ("plugins", "plugins.cron_providers"):
            if parent not in sys.modules:
                parent_path = Path(__file__).parent
                if parent == "plugins":
                    parent_path = parent_path.parent
                parent_init = parent_path / "__init__.py"
                if parent_init.exists():
                    spec = importlib.util.spec_from_file_location(
                        parent, str(parent_init),
                        submodule_search_locations=[str(parent_path)]
                    )
                    if spec:
                        parent_mod = importlib.util.module_from_spec(spec)
                        sys.modules[parent] = parent_mod
                        try:
                            spec.loader.exec_module(parent_mod)
                        except Exception:
                            logger.debug("Suppressed exception", exc_info=True)

        # User-installed plugins need their synthetic parent registered the
        # same way, or relative imports inside the plugin cannot resolve.
        if not _is_bundled:
            _register_synthetic_package(_USER_NAMESPACE, [])

        # Now load the provider module
        spec = importlib.util.spec_from_file_location(
            module_name, str(init_file),
            submodule_search_locations=[str(provider_dir)]
        )
        if not spec:
            return None

        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        loaded_submodules = []

        # Register submodules so relative imports work
        # e.g., "from ._nas_client import NasCronClient" in the chronos plugin
        for sub_file in provider_dir.glob("*.py"):
            if sub_file.name == "__init__.py":
                continue
            sub_name = sub_file.stem
            full_sub_name = f"{module_name}.{sub_name}"
            if full_sub_name not in sys.modules:
                sub_spec = importlib.util.spec_from_file_location(
                    full_sub_name, str(sub_file)
                )
                if sub_spec:
                    sub_mod = importlib.util.module_from_spec(sub_spec)
                    sys.modules[full_sub_name] = sub_mod
                    try:
                        sub_spec.loader.exec_module(sub_mod)
                        loaded_submodules.append((sub_name, sub_mod))
                    except Exception as e:
                        logger.debug("Failed to load submodule %s: %s", full_sub_name, e)

        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            logger.debug("Failed to exec_module %s: %s", module_name, e)
            sys.modules.pop(module_name, None)
            return None

        # Manual importlib loading bypasses the normal import machinery that
        # binds child modules onto their parent packages. Restore that shape so
        # later dotted imports and pytest monkeypatch paths resolve normally.
        parent_name, child_name = module_name.rsplit(".", 1)
        parent_mod = sys.modules.get(parent_name)
        if parent_mod is not None:
            setattr(parent_mod, child_name, mod)
        for sub_name, sub_mod in loaded_submodules:
            setattr(mod, sub_name, sub_mod)

    # Try register(ctx) pattern first (how our plugins are written)
    if hasattr(mod, "register"):
        collector = _ProviderCollector()
        try:
            mod.register(collector)
            if collector.provider:
                return collector.provider
        except Exception as e:
            logger.debug("register() failed for %s: %s", name, e)

    # Fallback: find a CronScheduler subclass and instantiate it
    from cron.scheduler_provider import CronScheduler
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name, None)
        if (isinstance(attr, type) and issubclass(attr, CronScheduler)
                and attr is not CronScheduler):
            try:
                return attr()
            except Exception:
                logger.debug("Suppressed exception", exc_info=True)

    return None


class _ProviderCollector:
    """Fake plugin context that captures register_cron_scheduler calls."""

    def __init__(self):
        self.provider = None

    def register_cron_scheduler(self, provider):
        self.provider = provider

    # No-op for other registration methods
    def register_tool(self, *args, **kwargs):
        pass

    def register_hook(self, *args, **kwargs):
        pass

    def register_memory_provider(self, *args, **kwargs):
        pass

    def register_cli_command(self, *args, **kwargs):
        pass

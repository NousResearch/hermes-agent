"""Cron scheduler provider discovery: bundled ``plugins/cron_providers/<name>/`` then user
``$HERMES_HOME/plugins/<name>/`` (bundled wins on collision). The built-in InProcessCronScheduler
is core, not discovered here, so the fallback can't be removed; one provider is active
(``cron.provider`` in config.yaml, empty = built-in)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from plugins import plugin_loader as _loader

logger = logging.getLogger(__name__)

_CRON_PLUGINS_DIR = Path(__file__).parent
# Synthetic parent package for user-installed providers (keeps them out of the bundled namespace).
_USER_NAMESPACE = "_hermes_user_cron"


def _is_cron_provider_dir(path: Path) -> bool:
    """Cheap text heuristic: ``__init__.py`` mentions the cron scheduler contract."""
    init_file = path / "__init__.py"
    if not init_file.exists():
        return False
    try:
        source = init_file.read_text(errors="replace", encoding="utf-8")[:8192]
        return "register_cron_scheduler" in source or "CronScheduler" in source
    except Exception:
        return False


def _user_provider_dirs() -> List[Path]:
    """User-installed ``$HERMES_HOME/plugins/<name>/`` dirs that look like cron providers."""
    user_dir = _loader.user_plugins_dir()
    return [c for c in _loader.iter_plugin_dirs(user_dir) if _is_cron_provider_dir(c)] if user_dir else []


def _iter_provider_dirs() -> List[Tuple[str, Path]]:
    """``(name, path)`` for bundled then user providers; bundled wins on collisions."""
    dirs = [(child.name, child) for child in _loader.iter_plugin_dirs(_CRON_PLUGINS_DIR)]
    seen = {name for name, _ in dirs}
    dirs.extend((child.name, child) for child in _user_provider_dirs() if child.name not in seen)
    return dirs


def find_provider_dir(name: str) -> Optional[Path]:
    """Resolve a provider name to its directory (bundled first, then user-installed)."""
    bundled = _CRON_PLUGINS_DIR / name
    if bundled.is_dir() and (bundled / "__init__.py").exists():
        return bundled
    user_dir = _loader.user_plugins_dir()
    user = user_dir / name if user_dir else None
    return user if user and user.is_dir() and _is_cron_provider_dir(user) else None


def discover_cron_schedulers() -> List[Tuple[str, str, bool]]:
    """Return ``[(name, description, is_available), ...]`` for all discovered providers."""
    return [(name, _loader.read_plugin_description(child),
             _loader.probe_availability(lambda c=child: _load_provider_from_dir(c)))
            for name, child in _iter_provider_dirs()]


def load_cron_scheduler(name: str) -> Optional["CronScheduler"]:  # noqa: F821
    """Load a CronScheduler instance by name; None if not found or it fails to load."""
    provider_dir = find_provider_dir(name)
    if not provider_dir:
        logger.debug("Cron provider '%s' not found in bundled or user plugins", name)
        return None
    return _loader.load_named(
        name, provider_dir, _load_provider_from_dir,
        kind="Cron provider", noun="provider", logger=logger)


def _load_provider_from_dir(provider_dir: Path) -> Optional["CronScheduler"]:  # noqa: F821
    """Import a provider module and extract its CronScheduler (register(ctx) or subclass)."""
    from cron.scheduler_provider import CronScheduler
    name = provider_dir.name
    is_bundled = _CRON_PLUGINS_DIR in provider_dir.parents or provider_dir.parent == _CRON_PLUGINS_DIR
    module_name = f"plugins.cron_providers.{name}" if is_bundled else f"{_USER_NAMESPACE}.{name}"
    mod = _loader.load_plugin_module(
        module_name, provider_dir, parents=("plugins", "plugins.cron_providers"), logger=logger,
        synthetic_namespace=None if is_bundled else _USER_NAMESPACE)
    return mod and _loader.instance_from_module(
        mod, collector=_ProviderCollector(), collected_attr="provider",
        base_cls=CronScheduler, name=name, logger=logger)


class _ProviderCollector(_loader.NoopPluginContext):
    """Fake plugin context that captures register_cron_scheduler calls."""

    def __init__(self):
        self.provider = None

    def register_cron_scheduler(self, provider):
        self.provider = provider


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import importlib.util  # noqa: F401,E402
import sys  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----

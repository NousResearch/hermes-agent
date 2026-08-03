"""Provider module registry.

Provider profiles can live in three places:

1. Bundled plugins: ``plugins/model-providers/<name>/`` (shipped with hermes-agent)
2. User plugins: ``$HERMES_HOME/plugins/model-providers/<name>/``
3. Opt-in project plugins: ``./.hermes/plugins/model-providers/<name>/``

Each plugin directory contains:
  - ``__init__.py`` — calls ``register_provider(profile)`` at import
  - ``plugin.yaml`` — manifest (name, kind: model-provider, version, description)

Discovery is lazy. Manifest identity and activation are evaluated before any
plugin code is imported. Bundled profiles are on by default; user and project
profiles must be listed in ``plugins.enabled``. Explicit disable always wins,
and safe mode imports bundled profiles only. Active user/project plugins
override bundled plugins on canonical-key collision.

For backward compatibility, explicitly enabled ``providers/*.py`` files
(other than ``base.py`` and ``__init__.py``) are still discovered via
``pkgutil.iter_modules``. This lets out-of-tree users keep a single-file
profile in an editable install without making it execute by default. New
profiles should prefer the plugin layout.

Usage::

    from providers import get_provider_profile
    profile = get_provider_profile("nvidia")   # ProviderProfile or None
    profile = get_provider_profile("kimi")     # checks name + aliases
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from hermes_cli.plugin_activation import PluginActivationState
from providers.base import OMIT_TEMPERATURE, ProviderProfile  # noqa: F401
from utils import env_var_enabled, fast_safe_load

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, ProviderProfile] = {}
_ALIASES: dict[str, str] = {}
_PROVIDER_LIST_CACHE: list[ProviderProfile] | None = None
_discovered = False
_discovering = False
_ACTIVATION_STATE: PluginActivationState | None = None
_DISCOVERY_FINGERPRINT: tuple[object, ...] | None = None
_DISCOVERY_LOCK = threading.RLock()
_IMPORTED_PROVIDER_MODULES: set[str] = set()
_PROVIDER_REFRESH_HOOKS: list[Callable[[], None]] = []
_PLUGIN_MANAGED_PROVIDER_IDS: set[str] = set()
_PROVIDER_PROFILE_ORIGINS: dict[str, tuple[str, str]] = {}

# Repo-root ``plugins/model-providers/`` — populated at discovery time.
_BUNDLED_PLUGINS_DIR = (
    Path(__file__).resolve().parent.parent / "plugins" / "model-providers"
)


def register_provider(profile: ProviderProfile) -> None:
    """Register a provider profile by name and aliases.

    Later registrations with the same name replace earlier ones — so user
    plugins under ``$HERMES_HOME/plugins/model-providers/`` can override
    bundled profiles without editing repo code.
    """
    global _PROVIDER_LIST_CACHE
    with _DISCOVERY_LOCK:
        _REGISTRY[profile.name] = profile
        for alias in profile.aliases:
            _ALIASES[alias] = profile.name
        _PROVIDER_LIST_CACHE = None


def get_provider_profile(name: str) -> ProviderProfile | None:
    """Look up a provider profile by name or alias.

    Returns None if the provider has no profile (falls back to generic).
    """
    _ensure_providers_discovered()
    with _DISCOVERY_LOCK:
        canonical = _ALIASES.get(name, name)
        return _REGISTRY.get(canonical)


def list_providers() -> list[ProviderProfile]:
    """Return all registered provider profiles (one per canonical name)."""
    global _PROVIDER_LIST_CACHE
    _ensure_providers_discovered()
    with _DISCOVERY_LOCK:
        if _PROVIDER_LIST_CACHE is not None:
            return list(_PROVIDER_LIST_CACHE)
        # Deduplicate: _REGISTRY has canonical names; _ALIASES points to same objects
        seen: set[int] = set()
        result: list[ProviderProfile] = []
        for profile in _REGISTRY.values():
            pid = id(profile)
            if pid not in seen:
                seen.add(pid)
                result.append(profile)
        _PROVIDER_LIST_CACHE = result
        return list(result)


def register_provider_refresh_hook(callback: Callable[[], None]) -> None:
    """Register an in-process index derived from provider discovery."""
    with _DISCOVERY_LOCK:
        if callback not in _PROVIDER_REFRESH_HOOKS:
            _PROVIDER_REFRESH_HOOKS.append(callback)


def get_provider_discovery_identity() -> tuple[object, ...]:
    """Return the active discovery identity for downstream cache keys."""
    _ensure_providers_discovered()
    with _DISCOVERY_LOCK:
        fingerprint = _DISCOVERY_FINGERPRINT or ()
        if not fingerprint or not isinstance(
            fingerprint[0], PluginActivationState
        ):
            return fingerprint
        state = fingerprint[0]
        return (
            state.safe_mode,
            None if state.enabled is None else tuple(sorted(state.enabled)),
            tuple(sorted(state.disabled)),
            *fingerprint[1:],
            tuple(sorted(_PROVIDER_PROFILE_ORIGINS.items())),
        )


def get_provider_profile_origin(name: str) -> tuple[str, str] | None:
    """Return ``(source, path)`` for the active provider profile."""
    _ensure_providers_discovered()
    with _DISCOVERY_LOCK:
        canonical = _ALIASES.get(name, name)
        return _PROVIDER_PROFILE_ORIGINS.get(canonical)


def invalidate_provider_discovery() -> None:
    """Rebuild providers and notify indexes after activation changes."""
    global _discovered, _ACTIVATION_STATE, _DISCOVERY_FINGERPRINT
    with _DISCOVERY_LOCK:
        _discovered = False
        _ACTIVATION_STATE = None
        _DISCOVERY_FINGERPRINT = None
    _ensure_providers_discovered()


def is_plugin_managed_provider_id(provider_id: str) -> bool:
    """Return whether any model-provider plugin declares *provider_id*."""
    _ensure_providers_discovered()
    with _DISCOVERY_LOCK:
        return provider_id in _PLUGIN_MANAGED_PROVIDER_IDS


def is_provider_plugin_active(provider_id: str) -> bool:
    """Return whether a plugin-managed provider is active and registered."""
    _ensure_providers_discovered()
    with _DISCOVERY_LOCK:
        if provider_id not in _PLUGIN_MANAGED_PROVIDER_IDS:
            return True
        return provider_id in _REGISTRY


def _current_activation_state() -> PluginActivationState:
    """Late-bind config.py to avoid its eager provider-injection cycle."""
    try:
        from hermes_cli.config import load_plugin_activation_state

        return load_plugin_activation_state()
    except Exception:
        return PluginActivationState(
            safe_mode=env_var_enabled("HERMES_SAFE_MODE"),
        )


def _ensure_providers_discovered() -> None:
    """Refresh when activation or any discovery root changes."""
    state = _current_activation_state()
    user_dir = _user_plugins_dir()
    project_dir = _project_plugins_dir()
    fingerprint = (
        state,
        _path_identity(_BUNDLED_PLUGINS_DIR),
        _path_identity(user_dir),
        _path_identity(project_dir),
    )
    callbacks: tuple[Callable[[], None], ...] = ()
    with _DISCOVERY_LOCK:
        if _discovered and _DISCOVERY_FINGERPRINT == fingerprint:
            return
        _discover_providers(
            state,
            user_dir=user_dir,
            project_dir=project_dir,
            fingerprint=fingerprint,
        )
        callbacks = tuple(_PROVIDER_REFRESH_HOOKS)

    for callback in callbacks:
        try:
            callback()
        except Exception:
            logger.warning(
                "Failed to refresh a provider-derived registry",
                exc_info=True,
            )


def _path_identity(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve())
    except (OSError, RuntimeError):
        return str(path.absolute())


def _user_plugins_dir() -> Path | None:
    """Return ``$HERMES_HOME/plugins/model-providers/`` if it exists."""
    try:
        from hermes_constants import get_hermes_home

        d = get_hermes_home() / "plugins" / "model-providers"
        return d if d.is_dir() else None
    except Exception:
        return None


def _project_plugins_dir() -> Path | None:
    """Return the opt-in project model-provider directory when present."""
    if not env_var_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
        return None
    directory = Path.cwd() / ".hermes" / "plugins" / "model-providers"
    return directory if directory.is_dir() else None


@dataclass(frozen=True)
class _ProviderPlugin:
    path: Path
    source: str
    key: str
    name: str
    provider_ids: frozenset[str]


def _provider_plugin(plugin_dir: Path, source: str) -> _ProviderPlugin:
    """Read provider identities without importing executable plugin code."""
    key = f"model-providers/{plugin_dir.name}"
    name = key
    provider_ids = {plugin_dir.name}
    manifest_file = plugin_dir / "plugin.yaml"
    if not manifest_file.exists():
        manifest_file = plugin_dir / "plugin.yml"
    if manifest_file.exists():
        try:
            data = fast_safe_load(manifest_file.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                manifest_name = data.get("name")
                if isinstance(manifest_name, str) and manifest_name.strip():
                    name = manifest_name.strip()
                declared_ids = data.get("provider_ids")
                if isinstance(declared_ids, list):
                    normalized_ids = {
                        value.strip()
                        for value in declared_ids
                        if isinstance(value, str) and value.strip()
                    }
                    if normalized_ids:
                        provider_ids = normalized_ids
        except Exception:
            logger.debug(
                "Could not parse provider plugin manifest identity: %s",
                plugin_dir,
                exc_info=True,
            )
    return _ProviderPlugin(
        path=plugin_dir,
        source=source,
        key=key,
        name=name,
        provider_ids=frozenset(provider_ids),
    )


def _import_plugin_dir(plugin_dir: Path, source: str) -> None:
    """Import a single plugin directory so it self-registers.

    ``source`` is "bundled", "user", or "project".
    """
    global _PROVIDER_LIST_CACHE

    init_file = plugin_dir / "__init__.py"
    if not init_file.exists():
        return

    # Give bundled plugins a stable import path (``plugins.model_providers.<name>``)
    # so relative imports within the plugin work. User plugins load via
    # ``importlib.util.spec_from_file_location`` with a unique module name so
    # multiple HERMES_HOME profiles don't alias each other.
    safe_name = plugin_dir.name.replace("-", "_")
    if source == "bundled":
        module_name = f"plugins.model_providers.{safe_name}"
    else:
        module_name = f"_hermes_{source}_provider_{safe_name}"

    if module_name in sys.modules:
        _IMPORTED_PROVIDER_MODULES.add(module_name)
        return  # already imported

    registry_snapshot = dict(_REGISTRY)
    aliases_snapshot = dict(_ALIASES)
    origins_snapshot = dict(_PROVIDER_PROFILE_ORIGINS)
    cache_snapshot = (
        None if _PROVIDER_LIST_CACHE is None else list(_PROVIDER_LIST_CACHE)
    )
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, init_file, submodule_search_locations=[str(plugin_dir)]
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        origin = (source, _path_identity(plugin_dir))
        for provider_id, profile in _REGISTRY.items():
            if registry_snapshot.get(provider_id) is not profile:
                _PROVIDER_PROFILE_ORIGINS[provider_id] = origin
        _IMPORTED_PROVIDER_MODULES.add(module_name)
    except Exception as exc:
        logger.warning(
            "Failed to load %s provider plugin %s: %s", source, plugin_dir.name, exc
        )
        _REGISTRY.clear()
        _REGISTRY.update(registry_snapshot)
        _ALIASES.clear()
        _ALIASES.update(aliases_snapshot)
        _PROVIDER_PROFILE_ORIGINS.clear()
        _PROVIDER_PROFILE_ORIGINS.update(origins_snapshot)
        _PROVIDER_LIST_CACHE = cache_snapshot
        for imported_name in tuple(sys.modules):
            if imported_name == module_name or imported_name.startswith(
                f"{module_name}."
            ):
                sys.modules.pop(imported_name, None)


def _discover_providers(
    state: PluginActivationState,
    *,
    user_dir: Path | None,
    project_dir: Path | None,
    fingerprint: tuple[object, ...],
) -> None:
    """Rebuild provider discovery from one canonical activation snapshot."""
    global _discovered, _discovering, _ACTIVATION_STATE
    global _DISCOVERY_FINGERPRINT, _PROVIDER_LIST_CACHE
    if _discovering:
        return

    _discovering = True
    try:
        for prefix in tuple(_IMPORTED_PROVIDER_MODULES):
            for module_name in tuple(sys.modules):
                if module_name == prefix or module_name.startswith(f"{prefix}."):
                    sys.modules.pop(module_name, None)
        _IMPORTED_PROVIDER_MODULES.clear()
        _REGISTRY.clear()
        _ALIASES.clear()
        _PROVIDER_LIST_CACHE = None
        _PLUGIN_MANAGED_PROVIDER_IDS.clear()
        _PROVIDER_PROFILE_ORIGINS.clear()

        candidates: list[_ProviderPlugin] = []
        for directory, source in (
            (_BUNDLED_PLUGINS_DIR, "bundled"),
            (user_dir, "user"),
            (project_dir, "project"),
        ):
            if state.safe_mode and source != "bundled":
                continue
            if directory is None or not directory.is_dir():
                continue
            for child in sorted(directory.iterdir()):
                if not child.is_dir() or child.name.startswith(("_", ".")):
                    continue
                candidates.append(_provider_plugin(child, source))

        winners: dict[str, _ProviderPlugin] = {}
        grouped: dict[str, list[_ProviderPlugin]] = {}
        for plugin in candidates:
            winners[plugin.key] = plugin
            grouped.setdefault(plugin.key, []).append(plugin)
            _PLUGIN_MANAGED_PROVIDER_IDS.update(plugin.provider_ids)

        for key, winner in winners.items():
            if not state.is_active(
                name=winner.name,
                key=winner.key,
                source=winner.source,
                kind="model-provider",
            ):
                logger.debug(
                    "Skipping inactive provider plugin winner '%s' (%s)",
                    key,
                    winner.source,
                )
                continue

            # Load active sources in precedence order. Bundled profiles load
            # first so an enabled user/project override can replace selected
            # IDs without deleting sibling profiles from a multi-ID bundle.
            for plugin in grouped[key]:
                if not state.is_active(
                    name=plugin.name,
                    key=plugin.key,
                    source=plugin.source,
                    kind="model-provider",
                ):
                    continue
                _import_plugin_dir(plugin.path, plugin.source)

        # Legacy single-file profiles are a compatibility extension path. Safe
        # mode must not execute them because their provenance is unknowable.
        if not state.safe_mode:
            try:
                import pkgutil

                import providers as _pkg

                for _importer, modname, _ispkg in pkgutil.iter_modules(_pkg.__path__):
                    if modname.startswith("_") or modname == "base":
                        continue
                    _PLUGIN_MANAGED_PROVIDER_IDS.add(modname)
                    if not state.is_active(
                        name=modname,
                        key=f"model-providers/{modname}",
                        source="legacy",
                        kind="model-provider",
                    ):
                        continue
                    registry_snapshot = dict(_REGISTRY)
                    aliases_snapshot = dict(_ALIASES)
                    origins_snapshot = dict(_PROVIDER_PROFILE_ORIGINS)
                    cache_snapshot = (
                        None
                        if _PROVIDER_LIST_CACHE is None
                        else list(_PROVIDER_LIST_CACHE)
                    )
                    module_name = f"providers.{modname}"
                    try:
                        module = importlib.import_module(module_name)
                        origin = (
                            "legacy",
                            _path_identity(
                                Path(getattr(module, "__file__", None) or modname)
                            ),
                        )
                        for provider_id, profile in _REGISTRY.items():
                            if registry_snapshot.get(provider_id) is not profile:
                                _PROVIDER_PROFILE_ORIGINS[provider_id] = origin
                                _PLUGIN_MANAGED_PROVIDER_IDS.add(provider_id)
                        _IMPORTED_PROVIDER_MODULES.add(module_name)
                    except Exception as exc:
                        _REGISTRY.clear()
                        _REGISTRY.update(registry_snapshot)
                        _ALIASES.clear()
                        _ALIASES.update(aliases_snapshot)
                        _PROVIDER_PROFILE_ORIGINS.clear()
                        _PROVIDER_PROFILE_ORIGINS.update(origins_snapshot)
                        _PROVIDER_LIST_CACHE = cache_snapshot
                        for imported_name in tuple(sys.modules):
                            if imported_name == module_name or imported_name.startswith(
                                f"{module_name}."
                            ):
                                sys.modules.pop(imported_name, None)
                        logger.warning(
                            "Failed to import legacy provider module %s: %s",
                            modname,
                            exc,
                        )
            except Exception:
                pass

        _ACTIVATION_STATE = state
        _DISCOVERY_FINGERPRINT = fingerprint
        _discovered = True
    except BaseException:
        _ACTIVATION_STATE = None
        _DISCOVERY_FINGERPRINT = None
        _discovered = False
        raise
    finally:
        _discovering = False

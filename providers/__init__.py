"""Provider module registry.

Provider profiles can live in three places:

1. Bundled plugins: ``plugins/model-providers/<name>/`` (shipped with hermes-agent)
2. Enabled Python packages exposing ``hermes_agent.model_providers`` entry points
3. User plugins: ``$HERMES_HOME/plugins/model-providers/<name>/``

Each plugin directory contains:
  - ``__init__.py`` — calls ``register_provider(profile)`` at import
  - ``plugin.yaml`` — manifest (name, kind: model-provider, version, description)

Discovery is lazy: the first call to ``get_provider_profile()`` or
``list_providers()`` scans all sources. User paths suppress same-key package
entry points before invocation, then user registrations can override bundled
profiles without editing the repo.

For backward compatibility, ``providers/*.py`` files (other than ``base.py``
and ``__init__.py``) are still discovered via ``pkgutil.iter_modules``.
This lets out-of-tree users drop a single-file profile into an editable
install without the plugin dir structure. New profiles should prefer the
plugin layout.

Usage::

    from providers import get_provider_profile
    profile = get_provider_profile("nvidia")   # ProviderProfile or None
    profile = get_provider_profile("kimi")     # checks name + aliases
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import sys
from pathlib import Path

import yaml

from providers.base import OMIT_TEMPERATURE, ProviderProfile  # noqa: F401

logger = logging.getLogger(__name__)

MODEL_PROVIDER_ENTRY_POINTS_GROUP = "hermes_agent.model_providers"

_REGISTRY: dict[str, ProviderProfile] = {}
_ALIASES: dict[str, str] = {}
_discovered = False

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
    _REGISTRY[profile.name] = profile
    for alias in profile.aliases:
        _ALIASES[alias] = profile.name


def get_provider_profile(name: str) -> ProviderProfile | None:
    """Look up a provider profile by name or alias.

    Returns None if the provider has no profile (falls back to generic).
    """
    if not _discovered:
        _discover_providers()
    canonical = _ALIASES.get(name, name)
    return _REGISTRY.get(canonical)


def list_providers() -> list[ProviderProfile]:
    """Return all registered provider profiles (one per canonical name)."""
    if not _discovered:
        _discover_providers()
    # Deduplicate: _REGISTRY has canonical names; _ALIASES points to same objects
    seen: set[int] = set()
    result: list[ProviderProfile] = []
    for profile in _REGISTRY.values():
        pid = id(profile)
        if pid not in seen:
            seen.add(pid)
            result.append(profile)
    return result


def _user_plugins_dir() -> Path | None:
    """Return ``$HERMES_HOME/plugins/model-providers/`` if it exists."""
    try:
        from hermes_constants import get_hermes_home

        d = get_hermes_home() / "plugins" / "model-providers"
        return d if d.is_dir() else None
    except Exception:
        return None


def _import_plugin_dir(plugin_dir: Path, source: str) -> None:
    """Import a single plugin directory so it self-registers.

    ``source`` is "bundled" or "user", used only for log messages.
    """
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
        module_name = f"_hermes_user_provider_{safe_name}"

    if module_name in sys.modules:
        return  # already imported

    try:
        spec = importlib.util.spec_from_file_location(
            module_name, init_file, submodule_search_locations=[str(plugin_dir)]
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        logger.warning(
            "Failed to load %s provider plugin %s: %s", source, plugin_dir.name, exc
        )
        sys.modules.pop(module_name, None)


def _provider_is_active(
    identities: set[str],
    *,
    enabled: set[str],
    disabled: set[str],
    opt_in: bool,
) -> bool:
    """Return whether a provider is active under its known identities."""
    if identities & disabled:
        return False
    return bool(identities & enabled) if opt_in else True


def _load_package_providers(
    enabled: set[str], disabled: set[str], blocked: set[str]
) -> None:
    """Load explicitly enabled dedicated model-provider entry points."""
    try:
        entry_points = list(
            importlib.metadata.entry_points().select(
                group=MODEL_PROVIDER_ENTRY_POINTS_GROUP
            )
        )
    except Exception as exc:
        logger.warning("Skipping package provider discovery: %s", exc)
        return

    counts: dict[str, int] = {}
    for entry_point in entry_points:
        counts[entry_point.name] = counts.get(entry_point.name, 0) + 1

    for entry_point in entry_points:
        canonical = f"model-providers/{entry_point.name}"
        if canonical in blocked:
            continue
        if counts[entry_point.name] > 1:
            logger.warning(
                "Skipping ambiguous package provider entry point %s",
                entry_point.name,
            )
            continue
        if not _provider_is_active(
            {canonical}, enabled=enabled, disabled=disabled, opt_in=True
        ):
            continue
        try:
            register = entry_point.load()
            if callable(register):
                register()
        except Exception as exc:
            logger.warning(
                "Failed to load package provider entry point %s: %s",
                entry_point.name,
                exc,
            )


def _user_provider_identities(plugin_dir: Path) -> set[str]:
    """Return canonical, directory-leaf, and manifest identities."""
    identities = {
        plugin_dir.name,
        f"model-providers/{plugin_dir.name}",
    }
    manifest_path = plugin_dir / "plugin.yaml"
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict) and isinstance(manifest.get("name"), str):
            identities.add(manifest["name"])
    except (OSError, yaml.YAMLError):
        pass
    return identities


def _discover_providers() -> None:
    """Populate the registry by importing every provider plugin.

    Order:
      1. Bundled plugins at ``<repo>/plugins/model-providers/<name>/``
      2. Enabled ``hermes_agent.model_providers`` package entry points
      3. User plugins at ``$HERMES_HOME/plugins/model-providers/<name>/``
      4. Legacy per-file modules at ``providers/<name>.py`` (back-compat)

    Directory plugins call ``register_provider()`` at module-level. User paths
    suppress same-key package entry points; later user registrations can still
    override bundled profiles.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True

    # 1. Bundled plugins — shipped with hermes-agent.
    if _BUNDLED_PLUGINS_DIR.is_dir():
        for child in sorted(_BUNDLED_PLUGINS_DIR.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            _import_plugin_dir(child, "bundled")

    # Read activation without defaults or fail-open parsing. A malformed or
    # unreadable existing config must not authorize untrusted package/user code.
    try:
        from hermes_cli.config import read_plugin_activation_config_strict

        _config, _plugins, enabled, disabled = read_plugin_activation_config_strict()
        activation_readable = True
    except Exception as exc:
        logger.warning("Skipping external provider plugins: %s", exc)
        enabled = set()
        disabled = set()
        activation_readable = False

    # 2. User plugin paths reserve their canonical package keys even when
    # disabled, so one activation token can never execute hidden package code.
    user_dir = _user_plugins_dir() if activation_readable else None
    blocked_package_keys = (
        {
            f"model-providers/{child.name}"
            for child in user_dir.iterdir()
            if child.is_dir() and not child.name.startswith(("_", "."))
        }
        if user_dir is not None
        else set()
    )

    # 3. Dedicated package providers are opt-in.
    if activation_readable:
        _load_package_providers(enabled, disabled, blocked_package_keys)

    # 4. User plugins — under $HERMES_HOME/plugins/model-providers/<name>/.
    #    These can override any bundled profile of the same name (last-writer-wins
    #    in register_provider()).
    if user_dir is not None:
        for child in sorted(user_dir.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            if not _provider_is_active(
                _user_provider_identities(child),
                enabled=enabled,
                disabled=disabled,
                opt_in=False,
            ):
                continue
            _import_plugin_dir(child, "user")

    # 5. Legacy single-file profiles at providers/<name>.py. Kept for
    #    back-compat — if someone drops a ``providers/foo.py`` into an
    #    editable install, it still works without the plugin layout.
    try:
        import pkgutil

        import providers as _pkg

        for _importer, modname, _ispkg in pkgutil.iter_modules(_pkg.__path__):
            if modname.startswith("_") or modname == "base":
                continue
            try:
                importlib.import_module(f"providers.{modname}")
            except ImportError as exc:
                logger.warning(
                    "Failed to import legacy provider module %s: %s", modname, exc
                )
    except Exception:
        pass

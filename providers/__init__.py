"""Provider module registry.

Provider profiles can live in two places:

1. Bundled plugins: ``plugins/model-providers/<name>/`` (shipped with hermes-agent)
2. User plugins: ``$HERMES_HOME/plugins/model-providers/<name>/``

Each plugin directory contains:
  - ``__init__.py`` — calls ``register_provider(profile)`` at import
  - ``plugin.yaml`` — manifest (name, kind: model-provider, version, description)

Discovery is lazy: the first call to ``get_provider_profile()`` or
``list_providers()`` scans both locations and imports every plugin. User
plugins override bundled plugins on name collision (last-writer-wins), so
third parties can monkey-patch or replace any built-in profile without
editing the repo.

An isolated runtime can instead call
``configure_isolated_provider_discovery()`` before the first registry use.
That one-way pin permits only the named bundled provider directories and
disables user and legacy discovery for the lifetime of the process.

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
import importlib.util
import logging
import re
import sys
import threading
from pathlib import Path
from types import ModuleType

from providers.base import OMIT_TEMPERATURE, ProviderProfile  # noqa: F401

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, ProviderProfile] = {}
_ALIASES: dict[str, str] = {}
_discovered = False
_discovery_error: str | None = None

# Isolated discovery is deliberately process-global and one-way.  The gateway
# configures it before provider resolution, after which neither a user plugin
# nor a later registration can broaden or replace the selected provider set.
_isolated_provider_allowlist: frozenset[str] | None = None
_isolated_registration_target: str | None = None
_isolated_discovery_validated = False
_DISCOVERY_LOCK = threading.RLock()
_SAFE_PROVIDER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Repo-root ``plugins/model-providers/`` — populated at discovery time.
_BUNDLED_PLUGINS_DIR = (
    Path(__file__).resolve().parent.parent / "plugins" / "model-providers"
)


class ProviderDiscoveryIsolationError(RuntimeError):
    """The isolated provider registry could not preserve its closed set."""


def configure_isolated_provider_discovery(
    allowed_bundled_providers: frozenset[str],
) -> None:
    """Pin discovery to an exact set of bundled provider plugins.

    This must be called before any provider is registered or discovered.  It
    synchronously imports and origin-validates the exact bundled set before
    returning, so callers can publish readiness only after the provider
    boundary is real.  The pin is intentionally one-way: repeating the exact
    same successfully validated pin is harmless, but changing or clearing it
    requires a new process.  Isolated discovery never inspects ``HERMES_HOME``
    provider plugins or legacy ``providers/*.py`` modules.

    ``frozenset`` is required so the caller cannot mutate the authorization
    set after this boundary accepts it.
    """
    global _isolated_provider_allowlist

    if not isinstance(allowed_bundled_providers, frozenset):
        raise TypeError("allowed_bundled_providers must be a frozenset")
    if not allowed_bundled_providers:
        raise ValueError("at least one bundled provider must be allowed")
    invalid = sorted(
        repr(name)
        for name in allowed_bundled_providers
        if not isinstance(name, str) or not _SAFE_PROVIDER_NAME.fullmatch(name)
    )
    if invalid:
        raise ValueError(f"invalid bundled provider names: {invalid!r}")

    with _DISCOVERY_LOCK:
        if _isolated_provider_allowlist is not None:
            if _isolated_provider_allowlist != allowed_bundled_providers:
                raise ProviderDiscoveryIsolationError(
                    "isolated provider discovery is already pinned and cannot be changed"
                )
            if _discovery_error is not None:
                raise ProviderDiscoveryIsolationError(_discovery_error)
            if _isolated_discovery_validated:
                return
            raise ProviderDiscoveryIsolationError(
                "isolated provider discovery has not completed successfully"
            )
        if _discovered or _REGISTRY or _ALIASES or _discovery_error is not None:
            raise ProviderDiscoveryIsolationError(
                "isolated provider discovery must be configured before registry use"
            )
        _isolated_provider_allowlist = frozenset(allowed_bundled_providers)
        _ensure_providers_discovered()
        if not _isolated_discovery_validated:
            raise ProviderDiscoveryIsolationError(
                "isolated provider discovery validation did not complete"
            )


def register_provider(profile: ProviderProfile) -> None:
    """Register a provider profile by name and aliases.

    Later registrations with the same name replace earlier ones — so user
    plugins under ``$HERMES_HOME/plugins/model-providers/`` can override
    bundled profiles without editing repo code.
    """
    with _DISCOVERY_LOCK:
        if _isolated_provider_allowlist is not None:
            if _isolated_registration_target is None:
                raise ProviderDiscoveryIsolationError(
                    "provider registration is closed after isolated discovery is configured"
                )
            if not isinstance(profile, ProviderProfile):
                raise ProviderDiscoveryIsolationError(
                    "isolated provider plugins must register a ProviderProfile"
                )
            if profile.name != _isolated_registration_target:
                raise ProviderDiscoveryIsolationError(
                    "isolated provider plugin registered an unexpected provider: "
                    f"expected {_isolated_registration_target!r}, got {profile.name!r}"
                )

        _REGISTRY[profile.name] = profile
        for alias in profile.aliases:
            _ALIASES[alias] = profile.name


def get_provider_profile(name: str) -> ProviderProfile | None:
    """Look up a provider profile by name or alias.

    Returns None if the provider has no profile (falls back to generic).
    """
    with _DISCOVERY_LOCK:
        _ensure_providers_discovered()
        canonical = _ALIASES.get(name, name)
        return _REGISTRY.get(canonical)


def list_providers() -> list[ProviderProfile]:
    """Return all registered provider profiles (one per canonical name)."""
    with _DISCOVERY_LOCK:
        _ensure_providers_discovered()
        # Deduplicate: _REGISTRY has canonical names; aliases point to the same
        # canonical entries.
        seen: set[int] = set()
        result: list[ProviderProfile] = []
        for profile in _REGISTRY.values():
            pid = id(profile)
            if pid not in seen:
                seen.add(pid)
                result.append(profile)
        return result


def _ensure_providers_discovered() -> None:
    if _discovery_error is not None:
        raise ProviderDiscoveryIsolationError(_discovery_error)
    if not _discovered:
        _discover_providers()


def _user_plugins_dir() -> Path | None:
    """Return ``$HERMES_HOME/plugins/model-providers/`` if it exists."""
    try:
        from hermes_constants import get_hermes_home

        d = get_hermes_home() / "plugins" / "model-providers"
        return d if d.is_dir() else None
    except Exception:
        return None


def _import_plugin_dir(
    plugin_dir: Path,
    source: str,
    *,
    raise_on_error: bool = False,
) -> ModuleType | None:
    """Import a single plugin directory so it self-registers.

    ``source`` is "bundled" or "user", used only for log messages.
    """
    init_file = plugin_dir / "__init__.py"
    if not init_file.exists():
        if raise_on_error:
            raise ProviderDiscoveryIsolationError(
                f"provider plugin {plugin_dir.name!r} has no __init__.py"
            )
        return None

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
        return sys.modules[module_name]  # already imported

    try:
        spec = importlib.util.spec_from_file_location(
            module_name, init_file, submodule_search_locations=[str(plugin_dir)]
        )
        if spec is None or spec.loader is None:
            if raise_on_error:
                raise ProviderDiscoveryIsolationError(
                    f"cannot construct import spec for provider {plugin_dir.name!r}"
                )
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        logger.warning(
            "Failed to load %s provider plugin %s: %s", source, plugin_dir.name, exc
        )
        sys.modules.pop(module_name, None)
        if raise_on_error:
            if isinstance(exc, ProviderDiscoveryIsolationError):
                raise
            raise ProviderDiscoveryIsolationError(
                f"failed to load isolated provider {plugin_dir.name!r}"
            ) from exc
        return None


def _discover_isolated_providers(allowlist: frozenset[str]) -> None:
    """Import and verify exactly *allowlist* from the bundled provider tree."""
    global _isolated_registration_target

    try:
        bundled_root = _BUNDLED_PLUGINS_DIR.resolve(strict=True)
    except OSError as exc:
        raise ProviderDiscoveryIsolationError(
            "bundled model-provider directory is unavailable"
        ) from exc

    for provider_name in sorted(allowlist):
        plugin_dir = _BUNDLED_PLUGINS_DIR / provider_name
        try:
            resolved_dir = plugin_dir.resolve(strict=True)
        except OSError as exc:
            raise ProviderDiscoveryIsolationError(
                f"required bundled provider {provider_name!r} is unavailable"
            ) from exc
        if (
            not plugin_dir.is_dir()
            or plugin_dir.is_symlink()
            or resolved_dir.parent != bundled_root
        ):
            raise ProviderDiscoveryIsolationError(
                f"bundled provider {provider_name!r} has an invalid origin"
            )

        init_file = plugin_dir / "__init__.py"
        if init_file.is_symlink():
            raise ProviderDiscoveryIsolationError(
                f"bundled provider {provider_name!r} has a symlinked entry point"
            )

        _isolated_registration_target = provider_name
        try:
            module = _import_plugin_dir(
                plugin_dir, "bundled", raise_on_error=True
            )
        finally:
            _isolated_registration_target = None

        module_file = getattr(module, "__file__", None)
        try:
            actual_module_file = Path(module_file).resolve(strict=True)
            expected_module_file = init_file.resolve(strict=True)
        except (OSError, TypeError) as exc:
            raise ProviderDiscoveryIsolationError(
                f"bundled provider {provider_name!r} has no verifiable module origin"
            ) from exc
        if actual_module_file != expected_module_file:
            raise ProviderDiscoveryIsolationError(
                f"bundled provider {provider_name!r} loaded from an unexpected origin"
            )

    registered = frozenset(_REGISTRY)
    if registered != allowlist:
        raise ProviderDiscoveryIsolationError(
            "isolated provider registry does not match its pin: "
            f"expected {sorted(allowlist)!r}, got {sorted(registered)!r}"
        )
    if not set(_ALIASES.values()).issubset(allowlist):
        raise ProviderDiscoveryIsolationError(
            "isolated provider registry contains an alias outside its pin"
        )


def _discover_providers() -> None:
    """Populate the registry by importing every provider plugin.

    Order:
      1. Bundled plugins at ``<repo>/plugins/model-providers/<name>/``
      2. User plugins at ``$HERMES_HOME/plugins/model-providers/<name>/``
      3. Legacy per-file modules at ``providers/<name>.py`` (back-compat)

    Each step imports its plugins, which call ``register_provider()`` at
    module-level. Later steps win on name collision.
    """
    global _discovered, _discovery_error, _isolated_discovery_validated
    if _discovered:
        return
    _discovered = True

    if _isolated_provider_allowlist is not None:
        try:
            _discover_isolated_providers(_isolated_provider_allowlist)
            _isolated_discovery_validated = True
        except Exception as exc:
            _REGISTRY.clear()
            _ALIASES.clear()
            _discovery_error = f"isolated provider discovery failed: {exc}"
            raise ProviderDiscoveryIsolationError(_discovery_error) from exc
        return

    # 1. Bundled plugins — shipped with hermes-agent.
    if _BUNDLED_PLUGINS_DIR.is_dir():
        for child in sorted(_BUNDLED_PLUGINS_DIR.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            _import_plugin_dir(child, "bundled")

    # 2. User plugins — under $HERMES_HOME/plugins/model-providers/<name>/.
    #    These can override any bundled profile of the same name (last-writer-wins
    #    in register_provider()).
    user_dir = _user_plugins_dir()
    if user_dir is not None:
        for child in sorted(user_dir.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            _import_plugin_dir(child, "user")

    # 3. Legacy single-file profiles at providers/<name>.py. Kept for
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

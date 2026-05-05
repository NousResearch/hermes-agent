"""Provider module registry.

Auto-discovers ProviderProfile instances from providers/*.py modules.
Each module should define a module-level PROVIDER or PROVIDERS list.

Usage:
    from providers import get_provider_profile
    profile = get_provider_profile("nvidia")  # returns ProviderProfile or None
    profile = get_provider_profile("kimi")    # checks name + aliases
"""

from __future__ import annotations

from providers.base import OMIT_TEMPERATURE, ProviderProfile  # noqa: F401

_REGISTRY: dict[str, ProviderProfile] = {}
_ALIASES: dict[str, str] = {}
_discovered = False


def register_provider(profile: ProviderProfile) -> None:
    """Register a provider profile by name and aliases."""
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


def _discover_providers() -> None:
    """Import all provider modules to trigger registration."""
    global _discovered
    if _discovered:
        return
    _discovered = True

    import importlib
    import pkgutil

    import providers as _pkg

    for _importer, modname, _ispkg in pkgutil.iter_modules(_pkg.__path__):
        if modname.startswith("_") or modname == "base":
            continue
        try:
            importlib.import_module(f"providers.{modname}")
        except ImportError as e:
            import logging

            logging.getLogger(__name__).warning(
                "Failed to import provider module %s: %s", modname, e
            )

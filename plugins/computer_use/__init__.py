"""Computer-use provider discovery: bundled ``plugins/computer_use/<name>/`` (the built-in ``cua`` driver lives
here too) then user ``$HERMES_HOME/plugins/<name>/`` (bundled wins on collision). Exactly one provider is active:
``computer_use.backend`` in the active profile's config.yaml names its directory (default ``cua``), read per call.
Only that provider is imported; the others stay on disk, so a plugin that is installed but not selected is just an
option in ``hermes tools``. A configured name that resolves to nothing fails; there is no fallback driver."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from plugins import plugin_loader as _loader

if TYPE_CHECKING:
    from tools.computer_use.backend import ComputerUseProvider

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "cua"
_CU_PLUGINS_DIR = Path(__file__).parent
# Synthetic parent package for user-installed providers (keeps them out of the bundled namespace).
_USER_NAMESPACE = "_hermes_user_computer_use"


def _is_computer_use_dir(path: Path) -> bool:
    """Cheap text heuristic: ``__init__.py`` mentions the computer-use provider contract."""
    try:
        source = (path / "__init__.py").read_text(errors="replace", encoding="utf-8-sig")[:8192]
    except OSError:
        return False
    return "register_computer_use_provider" in source or "ComputerUseProvider" in source


def _iter_provider_dirs() -> list[tuple[str, Path]]:
    """``(name, path)`` for bundled then user providers; bundled wins on collisions."""
    dirs = [(child.name, child) for child in _loader.iter_plugin_dirs(_CU_PLUGINS_DIR)]
    seen = {name for name, _ in dirs}
    user_dir = _loader.user_plugins_dir()
    if user_dir:
        dirs.extend((child.name, child) for child in _loader.iter_plugin_dirs(user_dir)
                    if child.name not in seen and _is_computer_use_dir(child))
    return dirs


def discover_computer_use_providers() -> list[tuple[str, str]]:
    """``[(name, description), ...]`` from plugin.yaml only: listing never imports a provider."""
    return [(name, _loader.read_plugin_description(child)) for name, child in _iter_provider_dirs()]


def find_provider_dir(name: str) -> Optional[Path]:
    """Resolve a provider name to its directory (bundled first, then user-installed)."""
    return next((path for found, path in _iter_provider_dirs() if found == name), None)


def load_computer_use_provider(name: str) -> Optional[ComputerUseProvider]:
    """Import the named provider and return its ComputerUseProvider; None if not found or it fails to load."""
    provider_dir = find_provider_dir(name)
    if provider_dir is None:
        return None
    return _loader.load_named(name, provider_dir, _load_provider_from_dir, kind="Computer-use provider",
                              noun="provider", logger=logger)


def _load_provider_from_dir(provider_dir: Path) -> Optional[ComputerUseProvider]:
    # Providers hand back live driver sessions, so they load in-process only: load_plugin_module refuses user
    # code under ``plugins.isolation: host`` (warning + None), which surfaces as "could not be loaded".
    from tools.computer_use.backend import ComputerUseProvider
    name = provider_dir.name
    is_bundled = provider_dir.parent == _CU_PLUGINS_DIR
    mod = _loader.load_plugin_module(
        f"plugins.computer_use.{name}" if is_bundled else f"{_USER_NAMESPACE}.{name}", provider_dir,
        parents=("plugins", "plugins.computer_use"), logger=logger,
        synthetic_namespace=None if is_bundled else _USER_NAMESPACE)
    return mod and _loader.instance_from_module(
        mod, collector=_ProviderCollector(), collected_attr="provider", base_cls=ComputerUseProvider,
        name=name, logger=logger)


class _ProviderCollector(_loader.NoopPluginContext):
    """Fake plugin context that captures ``register_computer_use_provider``."""

    def __init__(self):
        self.provider = None

    def register_computer_use_provider(self, provider):
        self.provider = provider


def configured_backend_name() -> str:
    """``computer_use.backend`` from the active profile's config.yaml (``cua`` when unset)."""
    from hermes_cli.config import load_config_readonly

    raw = ((load_config_readonly() or {}).get("computer_use") or {}).get("backend")
    return raw.strip() if isinstance(raw, str) and raw.strip() else DEFAULT_BACKEND


def get_active_provider() -> ComputerUseProvider:
    """The one provider ``computer_use.backend`` selects; ``LookupError`` naming the available ones otherwise."""
    name = configured_backend_name()
    provider = load_computer_use_provider(name)
    if provider is None:
        available = ", ".join(found for found, _ in discover_computer_use_providers())
        state = "could not be loaded (see the log)" if find_provider_dir(name) else "is not installed"
        raise LookupError(
            f"computer_use.backend is {name!r}, but that computer-use provider {state} (available: {available}). "
            "Install it under ~/.hermes/plugins/<name>/ or pick a backend with `hermes tools`.")
    return provider

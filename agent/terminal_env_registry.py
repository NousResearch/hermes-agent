"""Terminal Environment Registry.

Central map of registered pluggable terminal backends, populated by plugins via
:meth:`PluginContext.register_terminal_environment_provider` and consumed by
:func:`tools.terminal_tool_backends._create_environment` plus the classification helpers across the
terminal/file/approval/prompt surfaces. Unlike the image/video/web/browser registries there
is **no active-provider resolution**: the active backend is whatever ``TERMINAL_ENV`` /
``terminal.backend`` names. Built-in names are reserved (registration raises) so a plugin can
never shadow the in-tree docker/modal/... implementations. Scope semantics mirror
:mod:`agent.browser_registry` (per-profile scope or the global base map).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from agent.provider_registry import ProviderRegistry, lower_key
from agent.terminal_env_provider import TerminalEnvironmentProvider

logger = logging.getLogger(__name__)


#: Names owned by in-tree backends in tools/environments/ — never
#: registrable by plugins. Includes internal-mode aliases (managed_modal).
BUILTIN_BACKEND_NAMES = frozenset({
    "local", "docker", "singularity", "modal", "managed_modal",
    "daytona", "vercel_sandbox", "ssh",
})


def _reject_builtin_collision(name: str) -> None:
    raise ValueError(f"Terminal backend name '{name}' is reserved for the built-in {name} backend "
                     "and cannot be registered by a plugin")


_registry: ProviderRegistry[TerminalEnvironmentProvider] = ProviderRegistry(
    label="Terminal environment",
    provider_cls=TerminalEnvironmentProvider,
    logger=logger,
    normalize=lower_key,
    builtin_names=BUILTIN_BACKEND_NAMES,
    on_builtin_collision=_reject_builtin_collision,
)
_registry.export(globals())


def plugin_backend_names(*, scope: Optional[str] = None) -> List[str]:
    """Names of all registered plugin backends (sorted)."""
    return [p.name.strip().lower() for p in _registry.list_providers(scope=scope)]


def provider_flag(name: str, attr: str, default=False):
    """Read a classification attribute off the provider for *name*.

    Fail-soft: unknown backend or a raising property returns *default* so a misbehaving
    plugin degrades to built-in-equivalent behavior instead of taking the terminal tool down.
    """
    provider = _registry.get_provider(name)
    if provider is None:
        return default
    try:
        return getattr(provider, attr, default)
    except Exception:
        logger.debug("Terminal environment provider '%s' attribute '%s' raised", name, attr, exc_info=True)
        return default


def plugin_strip_env_keys() -> frozenset:
    """Union of every registered provider's ``strip_env_keys`` — across ALL scopes, not
    just the active backend: a token in the process environment is strippable regardless of
    which backend is selected (as MODAL_*/DAYTONA_API_KEY sit in the static tier-1 set)."""
    keys: set = set()
    with _registry._lock:
        all_providers = list(_registry._providers.values())
        for scoped in _registry._scoped_providers.values():
            all_providers.extend(scoped.values())
    for provider in all_providers:
        try:
            keys.update(provider.strip_env_keys)
        except Exception:
            logger.debug("Terminal environment provider strip_env_keys raised", exc_info=True)
    return frozenset(keys)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Dict  # noqa: F401,E402
import threading  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'hermes_home_key': ('hermes_constants', 'hermes_home_key'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----

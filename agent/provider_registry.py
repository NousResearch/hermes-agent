"""Shared engine behind the ``agent.*_registry`` provider registries.

Every pluggable-backend registry has the same shape: a global name->provider map
plus per-profile *scoped* maps (multiplexed gateways), a lock, registration with
re-registration logging, and the snapshot/restore pair :mod:`hermes_cli.plugins`
uses to unwind a plugin. Each ``*_registry`` module instantiates one
:class:`ProviderRegistry` and re-exports its bound methods under the historical
module-level names via :meth:`ProviderRegistry.export`, so ``patch("agent.x_registry.get_provider")``
targets and the ``_providers`` / ``_scoped_providers`` / ``_lock`` test hooks are unchanged.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, FrozenSet, Generic, List, Optional, TypeVar

from hermes_constants import hermes_home_key

P = TypeVar("P")


def lower_key(name: str) -> str:
    return name.strip().lower()


class ProviderRegistry(Generic[P]):
    """Global + per-scope provider map with plugin snapshot/restore support.

    ``normalize`` is ``str.strip`` or ``lower_key`` (case-insensitive registries mirror
    how their dispatcher normalizes the configured name). ``builtin_names`` are reserved
    for in-tree implementations; a collision calls ``on_builtin_collision(key)`` and, if
    that returns, skips registration. ``logger`` is the owning module's so record names
    stay per-registry.
    """

    def __init__(
        self, *, label: str, provider_cls: type, logger: logging.Logger,
        normalize: Callable[[str], str] = str.strip, builtin_names: FrozenSet[str] = frozenset(),
        on_builtin_collision: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.label = label
        self.provider_cls = provider_cls
        self.logger = logger
        self.normalize = normalize
        self.builtin_names = builtin_names
        self._on_builtin_collision = on_builtin_collision
        self._providers: Dict[str, P] = {}
        self._scoped_providers: Dict[str, Dict[str, P]] = {}
        self._generation = 0
        self._scoped_generations: Dict[str, int] = {}
        self._lock = threading.Lock()
        # "TTS provider" but "Registered browser provider": acronyms keep their case.
        self._log_label = label if label.isupper() else label[0].lower() + label[1:]

    def _target(self, scope: Optional[str], *, create: bool) -> Dict[str, P]:
        if scope is None:
            return self._providers
        if create:
            return self._scoped_providers.setdefault(scope, {})
        return self._scoped_providers.get(scope, {})

    def _bump(self, scope: Optional[str]) -> None:
        if scope is None:
            self._generation += 1
        else:
            self._scoped_generations[scope] = self._scoped_generations.get(scope, 0) + 1

    def register(self, provider: P, *, scope: Optional[str] = None) -> None:
        """Register a provider; same-name re-registration overwrites (hot reload)."""
        if not isinstance(provider, self.provider_cls):
            article = "an" if self.provider_cls.__name__[0] in "AEIOU" else "a"
            raise TypeError(
                f"register_provider() expects {article} {self.provider_cls.__name__} "
                f"instance, got {type(provider).__name__}"
            )
        raw_name = getattr(provider, "name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"{self.label} provider .name must be a non-empty string")
        key = self.normalize(raw_name)
        if key in self.builtin_names:
            if self._on_builtin_collision is not None:
                self._on_builtin_collision(key)
            return
        with self._lock:
            target = self._target(scope, create=True)
            existing = target.get(key)
            target[key] = provider
            self._bump(scope)
        if existing is not None:
            self.logger.debug(
                f"{self.label} provider '%s' re-registered (was %r)", key, type(existing).__name__,
            )
        else:
            self.logger.debug(
                f"Registered {self._log_label} provider '%s' (%s)", key, type(provider).__name__,
            )

    def merged(self, scope: Optional[str] = None) -> Dict[str, P]:
        """Global map overlaid with the active profile's scoped map (a copy)."""
        with self._lock:
            merged = dict(self._providers)
            merged.update(self._scoped_providers.get(scope or hermes_home_key(), {}))
        return merged

    def list_providers(self, *, scope: Optional[str] = None) -> List[P]:
        """Return all registered providers, sorted by name."""
        return sorted(self.merged(scope).values(), key=lambda p: p.name)

    def get_provider(self, name: str, *, scope: Optional[str] = None) -> Optional[P]:
        """Return the provider registered under *name* (scoped first), or None."""
        if not isinstance(name, str):
            return None
        key = self.normalize(name)
        with self._lock:
            return (
                self._scoped_providers.get(scope or hermes_home_key(), {}).get(key)
                or self._providers.get(key)
            )

    def registry_generation(self, *, scope: Optional[str] = None) -> tuple:
        """Cache fingerprint ``(global_generation, scoped_generation)``."""
        active_scope = scope or hermes_home_key()
        with self._lock:
            return self._generation, self._scoped_generations.get(active_scope, 0)

    def snapshot_registration(self, name: str, *, scope: Optional[str] = None) -> Optional[P]:
        """Exact-slot lookup (no global fallback) used to detect plugin ownership."""
        with self._lock:
            return self._target(scope, create=False).get(self.normalize(name))

    def restore_registration(
        self, name: str, current: P, previous: Optional[P], *, scope: Optional[str] = None
    ) -> bool:
        """Restore *previous* only when *current* is still installed under *name*."""
        key = self.normalize(name)
        with self._lock:
            target = self._target(scope, create=True)
            if target.get(key) is not current:
                return False
            if previous is None:
                target.pop(key, None)
            else:
                target[key] = previous
            self._bump(scope)
            if scope is not None and not target:
                self._scoped_providers.pop(scope, None)
        return True

    def reset_for_tests(self) -> None:
        """Clear every registration. **Test-only.**"""
        with self._lock:
            self._providers.clear()
            self._scoped_providers.clear()
            self._scoped_generations.clear()
            self._generation += 1

    def export(self, namespace: Dict[str, Any]) -> None:
        """Bind the historical module-level API (+ ``_providers``/``_scoped_providers``/
        ``_lock`` test hooks) into a ``*_registry`` module namespace."""
        namespace.update(
            _providers=self._providers, _scoped_providers=self._scoped_providers, _lock=self._lock,
            register_provider=self.register, list_providers=self.list_providers,
            get_provider=self.get_provider, snapshot_registration=self.snapshot_registration,
            restore_registration=self.restore_registration,
            registry_generation=self.registry_generation, _reset_for_tests=self.reset_for_tests,
        )


def is_available_safe(
    provider: Any, logger: logging.Logger, fmt: str, *, level: int = logging.DEBUG, exc_info: bool = False,
) -> bool:
    """``bool(provider.is_available())`` that treats a raising provider as unavailable."""
    try:
        return bool(provider.is_available())
    except Exception as exc:  # noqa: BLE001
        logger.log(level, fmt, provider.name, exc, exc_info=exc_info)
        return False


def configured_provider_name(section: str, logger: logging.Logger) -> Optional[str]:
    """Read ``<section>.provider`` from config.yaml, mapping the managed Nous
    selection to ``fal`` (the FAL plugin services it via the managed gateway)."""
    configured: Optional[str] = None
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        block = cfg.get(section) if isinstance(cfg, dict) else None
        raw = block.get("provider") if isinstance(block, dict) else None
        if isinstance(raw, str) and raw.strip():
            configured = raw.strip()
    except Exception as exc:
        logger.debug("Could not read %s.provider from config: %s", section, exc)
    if configured:
        try:
            from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER
            if configured.lower() == NOUS_MANAGED_PROVIDER:
                configured = "fal"
        except Exception:  # pragma: no cover — helpers are in-repo
            pass
    return configured

"""Pure plugin activation decisions.

Configuration I/O belongs to :mod:`hermes_cli.config`.  This module only
normalizes the already-loaded ``plugins`` section and applies the shared
source/kind policy, so runtime discovery and management surfaces cannot drift
without introducing another config reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


_BUNDLED_DEFAULT_KINDS = frozenset({"backend", "platform", "model-provider"})


def _string_set(value: Any) -> frozenset[str]:
    """Return non-empty string entries from a config list."""
    if not isinstance(value, list):
        return frozenset()
    return frozenset(
        item.strip() for item in value if isinstance(item, str) and item.strip()
    )


@dataclass(frozen=True)
class PluginActivationState:
    """Immutable activation inputs derived from canonical config."""

    enabled: frozenset[str] | None = None
    disabled: frozenset[str] = frozenset()
    safe_mode: bool = False

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        safe_mode: bool = False,
    ) -> "PluginActivationState":
        plugins = config.get("plugins") if isinstance(config, dict) else None
        if not isinstance(plugins, dict):
            return cls(safe_mode=safe_mode)

        raw_enabled = plugins.get("enabled")
        enabled = _string_set(raw_enabled) if isinstance(raw_enabled, list) else None
        return cls(
            enabled=enabled,
            disabled=_string_set(plugins.get("disabled")),
            safe_mode=safe_mode,
        )

    @staticmethod
    def identities(
        name: str = "",
        key: str = "",
        aliases: Iterable[str] = (),
    ) -> frozenset[str]:
        """Return the canonical config identities accepted for one plugin."""
        values = (name, key, *aliases)
        return frozenset(
            value.strip()
            for value in values
            if isinstance(value, str) and value.strip()
        )

    def status(
        self,
        *,
        name: str = "",
        key: str = "",
        source: str,
        kind: str,
        aliases: Iterable[str] = (),
    ) -> str:
        """Return ``enabled``, ``disabled``, or ``not enabled``."""
        identities = self.identities(name, key, aliases)
        if identities & self.disabled:
            return "disabled"

        normalized_source = (source or "").strip().lower()
        normalized_kind = (kind or "standalone").strip().lower()
        if self.safe_mode:
            # General PluginManager discovery is disabled wholesale in safe
            # mode. Bundled model profiles remain available so Hermes can
            # still resolve the explicitly selected inference provider.
            if not (
                normalized_source == "bundled" and normalized_kind == "model-provider"
            ):
                return "not enabled"
        if normalized_source == "bundled" and normalized_kind in _BUNDLED_DEFAULT_KINDS:
            return "enabled"
        if self.enabled is not None and identities & self.enabled:
            return "enabled"
        return "not enabled"

    def is_active(self, **plugin: Any) -> bool:
        """Return whether a plugin may execute under this state."""
        return self.status(**plugin) == "enabled"

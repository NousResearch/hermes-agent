"""Runtime configuration resolution for the Discord adapter."""

from __future__ import annotations

import math
from typing import Any, Optional


class RuntimeConfigMixin:
    """Resolve profile-aware runtime settings without owning adapter lifecycle."""

    def _config_value(self, key: str, default: Any, *, env_key: Optional[str] = None) -> Any:
        """Resolve a liveness value from profile config, legacy env, or default."""
        extra = self.config.extra if isinstance(getattr(self.config, "extra", None), dict) else {}
        value = extra.get(key)
        if value is None and env_key:
            from .. import adapter as _adapter
            value = _adapter._scoped_gate_env(env_key) or None
        return default if value is None or value == "" else value

    def _finite_positive_config_float(
        self, key: str, default: float, *, env_key: Optional[str] = None
    ) -> float:
        """Resolve a finite positive liveness duration; invalid values disable it."""
        try:
            value = float(self._config_value(key, default, env_key=env_key))
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) and value > 0 else 0.0

    def _config_int(self, key: str, default: int, *, env_key: Optional[str] = None) -> int:
        """Resolve a positive liveness count; invalid values disable it."""
        value = self._config_value(key, default, env_key=env_key)
        if isinstance(value, bool):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

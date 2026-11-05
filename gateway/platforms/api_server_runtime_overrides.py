"""Merge resolved provider/runtime fields into the agent kwargs ``api_server`` builds per request.

Lifted from ``platforms/api_server.py`` to keep that facade under its frozen line cap (root
AGENTS.md god-file guidance: move code into a topical sibling).
"""

from typing import Any, Optional

_RUNTIME_AGENT_OVERRIDE_KEYS = (
    "api_key", "base_url", "provider", "api_mode", "command", "args", "credential_pool")


def _apply_runtime_agent_overrides(
    runtime_kwargs: dict[str, Any], overrides: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Merge resolved provider/runtime fields into ``runtime_kwargs`` in place."""
    if not isinstance(overrides, dict):
        return runtime_kwargs
    for key in _RUNTIME_AGENT_OVERRIDE_KEYS:
        value = overrides.get(key)
        if value is None:
            continue
        runtime_kwargs[key] = list(value) if key == "args" and isinstance(value, (list, tuple)) else value
    return runtime_kwargs

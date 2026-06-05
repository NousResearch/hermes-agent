"""Helpers for interpreting memory-related config."""

from __future__ import annotations

from typing import Any, Mapping


def builtin_memory_tool_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return whether the built-in global memory tool should be exposed.

    The option defaults to enabled for backward compatibility.  Only an
    explicit ``memory.builtin_tool.enabled: false`` disables the tool.
    """
    if not isinstance(config, Mapping):
        return True
    memory_config = config.get("memory", {})
    if not isinstance(memory_config, Mapping):
        return True
    builtin_tool = memory_config.get("builtin_tool", {})
    if not isinstance(builtin_tool, Mapping):
        return True
    return bool(builtin_tool.get("enabled", True))

"""Lightweight registry for the active LinearAgentAdapter instance.

This allows the linear_agent mutation tools to reach the live authenticated
client without importing gateway internals or creating circular dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from tools.registry import registry as _tool_registry

if TYPE_CHECKING:
    from .adapter import LinearAgentAdapter

# The active adapter is stored as an attribute on the shared tool registry
# object rather than in a module-level global. This plugin can be imported
# under two module identities — ``plugins.platforms.linear_agent.*`` (direct
# imports) and ``hermes_plugins.linear_agent.*`` (Hermes plugin discovery) —
# and a module global would exist once per identity, so the adapter stored by
# one copy would be invisible to tool handlers registered by the other.
# ``tools.registry`` is always imported by its canonical name, so its
# registry object is a process-wide singleton both copies share.
_ADAPTER_ATTR = "_linear_agent_active_adapter"


def set_active_adapter(adapter: Optional["LinearAgentAdapter"]) -> None:
    """Register (or clear) the currently connected LinearAgentAdapter."""
    setattr(_tool_registry, _ADAPTER_ATTR, adapter)


def get_active_adapter() -> "LinearAgentAdapter":
    """Return the currently connected LinearAgentAdapter.

    Raises RuntimeError if no Linear Agent platform is active.
    """
    adapter = getattr(_tool_registry, _ADAPTER_ATTR, None)
    if adapter is None:
        raise RuntimeError(
            "linear_agent platform is not currently connected. "
            "Make sure the Linear Agent platform is enabled and the gateway is running."
        )
    return adapter

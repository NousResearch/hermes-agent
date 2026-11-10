"""Connector integration boundary for managed gateway accounts and local MCP servers.

Only the names below are cross-package surface; imports beyond it need a design decision.
Siblings: ``contract`` (states, actors, transition table), ``operation`` (the record),
``live`` (open operation per session), ``run`` (the lifecycle loop), ``managed`` / ``mcp``
(per-kind hooks), ``targets``, ``search``, ``dispatch``, ``gateway/`` (HTTP wire + client).

The surface is exported lazily (PEP 562): eager submodule imports here deadlocked the
concurrent registry scan and the Group Chat worker (see test_connectors_import_deadlock.py).
"""

import importlib

# name -> module owning it: the single source of truth for __all__ and __getattr__.
_LAZY_EXPORTS: dict[str, str] = {
    "CONNECTOR_BATCH_SENTINEL": "hermes_agent.tools.connectors.gateway.names",
    "MANAGE_CONNECTIONS_SCHEMA": "hermes_agent.tools.connectors.tool",
    "connector_describe": "hermes_agent.tools.connectors.gateway.bridge",
    "connector_search_hits": "hermes_agent.tools.connectors.gateway.bridge",
    "connectors_available": "hermes_agent.tools.connectors.gateway.config",
    "dispatch_connector_batch": "hermes_agent.tools.connectors.dispatch",
    "dispatch_connector_call": "hermes_agent.tools.connectors.dispatch",
    "ensure_guest_identity": "hermes_agent.tools.connectors.gateway.config",
    "guest_identity_pending": "hermes_agent.tools.connectors.gateway.config",
    "is_connector_name": "hermes_agent.tools.connectors.gateway.names",
    "manage_connections": "hermes_agent.tools.connectors.tool",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Resolve and cache a public name on first access (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

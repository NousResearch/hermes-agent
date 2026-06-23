"""Helpers for configured MCP server toolset names.

MCP discovery registers runtime toolsets as ``mcp-<server>`` and also exposes
bare server-name aliases. Several startup paths validate explicit toolset lists
before discovery has run, so they need a config-only view of both spellings.
"""

from __future__ import annotations

from collections.abc import Mapping


def _parse_enabled_flag(value: object, default: bool = True) -> bool:
    """Parse bool-like config values used by MCP server settings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _server_enabled(server_cfg: object) -> bool:
    """Return whether an MCP server config is enabled by config semantics."""
    if isinstance(server_cfg, Mapping):
        return _parse_enabled_flag(server_cfg.get("enabled", True), default=True)
    return True


def mcp_toolset_aliases_for_servers(
    mcp_servers: object,
    *,
    enabled: bool | None = None,
) -> set[str]:
    """Return configured MCP toolset spellings for startup validation.

    Args:
        mcp_servers: The ``config.yaml`` ``mcp_servers`` mapping.
        enabled: ``True`` for only enabled servers, ``False`` for only disabled
            servers, ``None`` for every configured server.

    Each configured server contributes both accepted toolset spellings:
    ``<server>`` and ``mcp-<server>``.  The latter is the runtime toolset name
    registered by MCP discovery; the former is the historical alias supported
    by earlier startup validators.
    """
    if not isinstance(mcp_servers, Mapping):
        return set()

    aliases: set[str] = set()
    for name, server_cfg in mcp_servers.items():
        name = str(name)
        is_enabled = _server_enabled(server_cfg)
        if enabled is not None and is_enabled is not enabled:
            continue
        aliases.add(name)
        aliases.add(f"mcp-{name}")
    return aliases


def split_configured_mcp_toolset_aliases(
    mcp_servers: object,
) -> tuple[set[str], set[str]]:
    """Return ``(enabled_aliases, disabled_aliases)`` for configured MCP servers."""
    return (
        mcp_toolset_aliases_for_servers(mcp_servers, enabled=True),
        mcp_toolset_aliases_for_servers(mcp_servers, enabled=False),
    )


def canonical_mcp_toolset_name(name: str) -> str:
    """Return the runtime MCP toolset name for either accepted spelling."""
    return name if name.startswith("mcp-") else f"mcp-{name}"

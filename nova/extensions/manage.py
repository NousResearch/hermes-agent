"""Granting and revoking MCP servers and plugins — by editing the bundle.

Same shape as :mod:`nova.agents.manage`, and for the same reason. The profile's
``config.yaml`` is *derived*: ``nova apply`` rewrites it from the bundle every time it
runs. A Control Centre that wrote ``mcp_servers`` straight into that file would produce a
grant which saves, persists, survives a reload — and silently disappears at the next apply.
So every function here transforms the agent's declaration and lets materialization compile
it, which is what makes the grant durable.

Each edit goes through :func:`nova.spec.writer.edit`, so the whole bundle is validated
before anything lands. Granting a server this deployment does not ship fails here with the
same message it would have failed with had somebody written the YAML by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nova.agents.manage import agent_file
from nova.errors import SpecError
from nova.spec.writer import BundleEdit, edit, validate_id

#: What may be granted or revoked, and nothing else.
ACTIONS = ("grant", "revoke")


def _extensions_block(document: dict[str, Any]) -> dict[str, Any]:
    block = document.get("extensions")
    return dict(block) if isinstance(block, dict) else {}


def _plugins_block(block: dict[str, Any]) -> dict[str, Any]:
    plugins = block.get("plugins")
    return dict(plugins) if isinstance(plugins, dict) else {}


def _prune(document: dict[str, Any], block: dict[str, Any]) -> None:
    """Write the block back, or remove it when it says nothing.

    An empty ``extensions: {}`` is noise in a file an operator reads by hand, and it is
    also a statement — "this agent has extensions configured" — that is not true.
    """
    plugins = block.get("plugins")
    if isinstance(plugins, dict) and not any(plugins.values()):
        block.pop("plugins", None)
    if any(block.values()):
        document["extensions"] = block
    else:
        document.pop("extensions", None)


def _require_agent(e: BundleEdit, agent_id: str) -> dict[str, Any]:
    document = e.read_yaml(agent_file(agent_id))
    if not document:
        raise SpecError(f"no agent {agent_id!r} in this bundle")
    return document


def set_mcp(root: Path, agent_id: str, server_id: str, *, granted: bool):
    """Grant or revoke one catalogue MCP server for one agent.

    The server id is checked against the runtime's catalogue by the bundle loader, not
    here — validating in two places would eventually mean validating differently in two
    places, and the loader is the one that also runs for a hand-edited file.
    """
    agent_id = validate_id(agent_id, what="agent id")
    server_id = str(server_id or "").strip()
    if not server_id:
        raise SpecError("name the MCP server to grant")

    def mutate(e: BundleEdit) -> None:
        document = _require_agent(e, agent_id)
        block = _extensions_block(document)
        current = [str(name) for name in (block.get("mcp") or [])]
        if granted:
            if server_id not in current:
                current.append(server_id)
        else:
            current = [name for name in current if name != server_id]
        if current:
            block["mcp"] = current
        else:
            block.pop("mcp", None)
        _prune(document, block)
        e.write_yaml(agent_file(agent_id), document)

    return edit(root, mutate)


def set_plugin(root: Path, agent_id: str, plugin_id: str, *, state: str):
    """Put one plugin into ``enable``, into ``disable``, or into neither.

    Three states rather than two, because the runtime has three. A bundled backend loads
    unless it is disabled, so "not listed" and "disabled" are genuinely different answers
    for it, and collapsing them into a checkbox would make one of them unreachable.
    """
    agent_id = validate_id(agent_id, what="agent id")
    plugin_id = str(plugin_id or "").strip()
    if not plugin_id:
        raise SpecError("name the plugin")
    if state not in ("enable", "disable", "default"):
        raise SpecError(
            f"{state!r} is not a plugin state. Use 'enable', 'disable' or 'default'"
        )

    def mutate(e: BundleEdit) -> None:
        document = _require_agent(e, agent_id)
        block = _extensions_block(document)
        plugins = _plugins_block(block)
        for key in ("enable", "disable"):
            names = [str(name) for name in (plugins.get(key) or []) if str(name) != plugin_id]
            if key == state:
                names.append(plugin_id)
            if names:
                plugins[key] = names
            else:
                plugins.pop(key, None)
        block["plugins"] = plugins
        _prune(document, block)
        e.write_yaml(agent_file(agent_id), document)

    return edit(root, mutate)


def grants_of(root: Path, agent_id: str) -> dict[str, Any]:
    """What this agent is currently granted, read back from the loaded bundle."""
    from nova.spec import load_bundle

    agent_id = validate_id(agent_id, what="agent id")
    bundle = load_bundle(Path(root))
    for agent in bundle.agents:
        if agent.id == agent_id:
            return {
                "agent_id": agent_id,
                "mcp": list(agent.extensions.mcp),
                "plugins": {
                    "enable": list(agent.extensions.plugins_enable),
                    "disable": list(agent.extensions.plugins_disable),
                },
            }
    raise SpecError(f"no agent {agent_id!r} in this bundle")

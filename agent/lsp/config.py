"""Build profile-local LSP definitions without mutating the built-in registry."""
from __future__ import annotations

import logging
import os
from typing import Any

from agent.lsp.servers import SERVERS, ServerContext, ServerDef, SpawnSpec, _make_spec
from agent.lsp.workspace import nearest_root

logger = logging.getLogger(__name__)


def _string_list(value: Any, *, allow_empty: bool = False) -> bool:
    return (isinstance(value, list) and (allow_empty or bool(value))
            and all(isinstance(v, str) and bool(v.strip()) for v in value))


def _custom_server(server_id: str, config: dict) -> ServerDef:
    command = tuple(config["command"])
    markers = tuple(config.get("root_markers", []))

    def build_spawn(root: str, ctx: ServerContext) -> SpawnSpec | None:
        from agent.lsp.install import _existing_binary
        binary = _existing_binary(os.path.expanduser(command[0]))
        if binary is None:
            return None
        return _make_spec(root, ctx, server_id, [binary, *command[1:]])

    return ServerDef(
        server_id=server_id,
        extensions=tuple(ext.lower() if ext.startswith(".") else ext for ext in config["extensions"]),
        resolve_root=lambda path, workspace: nearest_root(path, markers, ceiling=workspace) or workspace,
        build_spawn=build_spawn,
        description="Custom language server",
        language_id=config.get("language_id", ""),
    )


def configured_servers(servers_config: Any = None) -> list[ServerDef]:
    """Custom servers in YAML order take precedence; built-ins remain available.

    Called once when a service starts (or for a CLI inspection). Invalid custom
    entries cannot take down unrelated language servers or the post-write hook.
    """
    if servers_config is None:
        from hermes_cli.config import load_config_readonly
        lsp_config = load_config_readonly().get("lsp")
        servers_config = lsp_config.get("servers", {}) if isinstance(lsp_config, dict) else {}
    if not isinstance(servers_config, dict):
        return list(SERVERS)
    builtin_ids = {server.server_id for server in SERVERS}
    custom = []
    for server_id, config in servers_config.items():
        if server_id in builtin_ids:
            continue
        if not isinstance(server_id, str) or not server_id.strip() or not isinstance(config, dict):
            logger.warning("Ignoring lsp.servers.%s: expected a named server mapping", server_id)
            continue
        invalid = next((key for key, valid in {
            "command": _string_list(config.get("command")),
            "extensions": _string_list(config.get("extensions")),
            "root_markers": _string_list(config.get("root_markers", []), allow_empty=True),
            "language_id": isinstance(config.get("language_id", ""), str),
        }.items() if not valid), None)
        if invalid:
            logger.warning("Ignoring lsp.servers.%s: invalid %s", server_id, invalid)
            continue
        custom.append(_custom_server(server_id, config))
    return custom + list(SERVERS)

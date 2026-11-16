"""Keep the default profile's MCP servers away from secondary profiles (4c).

MCP server processes run once per gateway, started from the **default**
profile's ``mcp_servers`` with the default's credentials (Webflow, Stripe,
GitHub, …). Under ``gateway.multiplex_profiles`` a secondary profile whose
role grants ``*`` or an ``mcp-*`` toolset would otherwise act with the
default bot's credentials.

So, in a secondary profile's context, an MCP toolset survives only when that
profile's own ``config.yaml`` declares the server. Applied twice: when the
turn's toolsets are resolved (the model is never offered the tools) and at
dispatch (``model_tools`` backstop, fail closed). Inert with multiplex off
and for the default profile.

See docs/superpowers/specs/2026-11-16-multiplex-fork-hardening-design.md.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

MCP_TOOLSET_PREFIX = "mcp-"
MCP_TOOL_PREFIX = "mcp__"

_cfg_lock = threading.Lock()
_cfg_cache: Dict[str, Tuple[Tuple[int, int], Set[str]]] = {}


def is_secondary_profile_context() -> bool:
    """True when multiplexing is on and the current home is not the process's."""
    from agent.secret_scope import is_multiplex_active

    if not is_multiplex_active():
        return False
    from hermes_constants import get_process_hermes_home, hermes_home_key

    return hermes_home_key() != hermes_home_key(get_process_hermes_home())


def profile_declared_mcp_servers(home: Optional[Path] = None) -> Set[str]:
    """``mcp_servers`` keys from the current profile's own config.yaml."""
    from hermes_constants import get_hermes_home

    cfg = Path(home or get_hermes_home()) / "config.yaml"
    try:
        st = cfg.stat()
    except OSError:
        return set()
    key = str(cfg.resolve())
    fp = (st.st_mtime_ns, st.st_size)
    with _cfg_lock:
        hit = _cfg_cache.get(key)
        if hit is not None and hit[0] == fp:
            return set(hit[1])
    import yaml

    data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    servers = data.get("mcp_servers") if isinstance(data, dict) else None
    names = {str(k) for k in servers} if isinstance(servers, dict) else set()
    with _cfg_lock:
        _cfg_cache[key] = (fp, names)
    return set(names)


def mcp_server_for_toolset(toolset: str) -> Optional[str]:
    """Server name behind an MCP toolset (``mcp-<x>`` or its alias), else None."""
    name = str(toolset or "")
    if name.startswith(MCP_TOOLSET_PREFIX):
        return name[len(MCP_TOOLSET_PREFIX):]
    try:
        from tools.registry import registry

        target = registry.get_toolset_alias_target(name)
    except Exception:
        target = None
    if target and str(target).startswith(MCP_TOOLSET_PREFIX):
        return str(target)[len(MCP_TOOLSET_PREFIX):]
    return None


def _declared(server: str, declared: Set[str]) -> bool:
    if server in declared:
        return True
    # Toolset names keep the raw server name; tolerate the -/_ spelling split
    # that sanitize_mcp_name_component introduces in tool prefixes.
    return server.replace("-", "_") in {d.replace("-", "_") for d in declared}


def filter_foreign_mcp_toolsets(toolsets: Iterable[str]) -> List[str]:
    """Drop MCP toolsets a secondary profile does not declare itself.

    Returns the input unchanged (as a list) with multiplex off or in the
    default profile's context.
    """
    items = list(toolsets)
    if not is_secondary_profile_context():
        return items
    declared = profile_declared_mcp_servers()
    kept: List[str] = []
    for ts in items:
        server = mcp_server_for_toolset(ts)
        if server is not None and not _declared(server, declared):
            logger.info(
                "multiplex: dropping default-profile MCP toolset %r from a "
                "secondary profile's turn (not in its mcp_servers)",
                ts,
            )
            continue
        kept.append(ts)
    return kept


def foreign_mcp_denial(tool_name: str) -> Optional[str]:
    """Dispatch backstop: deny a default-profile MCP tool in a secondary profile.

    Fails closed — any internal error on an ``mcp__`` tool in a secondary
    context denies.
    """
    try:
        if not is_secondary_profile_context():
            return None
    except Exception:
        return None
    try:
        from tools.registry import registry

        toolset = registry.get_toolset_for_tool(tool_name)
        server = mcp_server_for_toolset(toolset) if toolset else None
        if server is None:
            if str(tool_name).startswith(MCP_TOOL_PREFIX):
                server = str(tool_name)[len(MCP_TOOL_PREFIX):].split("__", 1)[0]
            else:
                return None
        if _declared(server, profile_declared_mcp_servers()):
            return None
    except Exception:
        if not str(tool_name).startswith(MCP_TOOL_PREFIX):
            return None
        logger.warning("multiplex MCP guard failed for %s; denying", tool_name, exc_info=True)
    return (
        f"⛔ '{tool_name}' belongs to an MCP server this profile does not "
        "configure. It is not available here."
    )

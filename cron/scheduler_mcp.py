"""Bounded observation of MCP recovery before a scheduled agent snapshots tools."""

import logging
import time

logger = logging.getLogger(__name__)


def _registered_and_ready(name, server, config):
    from tools.mcp_tool_registration import _make_tool_filter
    from tools.registry import registry

    # A session is published before tools/list finishes; ready alone can also
    # survive a transport failure. Check the current profile's dispatch entries.
    names = list(server._registered_tool_names)
    include = _make_tool_filter(name, config)
    return (
        server.session is not None
        and server._ready.is_set()
        and not server._reconnect_event.is_set()
        and (names or not any(include(tool.name) for tool in server._tools))
        and all(
            (entry := registry.get_entry(tool_name)) is not None
            and callable(entry.handler)
            for tool_name in names
        )
    )


def wait_for_recovery(deadline: float, job_id: str) -> list[str]:
    """Observe discovery's reconnect, never request another one.

    All eligible servers share the caller's deadline, including time already
    spent in discovery. Cold discovery retains its existing independent bounds.
    """
    from tools import mcp_tool as core
    from tools.mcp_tool_common import mcp_server_enabled
    from tools.mcp_tool_discovery import _config, _owner_secret_scope
    from tools.mcp_tool_registration import _existing_tool_names
    from tools.mcp_tool_scope import _resolve_server_key

    with _owner_secret_scope():
        configured = _config._load_mcp_config() or {}
    with core._lock:
        recovering = [
            (name, server)
            for name, config in configured.items()
            if mcp_server_enabled(config)
            and (server := core._servers.get(_resolve_server_key(name))) is not None
            and server._ever_connected
            and not (server._park_reason and "permanent" in server._park_reason)
        ]
    while recovering:
        recovering = [
            (name, server)
            for name, server in recovering
            if not _registered_and_ready(name, server, configured[name])
            and server._task is not None
            and not server._task.done()
            and not (server._park_reason and "permanent" in server._park_reason)
        ]
        remaining = deadline - time.monotonic()
        if not recovering or remaining <= 0:
            break
        time.sleep(min(0.05, remaining))
    if recovering:
        logger.warning(
            "Job '%s': MCP recovery deadline reached: %s",
            job_id,
            ", ".join(name for name, _ in recovering),
        )
    # Another discovery could wake a transport again. Selection stays with the agent.
    return _existing_tool_names()

"""Shared CLI/TUI-safe helpers for background MCP discovery."""

from __future__ import annotations

import threading
from typing import Optional

_mcp_discovery_lock = threading.Lock()
_mcp_discovery_started = False
_mcp_discovery_thread: Optional[threading.Thread] = None


def _has_configured_mcp_servers() -> bool:
    """Cheap config probe so non-MCP users avoid importing the MCP stack."""
    try:
        from hermes_cli.config import read_raw_config

        mcp_servers = (read_raw_config() or {}).get("mcp_servers")
        return isinstance(mcp_servers, dict) and len(mcp_servers) > 0
    except Exception:
        # Be conservative: if config probing fails, try discovery in the
        # background so startup still can't block.
        return True


def start_background_mcp_discovery(*, logger, thread_name: str) -> None:
    """Spawn one shared background MCP discovery thread for this process."""
    global _mcp_discovery_started, _mcp_discovery_thread

    with _mcp_discovery_lock:
        if _mcp_discovery_started:
            return
        _mcp_discovery_started = True
        if not _has_configured_mcp_servers():
            return

        def _discover() -> None:
            try:
                from tools.mcp_tool import discover_mcp_tools

                discover_mcp_tools()
            except Exception:
                logger.debug("Background MCP tool discovery failed", exc_info=True)

        thread = threading.Thread(
            target=_discover,
            name=thread_name,
            daemon=True,
        )
        _mcp_discovery_thread = thread
        thread.start()


def wait_for_mcp_discovery(timeout: float = 0.75) -> None:
    """Briefly wait for background MCP discovery before the first tool snapshot.

    Kanban workers (detected via ``HERMES_KANBAN_TASK``) automatically use a
    longer timeout because they depend on MCP tools for task execution and run
    in the background where a blocking wait is acceptable.  See #43273.
    """
    import os

    thread = _mcp_discovery_thread
    if thread is None or not thread.is_alive():
        return
    # Kanban workers can afford to wait longer — MCP tools are critical
    # for task execution and the worker is a background subprocess.
    if os.environ.get("HERMES_KANBAN_TASK"):
        timeout = max(timeout, 60.0)
    thread.join(timeout=timeout)

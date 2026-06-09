"""Shared CLI/TUI-safe helpers for background MCP discovery."""

from __future__ import annotations

import os
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


def wait_for_mcp_discovery(timeout: float | None = None) -> None:
    """Briefly wait for background MCP discovery before the first tool snapshot.

    Timeout resolution order:
      1. Explicit ``timeout`` argument (callers like CLI can override)
      2. ``HERMES_MCP_DISCOVERY_TIMEOUT`` env var (seconds, user-tunable)
      3. Default 0.75s (original fast-start heuristic)
    """
    if timeout is None:
        try:
            timeout = float(os.environ.get("HERMES_MCP_DISCOVERY_TIMEOUT", "0.75"))
        except (TypeError, ValueError):
            timeout = 0.75
    thread = _mcp_discovery_thread
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=timeout)


# ── Late-binding MCP tool refresh ─────────────────────────────────────
# After the agent is built (with whatever tools were ready at the time),
# spawn a background thread that waits for MCP discovery to finish.  When
# it does, check whether new tools appeared and auto-refresh the agent's
# tool list.  This is the same logic as ``/reload-mcp`` but triggered
# automatically so slow MCP servers (lark ~8s, redis ~10s) land in the
# agent without blocking startup or requiring manual intervention.

_mcp_late_refresh_thread: Optional[threading.Thread] = None


def spawn_late_mcp_refresh(
    *,
    agent,
    logger,
    get_tool_definitions_fn: callable,
    on_refreshed: "callable | None" = None,
) -> None:
    """Spawn a background thread that auto-refreshes MCP tools once discovery finishes.

    Args:
        agent: The AIAgent instance whose ``.tools`` and ``.valid_tool_names``
               will be updated in-place.
        logger: Logger instance for debug/warning output.
        get_tool_definitions_fn: A callable that returns the current tool
                                 definitions list (e.g. ``model_tools.get_tool_definitions``).
        on_refreshed: Optional callback invoked after tools are refreshed.
                      Receives ``(added_count: int, total_count: int)``.
    """
    global _mcp_late_refresh_thread

    # Only spawn if discovery is still running
    thread = _mcp_discovery_thread
    if thread is None or not thread.is_alive():
        return

    # Avoid spawning multiple late-refresh threads
    with _mcp_discovery_lock:
        if _mcp_late_refresh_thread is not None and _mcp_late_refresh_thread.is_alive():
            return

    def _refresh() -> None:
        try:
            # Wait for discovery to fully complete (up to 120s)
            discovery_thread = _mcp_discovery_thread
            if discovery_thread is not None:
                discovery_thread.join(timeout=120.0)

            if discovery_thread and discovery_thread.is_alive():
                logger.debug("MCP discovery still running after 120s, skipping late refresh")
                return

            # Snapshot current tools from the registry
            current_tools = set()
            if hasattr(agent, "tools") and agent.tools:
                current_tools = {t["function"]["name"] for t in agent.tools}

            # Get fresh tool definitions from the registry
            new_defs = get_tool_definitions_fn(quiet_mode=True)
            new_tool_names = {t["function"]["name"] for t in new_defs} if new_defs else set()

            # Only update if new tools appeared
            added = new_tool_names - current_tools
            if not added:
                logger.debug("MCP late refresh: no new tools discovered")
                return

            # Update agent in-place (same as /reload-mcp)
            agent.tools = new_defs
            agent.valid_tool_names = new_tool_names

            logger.info(
                "MCP late refresh: %d new tool(s) added (%s)",
                len(added),
                ", ".join(sorted(added)[:5]) + ("..." if len(added) > 5 else ""),
            )

            if on_refreshed:
                try:
                    on_refreshed(len(added), len(new_tool_names))
                except Exception:
                    pass

        except Exception as exc:
            logger.debug("MCP late refresh failed: %s", exc)

    late_thread = threading.Thread(
        target=_refresh,
        name="mcp-late-refresh",
        daemon=True,
    )
    _mcp_late_refresh_thread = late_thread
    late_thread.start()

"""Shared CLI/TUI-safe helpers for background MCP discovery."""

from __future__ import annotations

import threading
from contextlib import nullcontext
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
                _discover_mcp_tools_without_interactive_oauth()
            except Exception:
                logger.debug("Background MCP tool discovery failed", exc_info=True)

        thread = threading.Thread(
            target=_discover,
            name=thread_name,
            daemon=True,
        )
        _mcp_discovery_thread = thread
        thread.start()


def _discover_mcp_tools_without_interactive_oauth() -> None:
    from tools.mcp_oauth import suppress_interactive_oauth
    from tools.mcp_tool import discover_mcp_tools

    with suppress_interactive_oauth():
        discover_mcp_tools()


def _resolve_discovery_timeout(explicit: "float | None") -> float:
    """Resolve the MCP discovery wait bound: explicit arg > config > default.

    Reads ``mcp_discovery_timeout`` from config.yaml, defaulting to the value in
    ``DEFAULT_CONFIG`` (single source of truth) when the key is absent. Kept lazy
    and fail-safe — a missing/invalid value or a broken config falls back to a
    short safe bound so startup can never hang or crash.
    """
    if explicit is not None:
        return explicit
    try:
        from hermes_cli.config import load_config, DEFAULT_CONFIG

        default = float(DEFAULT_CONFIG.get("mcp_discovery_timeout", 1.5))
        raw = (load_config() or {}).get("mcp_discovery_timeout", default)
        val = float(raw)
        return val if val > 0 else default
    except Exception:
        return 1.5


def _discover_mcp_tools_without_interactive_oauth() -> None:
    """Run MCP discovery without letting OAuth read from the user's stdin."""
    try:
        from tools.mcp_oauth import suppress_interactive_oauth
    except Exception:
        suppress_interactive_oauth = nullcontext

    with suppress_interactive_oauth():
        from tools.mcp_tool import discover_mcp_tools

        discover_mcp_tools()


def wait_for_mcp_discovery(timeout: "float | None" = None) -> None:
    """Wait for background MCP discovery before the first tool snapshot.

    ``thread.join(timeout)`` returns the INSTANT discovery completes, so this
    only ever blocks for the real connect time of a still-pending server —
    users with no MCP servers or fast servers pay ~0s.  The bound (from
    ``mcp_discovery_timeout`` in config) just caps the wait so a dead server
    can't freeze startup; servers that miss it are picked up by the automatic
    late-binding refresh.
    """
    thread = _mcp_discovery_thread
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=_resolve_discovery_timeout(timeout))

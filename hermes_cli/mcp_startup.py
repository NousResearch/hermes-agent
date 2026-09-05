"""Shared CLI/TUI-safe helpers for background MCP discovery."""

from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Optional

from hermes_constants import get_hermes_home_override, reset_hermes_home_override, set_hermes_home_override

_mcp_discovery_lock = threading.Lock()
_mcp_discovery_started = False
_mcp_discovery_thread: Optional[threading.Thread] = None
_mcp_discovery_deferred: Optional[threading.Timer] = None
# Process-wide MCP server-name allowlist derived from ``-t/--toolsets``.
# ``None`` = no filter (spawn every configured server). Set once at CLI
# startup by ``set_mcp_server_filter`` and honored by every discovery path
# in this module (inline, background, deferred), so a ``-t terminal``
# oneshot never cold-starts MCP subprocesses it cannot use.
_mcp_server_filter: Optional[list[str]] = None


def set_mcp_server_filter(toolsets: object) -> Optional[list[str]]:
    """Derive the MCP spawn allowlist from a ``-t/--toolsets`` value.

    Built-in toolset names in the list are harmless (they never match a
    configured ``mcp_servers`` key). ``all``/``*`` or an empty/absent value
    clears the filter. Returns the stored list for logging/tests.
    """
    global _mcp_server_filter
    names: list[str] = []
    if isinstance(toolsets, str):
        names = [t.strip() for t in toolsets.split(",") if t.strip()]
    elif isinstance(toolsets, (list, tuple, set)):
        for item in toolsets:
            names.extend(t.strip() for t in str(item).split(",") if t.strip())
    if not names or "all" in names or "*" in names:
        _mcp_server_filter = None
    else:
        _mcp_server_filter = names
    return _mcp_server_filter


def get_mcp_server_filter() -> Optional[list[str]]:
    return _mcp_server_filter


def _has_configured_mcp_servers() -> bool:
    """Cheap config probe so non-MCP users avoid importing the MCP stack."""
    try:
        from hermes_cli.config import read_raw_config

        raw_config = read_raw_config() or {}
        if isinstance(raw_config.get("mcp_servers"), dict) and raw_config["mcp_servers"]:
            return True
        from hermes_cli.agent_plugins import has_enabled_agent_plugin_mcp

        return has_enabled_agent_plugin_mcp(raw_config)
    except Exception:
        return True  # conservative: still try discovery in the background; startup can't block


def _any_mcp_connected() -> bool:
    from tools.mcp_tool_discovery import get_mcp_status

    return any(entry.get("connected") for entry in (get_mcp_status() or []))


def start_background_mcp_discovery(*, logger, thread_name: str) -> None:
    """Spawn one shared background MCP discovery thread for this process.

    If the first run exits without connecting any server (e.g. startup cancellation / OOM restart),
    later calls may retry instead of pinning the process in "already started" with zero MCP tools.
    """
    global _mcp_discovery_started, _mcp_discovery_thread

    with _mcp_discovery_lock:
        if _mcp_discovery_started:
            thread = _mcp_discovery_thread
            if thread is not None and thread.is_alive():
                return
            try:
                if _any_mcp_connected():
                    return
            except Exception:
                return
            logger.warning(
                "Background MCP discovery previously exited with no connected "
                "servers; retrying discovery thread"
            )
            _mcp_discovery_started = False
            _mcp_discovery_thread = None

        _mcp_discovery_started = True
        if not _has_configured_mcp_servers():
            return

        # Re-install the caller's context-local HERMES_HOME override (multi-profile dashboard/desktop
        # backends) inside the thread: ContextVars don't propagate into bare threads, so a session
        # switched to profile X would otherwise discover the LAUNCH profile's mcp_servers.
        # The config gate above already runs on the caller's thread, so it sees the same override. See
        # #67605.
        home_override = get_hermes_home_override()

        def _discover() -> None:
            token = set_hermes_home_override(home_override)
            try:
                _discover_mcp_tools_without_interactive_oauth()
                try:
                    if not _any_mcp_connected():
                        logger.warning("Background MCP discovery completed with zero connected servers")
                except Exception:
                    logger.debug("Failed to inspect MCP status after background discovery", exc_info=True)
            except Exception:
                logger.debug("Background MCP tool discovery failed", exc_info=True)
            finally:
                reset_hermes_home_override(token)
                with _mcp_discovery_lock:
                    global _mcp_discovery_thread
                    _mcp_discovery_thread = None

        thread = threading.Thread(target=_discover, name=thread_name, daemon=True)
        _mcp_discovery_thread = thread
        thread.start()


def _resolve_discovery_timeout(explicit: "float | None", *, single_query: bool = False) -> float:
    """Resolve the MCP discovery wait bound: explicit arg > config.yaml > ``DEFAULT_CONFIG``.

    Lazy and fail-safe: a missing/invalid value or broken config falls back to a short bound so
    startup can never hang or crash.
    """
    if explicit is not None:
        return explicit
    key = "mcp_single_query_discovery_timeout" if single_query else "mcp_discovery_timeout"
    fallback = 15.0 if single_query else 1.5
    try:
        from hermes_cli.config import load_config, DEFAULT_CONFIG

        default = float(DEFAULT_CONFIG.get(key, fallback))
    except Exception:
        return fallback
    try:
        val = float((load_config() or {}).get(key, default))
        return val if val > 0 else default
    except Exception:
        return default


def _discover_mcp_tools_without_interactive_oauth() -> None:
    """Run MCP discovery without letting OAuth read from the user's stdin."""
    try:
        from tools.mcp_oauth import suppress_interactive_oauth
    except Exception:
        suppress_interactive_oauth = nullcontext

    with suppress_interactive_oauth():
        from tools.mcp_tool_discovery import discover_mcp_tools

        # Only pass the kwarg when a filter is set: many tests (and any
        # out-of-tree caller) stub discover_mcp_tools with a zero-arg
        # callable, and the unfiltered call shape is unchanged.
        if _mcp_server_filter is None:
            discover_mcp_tools()
        else:
            discover_mcp_tools(allowed_mcp_names=_mcp_server_filter)


def defer_background_mcp_discovery(*, logger, thread_name: str, delay: float) -> None:
    """Arm ``start_background_mcp_discovery`` to run ``delay`` seconds from now.

    Used by the Desktop ``serve`` backend after its socket is announced: the thread's first act is
    the ~350ms ``mcp`` SDK import, which would hold the GIL against the renderer's connect + first
    hydration reads (or the web_server import) if started earlier.
    """
    global _mcp_discovery_deferred
    with _mcp_discovery_lock:
        if _mcp_discovery_started or _mcp_discovery_deferred is not None:
            return

        def _fire() -> None:
            global _mcp_discovery_deferred
            with _mcp_discovery_lock:
                _mcp_discovery_deferred = None
            start_background_mcp_discovery(logger=logger, thread_name=thread_name)

        timer = threading.Timer(delay, _fire)
        timer.daemon = True
        timer.name = f"{thread_name}-deferred"
        _mcp_discovery_deferred = timer
        timer.start()


def _start_deferred_mcp_discovery_now() -> None:
    """Run an armed deferred start immediately (idempotent, thread-safe)."""
    with _mcp_discovery_lock:
        timer = _mcp_discovery_deferred
    if timer is None:
        return
    timer.cancel()
    timer.function()


def wait_for_mcp_discovery(timeout: "float | None" = None, *, single_query: bool = False) -> None:
    """Wait for background MCP discovery before the first tool snapshot.

    ``join`` returns the instant discovery completes, so this only blocks for a still-pending
    server's real connect time. ``single_query`` uses ``mcp_single_query_discovery_timeout``
    (15s vs 1.5s) because one-shot sessions have no second turn to recover.
    """
    _start_deferred_mcp_discovery_now()
    thread = _mcp_discovery_thread
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=_resolve_discovery_timeout(timeout, single_query=single_query))


def mcp_discovery_in_flight() -> bool:
    """True if THIS module's discovery thread is still running.

    Mirrors ``tui_gateway.entry.mcp_discovery_in_flight``; surfaces that start discovery here
    (desktop, dashboard sidecar) populate this thread, so the late-refresh scheduler consults both.

    Those processes populate THIS module's ``_mcp_discovery_thread``, not ``tui_gateway.entry``'s, so the
    late-refresh scheduler must consult both to decide whether a slow server's tools are still pending (see
    #51587).
    """
    thread = _mcp_discovery_thread
    return thread is not None and thread.is_alive()


def join_mcp_discovery(timeout: "float | None" = None) -> bool:
    """Block up to ``timeout`` for THIS module's discovery; True once complete, False if still
    running. For the off-critical-path late-refresh waiter (accepts a long wait, reports outcome)."""
    thread = _mcp_discovery_thread
    if thread is None:
        return True
    thread.join(timeout=timeout)
    return not thread.is_alive()


def ensure_mcp_discovery_before_agent_build(
    *,
    logger,
    timeout: "float | None" = None,
    single_query: bool = False,
    thread_name: str = "cli-mcp-discovery") -> None:
    """Give configured MCP tools a bounded chance to register before AIAgent.

    Non-interactive first turns (``chat -q``, ``hermes -z``) can construct ``AIAgent`` before any
    path started discovery, and ``wait_for_mcp_discovery()`` only joins an existing thread — so
    start discovery if needed, then wait up to the configured bound.
    """
    try:
        start_background_mcp_discovery(logger=logger, thread_name=thread_name)
        wait_for_mcp_discovery(timeout=timeout, single_query=single_query)
    except Exception:
        logger.debug("MCP discovery readiness check failed before agent build", exc_info=True)

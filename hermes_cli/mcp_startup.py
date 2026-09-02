"""Shared CLI/TUI-safe helpers for background MCP discovery."""

from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Optional

_mcp_discovery_lock = threading.Lock()
_mcp_discovery_started = False
_mcp_discovery_thread: Optional[threading.Thread] = None
_mcp_discovery_deferred: Optional[threading.Timer] = None


def _has_configured_mcp_servers() -> bool:
    """Cheap config probe so non-MCP users avoid importing the MCP stack."""
    try:
        from hermes_cli.config import read_raw_config

        raw_config = read_raw_config() or {}
        mcp_servers = raw_config.get("mcp_servers")
        if isinstance(mcp_servers, dict) and len(mcp_servers) > 0:
            return True
        from hermes_cli.agent_plugins import has_enabled_agent_plugin_mcp

        return has_enabled_agent_plugin_mcp(raw_config)
    except Exception:
        # Be conservative: if config probing fails, try discovery in the
        # background so startup still can't block.
        return True


def start_background_mcp_discovery(*, logger, thread_name: str) -> None:
    """Spawn one shared background MCP discovery thread for this process.

    If the first background discovery run exits without connecting any MCP
    server (for example after startup cancellation / OOM restart), later calls
    are allowed to retry instead of permanently pinning the process in a
    "discovery already started" state with zero MCP tools.
    """
    global _mcp_discovery_started, _mcp_discovery_thread

    with _mcp_discovery_lock:
        if _mcp_discovery_started:
            thread = _mcp_discovery_thread
            if thread is not None and thread.is_alive():
                return
            try:
                from tools.mcp_tool import get_mcp_status

                status = get_mcp_status() or []
                if any(entry.get("connected") for entry in status):
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

        # Capture the caller's context-local HERMES_HOME override (profile
        # scoping in multi-profile processes like the dashboard/desktop
        # backend) and re-install it inside the discovery thread. ContextVars
        # do not propagate into bare threads, so without this a session
        # "switched" to profile X would discover the LAUNCH profile's
        # mcp_servers instead (#67605). The config gate above already runs on
        # the caller's thread, so it sees the same override.
        try:
            from hermes_constants import get_hermes_home_override

            home_override = get_hermes_home_override()
        except Exception:
            home_override = None

        def _discover() -> None:
            token = None
            try:
                from hermes_constants import set_hermes_home_override

                token = set_hermes_home_override(home_override)
            except Exception:
                token = None
            try:
                _discover_mcp_tools_without_interactive_oauth()
                try:
                    from tools.mcp_tool import get_mcp_status
                    status = get_mcp_status() or []
                    if not any(entry.get("connected") for entry in status):
                        logger.warning(
                            "Background MCP discovery completed with zero connected servers"
                        )
                except Exception:
                    logger.debug("Failed to inspect MCP status after background discovery", exc_info=True)
            except Exception:
                logger.debug("Background MCP tool discovery failed", exc_info=True)
            finally:
                if token is not None:
                    try:
                        from hermes_constants import reset_hermes_home_override

                        reset_hermes_home_override(token)
                    except Exception:
                        pass
                with _mcp_discovery_lock:
                    global _mcp_discovery_thread, _mcp_discovery_started
                    _mcp_discovery_thread = None

        thread = threading.Thread(
            target=_discover,
            name=thread_name,
            daemon=True,
        )
        _mcp_discovery_thread = thread
        thread.start()


def _resolve_discovery_timeout(
    explicit: "float | None", *, single_query: bool = False
) -> float:
    """Resolve the MCP discovery wait bound: explicit arg > config > default.

    Reads ``mcp_discovery_timeout`` from config.yaml, defaulting to the value in
    ``DEFAULT_CONFIG`` (single source of truth) when the key is absent. Kept lazy
    and fail-safe — a missing/invalid value or a broken config falls back to a
    short safe bound so startup can never hang or crash.

    When ``single_query`` is True (``hermes -z "..."`` / ``-q``), the larger
    ``mcp_single_query_discovery_timeout`` bound is used instead. In single-query
    mode there is only ONE turn, so the between-turns late-binding refresh never
    runs — a server that misses the small interactive bound would be invisible to
    the LLM for the whole session. The wait still returns the instant discovery
    completes (see ``wait_for_mcp_discovery``), so fast servers pay ~0s; the
    larger bound only caps how long a genuinely slow cold-start may block.
    """
    if explicit is not None:
        return explicit
    key = (
        "mcp_single_query_discovery_timeout"
        if single_query
        else "mcp_discovery_timeout"
    )
    fallback = 15.0 if single_query else 1.5
    try:
        from hermes_cli.config import load_config, DEFAULT_CONFIG

        default = float(DEFAULT_CONFIG.get(key, fallback))
        try:
            raw = (load_config() or {}).get(key, default)
            val = float(raw)
            return val if val > 0 else default
        except Exception:
            return default
    except Exception:
        return fallback


def _discover_mcp_tools_without_interactive_oauth() -> None:
    """Run MCP discovery without letting OAuth read from the user's stdin."""
    try:
        from tools.mcp_oauth import suppress_interactive_oauth
    except Exception:
        suppress_interactive_oauth = nullcontext

    with suppress_interactive_oauth():
        from tools.mcp_tool import discover_mcp_tools

        discover_mcp_tools()


def defer_background_mcp_discovery(*, logger, thread_name: str, delay: float) -> None:
    """Arm ``start_background_mcp_discovery`` to run ``delay`` seconds from now.

    Used by the Desktop ``serve`` backend after its socket is announced: the
    discovery thread's first act is the ~350ms ``mcp`` SDK import, which holds
    the GIL against the renderer's connect + first hydration reads if it starts
    at bind time, and against the web_server import if it starts before. Any
    consumer that needs discovery sooner (``wait_for_mcp_discovery`` from an
    agent build) fires the deferred start immediately, so the bounded join and
    the late-binding refresh behave exactly as if it had been started eagerly.
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


def wait_for_mcp_discovery(
    timeout: "float | None" = None, *, single_query: bool = False
) -> None:
    """Wait for background MCP discovery before the first tool snapshot.

    ``thread.join(timeout)`` returns the INSTANT discovery completes, so this
    only ever blocks for the real connect time of a still-pending server —
    users with no MCP servers or fast servers pay ~0s.  The bound (from
    ``mcp_discovery_timeout`` in config) just caps the wait so a dead server
    can't freeze startup; servers that miss it are picked up by the automatic
    late-binding refresh.

    When ``single_query`` is True, the bound comes from
    ``mcp_single_query_discovery_timeout`` instead (default 15s vs 1.5s
    interactive) because one-shot sessions have no second turn to recover.
    """
    _start_deferred_mcp_discovery_now()
    thread = _mcp_discovery_thread
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=_resolve_discovery_timeout(timeout, single_query=single_query))


def mcp_discovery_in_flight() -> bool:
    """Return True if THIS module's background discovery thread is still running.

    Mirrors ``tui_gateway.entry.mcp_discovery_in_flight`` for the surfaces that
    start discovery through ``start_background_mcp_discovery`` here (the desktop
    app + dashboard WebSocket sidecar via ``tui_gateway/ws.py``, and
    ``hermes dashboard``).  Those processes populate THIS module's
    ``_mcp_discovery_thread``, not ``tui_gateway.entry``'s, so the late-refresh
    scheduler must consult both to decide whether a slow server's tools are
    still pending (see #51587).
    """
    thread = _mcp_discovery_thread
    return thread is not None and thread.is_alive()


def join_mcp_discovery(timeout: "float | None" = None) -> bool:
    """Block until THIS module's background discovery finishes, up to ``timeout``.

    Returns True if discovery has completed (thread absent or no longer alive),
    False if it is still running after the timeout.  Unlike
    ``wait_for_mcp_discovery`` this accepts an unbounded/long wait and reports
    the outcome, for the off-critical-path late-refresh waiter.
    """
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
    thread_name: str = "cli-mcp-discovery",
) -> None:
    """Give configured MCP tools a bounded chance to register before AIAgent.

    Non-interactive first turns (``chat -q``, ``hermes -z``) can construct
    ``AIAgent`` before the normal banner or tool-list paths touch
    ``get_tool_definitions()``.  Because the agent snapshots its tool
    registry at construction time, the first and only model turn can miss
    native ``mcp__...`` tools even when the MCP server is healthy.

    ``wait_for_mcp_discovery()`` only joins an already-created discovery
    thread, so it no-ops if a direct/single-query path reaches agent
    construction before MCP startup created that thread.  This helper makes
    the construction site self-sufficient: start discovery if needed, then
    wait up to the configured bound.

    When ``single_query`` is True, the larger
    ``mcp_single_query_discovery_timeout`` bound is used (default 15s vs 1.5s
    interactive) because one-shot sessions have no second turn to recover.

    Failures are swallowed so a broken MCP config never aborts agent
    construction — the agent runs without MCP tools, same as before.
    """
    try:
        start_background_mcp_discovery(
            logger=logger,
            thread_name=thread_name,
        )
        wait_for_mcp_discovery(timeout=timeout, single_query=single_query)
    except Exception:
        logger.debug(
            "MCP discovery readiness check failed before agent build",
            exc_info=True,
        )

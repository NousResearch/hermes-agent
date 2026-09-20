"""Desktop `serve` starts background MCP discovery only after the socket binds.

The MCP SDK import (~350ms) used to run on a thread started BEFORE
web_server was imported, holding the GIL against the main thread's own
import path and delaying the READY sentinel the Desktop waits on.
"""

from __future__ import annotations

import logging
import sys
import threading
import types

import hermes_cli.mcp_startup as mcp_startup
import hermes_cli.web_server as web_server
import hermes_cli.web_server_lifecycle as web_server_lifecycle
from tests.hermes_cli.test_dashboard_auth_gate import _stub_uvicorn_run


def _reset_discovery_state(monkeypatch):
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_started", set())
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", {})
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_deferred", None)


def test_desktop_serve_arms_mcp_discovery_only_after_ready_sentinel(monkeypatch):
    _reset_discovery_state(monkeypatch)
    order: list[str] = []
    monkeypatch.setattr(
        mcp_startup,
        "start_background_mcp_discovery",
        lambda *, logger, thread_name: order.append("discovery:" + thread_name),
    )
    monkeypatch.setattr(web_server, "_write_machine_sentinel_line", lambda line: order.append("sentinel"))
    monkeypatch.setattr(web_server_lifecycle, "_write_machine_sentinel_line", lambda line: order.append("sentinel"))
    _stub_uvicorn_run(monkeypatch)

    web_server.start_server(
        host="127.0.0.1", port=0, open_browser=False, headless=True,
        start_mcp_discovery_after_bind=True,
    )
    timer = mcp_startup._mcp_discovery_deferred
    assert order == ["sentinel"] and isinstance(timer, threading.Timer)
    timer.cancel()
    # An agent build inside the delay window pulls discovery forward itself.
    mcp_startup.wait_for_mcp_discovery(timeout=0)
    assert order == ["sentinel", "discovery:dashboard-mcp-discovery"]
    assert mcp_startup._mcp_discovery_deferred is None

    # Without the flag (dashboard / non-Desktop serve) start_server does not
    # start discovery itself — cmd_dashboard's pre-import path still owns it.
    order.clear()
    _reset_discovery_state(monkeypatch)
    web_server.start_server(host="127.0.0.1", port=0, open_browser=False, headless=True)
    assert order == ["sentinel"] and mcp_startup._mcp_discovery_deferred is None


def test_deferred_discovery_fires_once_and_is_idempotent(monkeypatch):
    _reset_discovery_state(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        mcp_startup,
        "start_background_mcp_discovery",
        lambda *, logger, thread_name: calls.append(thread_name),
    )
    log = logging.getLogger("test")
    mcp_startup.defer_background_mcp_discovery(logger=log, thread_name="t", delay=60)
    mcp_startup.defer_background_mcp_discovery(logger=log, thread_name="t", delay=60)  # second arm is a no-op
    first = mcp_startup._mcp_discovery_deferred
    mcp_startup.start_deferred_mcp_discovery_now()
    mcp_startup.start_deferred_mcp_discovery_now()
    assert calls == ["t"]
    assert first is not None and mcp_startup._mcp_discovery_deferred is None


def _stub_dashboard_runtime(monkeypatch):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(main_mod, "_resolve_dashboard_web_dist", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_quietly", lambda: None)
    monkeypatch.setitem(sys.modules, "fastapi", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "hermes_logging", types.SimpleNamespace(setup_logging=lambda **_k: None))
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", types.SimpleNamespace(discover_plugins=lambda: None))
    return main_mod


def test_standalone_dashboard_boot_arms_discovery_without_starting_it(monkeypatch):
    """#58733: an idle, unvisited `hermes dashboard` must not spawn the configured MCP servers."""
    _reset_discovery_state(monkeypatch)
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    main_mod = _stub_dashboard_runtime(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        mcp_startup, "start_background_mcp_discovery", lambda *, logger, thread_name: calls.append(thread_name)
    )

    after_bind = main_mod._dashboard_prepare_runtime(types.SimpleNamespace(skip_build=True), False)

    assert after_bind is False and calls == []
    armed = mcp_startup._mcp_discovery_deferred
    assert isinstance(armed, threading.Timer) and not armed.is_alive()


def test_first_gateway_ws_client_starts_the_armed_discovery_once(monkeypatch):
    import asyncio

    import hermes_cli.web_routers.chat_ws as chat_ws

    _reset_discovery_state(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        mcp_startup, "start_background_mcp_discovery", lambda *, logger, thread_name: calls.append(thread_name)
    )
    mcp_startup.defer_background_mcp_discovery(
        logger=logging.getLogger("test"), thread_name="dashboard-mcp-discovery", delay=None
    )
    assert calls == []

    async def _allowed(ws):
        return True

    async def _handle_ws(ws, **kwargs):
        return None

    monkeypatch.setattr(chat_ws, "_close_unless_sidecar_allowed", _allowed)
    monkeypatch.setattr("tui_gateway.ws.handle_ws", _handle_ws)
    asyncio.run(chat_ws.gateway_ws(object()))
    asyncio.run(chat_ws.gateway_ws(object()))

    assert calls == ["dashboard-mcp-discovery"]
    assert mcp_startup._mcp_discovery_deferred is None

"""Desktop `serve` starts background MCP discovery only after the socket binds.

The MCP SDK import (~350ms) used to run on a thread started BEFORE
web_server was imported, holding the GIL against the main thread's own
import path and delaying the READY sentinel the Desktop waits on.
"""

from __future__ import annotations

import logging
import threading

import hermes_cli.mcp_startup as mcp_startup
import hermes_cli.web_server as web_server
import hermes_cli.web_server_lifecycle as web_server_lifecycle
from tests.hermes_cli.test_dashboard_auth_gate import _stub_uvicorn_run


def _reset_discovery_state(monkeypatch):
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_started", False)
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", None)
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
    mcp_startup._start_deferred_mcp_discovery_now()
    mcp_startup._start_deferred_mcp_discovery_now()
    assert calls == ["t"]
    assert first is not None and mcp_startup._mcp_discovery_deferred is None

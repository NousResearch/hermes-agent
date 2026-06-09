"""Tests for MCP tool discovery in Dashboard/Desktop mode (fixes #42694).

When the dashboard starts via ``start_server()`` → ``uvicorn.run()``, the TUI
entrypoint's ``main()`` is never called, so the MCP discovery thread that
``main()`` spawns is missing.  ``start_server()`` must call
``start_background_mcp_discovery()`` so that Desktop/Dashboard sessions see
configured MCP servers.
"""

from unittest.mock import patch

import pytest


def _stub_uvicorn_run(monkeypatch):
    """Replace uvicorn.run with a no-op so start_server returns immediately."""
    import uvicorn

    captured: dict = {}

    def _fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", _fake_run)
    return captured


class TestStartServerMcpDiscovery:
    """start_server() must trigger background MCP discovery."""

    def test_start_server_calls_mcp_discovery(self, monkeypatch):
        """start_server calls start_background_mcp_discovery before uvicorn.run."""
        from hermes_cli import web_server

        _stub_uvicorn_run(monkeypatch)
        with patch(
            "hermes_cli.mcp_startup.start_background_mcp_discovery",
        ) as mock_discover:
            web_server.start_server(
                host="127.0.0.1", port=9119,
                open_browser=False, allow_public=False,
            )
            mock_discover.assert_called_once()
            call_kwargs = mock_discover.call_args
            assert "logger" in call_kwargs.kwargs
            assert call_kwargs.kwargs["thread_name"] == "dashboard-mcp-discovery"

    def test_start_server_mcp_discovery_failure_does_not_block(self, monkeypatch):
        """If MCP discovery import/start fails, start_server still proceeds."""
        from hermes_cli import web_server

        _stub_uvicorn_run(monkeypatch)
        with patch(
            "hermes_cli.mcp_startup.start_background_mcp_discovery",
            side_effect=ImportError("no mcp"),
        ):
            # Should not raise — the except block swallows the error
            web_server.start_server(
                host="127.0.0.1", port=9119,
                open_browser=False, allow_public=False,
            )


class TestWaitForMcpDiscoveryFallback:
    """wait_for_mcp_discovery() falls back to mcp_startup thread when the
    local (entry.py) thread is None — the Dashboard/Desktop path."""

    def test_fallback_to_mcp_startup_thread(self):
        """When entry._mcp_discovery_thread is None, delegates to mcp_startup."""
        import tui_gateway.entry as entry

        saved = entry._mcp_discovery_thread
        try:
            entry._mcp_discovery_thread = None
            with patch(
                "hermes_cli.mcp_startup.wait_for_mcp_discovery",
            ) as mock_wait:
                entry.wait_for_mcp_discovery(timeout=0.5)
                mock_wait.assert_called_once_with(timeout=0.5)
        finally:
            entry._mcp_discovery_thread = saved

    def test_local_thread_takes_precedence(self):
        """When entry._mcp_discovery_thread is set, it's used directly."""
        import threading
        import time

        import tui_gateway.entry as entry

        saved = entry._mcp_discovery_thread
        try:
            t = threading.Thread(target=lambda: time.sleep(0.05), daemon=True)
            t.start()
            entry._mcp_discovery_thread = t
            with patch(
                "hermes_cli.mcp_startup.wait_for_mcp_discovery",
            ) as mock_wait:
                entry.wait_for_mcp_discovery(timeout=1.0)
                mock_wait.assert_not_called()
            assert not t.is_alive()
        finally:
            entry._mcp_discovery_thread = saved

    def test_fallback_import_failure_is_swallowed(self):
        """If mcp_startup import fails, the fallback is silently skipped."""
        import tui_gateway.entry as entry

        saved = entry._mcp_discovery_thread
        try:
            entry._mcp_discovery_thread = None
            with patch(
                "builtins.__import__",
                side_effect=ImportError("no module"),
            ):
                # Should not raise
                entry.wait_for_mcp_discovery(timeout=0.1)
        finally:
            entry._mcp_discovery_thread = saved

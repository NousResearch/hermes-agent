"""Tests for ``hermes_cli.web_server._dashboard_health_middleware``.

The dashboard health rollup (``DASHBOARD_HEALTH``) counts a returned 5xx but exports
no path, so a ``components.dashboard`` degraded verdict on ``/api/status`` cannot be
traced to the route that caused it. The middleware's warning line is the access log:
one line per 5xx, carrying the route and the status.
"""

import asyncio
import logging

import pytest


class TestDashboardHealthMiddlewareLog:
    """One warning line per returned 5xx: route and status, nothing else."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, _isolate_hermes_home):
        try:
            import hermes_cli.web_server as ws
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        # Fresh state holder per test so counters don't leak across tests.
        monkeypatch.setattr(ws, "DASHBOARD_HEALTH", ws.DashboardHealth())
        self.ws = ws

    @staticmethod
    def _request(path: str):
        """A minimal ASGI request for driving the middleware directly."""
        from starlette.requests import Request

        return Request(
            {
                "type": "http",
                "method": "GET",
                "scheme": "http",
                "server": ("testserver", 80),
                "client": ("127.0.0.1", 12345),
                "path": path,
                "query_string": b"",
                "headers": [],
            }
        )

    def _call_middleware(self, status_code: int):
        """Run ``_dashboard_health_middleware`` over a stub downstream that answers
        ``status_code``; returns the response it passes through unaltered."""
        from starlette.responses import JSONResponse

        async def _call_next(request):
            return JSONResponse(status_code=status_code, content={"detail": "down"})

        return asyncio.run(
            self.ws._dashboard_health_middleware(self._request("/api/broken"), _call_next)
        )

    def test_5xx_response_is_logged_with_path_and_status(self, caplog):
        """A returned 5xx is counted by the rollup and written once to the log with
        its route — the line a degraded ``components.dashboard`` is diagnosed from."""
        with caplog.at_level("WARNING", logger="hermes_cli.web_server"):
            response = self._call_middleware(503)
        assert response.status_code == 503
        warnings = [
            record
            for record in caplog.records
            if record.name == "hermes_cli.web_server" and record.levelno >= logging.WARNING
        ]
        assert len(warnings) == 1
        assert "/api/broken" in warnings[0].getMessage()
        assert "503" in warnings[0].getMessage()

    def test_4xx_response_is_not_logged_as_an_error(self, caplog):
        """Client errors are not the rollup's subject — only 5xx warn."""
        with caplog.at_level("WARNING", logger="hermes_cli.web_server"):
            response = self._call_middleware(404)
        assert response.status_code == 404
        assert not [
            record
            for record in caplog.records
            if record.name == "hermes_cli.web_server" and record.levelno >= logging.WARNING
        ]

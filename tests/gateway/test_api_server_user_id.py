"""Tests for X-Hermes-User-Id (end-user attribution on /api/sessions/{id}/chat[/stream]).

A frontend proxying many human users through one Hermes deployment can declare which end
user is actually chatting; it threads through to AIAgent(user_id=...) for observability
plugins (e.g. Langfuse's per-user views) to pick up. Purely observability-facing -- it
grants nothing and is never used for authorization.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Minimal app wiring the one route these tests need."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


class TestUserIdHeader:
    @pytest.mark.asyncio
    async def test_session_chat_stream_threads_caller_user_id(self, adapter):
        """X-Hermes-User-Id reaches _run_agent as caller_user_id."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(adapter, "_get_existing_session_or_404", return_value=({"id": "s1"}, None)),
                patch.object(adapter, "_conversation_history_for_session", return_value=[]),
                patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run,
            ):
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                resp = await cli.post(
                    "/api/sessions/s1/chat/stream",
                    headers={"X-Hermes-User-Id": "alice@example.com"},
                    json={"message": "hi"},
                )
                assert resp.status == 200

        kwargs = mock_run.call_args.kwargs
        assert kwargs["caller_user_id"] == "alice@example.com"

    @pytest.mark.asyncio
    async def test_session_chat_stream_without_user_id_header_passes_none(self, adapter):
        """Absent header -> None, not an empty string (so AIAgent's own user_id default holds)."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(adapter, "_get_existing_session_or_404", return_value=({"id": "s1"}, None)),
                patch.object(adapter, "_conversation_history_for_session", return_value=[]),
                patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run,
            ):
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                resp = await cli.post("/api/sessions/s1/chat/stream", json={"message": "hi"})
                assert resp.status == 200

        kwargs = mock_run.call_args.kwargs
        assert kwargs["caller_user_id"] is None

    def test_extract_caller_user_id_rejects_control_characters(self, adapter):
        """Same header-injection guard as the other caller-supplied headers (X-Hermes-Session-Key)."""
        request = MagicMock()
        request.headers = {"X-Hermes-User-Id": "alice@example.com\r\nX-Injected: evil"}
        assert adapter._extract_caller_user_id(request) is None

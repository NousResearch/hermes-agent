"""
Tests verifying that exception messages containing secrets (API keys, tokens)
are NOT leaked to HTTP clients via error responses.

Covers Bug B1: exception messages in api_server.py were returned verbatim
to callers and written to logs without redaction.
"""

from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A realistic API key pattern that redact_sensitive_text will catch (sk- prefix)
_SECRET_KEY = "sk-proj-abc123XYZsecrettoken9999"


def _make_adapter() -> APIServerAdapter:
    config = PlatformConfig(enabled=True)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    return app


# ---------------------------------------------------------------------------
# /v1/chat/completions — secret must not appear in error response
# ---------------------------------------------------------------------------


class TestChatCompletionsSecretRedact:
    @pytest.mark.asyncio
    async def test_exception_with_api_key_not_in_response_body(self):
        """When _run_agent raises with an API key in the message,
        the HTTP response body must NOT contain that key."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        async def _raise_with_secret(**kwargs):
            raise RuntimeError(
                f"bad value for key='{_SECRET_KEY}' passed to upstream provider"
            )

        with patch.object(adapter, "_run_agent", side_effect=_raise_with_secret):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )

                assert resp.status == 500
                raw = await resp.text()
                assert _SECRET_KEY not in raw, (
                    f"Secret key '{_SECRET_KEY}' must not appear in HTTP response body"
                )

    @pytest.mark.asyncio
    async def test_exception_error_response_is_generic(self):
        """The error message returned to the client must be a generic string,
        not the raw exception message."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        async def _raise_with_secret(**kwargs):
            raise RuntimeError(
                f"upstream API rejected key={_SECRET_KEY}"
            )

        with patch.object(adapter, "_run_agent", side_effect=_raise_with_secret):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )

                assert resp.status == 500
                data = await resp.json()
                error_message = data.get("error", {}).get("message", "")
                assert _SECRET_KEY not in error_message, (
                    f"Secret key must not appear in error message: {error_message!r}"
                )
                assert "Internal server error" in error_message


# ---------------------------------------------------------------------------
# /v1/responses — secret must not appear in error response
# ---------------------------------------------------------------------------


class TestResponsesSecretRedact:
    @pytest.mark.asyncio
    async def test_exception_with_api_key_not_in_response_body(self):
        """When _run_agent raises with an API key in the message,
        the HTTP response body must NOT contain that key."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        async def _raise_with_secret(**kwargs):
            raise RuntimeError(
                f"provider error: invalid api_key={_SECRET_KEY}"
            )

        with patch.object(adapter, "_run_agent", side_effect=_raise_with_secret):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "hello",
                    },
                )

                assert resp.status == 500
                raw = await resp.text()
                assert _SECRET_KEY not in raw, (
                    f"Secret key '{_SECRET_KEY}' must not appear in HTTP response body"
                )

    @pytest.mark.asyncio
    async def test_exception_error_response_is_generic(self):
        """The error message returned to the client must be generic."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        async def _raise_with_secret(**kwargs):
            raise ValueError(
                f"token validation failed for {_SECRET_KEY}"
            )

        with patch.object(adapter, "_run_agent", side_effect=_raise_with_secret):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "hello",
                    },
                )

                assert resp.status == 500
                data = await resp.json()
                error_message = data.get("error", {}).get("message", "")
                assert _SECRET_KEY not in error_message, (
                    f"Secret key must not appear in error message: {error_message!r}"
                )
                assert "Internal server error" in error_message

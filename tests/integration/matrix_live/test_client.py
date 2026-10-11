"""Fixture clients return rate-limit errors without repeating requests indefinitely."""

import asyncio
from dataclasses import replace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from nio import SyncError

from tests.integration.matrix_live.conftest import MatrixAccount, _register


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "path"),
    [
        ("restore", "/_matrix/client/v3/sync"),
        ("register", "/_matrix/client/v3/register"),
    ],
)
async def test_fixture_client_returns_rate_limit_without_retrying(
    operation: str, path: str
) -> None:
    requests: list[str] = []

    async def rate_limit(request: web.Request) -> web.Response:
        requests.append(request.path)
        return web.json_response(
            {
                "errcode": "M_LIMIT_EXCEEDED",
                "error": "fixture rate limit",
                "retry_after_ms": 50,
            },
            status=429,
        )

    application = web.Application()
    application.router.add_route("*", "/{path:.*}", rate_limit)
    async with TestServer(application) as server:
        url = str(server.make_url(""))
        if operation == "register":
            with pytest.raises(AssertionError, match="M_LIMIT_EXCEEDED"):
                await asyncio.wait_for(_register(url, "fixture"), timeout=2)
        else:
            client = MatrixAccount(
                "@fixture:matrix.test", "fixture-device", "fixture-token"
            ).client(url)
            try:
                response = await asyncio.wait_for(client.sync(timeout=1), timeout=2)
            finally:
                await client.close()
            assert replace(response) == SyncError(
                "fixture rate limit", "M_LIMIT_EXCEEDED", retry_after_ms=50
            )

    assert requests == [path]

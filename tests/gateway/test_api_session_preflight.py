"""The browser can send the advertised continuation header only from allowed origins."""

import secrets

import pytest
from aiohttp import ClientSession

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _STATIC_FEATURE_FLAGS


@pytest.mark.asyncio
async def test_continuation_preflight_preserves_origin_allowlist():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": 0, "key": secrets.token_hex(32),
        "cors_origins": ["https://allowed.example"],
    }))
    assert await adapter.connect()
    try:
        port = adapter._site._server.sockets[0].getsockname()[1]
        header = _STATIC_FEATURE_FLAGS["session_continuity_header"]
        async with ClientSession() as client:
            for origin, expected in [("https://allowed.example", 200), ("https://denied.example", 403)]:
                async with client.options(f"http://127.0.0.1:{port}/v1/chat/completions", headers={
                    "Origin": origin, "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": f"Authorization, Content-Type, {header}",
                }) as response:
                    assert response.status == expected
                    if expected == 200:
                        allowed = {h.strip().lower() for h in response.headers["Access-Control-Allow-Headers"].split(",")}
                        assert header.lower() in allowed
                        assert response.headers["Access-Control-Allow-Origin"] == origin
                    else:
                        assert "Access-Control-Allow-Origin" not in response.headers
    finally:
        await adapter.disconnect()

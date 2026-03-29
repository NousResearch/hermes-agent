import time
import asyncio
import pytest


@pytest.mark.asyncio
async def test_token_refresh(monkeypatch):
    # Mock aiohttp ClientSession to simulate OAuth2 token response
    class DummyResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def json(self):
            return {"access_token": "abc123", "expires_in": 7200}

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def post(self, *args, **kwargs):
            # Return an awaitable response compatible with "async with session.post(...) as resp" pattern
            return DummyResponse()

    monkeypatch.setattr("aiohttp.ClientSession", DummySession)

    # Import here to ensure the monkeypatch is active before class import if it references aiohttp
    from gateway.platforms.qq import QQAdapter

    adapter = QQAdapter(
        app_id="id",
        app_secret="secret",
        refresh_token="rf",
        token=None,
        ws_url="wss://example",
    )
    token = await adapter.get_token()
    assert adapter.token == "abc123"
    assert isinstance(adapter.token_expiry, float)
    assert adapter.token_expiry > time.time()

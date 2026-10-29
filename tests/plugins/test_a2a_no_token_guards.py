"""Edges of the A2A no-token browser and Host guards: malformed Host values fail closed, a POST
without a Content-Type is refused, and a token-authenticated listener is left unchanged."""

from __future__ import annotations

import asyncio
from email.message import Message

import pytest

from plugins.platforms.a2a import security
from tests.plugins.test_a2a_plugin import _get_json, _make_live_adapter, _post_json, _send_body


@pytest.mark.parametrize("value", ["", "[::1", "127.0.0.1:notaport", "user@127.0.0.1", "127.0.0.1, rebound.example"])
def test_no_token_host_check_fails_closed_on_malformed_values(value):
    assert security.is_loopback_host_header(value, "https://agent.example.test/") is False


def test_no_token_post_without_content_type_is_refused():
    headers = Message()
    assert security.browser_post_refusal(headers)
    headers["Content-Type"] = "application/json; charset=utf-8"
    assert security.browser_post_refusal(headers) is None


def test_token_listener_keeps_serving_any_host(monkeypatch):
    """The Host and browser guards apply only without a token; an authenticated listener is unchanged."""
    monkeypatch.setenv("A2A_BEARER_TOKEN", "topsecret")
    monkeypatch.setenv("A2A_HOST", "127.0.0.1")
    adapter, base = _make_live_adapter(monkeypatch)

    async def run():
        assert await adapter.connect() is True
        try:
            health = await asyncio.to_thread(_get_json, base + "/health", {"Host": "rebound.example"})
            resp = await asyncio.to_thread(
                _post_json, base + "/", _send_body("hello"),
                {"Host": "rebound.example", "Authorization": "Bearer topsecret"})
            return health, resp
        finally:
            await adapter.disconnect()

    health, resp = asyncio.run(run())
    assert health["status"] == "ok"
    assert resp["result"]["status"]["state"] == "TASK_STATE_COMPLETED"

"""QQBot adapter typing-indicator tests: passive reply → active-mode fallback."""

import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest

from gateway.config import PlatformConfig


def _make_adapter():
    from gateway.platforms.qqbot import QQAdapter
    adapter = QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "a", "client_secret": "b"}))
    adapter._running = True
    adapter._ws = SimpleNamespace(closed=False)
    return adapter


def _connected_adapter(bodies):
    """Adapter whose _api_request records bodies; msg_id-bearing bodies raise once."""
    adapter = _make_adapter()

    async def fake_api_request(method, path, body=None, timeout=None):
        bodies.append(body)
        if "msg_id" in body:
            raise RuntimeError("QQ Bot API error [400] msg_id expired")
        return {"id": "notify_1"}

    adapter._api_request = fake_api_request
    return adapter


class TestSendTypingFallback:
    @pytest.mark.asyncio
    async def test_passive_used_first_when_msg_id_fresh(self):
        bodies = []
        adapter = _make_adapter()

        async def ok(method, path, body=None, timeout=None):
            bodies.append(body)
            return {"id": "notify_1"}

        adapter._api_request = ok
        adapter._last_msg_id["u1"] = "msg-fresh"
        await adapter.send_typing("u1")
        assert len(bodies) == 1
        assert bodies[0]["msg_id"] == "msg-fresh"
        assert bodies[0]["msg_type"] == 6

    @pytest.mark.asyncio
    async def test_active_notify_when_passive_window_expired(self):
        bodies = []
        adapter = _connected_adapter(bodies)
        adapter._last_msg_id["u1"] = "msg-old"
        await adapter.send_typing("u1")
        # 1st attempt passive (raises), 2nd active WITHOUT msg_id must succeed
        assert len(bodies) == 2
        assert "msg_id" not in bodies[1]
        assert bodies[1]["input_notify"]["input_type"] == 1

    @pytest.mark.asyncio
    async def test_dead_msg_id_skipped_on_later_ticks(self):
        bodies = []
        adapter = _connected_adapter(bodies)
        adapter._last_msg_id["u1"] = "msg-old"
        await adapter.send_typing("u1")
        bodies.clear()
        adapter._typing_sent_at.clear()  # bypass debounce for the simulated next tick
        await adapter.send_typing("u1")
        # known-dead msg_id: go straight to active, one call only
        assert len(bodies) == 1
        assert "msg_id" not in bodies[0]

    @pytest.mark.asyncio
    async def test_new_msg_id_retries_passive(self):
        adapter = _make_adapter()
        bodies = []

        async def fake_api_request(method, path, body=None, timeout=None):
            bodies.append(body)
            if (body or {}).get("msg_id") == "msg-old":
                raise RuntimeError("QQ Bot API error [400] msg_id expired")
            return {"id": "notify_1"}

        adapter._api_request = fake_api_request
        adapter._last_msg_id["u1"] = "msg-old"
        await adapter.send_typing("u1")
        adapter._typing_sent_at.clear()
        adapter._last_msg_id["u1"] = "msg-new"  # inbound reply refreshed the window
        await adapter.send_typing("u1")
        assert bodies[-1]["msg_id"] == "msg-new"

    @pytest.mark.asyncio
    async def test_no_msg_id_goes_active_directly(self):
        bodies = []
        adapter = _make_adapter()

        async def ok(method, path, body=None, timeout=None):
            bodies.append(body)
            return {"id": "notify_1"}

        adapter._api_request = ok
        await adapter.send_typing("u1")
        assert len(bodies) == 1 and "msg_id" not in bodies[0]

    @pytest.mark.asyncio
    async def test_debounce_holds_within_window(self):
        bodies = []
        adapter = _make_adapter()

        async def ok(method, path, body=None, timeout=None):
            bodies.append(body)
            return {"id": "notify_1"}

        adapter._api_request = ok
        await adapter.send_typing("u1")
        await adapter.send_typing("u1")  # inside the 50s debounce
        assert len(bodies) == 1

    @pytest.mark.asyncio
    async def test_group_chats_stay_silent(self):
        adapter = _make_adapter()
        adapter._api_request = mock.AsyncMock()
        adapter._chat_type_map["g1"] = "group"
        await adapter.send_typing("g1")
        adapter._api_request.assert_not_awaited()

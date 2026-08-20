"""Tests for WhatsApp typing presence lifecycle."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


class _AsyncResponse:
    async def __aenter__(self):
        return MagicMock(status=200)

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def adapter():
    instance: Any = WhatsAppAdapter(
        PlatformConfig(enabled=True, extra={"session_name": "test"})
    )
    instance._running = True
    instance._check_managed_bridge_exit = AsyncMock(return_value=False)
    instance._http_session = MagicMock()
    instance._http_session.post = MagicMock(return_value=_AsyncResponse())
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "state"),
    [("send_typing", "composing"), ("stop_typing", "paused")],
)
async def test_typing_presence_posts_explicit_state_and_normalized_jid(
    adapter, method_name, state
):
    await getattr(adapter, method_name)("15551234567")

    call = adapter._http_session.post.call_args
    assert call.args[0] == "http://127.0.0.1:3000/typing"
    assert call.kwargs["json"] == {
        "chatId": "15551234567@s.whatsapp.net",
        "state": state,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["send_typing", "stop_typing"])
@pytest.mark.parametrize("guard", ["not_running", "no_session"])
async def test_typing_presence_respects_adapter_guards(adapter, method_name, guard):
    post = adapter._http_session.post
    if guard == "not_running":
        adapter._running = False
    else:
        adapter._http_session = None

    await getattr(adapter, method_name)("15551234567")

    post.assert_not_called()
    adapter._check_managed_bridge_exit.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_typing_respects_managed_bridge_exit(adapter):
    adapter._check_managed_bridge_exit.return_value = "bridge exited"

    await adapter.stop_typing("15551234567")

    adapter._http_session.post.assert_not_called()


@pytest.mark.asyncio
async def test_stop_typing_failure_is_best_effort(adapter):
    adapter._http_session.post.side_effect = RuntimeError("bridge unavailable")

    await adapter.stop_typing("15551234567")

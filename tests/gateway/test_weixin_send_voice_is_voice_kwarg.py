"""Regression tests: WeixinAdapter.send_voice must accept the router's is_voice kwarg.

The gateway media dispatch loop (``gateway/platforms/base.py``) calls
``self.send_voice(chat_id=..., audio_path=..., metadata=..., is_voice=...)``
whenever ``should_send_media_as_audio`` routes an audio ``MEDIA:`` attachment
to the voice sender. An adapter whose ``send_voice`` accepts neither
``is_voice`` nor ``**kwargs`` raises ``TypeError`` at argument-binding time,
and the dispatch loop's ``except Exception`` handler swallows it — so the
audio silently never arrives.

Matrix was fixed for this in #99712 and Mattermost/LINE in #100021; these
tests pin the same contract for Weixin.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.weixin import WeixinAdapter


def test_send_voice_accepts_is_voice_kwarg():
    params = inspect.signature(WeixinAdapter.send_voice).parameters
    assert "is_voice" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    ), (
        "gateway media router calls send_voice(is_voice=...); "
        "WeixinAdapter must accept it"
    )


@pytest.mark.asyncio
async def test_router_kwarg_set_binds_and_sends_attachment():
    """The exact kwarg set from the base.py dispatch loop must bind and deliver."""
    adapter = WeixinAdapter.__new__(WeixinAdapter)
    adapter._send_session = object()
    adapter._token = "token"
    adapter._send_file = AsyncMock(return_value="msg-1")

    # A TypeError here is the regression: argument binding failed before any
    # send was attempted. Any other outcome means the kwarg was accepted.
    result = await adapter.send_voice(
        chat_id="C1",
        audio_path="/tmp/speech.ogg",
        metadata=None,
        is_voice=True,
    )

    assert result.success is True
    assert result.message_id == "msg-1"
    adapter._send_file.assert_awaited_once()


@pytest.mark.asyncio
async def test_disconnected_adapter_still_returns_controlled_failure():
    """is_voice must not disturb the not-connected guard."""
    adapter = WeixinAdapter.__new__(WeixinAdapter)
    adapter._send_session = None
    adapter._token = None

    result = await adapter.send_voice(
        chat_id="C1",
        audio_path="/tmp/speech.ogg",
        metadata=None,
        is_voice=True,
    )

    assert result.success is False
    assert result.error == "Not connected"

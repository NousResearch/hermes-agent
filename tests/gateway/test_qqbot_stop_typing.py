"""Tests for the QQBot typing-indicator cancel (stop_typing).

QQ's ``input_notify`` with ``input_type=1`` leaves a 60-second "typing…"
bubble even after the agent has replied. ``stop_typing`` cancels it with
``input_type=0`` so the reply doesn't sit under a stale indicator, and it
clears the send_typing debounce so the next indicator can fire.
"""

import pytest

from gateway.platforms.qqbot.adapter import QQAdapter, MSG_TYPE_INPUT_NOTIFY


class FakeAdapter(QQAdapter):
    """QQAdapter subclass with the network layer stubbed out."""

    def __init__(self, chat_type="c2c", connected=True, msg_id="m1"):
        if msg_id:
            self._last_msg_id = {"u1": msg_id}
        else:
            self._last_msg_id = {}
        self._typing_sent_at = {"u1": 123.0}
        self._connection = connected
        self._chat = chat_type
        self.calls = []
        self._next_seq = 0

    @property
    def is_connected(self):
        return self._connection

    @property
    def _log_tag(self):
        return "test"

    def _guess_chat_type(self, chat_id):
        return self._chat

    def _next_msg_seq(self, chat_id):
        self._next_seq += 1
        return self._next_seq

    async def _api_request(self, method, path, body):
        self.calls.append((method, path, body))
        return {}


class TestStopTyping:
    @pytest.mark.asyncio
    async def test_sends_cancel_payload_and_clears_debounce(self):
        a = FakeAdapter()
        await a.stop_typing("u1")

        assert len(a.calls) == 1, a.calls
        method, path, body = a.calls[0]
        assert method == "POST"
        assert path == "/v2/users/u1/messages"
        assert body["msg_type"] == MSG_TYPE_INPUT_NOTIFY
        assert body["input_notify"] == {"input_type": 0, "input_second": 0}
        assert body["msg_id"] == "m1"
        assert body["msg_seq"] == 1
        # The invite (debounce) must be cleared so a later send_typing fires.
        assert "u1" not in a._typing_sent_at

    @pytest.mark.asyncio
    async def test_typing_can_fire_again_after_stop(self):
        a = FakeAdapter()
        await a.send_typing("u1")
        assert len(a.calls) == 1
        await a.stop_typing("u1")
        await a.send_typing("u1")
        assert len(a.calls) == 3, a.calls
        assert a.calls[1][2]["input_notify"]["input_type"] == 0
        assert a.calls[2][2]["input_notify"]["input_type"] == 1

    @pytest.mark.asyncio
    async def test_noop_on_non_c2c(self):
        a = FakeAdapter(chat_type="group")
        await a.stop_typing("u1")
        assert a.calls == []

    @pytest.mark.asyncio
    async def test_noop_when_disconnected(self):
        a = FakeAdapter(connected=False)
        await a.stop_typing("u1")
        assert a.calls == []

    @pytest.mark.asyncio
    async def test_noop_without_last_msg_id(self):
        a = FakeAdapter(msg_id=None)
        await a.stop_typing("u1")
        assert a.calls == []

    @pytest.mark.asyncio
    async def test_api_failure_swallowed(self):
        class Boom(FakeAdapter):
            async def _api_request(self, method, path, body):
                raise RuntimeError("boom")

        a = Boom()
        await a.stop_typing("u1")  # must not raise

        class BoomSend(FakeAdapter):
            async def _api_request(self, method, path, body):
                raise RuntimeError("boom")

        b = BoomSend()
        await b.send_typing("u1")  # existing behavior, also silent

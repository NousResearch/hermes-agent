"""WhatsApp's typing indicator is withdrawn when a turn ends: it does not expire on its own as Telegram's does."""

from unittest.mock import MagicMock

import pytest

from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter


def _adapter_with_bridge():
    adapter = _make_adapter()
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(MagicMock(status=200)))
    adapter._typing_paused = set()  # the shared refresh loop's own state, which the bare test adapter lacks
    return adapter


def _typing_states(adapter) -> list[tuple[str, dict]]:
    return [(call.args[0], call.kwargs["json"]) for call in adapter._http_session.post.call_args_list]


@pytest.mark.asyncio
async def test_stop_typing_tells_the_bridge_the_indicator_is_paused():
    adapter = _adapter_with_bridge()

    await adapter.stop_typing("15551234567")

    assert _typing_states(adapter) == [
        ("http://127.0.0.1:3000/typing", {"chatId": "15551234567@s.whatsapp.net", "state": "paused"})]


@pytest.mark.asyncio
async def test_send_typing_still_shows_the_indicator():
    adapter = _adapter_with_bridge()

    await adapter.send_typing("15551234567")

    assert _typing_states(adapter) == [
        ("http://127.0.0.1:3000/typing", {"chatId": "15551234567@s.whatsapp.net", "state": "composing"})]


@pytest.mark.asyncio
async def test_the_end_of_a_turn_withdraws_the_indicator():
    """The shared refresh loop stops through ``stop_typing``: with the base no-op, nothing reached the bridge."""
    adapter = _adapter_with_bridge()

    await adapter._stop_typing_refresh("15551234567")

    assert ("http://127.0.0.1:3000/typing",
            {"chatId": "15551234567@s.whatsapp.net", "state": "paused"}) in _typing_states(adapter)


@pytest.mark.asyncio
async def test_stop_typing_is_quiet_when_the_bridge_is_down():
    adapter = _adapter_with_bridge()
    adapter._running = False

    await adapter.stop_typing("15551234567")

    assert _typing_states(adapter) == []

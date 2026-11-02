from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.session import build_session_key
from tests.gateway.test_active_session_text_merge import _make_adapter, _make_event


@pytest.mark.asyncio
@pytest.mark.parametrize("session_active", [False, True])
async def test_hook_only_wake_uses_inline_lane_without_queue_or_typing(session_active):
    adapter = _make_adapter()
    event = _make_event("@teammate candidate")
    event._suppress_unaddressed_reply = True
    session_key = build_session_key(event.source)
    adapter._message_handler = AsyncMock(return_value=None)
    adapter._busy_session_handler = AsyncMock(return_value=False)
    adapter._start_session_processing = MagicMock(return_value=True)
    if session_active:
        adapter._active_sessions[session_key] = __import__("asyncio").Event()

    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
    adapter._start_session_processing.assert_not_called()
    assert session_key not in adapter._pending_messages
    assert session_key not in adapter._text_debounce
    assert event._gateway_accepted is True

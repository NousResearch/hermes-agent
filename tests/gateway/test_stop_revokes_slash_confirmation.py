"""STOP dismisses pending slash confirmations without affecting other sessions."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tests.gateway.test_stop_ends_background_delegations import _Adapter
from tools import slash_confirm


@pytest.mark.asyncio
@pytest.mark.parametrize("session_state", ["idle", "busy"])
async def test_stop_revokes_pending_slash_confirmation(monkeypatch, session_state):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "u1")
    adapter = _Adapter()
    runner = GatewayRunner(config=GatewayConfig(platforms={
        Platform.TELEGRAM: PlatformConfig(enabled=True, token="fictional-token"),
    }))
    runner.adapters = {Platform.TELEGRAM: adapter}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", user_id="u1")
    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source)
    key = runner._session_key_for_source(source)
    other_source = SessionSource(platform=Platform.TELEGRAM, chat_id="c2", chat_type="dm", user_id="u1")
    other_key = runner._session_key_for_source(other_source)
    effect = AsyncMock(return_value="executed")
    other_effect = AsyncMock(return_value="other executed")
    slash_confirm.register(key, "fictional-confirm", "new", effect)
    slash_confirm.register(other_key, "other-confirm", "new", other_effect)
    try:
        if session_state == "busy":
            runner._session_state(key).turn.agent = MagicMock()
            handled, _ = await runner._hm_busy_slash_or_photo(event, source, key)
            assert handled
        else:
            await runner._handle_stop_command(event)
        late_result = await slash_confirm.resolve(key, "fictional-confirm", "once")
        assert late_result is None
        effect.assert_not_called()
        assert await slash_confirm.resolve(other_key, "other-confirm", "once") == "other executed"
        other_effect.assert_awaited_once_with("once")
    finally:
        slash_confirm.clear(key)
        slash_confirm.clear(other_key)

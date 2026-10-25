"""An admitted local input keeps its own identity in tool context."""
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource
from gateway.session_context import get_session_env


class ContextBound(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize('platform,source_id,expected', [
    (Platform.LOCAL, None, 'admitted-input'),
    (Platform.TELEGRAM, 'platform-anchor', 'platform-anchor'),
])
async def test_turn_binds_input_without_changing_shared_source(platform, source_id, expected):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={})
    runner.adapters = {}
    runner._hmwa_open_session = AsyncMock(return_value=(False, False))
    source = SessionSource(platform=platform, chat_id='chat', chat_type='dm', message_id=source_id)
    now = datetime.now(UTC)
    entry = SessionEntry(session_key='key', session_id='sid', created_at=now, updated_at=now)
    observed = []

    def bind(context):
        tokens = GatewayRunner._set_session_env(runner, context)
        try:
            observed.append(get_session_env('HERMES_SESSION_MESSAGE_ID'))
        finally:
            GatewayRunner._clear_session_env(runner, tokens)
        raise ContextBound

    runner._set_session_env = bind
    for message_id in ['admitted-input', 'next-input']:
        event = MessageEvent(text='hello', source=source, message_id=message_id)
        with pytest.raises(ContextBound):
            await runner._hmwa_prepare_turn(event, source, entry, 'key', 'key', 1)
        assert get_session_env('HERMES_SESSION_MESSAGE_ID') == ''
        assert source.message_id == source_id
    assert observed == ([expected, 'next-input'] if platform == Platform.LOCAL else [expected, expected])

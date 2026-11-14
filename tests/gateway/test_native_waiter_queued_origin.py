"""A native adapter must present the queued terminal turn, not its opener."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway import session_ingress
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from gateway.session_contract import SessionActor, SessionRef, Submission
from tests.gateway.test_base_auto_tts_profile_scope import _Adapter
from tests.gateway.test_prompt_attachments import _authority


@pytest.mark.asyncio
@pytest.mark.parametrize('opening,terminal,expected', [
    (MessageType.VOICE, MessageType.TEXT, False),
    (MessageType.TEXT, MessageType.VOICE, True),
])
async def test_native_adapter_uses_terminal_voice_origin(tmp_path, monkeypatch, opening, terminal, expected):
    started = asyncio.Event()
    release = asyncio.Event()

    async def answer(event):
        started.set()
        await release.wait()
        event.message_type = terminal
        event._queued_terminal_voice_projected = True
        return 'answer'

    authority = await _authority(tmp_path, monkeypatch, answer)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='c', chat_type='dm')
    authority.sessions['s'].source = source
    adapter = _Adapter()
    authority.runner._adapter_for_source = lambda source: adapter
    adapter._should_auto_tts_for_chat = lambda chat: True
    adapter._synthesize_auto_tts = AsyncMock(return_value=([], None))
    adapter.set_message_handler(lambda event: session_ingress.admit_message(authority, event))
    actor = SessionActor('human', 'p', frozenset({'session:submit'}), 't')

    async def admit_native(event):
        return await authority.submit(actor, Submission('m1', SessionRef('p', 's'), {'text': 'hi'}, 'queue'))

    monkeypatch.setattr(authority, 'admit_native', admit_native)
    event = MessageEvent(text='hi', source=source, message_type=opening, message_id='m1')
    # The real adapter marks this event for marker handoff when a runner is installed.
    adapter.gateway_runner = authority.runner
    task = asyncio.create_task(adapter._process_message_background(event, build_session_key(source)))
    await asyncio.wait_for(started.wait(), 5)
    release.set()
    await asyncio.wait_for(task, 10)
    await asyncio.wait_for(authority.sessions['s'].task, 10)
    assert adapter._synthesize_auto_tts.await_count == int(expected)
    assert adapter.sent == ['answer']
    assert event.source is source
    assert event.metadata == {}

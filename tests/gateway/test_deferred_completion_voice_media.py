"""Deferred completion speech uses the adapter's real response extraction."""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import session_ingress
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.test_auto_voice_reply_format import _make_runner
from tests.gateway.test_base_auto_tts_profile_scope import _Adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("human_voice", [False, True])
@pytest.mark.parametrize("kind", ['directive', 'media', 'local_path'])
async def test_settled_completion_speaks_only_extracted_text(monkeypatch, tmp_path, streamed, kind, human_voice):
    runner = _make_runner()
    adapter = _Adapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._voice_mode['telegram:123'] = 'voice_only'
    runner._session_key_for_source = lambda source: 'session'
    event = MessageEvent(text='completion', internal=True, metadata={'presentation_origin': 'voice'},
                         source=SessionSource(platform=Platform.TELEGRAM, chat_id='123'))
    if human_voice:
        event.internal = False
        event.message_type = MessageType.VOICE
        event._adapter_drains_after = False
    event._voice_result_committed = True
    file = tmp_path / 'private.png'
    file.write_bytes(b'image')
    # Isolate extraction from the host's configurable media-delivery allowlist.
    adapter.filter_media_delivery_paths = lambda files, session_key='': files
    adapter.filter_local_delivery_paths = lambda files, session_key='': files
    response = {'directive': 'Finished. [[as_document]]', 'media': f'Finished. MEDIA:{file}',
                'local_path': f'Finished. {file}'}[kind]
    event._deferred_voice_response = response
    event._deferred_voice_messages = []
    if streamed:
        event._deferred_voice_result = object()
    authority = SimpleNamespace(runner=runner, pending_deliveries={
        'id': (adapter, event, 'session', None if streamed else response, None)}, native_waiters=set())
    monkeypatch.setattr(session_ingress, '_held_markers', lambda authority: {})
    monkeypatch.setattr(session_ingress, '_execution_scope', AsyncMock(return_value=nullcontext()))
    monkeypatch.setattr(session_ingress, '_release_turn_marker', AsyncMock())
    monkeypatch.setattr(session_ingress, 'close_producer_turn', AsyncMock())
    async def send_text(event, key, text, metadata, ephemeral, ttl, record):
        record(SimpleNamespace(success=True))
    adapter._send_final_text = AsyncMock(side_effect=send_text)
    adapter._deliver_attachments = AsyncMock()
    spoken = []
    def tts(*, text, output_path):
        spoken.append(text)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b'audio')
        return json.dumps({'success': True, 'file_path': output_path})
    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', tts)
    runner._deliver_voice_reply = AsyncMock()
    await session_ingress.deliver_settled(authority, 'id')
    if kind == 'media':
        assert spoken == []
    else:
        assert spoken == ['Finished.']
        runner._deliver_voice_reply.assert_awaited_once()

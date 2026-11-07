"""Real worker/config/provider dispatch under alternating routed profile scopes."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


from agent.secret_scope import set_multiplex_active
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import _profile_runtime_scope
from gateway.session import SessionSource, build_session_key
from plugins.platforms.discord.adapter import DiscordAdapter


# Only the external Gemini request is replaced inside the child. Real config,
# provider selection, normalization, chunking, files, worker and cleanup run.
_GEMINI_STUB = '''
import json, os, pathlib, runpy, wave
import tools.tts_tool as t

def generate(text, output_path, config):
    home = pathlib.Path(os.environ['HERMES_HOME'])
    with (home / 'spoken.jsonl').open('a') as f:
        f.write(json.dumps({'text': text, 'key': os.environ.get('GEMINI_API_KEY'),
                            'provider': config['provider']}) + '\\n')
    with wave.open(output_path, 'wb') as audio:
        audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        audio.writeframes(b'\\0\\0' * 2400)

t._generate_gemini_tts = generate
runpy.run_module('gateway.platforms.base_auto_tts_worker', run_name='__main__')
'''


@pytest.mark.asyncio
async def test_worker_preserves_full_gemini_reply_and_profile_scope_a_b_a(tmp_path, monkeypatch):
    homes = {name: tmp_path / name for name in ('a', 'b')}
    for name, home in homes.items():
        home.mkdir()
        (home / '.env').write_text(f'GEMINI_API_KEY=test-{name}\n')
        (home / 'config.yaml').write_text(json.dumps({'tts': {'provider': 'gemini'}}))
    spawn = asyncio.create_subprocess_exec

    async def stub_request(*args, **kwargs):
        return await spawn(args[0], '-c', _GEMINI_STUB, **kwargs)

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', stub_request)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fake'))
    adapter._should_auto_tts_for_chat = lambda _: True
    adapter.send_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()
    adapter._run_processing_hook = AsyncMock()
    adapter._record_delivery_obligation = AsyncMock(return_value=None)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id='text-1'))
    delivered = []

    async def send_voice(**kwargs):
        path = Path(kwargs['audio_path'])
        assert path.exists() and path.stat().st_size > 0
        delivered.append(path)
        return SendResult(success=True)

    adapter.send_voice = send_voice
    adapter.gateway_runner = SimpleNamespace(
        _background_tasks=set(),
        _media_delivery_scope_for_source=lambda source: _profile_runtime_scope(
            homes[source.profile], hydrate_secrets=False),
    )
    reply = 'A complete reply keeps every spoken word. ' * 250
    adapter.set_message_handler(AsyncMock(return_value=reply))
    set_multiplex_active(True)
    try:
        for name in ('a', 'b', 'a'):
            event = MessageEvent(text='voice', message_type=MessageType.VOICE, message_id='1',
                                 source=SessionSource(platform=Platform.DISCORD, chat_id='123', profile=name))
            # Real post-handler delivery; no profile override ambient on the parent task.
            await adapter._process_message_background(event, build_session_key(event.source))
            await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks)), 10)
        assert adapter.send.await_count == 3  # no fallback notices
        assert len(delivered) >= 3
        for name, copies in (('a', 2), ('b', 1)):
            chunks = [json.loads(line) for line in (homes[name] / 'spoken.jsonl').read_text().splitlines()]
            expected = adapter.prepare_tts_text(reply).split() * copies
            assert ' '.join(chunk['text'] for chunk in chunks).split() == expected
            assert all(chunk['provider'] == 'gemini' and chunk['key'] == f'test-{name}' for chunk in chunks)
            assert not list((homes[name] / 'cache' / 'auto_tts').iterdir())
        assert all(not path.exists() for path in delivered)
    finally:
        await adapter.cancel_background_tasks()
        set_multiplex_active(False)

"""A held Codex turn reaches the chat only as the rewrite, also when the subprocess failed after
its agentMessage completed (the gateway still delivers that text). Real GatewayRunner and final-send
path; only the external Codex result is stubbed."""
import os
from pathlib import Path

import pytest

from tests.fakes.redactor_plugin import RAW, REWRITE, redactor_home
from tests.gateway.test_hold_stream_delivery import _Chat, _frames
from tests.gateway.test_run_progress_topics import _make_runner


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [None, 'codex app-server subprocess exited unexpectedly'], ids=['completed', 'failed_after_text'])
async def test_codex_held_partial_final_reaches_chat_only_rewritten(error, monkeypatch):
    import gateway.run as gateway_run
    from hermes_cli import plugins
    from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult
    from gateway.config import Platform, StreamingConfig
    from gateway.session import SessionSource
    from gateway.platforms.event import MessageEvent, MessageType

    home = Path(os.environ['HERMES_HOME'])
    redactor_home(home, 'http://127.0.0.1:9/v1', hold=True)
    plugins._reset_plugin_managers_for_tests()
    def run_turn(self, user_input, **kwargs):
        return TurnResult(final_text=RAW, projected_messages=[{'role': 'assistant', 'content': RAW}],
                          error=error, should_retire=error is not None,
                          thread_id='thread-held', turn_id='turn-held')
    monkeypatch.setattr(CodexAppServerSession, 'run_turn', run_turn)
    monkeypatch.setattr(CodexAppServerSession, 'ensure_started', lambda self: 'thread-held')
    monkeypatch.setattr(gateway_run, '_hermes_home', home)
    monkeypatch.setattr(gateway_run, '_resolve_runtime_agent_kwargs', lambda: {
        'api_key': 'sk-fake', 'base_url': 'http://127.0.0.1:9/v1', 'provider': 'custom',
        'api_mode': 'codex_app_server'})
    chat = _Chat()
    runner = _make_runner(chat)
    runner.config.streaming = StreamingConfig(enabled=True, edit_interval=0.01, buffer_threshold=1, cursor='')
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='c1', chat_type='dm', user_id='u1')
    event = MessageEvent(text='show config', source=source, message_type=MessageType.TEXT)
    key = 'agent:main:telegram:dm:c1'
    async def handler(event):
        result = await runner._run_agent(message=event.text, context_prompt='', history=[], source=source,
                                        session_id='s1', session_key=key)
        return None if result.get('already_sent') else result['final_response']
    chat.set_message_handler(handler)
    try:
        await chat._process_message_background(event, key)
    finally:
        plugins._reset_plugin_managers_for_tests()
    assert _frames(chat) == [('send', REWRITE)], _frames(chat)

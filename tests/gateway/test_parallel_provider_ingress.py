"""Ordinary native Discord ingress through real TurnRunner/AIAgent and a loopback model."""
import asyncio
import os
from pathlib import Path
import threading

import pytest

from gateway.config import GatewayConfig, Platform, StreamingConfig
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home
from tests.gateway.test_parallel_native_ingress import DiscordFixture, admissions, event
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_two_real_native_model_turns_overlap_keep_history_and_receipts(monkeypatch):
    import gateway.run as gateway_run
    entered = {name: threading.Event() for name in ('ONE-991', 'TWO-992')}
    release = threading.Event()

    def respond(request):
        users = [str(m.get('content', '')) for m in request['body'].get('messages', []) if m.get('role') == 'user']
        current = users[-1]
        name = next((n for n in entered if n in current), None)
        if name is not None:
            entered[name].set()
            assert release.wait(20), 'test release was not reached'
            return Text('ANSWER-' + name)
        return Text('CONTINUED')

    home = Path(os.environ['HERMES_HOME'])
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', 'alice,bob')
    with FakeLLMServer(respond, aux=lambda request: Text('Auxiliary')) as server:
        write_hermes_home(home, server.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW + '''
display:
  busy_input_mode: parallel
  tool_progress: off
  interim_assistant_messages: false
platform_toolsets:
  discord: [todo]
''')
        monkeypatch.setattr(gateway_run, '_hermes_home', home)
        monkeypatch.setattr(gateway_run, '_resolve_runtime_agent_kwargs', lambda: {
            'api_key': 'sk-fake-e2e', 'base_url': server.base_url,
            'provider': 'custom', 'api_mode': 'chat_completions'})
        runner = GatewayRunner(GatewayConfig())
        runner.config.streaming = StreamingConfig(enabled=False)
        adapter = DiscordFixture()
        adapter.gateway_runner = runner
        runner.adapters = {Platform.DISCORD: adapter}
        authority = await initialize_session_authority(runner, profile_id='default', instance_id='provider-parallel')
        runner._wire_adapter_handlers(adapter)
        runner._busy_input_mode = 'parallel'
        try:
            await adapter.handle_message(event(adapter, 'input-one', 'ONE-991'))
            await async_wait_until(entered['ONE-991'].is_set, timeout=20, what='first real model entered')
            await adapter.handle_message(event(adapter, 'input-two', 'TWO-992'))
            await async_wait_until(entered['TWO-992'].is_set, timeout=20, what='second model overlaps')
            assert len({row['target_session_id'] for row in admissions(authority)}) == 2
            release.set()
            await async_wait_until(lambda: len([s for s in adapter.sent if 'ANSWER-' in s]) == 2,
                                   timeout=30, what='both real replies')
            await async_wait_until(lambda: not adapter._background_tasks, timeout=30, what='native cleanup')
            requests = [r['body'] for r in server.requests if r['kind'] == 'main']
            for body in requests:
                user_text = str([m.get('content') for m in body['messages'] if m.get('role') == 'user'])
                assert not ('ONE-991' in user_text and 'TWO-992' in user_text)
            from gateway.session_parallel import conversation_scope
            from hermes_state_parallel import active_work
            source = runner._restored_source(next(iter(runner.session_store._entries.values())))
            assert active_work(authority.db, conversation_scope(source)) == []
        finally:
            release.set()
            await async_wait_until(lambda: not adapter._background_tasks, timeout=30, what='test cleanup')
            authority.db.close()

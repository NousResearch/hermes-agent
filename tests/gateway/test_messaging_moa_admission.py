"""A messaging ``/moa <prompt>`` is a durable FIFO turn of its own (#146026).

It ran in-process and wrote no ``session_admissions`` row, so after ``/new`` and only ``/moa``
turns a background result had no committed envelope to wake the chat with, and the turn could not
queue or be recovered. The preset now rides the turn's own admission and is armed only while that
row executes. Real GatewayRunner, Telegram-shaped adapter, SessionAuthority and drain over SQLite;
the agent turn is a stand-in that records the session route it would run on.
"""
import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session_results import execution_result
from hermes_state_runtime import list_session_admissions
from tests.gateway.test_shared_busy_receive_loop import Adapter
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_moa_prompt_is_admitted_and_its_preset_runs_on_that_turn_only(monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', 'fixture-user')
    runner = GatewayRunner(GatewayConfig())
    adapter = Adapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='moa')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='moa-chat', chat_type='dm', user_id='fixture-user')
    ran = []

    async def turn(event, source, quick_key, run_generation):
        override = runner._session_model_override(quick_key) or {}
        ran.append((event.text, override.get('provider')))
        if (captured := execution_result.get()) is not None:
            captured['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    def rows():
        return [(row['payload']['text'], row['status']) for sid in authority.sessions
                for row in list_session_admissions(authority.db, session_id=sid, pending_only=False)]
    try:
        await adapter.handle_message(MessageEvent(text='/moa compare the options', source=source, message_id='m1'))
        await async_wait_until(lambda: len(ran) == 1 and not adapter._background_tasks, what='the /moa turn ran')
        await adapter.handle_message(MessageEvent(text='and now?', source=source, message_id='m2'))
        await async_wait_until(lambda: len(ran) == 2 and not adapter._background_tasks, what='the next turn ran')
        assert rows() == [('compare the options', 'terminal'), ('and now?', 'terminal')]
        assert ran == [('compare the options', 'moa'), ('and now?', None)]
    finally:
        authority.db.close()

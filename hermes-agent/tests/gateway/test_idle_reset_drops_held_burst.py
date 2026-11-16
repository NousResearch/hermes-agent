"""An idle ``/new`` drops the text burst still held from the turn that just ended (#81370).

Queue-mode text sent during a turn is held for its burst window. When the turn ends inside that
window the session guard is gone, so ``/new`` takes the idle path; it must still drop the held text,
as the busy path does, or the old conversation's message becomes the new conversation's first turn.
Real GatewayRunner, DiscordAdapter, SessionAuthority and drain over SQLite; the turn is a stand-in.
"""
import asyncio

import pytest

from tests.waiting import async_wait_until

_WINDOW = 3.0  # the burst stays held well past the turn's end and the /new


async def _rig(monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id='sent')
    monkeypatch.setattr(adapter, 'send', send)
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter, busy_text_mode='queue', busy_text_timing=(_WINDOW, _WINDOW))
    running, finish, executed = asyncio.Event(), asyncio.Event(), []

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        if event.text == 'A':
            running.set()
            await finish.wait()
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    return adapter, authority, running, finish, executed


@pytest.mark.asyncio
@pytest.mark.parametrize('command,dropped,refused', [('new', True, False), ('status', False, False),
                                                     ('new', False, True)])
async def test_idle_command_after_a_turn_settles_the_held_text(monkeypatch, command, dropped, refused):
    """A refused /new (slash access allows this caller /status only) resets nothing, so it must not
    destroy the held text either: B still runs."""
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.session_ingress_burst import holds_busy_burst

    adapter, authority, running, finish, executed = await _rig(monkeypatch)
    source = adapter.build_source(chat_id='dm-42', chat_type='dm', user_id='42')
    if refused:
        adapter.gateway_runner.config.platforms[Platform.DISCORD] = PlatformConfig(
            enabled=True, extra={'allow_admin_from': ['99'], 'user_allowed_commands': ['status']})
        assert adapter.gateway_runner._check_slash_access(source, command) is not None
    await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await asyncio.wait_for(running.wait(), 10)
    key = adapter._event_session_key(MessageEvent(text='', source=source))
    held = asyncio.create_task(adapter.handle_message(MessageEvent(text='B', source=source, message_id='B')))
    await async_wait_until(lambda: holds_busy_burst(adapter, key), timeout=5, what='B held as a burst')
    finish.set()
    await async_wait_until(lambda: key not in adapter._active_sessions, timeout=5, what='the turn to end')
    assert holds_busy_burst(adapter, key), 'the burst window closed before the command: no idle case'
    await adapter.handle_message(MessageEvent(text=f'/{command}', source=source, message_id='C'))
    await asyncio.wait_for(held, _WINDOW + 10)
    if dropped:
        # health: allow HX013 -- absence proof: a leaked burst commits once its window closes
        await asyncio.sleep(_WINDOW + 1.0)
        assert 'B' not in executed, executed
    else:
        await async_wait_until(lambda: 'B' in executed, timeout=_WINDOW + 10, what='B still ran')
    await async_wait_until(lambda: all(live.task is None or live.task.done() for live in authority.sessions.values()),
                           timeout=10, what='the drains to exit')
    await asyncio.to_thread(authority.db.close)

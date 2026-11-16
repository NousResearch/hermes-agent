"""A queued messaging follow-up the drain refuses is told it was not answered, never that it is saved."""
import asyncio

import pytest


@pytest.mark.asyncio
async def test_refused_followups_are_each_told_they_were_not_answered(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import admit_message
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)  # B and C wait queued
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    waiters = []
    for name in ('B', 'C'):
        event = MessageEvent(text=name, source=source, message_id=name)
        with native_callback(runner, event, get_hermes_home()):
            waiters.append(asyncio.create_task(admit_message(authority, event)))
            for _ in range(200):
                if len(authority.waiters) == len(waiters):
                    break
                await asyncio.sleep(0.01)
    key = runner.session_store._generate_session_key(source)
    owner = runner.session_store.peek_session_id(key)
    db = authority.db
    # The chat moves to an older conversation while B and C wait (the real /resume handler).
    db.create_session('older', source='discord', user_id='42', session_key=key, chat_id='42', chat_type='dm')
    db.append_message('older', 'user', 'earlier')
    db.set_session_title('older', 'older-chat')
    resume = MessageEvent(text='/resume older-chat', source=source)
    with native_callback(runner, resume, get_hermes_home()):
        await runner._handle_message(resume)
    ran = []

    async def handle(event):
        ran.append(event.text)
        return 'ran'
    monkeypatch.setattr(runner, '_handle_message', handle)

    await asyncio.wait_for(SessionAuthority._drain(authority, SessionRef(authority.profile_id, owner)), 10)
    replies = await asyncio.wait_for(asyncio.gather(*waiters), 5)

    rows = {r['request_id']: (r['status'], r['outcome'])
            for r in list_session_admissions(db, session_id=owner, pending_only=False)}
    assert (ran, rows) == ([], {'B': ('terminal', 'cancelled'), 'C': ('terminal', 'cancelled')})
    for reply in replies:  # each refused message gets its own notice, and none claims it was saved
        assert reply and 'saved' not in reply and 'paused' not in reply, replies
    assert not authority.sessions[owner].pause_notified

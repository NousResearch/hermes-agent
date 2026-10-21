"""Automation wakes into an auto-created Discord thread must admit against the thread's
later in-thread envelopes.

The session ORIGIN (the channel message that auto-created the thread) carries
``auto_thread_created`` + ``auto_thread_initial_name`` so pre-existing or renamed threads are
never renamed again; every later in-thread message's committed envelope carries neither field.
Wake provenance compares the origin-rebuilt event source against the LAST committed envelope,
so those rename/display fields must stay out of the route-identity comparison — otherwise every
wake into an auto-created thread is refused ``admission_conflict`` and requeued forever (parent
agents never resume after their children finish; reproduced on a live Discord gateway).
"""
import pytest


async def _auto_thread_session_with_followup(monkeypatch):
    """A Discord thread session created by the auto-thread trigger message (rename metadata on
    the ORIGIN) followed by one plain in-thread message — the live ``auto_thread`` shape."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import claim_session_input, settle_session_input
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)

    origin_source = adapter.build_source(
        chat_id='77', chat_type='thread', thread_id='77', user_id='42',
        parent_chat_id='55', message_id='m-origin',
        auto_thread_created=True, auto_thread_initial_name='spawned thread')
    followup_source = adapter.build_source(
        chat_id='77', chat_type='thread', thread_id='77', user_id='42',
        parent_chat_id='55', message_id='m-followup')

    async def admit_and_settle(event):
        with native_callback(runner, event, get_hermes_home()):
            receipt = await authority.admit_native(event)
        claim = claim_session_input(authority.db, epoch=authority.epoch, session_id=receipt.ref.session_id)
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=receipt.admission_id,
                             generation=claim['generation'], outcome='completed')
        return receipt

    await admit_and_settle(MessageEvent(text='created the thread', source=origin_source,
                                        message_id='m-origin'))
    followup = await admit_and_settle(MessageEvent(text='later in-thread message',
                                                   source=followup_source, message_id='m-followup'))
    return runner, adapter, authority, origin_source, followup_source, followup.ref.session_id


@pytest.mark.asyncio
async def test_route_identity_ignores_auto_thread_rename_metadata(monkeypatch):
    """Origin (auto-thread creation message) and a later in-thread envelope are the same ROUTE."""
    from gateway.session_automation import _route_identity

    _, _, _, origin_source, followup_source, _ = await _auto_thread_session_with_followup(monkeypatch)
    assert _route_identity(origin_source) == _route_identity(followup_source)


@pytest.mark.asyncio
async def test_wake_into_auto_created_thread_admits_after_later_in_thread_messages(monkeypatch):
    """The wake's source is the origin-rebuilt source (rename metadata and all); it must admit
    against the newest in-thread envelope instead of being refused admission_conflict forever."""
    from gateway.platforms.event import MessageEvent

    runner, adapter, authority, origin_source, _, sid = await _auto_thread_session_with_followup(monkeypatch)
    route = runner.session_store._generate_session_key(origin_source)
    event = MessageEvent(text='finished', source=origin_source, internal=True,
                         metadata={'gateway_session_key': route, 'gateway_session_id': sid})
    admitted = await authority.admit_automation(adapter, event, 'completion')
    assert admitted.ref.session_id == sid

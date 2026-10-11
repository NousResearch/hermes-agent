"""A queued follower is not claimed while the settled turn's crash marker is still with its adapter."""
import asyncio
from functools import partial
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_follower_is_claimed_only_after_the_settled_turn_released_its_adopted_marker(tmp_path, monkeypatch):
    """The adapter that sends turn one's reply adopted its crash marker and releases it only once the
    reply is in the delivery ledger (seconds with auto-TTS). Claiming the follower first lets its
    mark_turn_active overwrite that one-slot marker: a kill in between leaves the follower's marker
    (cleared at boot as unknown) and turn one's persisted reply is never sent."""
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import AsyncSessionStore
    from gateway.session_contract import Principal, SessionRef, Submission
    from gateway.session_ingress import admit_message
    from tests.gateway.test_prompt_attachments import _authority

    one_running, follower_queued = asyncio.Event(), asyncio.Event()
    slot_at_follower = []

    async def answer(event):
        if event.text == 'two':
            slot_at_follower.append(store._entries[key].active_turn_token)
        await GatewayRunner._mark_durable_active_turn(runner, event, key)
        if event.text == 'one':
            one_running.set()
            await follower_queued.wait()
        return 'reply to ' + event.text
    authority = await _authority(tmp_path, monkeypatch, answer)
    store = authority.runner.session_store
    source = authority.sessions['s'].source
    key = store.get_or_create_session(source).session_key
    runner = SimpleNamespace(async_session_store=AsyncSessionStore(store))
    runner._clear_durable_active_turn = partial(GatewayRunner._clear_durable_active_turn, runner)
    authority.runner._clear_durable_active_turn = runner._clear_durable_active_turn
    adapter = SimpleNamespace(gateway_runner=runner)
    actor = Principal('human', 'p', frozenset({'session:submit'}), 't')
    ref = SessionRef('p', 's')

    async def admit_native(event):
        return await authority.submit(actor, Submission(event.text, ref, {'text': event.text}, 'queue'))
    monkeypatch.setattr(authority, 'admit_native', admit_native)

    async def adapter_lifecycle(event):
        # BasePlatformAdapter._process_message_background: hand-off on, await the reply, then
        # (auto-TTS, ledger write) release the adopted marker.
        event._turn_marker_handoff = True
        reply = await admit_message(authority, event)
        await asyncio.sleep(0.3)
        await BasePlatformAdapter._release_turn_marker(adapter, event)
        return reply

    first = asyncio.create_task(adapter_lifecycle(MessageEvent(text='one', source=source)))
    await asyncio.wait_for(one_running.wait(), 5)
    follower = await authority.submit(actor, Submission('two', ref, {'text': 'two'}, 'queue'))
    follower_queued.set()
    assert await asyncio.wait_for(first, 10) == 'reply to one'
    from hermes_state_runtime import get_session_admission
    async with asyncio.timeout(10):
        while get_session_admission(authority.db, admission_id=follower.admission_id)['status'] != 'terminal':
            await asyncio.sleep(0.02)
    assert slot_at_follower == [None], "the follower's marker overwrote turn one's before its reply was ledgered"

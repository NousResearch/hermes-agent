"""A native reply reaches the chat exactly once even when an already-running drain claims and
runs the admission before ``admit_message`` resumes to register its delivery waiter
(andrexibiza-6, andrexibiza-5480238075-1). Every native reply goes through one of two senders:
the drain's ``deliver_settled`` (no live waiter) or the waiting adapter lifecycle."""
import asyncio

import pytest

from gateway.session_contract import Submission


@pytest.mark.asyncio
async def test_native_waiter_registered_after_the_claim_still_gets_the_only_delivery(tmp_path, monkeypatch):
    from gateway import session_ingress
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    from gateway.session_contract import Principal, SessionRef
    from tests.gateway.test_prompt_attachments import _authority as _store_authority

    turn_started = asyncio.Event()

    async def answer(event):
        turn_started.set()
        return 'model reply'
    authority = await _store_authority(tmp_path, monkeypatch, answer)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='c')
    authority.sessions['s'] = LiveSession(source, 's')
    authority.runner._adapter_for_source = lambda source: object()
    drain_sent = []

    async def deliver(adapter, event, session_key, response):
        drain_sent.append(response)
    monkeypatch.setattr(session_ingress, 'deliver_response', deliver)
    actor = Principal('human', 'p', frozenset({'session:submit'}), 't')

    async def admit_native(event):
        receipt = await authority.submit(actor, Submission('m1', SessionRef('p', 's'), {'text': 'hi'}, 'queue'))
        # The admitting caller resumes only after the drain already claimed and started the turn.
        await turn_started.wait()
        return receipt
    monkeypatch.setattr(authority, 'admit_native', admit_native)
    adapter_sent = await asyncio.wait_for(
        session_ingress.admit_message(authority, MessageEvent(text='hi', source=source)), 10)
    await asyncio.wait_for(authority.sessions['s'].task, 10)
    deliveries = drain_sent + ([adapter_sent] if adapter_sent else [])
    assert deliveries == ['model reply'], f'the reply was sent {len(deliveries)} times: {deliveries}'
    assert not authority.native_waiters and not authority.pending_deliveries

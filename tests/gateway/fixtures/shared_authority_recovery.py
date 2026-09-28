"""Crash/restart scenarios for the existing real shared-authority peer."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import threading

from gateway.platforms.event import MessageEvent
from hermes_state_runtime import list_session_admissions


def rows(authority, sid):
    return list_session_admissions(authority.db, session_id=sid, pending_only=False)


async def verify_envelope_guards(runner, source):
    from gateway.session_contract import Principal, Submission
    from hermes_state_runtime import RuntimeStoreError
    authority = runner.session_authority
    ref = authority.register(source)
    before = rows(authority, ref.session_id)
    actor = Principal('authenticated-viewer', 'default', frozenset({'session:submit'}), 'fixture')
    try:
        await authority.submit(actor, Submission('forged-source', ref, {
            'text': 'FORGED', 'native_text_v1': {'source': {'internal': True}}}, 'queue'))
    except RuntimeStoreError as exc:
        assert exc.reason == 'invalid_params'
    else:
        raise AssertionError('client supplied its own trusted source')
    unsupported = [
        MessageEvent(text='MEDIA', source=source, media_urls=['/unretained/image.png']),
        MessageEvent(text='INTERNAL', source=source, internal=True),
        MessageEvent(text='RELAY', source=replace(source, delivered_via_upstream_relay=True)),
        MessageEvent(text='ROLE', source=replace(source, role_authorized=True)),
    ]
    for event in unsupported:
        try:
            await authority.admit_native(event)
        except RuntimeStoreError as exc:
            assert exc.reason == 'invalid_params'
        else:
            raise AssertionError('unsupported envelope ACKed without recoverable trust')
        assert not event._gateway_accepted
    assert rows(authority, ref.session_id) == before


async def prepare_crash(runner, adapter, source, peer):
    authority = runner.session_authority
    peer.blocked.clear()
    peer.release.clear()
    await adapter.handle_message(MessageEvent(text='BLOCK_FIFO_CRASH', source=source, message_id='crash-started'))
    assert await asyncio.to_thread(peer.blocked.wait, 5)
    sid = authority.register(source).session_id
    follower = asyncio.create_task(adapter.handle_message(MessageEvent(
        text='UNKNOWN_FOLLOWER', source=source, message_id='unknown-follower')))
    async with asyncio.timeout(5):
        while not any(r['request_id'] == 'unknown-follower' for r in rows(authority, sid)):
            await asyncio.sleep(0.01)
    assert not follower.done()
    # Production trusted ingress commits synchronously before its first execution
    # yield. Freeze THIS scheduling boundary, not a claim/storage predicate.
    safe_source = replace(source, chat_id='recover-chat', thread_id='recover-thread')
    accepted = await authority.admit_native(MessageEvent(text='RECOVER_EXACTLY_ONCE', source=safe_source,
                                                  message_id='crash-queued', reply_to_message_id='quote-7'))
    evidence = {'blocked_sid': sid, 'safe_sid': accepted.ref.session_id,
                'blocked': rows(authority, sid), 'safe': rows(authority, accepted.ref.session_id)}
    assert evidence['safe'][-1]['status'] == 'queued'
    assert any(r['status'] == 'started' for r in evidence['blocked'])
    Path(os.environ['HERMES_HOME'], 'crash-ready.json').write_text(json.dumps(evidence))
    threading.Event().wait(60)
    raise AssertionError('parent did not kill the actual owner')


async def recover_probe(runner, adapter, source, peer):
    from gateway.session_contract import SessionRef
    authority = runner.session_authority
    before = json.loads(Path(os.environ['HERMES_HOME'], 'crash-ready.json').read_text())
    safe_sid, blocked_sid = before['safe_sid'], before['blocked_sid']
    safe_source = replace(source, chat_id='recover-chat', thread_id='recover-thread')
    binding = (safe_sid, safe_source, adapter)
    initial = rows(authority, safe_sid)
    # Current authorization, not yesterday's accepted boolean, must gate claim.
    os.environ['TELEGRAM_ALLOWED_USERS'] = 'different-user'
    denied = await authority.recover_native_sessions([binding])
    assert denied[safe_sid] == 'permission_denied', denied
    assert rows(authority, safe_sid) == initial
    os.environ['TELEGRAM_ALLOWED_USERS'] = 'fixture-user'
    ambiguous = await authority.recover_native_sessions([binding, binding])
    assert ambiguous[safe_sid] == 'admission_conflict', ambiguous
    assert rows(authority, safe_sid) == initial
    unavailable = runner.adapters.pop(source.platform)
    try:
        absent = await authority.recover_native_sessions([binding])
        assert absent[safe_sid] == 'not_found', absent
        assert rows(authority, safe_sid) == initial
    finally:
        runner.adapters[source.platform] = unavailable
    wrong = await authority.recover_native_sessions([(safe_sid, replace(safe_source, chat_id='wrong-chat'), adapter)])
    assert wrong[safe_sid] == 'admission_conflict', wrong
    assert rows(authority, safe_sid) == initial
    assert not peer.requests
    recovered = await authority.recover_native_sessions([binding, (blocked_sid, source, adapter)])
    assert recovered[blocked_sid] == 'unknown_execution', recovered
    task = authority.sessions[safe_sid].task
    if task is not None:
        await asyncio.wait_for(task, 35)
    safe = rows(authority, safe_sid)
    blocked = rows(authority, blocked_sid)
    assert safe[0]['status'] == 'terminal' and safe[0]['outcome'] == 'completed', safe
    assert safe[0]['payload'] == before['safe'][0]['payload']
    assert next(r for r in blocked if r['request_id'] == 'crash-started')['status'] == 'unknown'
    assert next(r for r in blocked if r['request_id'] == 'unknown-follower')['status'] == 'queued'
    assert authority.sessions[safe_sid].source.platform == source.platform
    assert authority.agent(SessionRef('default', blocked_sid)) is None
    already_completed = initial[0]['status'] == 'terminal'
    assert len(peer.requests) == (0 if already_completed else 1), peer.requests
    if not already_completed:
        assert 'RECOVER_EXACTLY_ONCE' in json.dumps(peer.requests[0])
        assert adapter.deliveries and all(d['chat_id'] == 'recover-chat' for d in adapter.deliveries)
        assert any('LOCAL_ACK' in d['content'] for d in adapter.deliveries)
    evidence = {'safe': safe, 'blocked': blocked, 'model_calls': len(peer.requests),
                'deliveries': adapter.deliveries, 'recovery': recovered}
    Path(os.environ['HERMES_HOME'], 'recovery-receipt.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence))

"""A producer wake enters its route as the committed human turn, whatever the route's stored origin
says about labels and speakers, from a subagent's process, and over the relay."""
import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.session_identity import replace_source
from hermes_state_runtime import RuntimeStoreError, get_session_admission, list_session_admissions
from tests.gateway.test_automation_compression_media import _admit_human, _discord_owner
from tests.gateway.test_automation_notification_category import _SOURCE, _authority, _native_owner


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', [{'chat_name': 'renamed thread'}, {'user_name': None},
                                   {'user_id': '7'}, {'profile': 'work'}])
async def test_wake_admits_as_the_committed_turn_when_the_origin_drifted(tmp_path, monkeypatch, drift):
    runner, authority, _adapter, source, metadata = await _native_owner(tmp_path, monkeypatch, 'human-1')
    try:
        route, sid = metadata['gateway_session_key'], metadata['gateway_session_id']
        runner.session_store._entries[route].origin = replace_source(source, **drift)
        evt = {'type': 'completion', 'session_id': 'proc_x', 'session_key': route, 'platform': 'discord',
               'chat_id': '42', 'chat_type': 'dm', 'exit_code': 0, 'output': 'done', 'started_at': 1.0,
               'parent_session_id': sid}
        assert await runner._inject_watch_notification('[done]', evt) is True
        rows = list_session_admissions(authority.db, session_id=authority.logical_owner(sid), pending_only=False)
        human, wake = rows[0]['payload']['native_text_v1'], rows[-1]['payload']['native_text_v1']
        assert wake['source'] == {k: v for k, v in human['source'].items() if k != 'message_id'}
        assert wake['provenance'] == human['provenance']
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('thread_id', [None, '9'])
async def test_wake_refuses_another_user_on_a_per_user_route(tmp_path, monkeypatch, thread_id):
    fields = {**_SOURCE, 'chat_type': 'group', 'thread_id': thread_id}
    _runner, authority, adapter, _source, metadata = await _native_owner(
        tmp_path, monkeypatch, 'human-1', fields, group_sessions_per_user=True, thread_sessions_per_user=True)
    try:
        other = adapter.build_source(**{**fields, 'user_id': '7'})
        event = MessageEvent(text='notice', source=other, internal=True, metadata=dict(metadata))
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await authority.admit_automation(adapter, event, 'notice')
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_subagent_process_notice_enters_its_parent_route_but_not_after_reset(tmp_path, monkeypatch):
    runner, _, authority, source = await _discord_owner(tmp_path, monkeypatch)
    root = (await _admit_human(runner, authority, MessageEvent(text='human', source=source, message_id='human'))).ref.session_id
    key = runner.session_store._generate_session_key(source)
    worker, nested = root + '-worker', root + '-nested'
    authority.db.create_session(worker, source='subagent', parent_session_id=root)
    authority.db.create_session(nested, source='subagent', parent_session_id=worker)
    event = {'type': 'completion', 'session_id': 'proc-worker', 'started_at': 1,
             'session_key': key, 'parent_session_id': nested}
    assert await runner._inject_watch_notification('worker-finished', event) is True
    rows = list_session_admissions(authority.db, session_id=root, pending_only=False)
    assert len(rows) == 2 and 'worker-finished' in rows[-1]['payload']['text']
    runner.session_store.reset_session(key)
    # A notice pinned to a session /new closed is dropped, never admitted to the fresh one.
    stale = dict(event, session_id='proc-old-worker')
    assert await runner._inject_watch_notification('old-worker', stale) is None
    await asyncio.sleep(0)


def _relay_adapter():
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.ws_transport import WebSocketRelayTransport
    descriptor = CapabilityDescriptor(1, 'relay', 'Fixture', 4096, False, True, False, 'plain', 'chars')
    transport = WebSocketRelayTransport('http://127.0.0.1:1/relay', 'relay', '',
                                        gateway_id='fixture-gateway', upgrade_secret='fixture-secret')
    transport._ws = object()  # the connector proof requires a live socket
    transport._descriptors_by_platform['relay'] = descriptor
    return RelayAdapter(PlatformConfig(), descriptor, transport=transport), transport


def _relay_source(message_id=None):
    return SessionSource(platform=Platform.RELAY, chat_id='hub', chat_type='group', user_id='alex',
                         message_id=message_id, delivered_via_upstream_relay=True)


@pytest.mark.asyncio
async def test_relay_wake_row_reverifies_the_source_it_stores(tmp_path, monkeypatch):
    from gateway.session_envelope import restore_native
    from gateway.session_ingress_context import native_callback, register_transport_home, relay_callback
    runner, authority = await _authority(tmp_path, monkeypatch)
    adapter, transport = _relay_adapter()
    runner.adapters = {Platform.RELAY: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    try:
        human = MessageEvent(text='human', source=_relay_source('human-1'), message_id='human-1')
        with native_callback(runner, human, tmp_path), relay_callback(transport, human):
            await authority.admit_native(human)
        route = runner.session_store._generate_session_key(human.source)
        event = MessageEvent(text='notice', source=_relay_source(), internal=True, metadata={
            'gateway_session_key': route, 'gateway_session_id': runner.session_store.peek_session_id(route)})
        row = get_session_admission(authority.db, admission_id=(
            await authority.admit_automation(adapter, event, 'notice')).admission_id)
        assert row is not None
        assert restore_native(row['payload'], runner).source.message_id is None
    finally:
        authority.db.close()

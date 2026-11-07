"""Native uploads transfer only after durable admission to the adopted execution."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import threading
from unittest.mock import patch

import pytest


from gateway.session_authority import SessionAuthority, initialize_session_authority
from hermes_state_runtime import claim_session_input, list_session_admissions
from tests.gateway.test_native_image_steering import install_route, pixel_bytes, assert_provider_projections
from tests.agent.test_tool_call_incremental_persistence import _mock_response, _mock_tool_call, _durable_messages
from gateway.config import Platform
from tests.waiting import async_wait_until

pytest_plugins = ["tests.gateway.test_native_image_steering"]


async def owner(setup, monkeypatch):
    runner, agent, images = setup
    adapter, key, event = install_route(runner, agent, images, Platform.SLACK)
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='native-transfer')
    runner._wire_adapter_handlers(adapter)
    first = await authority.admit_native(replace(event, text='Inspect the first diagram', message_id='initial',
                                                media_urls=[], media_types=[]))
    sid = first.ref.session_id
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    authority.executing[sid] = row['generation']
    authority.adopt_agent(sid, row['generation'], agent)
    agent.session_id = sid
    agent._session_db = authority.db
    agent._session_db_created = True
    return runner, agent, images, adapter, key, event, authority, row


@pytest.mark.asyncio
@pytest.mark.parametrize('ingress', ['direct', 'adapter', 'relay'])
async def test_admitted_upload_reaches_active_request_from_retained_snapshot(setup, monkeypatch, ingress):
    runner, agent, images, adapter, key, event, authority, running = await owner(setup, monkeypatch)
    original = [p.read_bytes() for p in images]
    entered, release = threading.Event(), threading.Event()
    requests = []

    def dispatch(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return '{"result":"complete"}'

    def respond(**kwargs):
        requests.append(deepcopy(kwargs))
        if len(requests) == 1:
            return _mock_response(content='Inspecting', finish_reason='tool_calls', tool_calls=[
                _mock_tool_call('terminal', '{"command":"first"}', call_id='first'),
                _mock_tool_call('terminal', '{"command":"second"}', call_id='second')])
        return _mock_response(content='Revised answer')

    agent.client.chat.completions.create.side_effect = respond
    with patch('model_tools.handle_function_call', side_effect=dispatch):
        task = asyncio.create_task(asyncio.to_thread(agent.run_conversation, 'Inspect the first diagram'))
        try:
            await async_wait_until(entered.is_set, timeout=10, what='first tool')
            if ingress == 'direct':
                await asyncio.wait_for(runner._handle_message(event), 10)
            elif ingress == 'adapter':
                adapter._active_sessions[key] = asyncio.Event()
                adapter._session_tasks[key] = asyncio.current_task()
                await asyncio.wait_for(adapter.handle_message(event), 10)
                assert agent.steer('Focus on geometry')
            else:
                from tests.gateway.test_native_image_steering import deliver_relay
                await deliver_relay(runner, event)
            await async_wait_until(lambda: len(list_session_admissions(
                authority.db, session_id=agent.session_id, pending_only=False)) == 2,
                timeout=10, what='durable upload')
            rows = list_session_admissions(authority.db, session_id=agent.session_id, pending_only=False)
            assert len(rows) == 2, 'image steering must be durably admitted before delivery'
            assert rows[1]['outcome'] == 'steering'
            assert rows[1]['generation'] == running['generation']
            for path in images:
                path.unlink()
        finally:
            release.set()
            result = await asyncio.wait_for(task, 15)
    assert result['final_response'] == 'Revised answer'
    assert len(requests) == 2
    assert requests[-1]['messages'][:len(requests[0]['messages'])] == requests[0]['messages']
    assert requests[-1]['tools'] == requests[0]['tools']
    if ingress == 'adapter':
        assert 'Focus on geometry' in requests[-1]['messages'][-1]['content'][0]['text']
    assert pixel_bytes(requests[-1]['messages'][-1]['content']) == original
    replay = _durable_messages(authority.db.db_path, agent.session_id)
    from agent.image_replay import replay_recent_images
    assert isinstance(replay[-2]['content'], str)
    assert replay_recent_images(replay, 3) == 2
    for messages in (requests[-1]['messages'], replay[:-1]):
        assert_provider_projections(messages, original)
    from gateway.session_queued_controls import steer_transfers
    with authority.sessions[agent.session_id].event_stream.lock:
        transfers = steer_transfers(authority, agent.session_id, running['generation'])
    assert transfers == {rows[1]['admission_id']: True}
    from hermes_state_runtime import settle_session_input, get_session_admission
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=running['admission_id'],
                         generation=running['generation'], outcome='completed', _transfers=transfers)
    from hermes_state import SessionDB
    path = authority.db.db_path
    authority.db.close()
    with SessionDB(db_path=path) as reopened:
        assert get_session_admission(reopened, admission_id=rows[1]['admission_id'])['outcome'] == 'steered'

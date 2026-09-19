"""Real facade lease interruption through the gateway's result reprojection."""
import copy
import json
from types import MethodType

import pytest

from gateway.session_api_turn import api_execution
from tests.gateway.test_files_live_provenance import live


@pytest.mark.parametrize('files', [True, False], ids=['files', 'ordinary-cached'])
@pytest.mark.parametrize('replacement', [False, True], ids=['cached-history', 'replacement-history'])
@pytest.mark.parametrize('hard_stop', [False, True], ids=['carry', 'hard-stop'])
def test_gateway_carries_unadmitted_input_to_following_sql(
        live, monkeypatch, record_property, files, replacement, hard_stop):
    from run_agent import AIAgent
    from agent.conversation_loop import run_conversation
    live.admit('/private/previous-files.txt', 'previous-files')
    live.ctx.history = live.store.load_transcript('owned')
    agent, reused = live.turn._resolve_turn_agent(live.route, 'api_server', '', 3, None, {})
    assert reused and agent is live.agent
    history, observed, _ = live.turn._load_turn_history(agent, reused)
    if replacement:
        history = [{'role': 'user', 'content': 'replacement question'},
                   {'role': 'assistant', 'content': 'replacement answer', 'api_content': 'ordinary sidecar'}]
    history_before = copy.deepcopy(history)
    canonical_before = copy.deepcopy(agent._session_messages)
    live.sent.clear()
    assert live.db.acquire_session_turn_lease('owned', 'conflicting-holder', ttl_seconds=30, wait_seconds=0)
    monkeypatch.setattr('agent.relay_runtime.SESSION_COORDINATOR.acquire_conversation',
                        lambda **kw: pytest.fail('coordinator must not run'))
    agent.run_conversation = MethodType(AIAgent.run_conversation, agent)
    agent.interrupt('following input', hard_cancel=hard_stop)
    safe = 'accepted prompt\n\n[Attached file: "current.txt"]'
    private = '/private/unadmitted-current.txt'
    data = 'data:image/png;base64,Y2Fycnk='
    live.ctx.message = 'ordinary input with sidecar'
    live.ctx.inbound_message_id = 'carried-turn'
    live.ctx.persist_user_display_kind = 'internal_notification'
    live.ctx.persist_user_display_metadata = {'origin': 'gateway-lease'}
    token = api_execution.set({'content': [{'type': 'text', 'text': private},
        {'type': 'image_url', 'image_url': {'url': data}}],
        'files_persist_user_message': safe} if files else None)
    try:
        result = live.turn._run_conversation_with_approval(
            agent, history, observed, None if files else 'ordinary caption', 1234.5)
    finally:
        api_execution.reset(token)
    assert live.sent == []
    assert agent._session_messages == canonical_before
    assert history == history_before
    assert result.get('interrupted') is True, result
    assert result['interrupt_message'] == 'following input'
    assert result['api_calls'] == 0 and result['completed'] is False
    assert result['messages'][:len(history)] == history
    assert result['messages'] is not history
    assert private not in json.dumps(result) and data not in json.dumps(result)
    record_property('gateway_unadmitted_result', json.dumps(result))
    live.db.release_session_turn_lease('owned', 'conflicting-holder')
    if hard_stop:
        assert result['messages'] == history
        assert not any(r['platform_message_id'] == 'carried-turn' for r in live.db.get_messages('owned'))
        return
    carried = result['messages'][-1]
    assert carried['content'] == (safe if files else 'ordinary caption')
    assert carried.get('api_content') == (None if files else live.ctx.message)
    assert carried['_persist_after_admission_interrupt'] is True
    assert carried['timestamp'] == 1234.5
    assert carried['platform_message_id'] == 'carried-turn'
    assert carried['display_metadata'] == {'origin': 'gateway-lease'}
    assert carried['display_kind'] == 'internal_notification'
    # Real following gateway boundary and core flush; bypass only facade admission
    # because the test must not start a coordinator or lease refresher.
    agent.run_conversation = MethodType(run_conversation, agent)
    live.ctx.message = 'following input'
    live.ctx.inbound_message_id = 'following-turn'
    following = live.turn._run_conversation_with_approval(
        agent, copy.deepcopy(result['messages']), None, None, None)
    rows = live.db.get_messages('owned')
    stored = [r for r in rows if r['platform_message_id'] == 'carried-turn']
    assert len(stored) == 1
    for key in ('content', 'timestamp', 'platform_message_id', 'display_metadata', 'display_kind'):
        assert stored[0][key] == carried[key]
    assert stored[0]['api_content'] == carried.get('api_content')
    assert private not in json.dumps(rows, default=str) and data not in json.dumps(rows, default=str)
    # Returned snapshot mutation cannot affect canonical rows or the next result.
    result['messages'][-1]['content'] = 'caller mutation'
    assert 'caller mutation' not in json.dumps(agent._session_messages)
    assert 'caller mutation' not in json.dumps(following)
    record_property('following_sql', json.dumps(rows, default=str))


@pytest.mark.parametrize('fault', ['clone', 'tampered', 'foreign', 'stale', 'raw',
                                 'tampered-envelope', 'previous-invocation'])
def test_gateway_refuses_unproven_carry_results(live, monkeypatch, fault):
    from run_agent import AIAgent
    live.admit('/private/previous.txt', 'previous')
    assert live.db.acquire_session_turn_lease('owned', 'other', ttl_seconds=30, wait_seconds=0)
    monkeypatch.setattr('agent.relay_runtime.SESSION_COORDINATOR.acquire_conversation',
                        lambda **kw: pytest.fail('coordinator must not run'))
    agent = live.agent
    facade = MethodType(AIAgent.run_conversation, agent)
    def produce(message, **kwargs):
        agent.interrupt('following', hard_cancel=fault in ('tampered-envelope', 'previous-invocation'))
        return facade(message, **kwargs)
    agent.run_conversation = produce
    stale = live.turn._run_conversation_with_approval(agent, [], None, None, None)
    def alternate(message, **kwargs):
        if fault == 'stale':
            return stale
        if fault == 'raw':
            return {'messages': [{'role': 'user', 'content': '/private/raw-alternate'}],
                    'interrupted': True, 'trusted_carry': True}
        if fault == 'foreign':
            foreign = copy.copy(agent)
            foreign.interrupt('following')
            return AIAgent.run_conversation(foreign, message, **kwargs)
        result = produce(message, **kwargs)
        if fault == 'tampered-envelope':
            result['interrupt_message'] = '/private/tampered-envelope'
            return result
        if fault == 'previous-invocation':
            produce(message, **kwargs)
            return result
        if fault == 'clone':
            return copy.deepcopy(result)
        result['messages'][-1]['content'] = '/private/tampered'
        return result
    agent.run_conversation = alternate
    result = live.turn._run_conversation_with_approval(agent, [], None, None, None)
    assert result['failure_reason'] == 'files_result_unavailable'
    assert result['messages'] == []

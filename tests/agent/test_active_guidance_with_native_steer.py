"""Native steering and receipt-bearing guidance share the real pre-API boundary."""
from copy import deepcopy


def test_guidance_and_native_steer_preserve_prefix_and_deliver_once(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from agent.active_turn_guidance import GuidanceInbox
    from agent.turn_context import _reset_per_turn_agent_state
    from agent.turn_iteration_prep import prepare_iteration
    from hermes_state import SessionDB
    from run_agent import AIAgent

    agent = AIAgent(session_db=SessionDB(db_path=tmp_path / 'proof.db'),
                    model='test-model', provider='openai-compat', api_key='test',
                    base_url='http://127.0.0.1:1/v1', max_iterations=4,
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    try:
        _reset_per_turn_agent_state(agent)
        inbox = GuidanceInbox('owner:1', 'session')
        target = inbox.start('primary')
        request = {'runtimeInstanceId': target['runtimeInstanceId'], 'sessionId': 'session',
                   'expectedTurnId': target['turnId'], 'messageId': 'guidance', 'text': 'LITERAL_GUIDANCE'}
        assert inbox.request(request)['state'] == 'pending_incorporation'
        agent._active_turn_guidance = inbox
        assert agent.steer('NATIVE_STEER')
        messages = [
            {'role': 'user', 'content': 'original objective'},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'call', 'content': 'retained tool result'},
        ]
        prefix = deepcopy(messages)
        prepared = prepare_iteration(agent, messages=messages, api_call_count=1,
                                     user_message='original objective', current_turn_user_idx=0)
        assert prepared.messages[:len(prefix)] == prefix
        tail = prepared.messages[len(prefix):]
        assert len(tail) == 1 and tail[0]['role'] == 'user'
        assert 'LITERAL_GUIDANCE' not in tail[0]['content']
        assert tail[0]['content'].count('NATIVE_STEER') == 1
        assert inbox.request(request, lookup=True)['state'] == 'pending_incorporation'
        native_prefix = deepcopy(prepared.messages)
        messages = prepared.messages + [
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'next', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'next', 'content': 'next tool result'},
        ]
        prepared = prepare_iteration(agent, messages=messages, api_call_count=2,
                                     user_message='original objective', current_turn_user_idx=0)
        assert prepared.messages[:len(native_prefix)] == native_prefix
        assert prepared.messages[-1] == {'role': 'user', 'content': 'LITERAL_GUIDANCE'}
        assert inbox.request(request, lookup=True)['state'] == 'incorporated'
        # A second preparation cannot consume either source a second time.
        settled = deepcopy(prepared.messages)
        again = prepare_iteration(agent, messages=prepared.messages, api_call_count=3,
                                  user_message='original objective', current_turn_user_idx=0)
        assert again.messages == settled
    finally:
        agent._session_db.close()

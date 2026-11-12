"""A declared API conversation restores at its current physical compression tip."""
from gateway.session_api import restore_api_session
from gateway.session_api_turn import admit_api_turn


def test_declared_api_root_restores_compression_tip_and_retries_same_admission(api, owner):
    kwargs = dict(session_id='s', request_id='one', user_message='hello', conversation_history=[],
                  gateway_session_key='conversation', bind_declared_conversation=True)
    _, ref, first = admit_api_turn(api, **kwargs)
    _, _, second = admit_api_turn(api, **{**kwargs, 'request_id': 'two', 'user_message': 'next'})
    owner.db.publish_compression_child(parent_session_id='s', child_session_id='child', source='api_server',
        messages=[{'role': 'assistant', 'content': 'summary'}], require_compression_lease=False)
    route = owner.sessions['s'].route
    owner.runner.session_store.advance_compression_session(route, 's', 'child')
    assert restore_api_session(owner, 's') == ref
    assert owner.runner.session_store.peek_session_id(route) == 'child'
    assert admit_api_turn(api, **kwargs)[2]['admission_id'] == first['admission_id']
    assert admit_api_turn(api, **{**kwargs, 'request_id': 'two', 'user_message': 'next'})[2]['admission_id'] == second['admission_id']

    # A restart may restore a persisted routing entry from before the final compression
    # publication. Any genuine ancestor may catch up, not only the original root.
    owner.db.publish_compression_child(parent_session_id='child', child_session_id='tip', source='api_server',
        messages=[{'role': 'assistant', 'content': 'new summary'}], require_compression_lease=False)
    assert restore_api_session(owner, 's') == ref
    assert owner.runner.session_store.peek_session_id(route) == 'tip'

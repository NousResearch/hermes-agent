"""Relay metadata a client sends on an API request reaches the canonical turn that executes it
(main forwarded ``body.metadata`` to ``run_conversation(relay_metadata=...)``)."""
import pytest

from gateway.session_ingress import execute_admission
from gateway.session_api_turn import admit_api_turn


@pytest.mark.asyncio
async def test_request_relay_metadata_reaches_the_executing_turn(api, owner):
    metadata = {'relay.trace': 'trace-7', 'tenant': 'acme'}
    _, ref, row = admit_api_turn(api, session_id='relay-meta', user_message='hello',
                                 conversation_history=[], relay_metadata=metadata)
    seen = {}

    async def handle(event):
        from gateway.session_api_turn import api_execution
        from gateway.session_results import execution_result
        seen['relay_metadata'] = api_execution.get()['relay_metadata']
        execution_result.get()['result'] = {'final_response': 'ok'}
        return 'ok'
    owner.runner._handle_message = handle
    await execute_admission(owner, ref, row)
    assert seen['relay_metadata'] == metadata


def test_turn_runner_hands_relay_metadata_to_run_conversation(api, owner):
    """The last hop: the TurnRunner passes the admitted relay metadata to run_conversation (N99)."""
    from types import SimpleNamespace

    from gateway.run_turn_runner import TurnRunner
    from gateway.session_api_turn import api_execution, prepare_api_execution
    from gateway.turn_context import TurnContext
    metadata = {'relay.trace': 'trace-8'}
    _, ref, row = admit_api_turn(api, session_id='relay-hop', user_message='hello',
                                 conversation_history=[], relay_metadata=metadata)
    seen = []

    def run(message, *, relay_metadata=None, **kwargs):
        seen.append(relay_metadata)
        return {'final_response': 'ok'}
    turn = TurnRunner(SimpleNamespace(session_authority=None, _consume_pending_native_image_paths=lambda key: []),
                      TurnContext(message='hello', session_id=ref.session_id, session_key='api',
                                  source=SimpleNamespace(user_id='u', user_name='U', is_bot=False)))
    for payload in (row['payload'], {'text': 'human'}):
        token = api_execution.set(prepare_api_execution(owner, ref, payload))
        try:
            turn._run_conversation_with_approval(SimpleNamespace(run_conversation=run), [], None, None, None)
        finally:
            api_execution.reset(token)
    assert seen == [metadata, None]

"""A watchdog timeout remains failed when the handler projects its diagnostic as text."""
from types import SimpleNamespace
import time

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_authority import LiveSession
from gateway.session_contract import SessionRef
from hermes_state_runtime import admit_session_input, claim_session_input, get_session_admission


@pytest.mark.asyncio
@pytest.mark.parametrize('platform', [Platform.LOCAL, Platform.TELEGRAM])
@pytest.mark.parametrize('late_result', [False, True])
async def test_timeout_diagnostic_keeps_failed_receipt_after_string_projection(owner, platform, late_result):
    from gateway.run import GatewayRunner
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin
    from gateway.session_ingress import execute_admission
    from gateway.session_results import execution_result, finish_result, admission_result
    from gateway.turn_context import TurnContext
    owner.db.create_session('s', source='cli')
    source = SessionSource(platform, 's', user_id='human')
    owner.sessions['s'] = LiveSession(source, 'route')
    owner.runner._adapter_for_source = lambda source: None
    owner.runner._agent_activity_summary = lambda agent: {'seconds_since_activity': 1800}
    owner.runner._is_intentional_silence = lambda *args: False
    diagnostic = []

    async def handle(event):
        raw = GatewayRunner._run_agent_timeout_result(owner.runner,
            SimpleNamespace(agent_timeout=1800), TurnContext(session_key='route'))
        response, _, _ = await GatewayTurnPersistenceMixin._hmwa_shape_agent_response(
            owner.runner, raw, source, [], SimpleNamespace(session_id='s'), None,
            None, 1, 's', platform.value, time.time())
        diagnostic.append(response)
        if late_result:
            # The abandoned executor shares the capture dict and can return while the handler
            # is still unwinding. Its later value cannot replace the owner's timeout verdict.
            execution_result.get()['result'] = {'final_response': 'late worker answer', 'completed': True}
        return response

    owner.runner._handle_message = handle
    admitted = admit_session_input(owner.db, epoch=owner.epoch, principal_id='human',
        session_id='s', request_id='timeout', payload={'text': 'work'})
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id='s')
    response = await execute_admission(owner, SessionRef(owner.profile_id, 's'), row)
    finish_result(owner.db, epoch=owner.epoch, row=row, response=response, outcome='completed',
                  result=owner.pending_results[admitted['admission_id']])
    saved = admission_result(owner.db, admitted['admission_id'])['result']
    assert get_session_admission(owner.db, admission_id=admitted['admission_id'])['outcome'] == 'failed'
    assert saved['failed'] is True and saved['completed'] is False
    assert saved['final_response'] == response == diagnostic[0]


@pytest.mark.asyncio
async def test_owner_notice_without_execution_is_still_completed(owner):
    from gateway.session_ingress import execute_admission
    from gateway.session_results import finish_result
    owner.db.create_session('notice', source='telegram')
    source = SessionSource(Platform.TELEGRAM, 'notice', user_id='human')
    owner.sessions['notice'] = LiveSession(source, 'notice-route')
    owner.runner._adapter_for_source = lambda source: None
    async def notice(event):
        return 'Gateway is paused by the operator.'
    owner.runner._handle_message = notice
    admitted = admit_session_input(owner.db, epoch=owner.epoch, principal_id='human',
        session_id='notice', request_id='notice', payload={'text': 'work'})
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id='notice')
    response = await execute_admission(owner, SessionRef(owner.profile_id, 'notice'), row)
    finish_result(owner.db, epoch=owner.epoch, row=row, response=response, outcome='completed',
                  result=owner.pending_results[admitted['admission_id']])
    assert get_session_admission(owner.db, admission_id=admitted['admission_id'])['outcome'] == 'completed'

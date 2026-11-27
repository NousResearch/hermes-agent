"""A stopped background turn cannot speak its interruption ahead of the next message."""

import pytest

from tests.gateway.session_claims import claim
from tests.gateway.test_automation_compression_media import _admit_human, _discord_owner


@pytest.mark.asyncio
@pytest.mark.parametrize('result, expected', [
    ({'interrupted': True, 'completed': False}, ''),
    ({'failed': True, 'completed': False}, 'A real failure.'),
    ({'completed': True}, 'A real failure.'),
])
async def test_interrupted_background_reply_is_not_delivered(tmp_path, monkeypatch, result, expected):
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import admission_row
    from gateway.session_ingress import execute_admission
    from gateway.session_results import execution_result

    runner, adapter, authority, source = await _discord_owner(tmp_path, monkeypatch)
    receipt = await _admit_human(runner, authority, MessageEvent(text='hello', source=source, message_id='human'))
    sid, ref = receipt.ref.session_id, receipt.ref
    human = authority.persistence.admissions.claim(epoch=authority.epoch, session_id=sid)
    authority.persistence.admissions.settle(epoch=authority.epoch, admission_id=human.admission_id,
                                            generation=human.generation, outcome='completed')
    event = MessageEvent(text='background process completed', source=source, internal=True,
                         metadata={'gateway_session_key': runner.session_store._generate_session_key(source),
                                   'gateway_session_id': sid})
    notice = await authority.admit_automation(adapter, event, 'notice')
    row = admission_row(authority.persistence.admissions.list(session_id=sid, pending_only=False)[-1])

    async def handle(event):
        execution_result.get()['result'] = {'final_response': 'A real failure.', **result}
        return 'A real failure.'

    monkeypatch.setattr(runner, '_handle_message', handle)
    assert await execute_admission(authority, ref, claim(authority, ref, row)) == expected
    assert (notice.admission_id in authority.pending_deliveries) is bool(expected)
    assert authority.pending_results[notice.admission_id]['result'] == {'final_response': 'A real failure.', **result}


@pytest.mark.asyncio
async def test_interrupted_human_turn_keeps_its_diagnostic(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import admission_row
    from gateway.session_ingress import execute_admission
    from gateway.session_results import execution_result

    runner, _, authority, source = await _discord_owner(tmp_path, monkeypatch)
    receipt = await _admit_human(runner, authority, MessageEvent(text='hello', source=source, message_id='human'))
    row = admission_row(authority.persistence.admissions.list(session_id=receipt.ref.session_id,
                                                             pending_only=False)[0])

    async def handle(event):
        execution_result.get()['result'] = {'final_response': 'Interrupted.', 'interrupted': True}
        return 'Interrupted.'

    monkeypatch.setattr(runner, '_handle_message', handle)
    assert await execute_admission(authority, receipt.ref, claim(authority, receipt.ref, row)) == 'Interrupted.'

"""A turn's ``message.start`` names its typed input, so a viewer that did not submit it can show it.

Desktop next to an editor's ACP client (or the CLI) streams the other surface's reply live, but
without this the prompt above it appears only after the transcript is read again.
"""
import asyncio

import pytest

from gateway.session_contract import Submission
from gateway.session_display import admitted_input
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority


@pytest.mark.asyncio
async def test_each_claim_names_its_typed_input_and_a_hidden_send_none(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    frames = []
    authority.sessions['s'].event_stream.observers.add(frames.append)

    async def execute(owner, ref, row):
        owner.pending_results[row['admission_id']] = {'result': {'final_response': 'done'}, 'usage': {}}
        return 'done'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    payloads = [{'text': 'from the editor'}, {'text': 'model-facing kickoff', 'display_kind': 'hidden'},
                {'text': 'and again'}]
    with db:
        for index, payload in enumerate(payloads):
            await authority.submit(ACTOR, Submission(f'sub-{index}', REF, payload, 'queue'))
        await asyncio.wait_for(authority._drain(REF), 10)

    starts = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.start']
    assert starts == [{'input': {'text': 'from the editor', 'submission_id': 'sub-0'}}, {},
                      {'input': {'text': 'and again', 'submission_id': 'sub-2'}}]


@pytest.mark.parametrize('payload', [
    # An automation or notification wake: its card rides session_automation.admission_display.
    {'text': 'Background process proc_1 finished', 'local_automation_v1': {'display_kind': 'process_complete'}},
    {'text': 'card prompt', 'kanban_card_v1': {'task_id': 't'}},
    # A messaging envelope: its row replays through the platform pipeline, which can rewrite the text.
    {'text': 'hi from telegram', 'native_text_v1': {'route': 'agent:main:telegram:dm:1', 'event': {}}},
    # Attachments only.
    {'text': '   ', 'attachments_v1': {'media': []}},
])
def test_inputs_not_typed_into_a_surface_name_none(payload):
    assert admitted_input({'payload': payload, 'request_id': 'req'}) == {}

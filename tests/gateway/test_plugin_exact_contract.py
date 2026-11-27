"""Authority and replay boundaries of exact plugin tool batches."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway.plugin_submission.execution import admission_scope, run, validate
pytest_plugins = ["tests.gateway.test_plugin_submission"]

pytestmark = pytest.mark.usefixtures('authority_cwd_in_tmp')


@pytest.mark.asyncio
@pytest.mark.parametrize('compressed', [False, True])
@pytest.mark.parametrize('output_size', [5, 70000])
async def test_exact_batch_replay_conflicts_on_changed_calls_and_projects_only_its_outputs(submission, compressed, output_size):
    ctx, _runner, authority, _source, identity = submission
    calls = [dict(call_id='one', name='terminal', arguments={'command': 'echo first'})]
    first = await asyncio.to_thread(ctx.execute_tools, calls, text='Run the requested command.', **identity)
    assert first['ok'] and first['status'] == 'queued'
    replay = await asyncio.to_thread(ctx.execute_tools, calls, text='Run the requested command.', **identity)
    assert replay['admission_id'] == first['admission_id']
    conflict = await asyncio.to_thread(ctx.execute_tools,
        [dict(call_id='one', name='terminal', arguments={'command': 'echo different'})],
        text='Run the requested command.', **identity)
    assert conflict['error'] == 'admission_conflict'
    ledger = authority.persistence.admissions
    human = ledger.claim(epoch=authority.epoch, session_id=identity['expected_session_id'])
    ledger.settle(epoch=authority.epoch, admission_id=human.admission_id,
                  generation=human.generation, outcome='completed')
    work = ledger.claim(epoch=authority.epoch, session_id=identity['expected_session_id'])
    assert work.admission_id == first['admission_id']
    ledger.settle(epoch=authority.epoch, admission_id=work.admission_id, generation=work.generation,
        outcome='completed', result={'result': {'final_response': '[SILENT]', '_compressed': compressed,
            'current_turn_user_idx': 1, 'messages': [
            {'role': 'tool', 'tool_call_id': 'one', 'content': 'unrelated'},
            {'role': 'user', 'content': work.payload['text']},
            {'role': 'tool', 'tool_call_id': 'one', 'content': 'x' * output_size}]}})
    done = await asyncio.to_thread(ctx.execute_tools, calls, text='Run the requested command.', **identity)
    assert done['outputs'][0]['output'] == 'x' * min(output_size, 64000)
    assert done['outputs_truncated'] is (output_size > 64000)
    assert bool(done['outputs'][0].get('truncated')) is (output_size > 64000)


def test_only_authoritative_automation_admission_can_select_exact_transport():
    spec = dict(text='Do the work.', calls=[dict(call_id='one', name='terminal', arguments={})])
    row = dict(principal_id='human:room', request_id=json.dumps(['plugin-tools', 'test', 'room', 'one']),
               payload={'text': json.dumps(spec)})
    assert admission_scope(row) is None
    row['principal_id'] = 'automation:room'
    called = []
    agent = SimpleNamespace(api_mode='unsupported', run_conversation=lambda *args, **kwargs: called.append(args))
    with admission_scope(row), pytest.raises(RuntimeError, match='require Chat'):
        run(agent, 'input', {})
    assert called == []
    run(agent, 'normal input', {})
    assert called == [('normal input',)]


@pytest.mark.parametrize('calls', [[], [{'call_id': 'one', 'name': 'terminal', 'arguments': []}],
    [dict(call_id='one', name='terminal', arguments={})] * 2])
def test_invalid_batch_refuses_before_admission(calls):
    with pytest.raises(ValueError):
        validate(dict(text='Perform the operation.', calls=calls))

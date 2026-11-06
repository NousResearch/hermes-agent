"""Canonical authority and its real authenticated transport with a loopback model."""
import asyncio
import json
from pathlib import Path

import pytest

from gateway.session_guidance import SessionGuidance
from tests.gateway.test_session_busy_controls import owner, admissions, settled
from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket


def identity(snapshot, message_id, text='literal'):
    return {'runtime_instance_id': snapshot['runtimeInstanceId'], 'expected_turn_id': snapshot['turnId'],
            'message_id': message_id, 'text': text}


def test_generations_keep_separate_engine_inboxes_and_retain_receipts():
    ledger = SessionGuidance('owner:1', 's')
    ledger.start({'admission_id': 'a', 'generation': 1})
    first_engine = ledger.current
    target = ledger.snapshot('a')
    params = identity(target, 'm', '/reset literal')
    admitted = ledger.request(params, lookup=False)
    assert admitted['state'] == 'pending_incorporation'
    first_engine.close(cancelled=True)
    ledger.start({'admission_id': 'b', 'generation': 2})
    assert ledger.request(params, lookup=True)['state'] == 'cancelled_before_incorporation'
    assert ledger.snapshot('a')['receipts'][0]['messageId'] == 'm'
    assert ledger.request(params, lookup=False)['state'] == 'cancelled_before_incorporation'
    assert ledger.request(identity(target, 'late'), lookup=False)['disposition'] == 'rejected_turn_mismatch'
    second_target = ledger.snapshot('b')
    assert ledger.request(identity(second_target, 'new'), lookup=False)['disposition'] == 'accepted_active'
    messages = [{'role': 'assistant', 'content': 'boundary'}]
    first_engine.incorporate(messages)
    assert len(messages) == 1
    ledger.current.incorporate(messages)
    assert messages[-1]['content'] == 'literal'
    assert ledger.request(identity(second_target, 'new', 'changed'), lookup=False)['disposition'] == 'message_id_conflict'
    restarted = SessionGuidance('owner:2', 's')
    restarted.start({'admission_id': 'recovered', 'generation': 2})
    assert restarted.request(identity(second_target, 'new'), lookup=True)['disposition'] == 'rejected_instance_mismatch'


@pytest.mark.platforms('linux')
def test_authenticated_owner_incorporates_concurrent_guidance_without_a_second_admission(tmp_path):
    with owner(tmp_path, {'platform_toolsets': {'acp': ['terminal']}}) as (home, peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                created = await rpc(ws, 'session.create', request_id='guidance', source='acp', toolsets=['terminal'],
                                    cwd=str(home), editor={'mcp_servers': [], 'edit_approval_policy': 'ask'})
                assert 'result' in created, created
                sid = created['result']['session_id']
                primary = (await rpc(ws, 'prompt.submit', session_id=sid, input_id='primary', text='BLOCK_STEER'))['result']
                assert await asyncio.to_thread(peer.blocked.wait, 20)
                target = (await rpc(ws, 'session.guidance.turn', session_id=sid, admission_id=primary['admission_id']))['result']
                assert target['state'] == 'started', target

                async def admit(message_id, text):
                    async with websocket(home, desc) as connection:
                        params = identity(target, message_id, text)
                        response = await rpc(connection, 'session.guidance.admit', session_id=sid, **params)
                        assert 'result' in response, response
                        return params, response['result']

                results = await asyncio.gather(admit('m1', '/model literal first'), admit('m2', '/reset literal second'))
                for params, receipt in results:
                    assert receipt['state'] == 'pending_incorporation'
                    assert (await rpc(ws, 'session.guidance.admit', session_id=sid, **params))['result'] == receipt
                peer.release.set()
                await settled(home)
                final = (await rpc(ws, 'session.guidance.turn', session_id=sid, admission_id=primary['admission_id']))['result']
                assert final['state'] == 'ended'
                assert [r['messageId'] for r in final['receipts']] == [r[1]['messageId'] for r in sorted(results, key=lambda r:r[1]['sequence'])]
                assert all(r['state'] == 'incorporated' for r in final['receipts'])
                assert len(admissions(home)) == 1
                assert len(peer.requests) == 2
                expected = '\n\n'.join(params['text'] for params, receipt in sorted(results, key=lambda r:r[1]['sequence']))
                assert peer.requests[1]['messages'][-1]['content'] == expected
                # A fresh authenticated connection recovers a lost response without resubmission.
                async with websocket(home, desc) as reconnected:
                    for params, receipt in results:
                        status = dict(params); status.pop('text')
                        reply = await rpc(reconnected, 'session.guidance.status', session_id=sid, **status)
                        assert reply['result']['state'] == 'incorporated', reply
                result = (await rpc(ws, 'prompt.receipt', session_id=sid, admission_id=primary['admission_id'], include_result=True))['result']
                assert result['status'] == 'terminal', result
                print(json.dumps({'target': target, 'final': final, 'primary': result}, default=str))
        asyncio.run(probe())


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('cancelled', [False, True])
def test_terminal_or_cancelled_guidance_never_runs_in_a_successor(tmp_path, cancelled):
    with owner(tmp_path) as (home, peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='end', source='acp', toolsets=[],
                                 cwd=str(home), editor={'mcp_servers': [], 'edit_approval_policy': 'ask'}))['result']['session_id']
                primary = (await rpc(ws, 'prompt.submit', session_id=sid, input_id='first', text='BLOCK_REDIRECT'))['result']
                assert await asyncio.to_thread(peer.blocked.wait, 20)
                target = (await rpc(ws, 'session.guidance.turn', session_id=sid, admission_id=primary['admission_id']))['result']
                params = identity(target, 'm', 'NEVER_IN_SUCCESSOR')
                assert (await rpc(ws, 'session.guidance.admit', session_id=sid, **params))['result']['state'] == 'pending_incorporation'
                expected = 'unconsumed_at_turn_end'
                if cancelled:
                    snapshot = (await rpc(ws, 'session.resume', session_id=sid))['result']
                    stopped = await rpc(ws, 'session.interrupt', session_id=sid,
                                        execution_generation=snapshot['execution_generation'])
                    assert 'result' in stopped, stopped
                    expected = 'cancelled_before_incorporation'
                peer.release.set()
                await settled(home)
                status = dict(params); status.pop('text')
                assert (await rpc(ws, 'session.guidance.status', session_id=sid, **status))['result']['state'] == expected
                late = await rpc(ws, 'session.guidance.admit', session_id=sid,
                                 **identity(target, 'late', 'NEVER_IN_SUCCESSOR'))
                assert late['result']['disposition'] == ('rejected_cancelled' if cancelled else 'rejected_turn_ended'), late
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='next', text='SUCCESSOR')
                await settled(home)
                assert (await rpc(ws, 'session.guidance.admit', session_id=sid, **params))['result']['state'] == expected
                assert all('NEVER_IN_SUCCESSOR' not in json.dumps(r['messages']) for r in peer.requests)
                assert len(admissions(home)) == 2
        asyncio.run(probe())


@pytest.mark.platforms('linux')
def test_owner_restart_loses_receipt_without_restoring_or_replaying_input(tmp_path):
    with owner(tmp_path) as (home, peer, desc):
        async def admit():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='restart', source='acp', toolsets=[],
                                 cwd=str(home), editor={'mcp_servers': [], 'edit_approval_policy': 'ask'}))['result']['session_id']
                primary = (await rpc(ws, 'prompt.submit', session_id=sid, input_id='first', text='BLOCK_REDIRECT'))['result']
                assert await asyncio.to_thread(peer.blocked.wait, 20)
                target = (await rpc(ws, 'session.guidance.turn', session_id=sid, admission_id=primary['admission_id']))['result']
                params = identity(target, 'm', 'NEVER_REPLAY')
                assert (await rpc(ws, 'session.guidance.admit', session_id=sid, **params))['result']['state'] == 'pending_incorporation'
                return sid, params
        sid, params = asyncio.run(admit())
    env = child_env()
    env.update(HOME=str(tmp_path / 'user'), USERPROFILE=str(tmp_path / 'user'), HERMES_HOME=str(home),
               OPENAI_API_KEY='loopback-only', PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    with daemon(Path(__file__).resolve().parents[2], home, env, barrier=False) as (_, new_desc):
        async def recover():
            async with websocket(home, new_desc) as ws:
                params.pop('text')
                response = await rpc(ws, 'session.guidance.status', session_id=sid, **params)
                if 'result' in response:
                    assert response['result']['disposition'] in {'unknown_message', 'rejected_instance_mismatch'}, response
                    assert response['result']['state'] == 'rejected', response
                else:
                    assert response['error']['message'] == 'not_found', response
                assert new_desc['instance_id'] != desc['instance_id']
        asyncio.run(recover())
    assert len(admissions(home)) == 1
    assert len(peer.requests) == 1

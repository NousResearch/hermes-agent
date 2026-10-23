"""A managed turn whose worker acknowledged its durable finish keeps its answer even when the
worker's interpreter is slow to exit or exits non-zero afterwards (interpreter teardown,
a lingering non-daemon thread, a Windows launcher): that is close()'s cleanup, not a lost
execution. Real child process, real SQLite authority."""
import asyncio
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.session_contract import Principal, SessionRef

CHILD = """import json, os, sys, time, psutil
me = psutil.Process()
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
def send(frame):
    out.write((json.dumps(frame) + '\\n').encode())
send({'type': 'hello', 'pid': me.pid, 'birth': me.create_time(), 'ancestors': [p.pid for p in me.parents()[:3]]})
for line in sys.stdin.buffer:
    frame = json.loads(line)
    if 'version' in frame:
        send({'type': 'result', 'result': {'final_response': 'FINISHED_ANSWER', 'model': 'm', 'failure_reason': 'billing',
              'billing_block': {'provider': 'custom', 'provider_label': 'Custom', 'model': 'm', 'billing_url': None,
                                'is_nous': False, 'message': 'Add credits'}}})
    elif frame == {'type': 'finish'}:
        send({'type': 'finished'})
        break
if sys.argv[1] == 'nonzero':
    os._exit(3)
time.sleep(60)  # interpreter teardown that outlives the owner's exit wait
"""


@pytest.mark.platforms("linux")
@pytest.mark.parametrize(('exit_shape', 'rotate'), [('nonzero', False), ('lingering', False), ('nonzero', True)])
def test_finished_worker_keeps_its_result_on_slow_or_nonzero_exit(tmp_path, monkeypatch, exit_shape, rotate):
    from gateway import session_managed_worker as managed
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Submission
    from gateway.session_policy import build_policy
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    # The worker's per-call usage writes, already committed by the time it reports 'finished'.
    db.update_token_counts('s', input_tokens=200, output_tokens=50, cache_read_tokens=800, api_call_count=3, model='m')
    runner = SimpleNamespace(_draining=False, _cached_agent_for=lambda route: None,
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='i', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='i'))
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human', chat_id='c'), 'route')
    actor = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
    ref = SessionRef('owned', 's')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    spawned = []

    class Child(subprocess.Popen):
        def __init__(self, args, **kw):
            super().__init__([sys.executable, '-c', CHILD, exit_shape], **kw)
            spawned.append(self)
    monkeypatch.setattr(managed.subprocess, 'Popen', Child)
    if rotate:
        reserve = managed.reserve_admission_worker

        def reserve_then_compress(authority, **kwargs):
            # A worker compression mid-turn: the transcript continues in a child session and the owner's
            # execution row follows it in the same transaction (hermes_state_worker_compression.py).
            scope = reserve(authority, **kwargs)
            db.end_session('s', 'compression')
            db.create_session('s-c1', source='test', parent_session_id='s')
            db._execute_write(lambda conn: conn.execute('UPDATE worker_executions SET session_id=? WHERE execution_id=?',
                                                        ('s-c1', scope['execution_id'])))
            db.update_token_counts('s-c1', input_tokens=100, output_tokens=10, api_call_count=1, model='m')
            return scope
        monkeypatch.setattr(managed, 'reserve_admission_worker', reserve_then_compress)
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {}, profile_terminal=False)

    async def scenario():
        await authority.submit(actor, Submission('turn', ref, {'text': 'go'}, 'queue'))
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        authority._stamp_claim(ref, authority.sessions['s'], row)  # the drain's claim stamp
        async with asyncio.timeout(40):
            return row, await managed.execute_managed(authority, ref, row, policy)
    with db:
        row, response = asyncio.run(scenario())
        status = db._read_one('SELECT status FROM session_admissions WHERE admission_id=?', (row['admission_id'],))[0]
    assert response == 'FINISHED_ANSWER'
    captured = authority.pending_results[row['admission_id']]
    assert captured['result']['final_response'] == 'FINISHED_ANSWER'
    assert status == 'started', 'the finished turn was marked unknown instead of reaching settlement'
    assert spawned[0].poll() is not None, "close() did not reap the worker"
    # A local viewer's message.complete gets the billing banner and turn-end usage, as in process.
    from gateway.session_settlement_recovery import completion_payload
    complete = completion_payload(row, {'outcome': 'completed'}, response, captured)
    assert complete['billing']['message'] == 'Add credits' and complete['failure_reason'] == 'billing'
    if rotate:
        assert complete['usage'] == {'model': 'm', 'input': 300, 'output': 60, 'reasoning': 0, 'prompt': 1100,
                                     'completion': 60, 'total': 1160, 'calls': 4, 'cache_hit_pct': 73}
    else:
        assert complete['usage'] == {'model': 'm', 'input': 200, 'output': 50, 'reasoning': 0, 'prompt': 1000,
                                     'completion': 50, 'total': 1050, 'calls': 3, 'cache_hit_pct': 80}

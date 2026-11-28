"""Independent review probes. Failures assert preserved safety contracts."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
from cron import scheduler as sched, scheduler_delivery as delivery, delivery_queue as queue
from cron import bot_chat_legacy as legacy, delivery_outcome
from tools import bot_live_delivery as live

OLD = Path(__file__).parent / 'fixtures' / 'approved_bot_chat_delivery.py'

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(queue, 'DELIVERY_DB', tmp_path / 'cron/deliveries.db')
    monkeypatch.setattr(sched, 'load_config', lambda: {'cron': {'bot_chat_durable_pending': True}})
    monkeypatch.setattr(live, 'find_canonical_live_owner', lambda home: {'session_id': 'bot'})
    (tmp_path / 'state.db').touch()
    return tmp_path


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipe readiness barrier")
def test_stripe_waiter_does_not_adopt_old_process_claim(store, monkeypatch):
    # The predecessor stripe now serializes before the O_EXCL publication point.
    from hermes_cli.active_sessions import _FileLock
    import select

    monkeypatch.setattr(sched, 'load_config', dict)
    job = dict(id='job', name='Daily', execution_id='execution')
    key = delivery._bot_chat_delivery_key(job, store, store)
    queue.get_status('initialize')
    # The old process completes its SQL absence check before the new process
    # reserves SQL; it still owns its striped arbitration lock through JSON create.
    code = '''
import importlib.util, json, sys
from pathlib import Path
from cron import delivery_queue as q
spec=importlib.util.spec_from_file_location('old_receipts',sys.argv[1])
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
home=Path(sys.argv[2]); key=sys.argv[3]
with m.ownership_arbitration(key):
    assert q.get_status(key) is None
    print('checked-no-sql',flush=True)
    sys.stdin.readline()
    record,created=m.claim_direct(key,{'id':'job','name':'Daily','execution_id':'execution'},'body','',home)
    assert created
    print('old-claim-committed',flush=True)
'''
    proc = subprocess.Popen([sys.executable, '-c', code, str(OLD), str(store), key],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    original_enter = _FileLock.__enter__
    crossed = []

    def line():
        assert select.select([proc.stdout], [], [], 15)[0], 'predecessor barrier timed out'
        return proc.stdout.readline().strip()

    def entering_stripe(lock):
        if not crossed:
            crossed.append(True)
            # Release BEFORE waiting for the stripe: releasing at os.open would
            # deadlock because the repaired producer cannot reach it yet.
            proc.stdin.write('publish\n'); proc.stdin.flush()
            assert line() == 'old-claim-committed'
        return original_enter(lock)

    admit = Mock(side_effect=lambda h, o, message, **kw: dict(status='settled', message=message))
    try:
        assert line() == 'checked-no-sql'
        monkeypatch.setattr(_FileLock, '__enter__', entering_stripe)
        monkeypatch.setattr(live, 'deliver_to_live_owner', admit)
        result = delivery._deliver_to_bot_chat(job, 'body', '')
        out, err = proc.communicate(timeout=15)
        assert proc.returncode == 0, (out, err)
    finally:
        if proc.poll() is None:
            proc.kill(); proc.communicate()
    print('RACE', {'crossed': crossed, 'new_rpc_calls': admit.call_count,
                   'final_receipt': legacy.read_pending(key), 'result': result})
    assert crossed == [True]
    assert admit.call_count == 0, 'new producer observed the predecessor claim but launched transport'
    assert legacy.read_pending(key)['status'] == 'claimed'
    assert legacy.read_pending(key)['direct'] is True
    assert 'claimed' in result


@pytest.mark.parametrize('mutate', ['content', 'name', 'destination', 'category'])
def test_legacy_consumption_checks_actual_render_against_retained_hash(store, monkeypatch, mutate):
    key = 'a' * 64
    job = {'id': 'job', 'name': 'Daily', 'execution_id': 'run'}
    record = legacy.defer(key, job, 'approved-body', store, for_failure=False)
    if mutate == 'content': record['content'] = 'different-body'
    if mutate == 'name': record['job']['name'] = 'Different Name'
    if mutate == 'destination':
        other = store / 'other'; other.mkdir(); (other / 'state.db').touch()
        record['home'] = str(other)
    if mutate == 'category': record['for_failure'] = True
    legacy.atomic_json_write(legacy._receipt_path(legacy._root(), key), record)
    admit = Mock(return_value={'status': 'queued'})
    monkeypatch.setattr(live, 'deliver_to_live_owner', admit)
    legacy.drain_legacy_pending()
    print('MUTATION', mutate, 'rpc_calls=', admit.call_count, 'receipt=', legacy.read_pending(key))
    assert admit.call_count == 0, 'stored hash was compared to itself, not the execution payload'


def test_opted_in_no_agent_producer_keeps_bounded_refusal_queue(store, monkeypatch):
    job = dict(id='job', name='Daily', execution_id='execution', no_agent=True, deliver='bot-chat')
    admit = Mock(side_effect=live.PreDialError('not dialed'))
    monkeypatch.setattr(live, 'deliver_to_live_owner', admit)
    delivery._deliver_to_bot_chat(job, 'body', '')
    for _ in range(13):
        legacy.drain_legacy_pending()
    key = delivery._bot_chat_delivery_key(job, store, store)
    print('REFUSALS', {'calls': admit.call_count, 'sql': queue.get_status(key), 'json': legacy.read_pending(key)})
    assert admit.call_count <= 12, 'opt-in bare Bot Chat bypasses the bounded outbox and retries every tick forever'


def test_opted_in_durable_producer_requires_execution_identity(store, monkeypatch):
    admit = Mock(side_effect=lambda h, o, message, **kw: dict(status='settled', message=message))
    monkeypatch.setattr(live, 'deliver_to_live_owner', admit)
    result = delivery._deliver_to_bot_chat({'id': 'job', 'deliver': 'bot-chat'}, 'body', '')
    print('NO_EXECUTION', result, 'rpc_calls=', admit.call_count)
    assert admit.call_count == 0, 'old enabled outbox requires a real execution_id'


def test_imported_digest_maps_to_original_execution_outcome(store, monkeypatch):
    job = dict(id='job', name='Daily', execution_id='real-execution', _bot_chat_pending=str(store))
    key = 'a' * 64
    queue.enqueue(key, job, 'body')
    def admitted(h, o, message, **kw):
        root = live._root(h); root.mkdir(parents=True, exist_ok=True)
        legacy.atomic_json_write(root / (key + '.json'), dict(id=key, delivery_id=key, status='settled', message=message))
        return dict(status='settled', message=message)
    monkeypatch.setattr(live, 'deliver_to_live_owner', admitted)
    assert sched.drain_delivery_queue({}, None) == 1
    print('DIGEST', {'queue': queue.get_status(key), 'digest_outcome': delivery_outcome.settled_outcome(key),
                     'real_outcome': delivery_outcome.settled_outcome('real-execution')})
    assert delivery_outcome.settled_outcome(key) is None
    assert delivery_outcome.settled_outcome('real-execution') == 'delivered', 'reference is stored but never used for execution lookup'


def test_cron_recovered_key_is_validated_before_mailbox_path_use(store, monkeypatch):
    key = 'a' * 32
    job = dict(id='job', execution_id='execution', _bot_chat_pending=str(store))
    queue.enqueue(key, job, 'body')
    root = live._root(store); root.mkdir(parents=True)
    touched = []
    old_read = live._read
    def read(path):
        touched.append(str(path))
        return old_read(path)
    monkeypatch.setattr(live, '_read', read)
    sched.drain_delivery_queue({}, None)
    print('NONCANONICAL', touched)
    assert touched == [], 'cron recovered key reached live 32-hex protocol before cron 64-hex validation'


def test_clean_upstream_pending_bot_row_does_not_report_inference_complete(store, monkeypatch):
    # Clean ae2a81cc wrote full jobs under actual execution IDs without a Bot Chat
    # envelope or hash. This is the persisted migration shape, not a new producer.
    job = dict(id='job', name='Daily', execution_id='execution', deliver='bot-chat')
    queue.enqueue('execution', job, 'body')
    key = delivery._bot_chat_delivery_key(job, store, store)
    def admit(h, o, message, **kw):
        root = live._root(h); root.mkdir(parents=True, exist_ok=True)
        result = dict(id=key, delivery_id=key, status='queued', message=message)
        legacy.atomic_json_write(root / (key + '.json'), result)
        return result
    monkeypatch.setattr(live, 'deliver_to_live_owner', admit)
    assert sched.drain_delivery_queue({}, None) == 1
    assert live.read_delivery_result(store, key)['status'] == 'queued'
    print('CLEAN_MIGRATION', {'queue': queue.get_status('execution'),
                            'outcome': delivery_outcome.settled_outcome('execution')})
    assert delivery_outcome.settled_outcome('execution') is None, 'clean-upstream Bot Chat rows still treat handoff as inference completion'
    assert queue.get_status('execution')['status'] == 'pending'
    assert queue.claim_next() is None
    root = live._root(store)
    legacy.atomic_json_write(root / (key + '.json'), dict(id=key, delivery_id=key, status='settled'))
    assert delivery_outcome.settled_outcome('execution') == 'delivered'
    assert queue.get_status('execution')['status'] == 'delivered'


def test_noncanonical_recovered_reference_cannot_settle_execution(store):
    key = 'a' * 32
    job = dict(id='job', execution_id='execution', _bot_chat_pending=str(store), _bot_chat_delivery_id=key)
    queue.enqueue('execution', job, 'body')
    queue.claim_next()
    queue._finish('execution', error=None)
    root = live._root(store); root.mkdir(parents=True)
    legacy.atomic_json_write(root / (key + '.json'), dict(id=key, delivery_id=key, status='settled'))
    result = delivery_outcome.settled_outcome('execution')
    print('NONCANONICAL_SETTLEMENT', result)
    assert result is None, 'a cron reference outside its canonical 64-hex identity grammar settled an execution'



def test_opted_in_producer_sql_backoff_exhaustion_and_flag_change(store, monkeypatch):
    job = dict(id='job', name='Daily', execution_id='execution', no_agent=True, deliver='bot-chat')
    admit = Mock(side_effect=live.PreDialError('not dialed'))
    monkeypatch.setattr(live, 'deliver_to_live_owner', admit)
    clock = [1000.0]
    monkeypatch.setattr(queue.time, 'time', lambda: clock[0])
    assert 'queued' in delivery._deliver_to_bot_chat(job, 'body', '')
    key = delivery._bot_chat_delivery_key(job, store, store)
    assert queue.get_status(key)['status'] == 'pending'
    assert legacy.read_pending(key) is None
    admit.assert_not_called()
    monkeypatch.setattr(sched, 'load_config', dict)
    for attempt, delay in enumerate([30, 60, 120, 240, 300, 300, 300, 300, 300, 300, 300, None], 1):
        delivery._deliver_to_bot_chat(dict(job), 'body', '')
        assert sched.drain_delivery_queue({}, None) == 1
        row = queue.get_status(key)
        assert row['status'] == 'pending' and row['retry_count'] == attempt
        assert row['next_attempt_at'] == (clock[0] + delay if delay else None)
        assert sched.drain_delivery_queue({}, None) == 0
        clock[0] = row['next_attempt_at'] or 1000000
    assert sched.drain_delivery_queue({}, None) == 0
    assert admit.call_count == 12
    assert legacy.read_pending(key) is None


def test_clean_mixed_bot_routes_wait_for_all_authority_completions(store, monkeypatch):
    from hermes_cli import profiles
    peer = store / 'profiles' / 'peer'
    peer.mkdir(parents=True)
    (peer / 'state.db').touch()
    monkeypatch.setattr(profiles, 'profile_exists', lambda name: True)
    monkeypatch.setattr(profiles, 'get_profile_dir', lambda name: peer)
    job = dict(id='job', name='Daily', execution_id='execution', deliver='bot-chat,bot-chat:peer')
    queue.enqueue('execution', job, 'body')
    calls = []
    def admit(home, owner, message, **kwargs):
        key = kwargs['delivery_id']
        root = live._root(home); root.mkdir(parents=True, exist_ok=True)
        result = dict(id=key, delivery_id=key, status='queued', message=message)
        legacy.atomic_json_write(root / (key + '.json'), result)
        calls.append((home, key))
        return result
    monkeypatch.setattr(live, 'deliver_to_live_owner', admit)
    assert sched.drain_delivery_queue({}, None) == 1
    assert len(calls) == 2
    assert queue.get_status('execution')['status'] == 'pending'
    assert delivery_outcome.settled_outcome('execution') is None
    assert queue.claim_next() is None
    for index, (home, key) in enumerate(calls):
        legacy.atomic_json_write(live._root(home) / (key + '.json'), dict(id=key, delivery_id=key, status='settled'))
        assert delivery_outcome.settled_outcome('execution') == (None if index == 0 else 'delivered')
    assert sched.drain_delivery_queue({}, None) == 0
    assert len(calls) == 2



@pytest.mark.parametrize('imported', [False, True])
@pytest.mark.parametrize('completion,expected', [('settled', 'delivered'), ('failed', 'failed'), ('suppressed', 'suppressed')])
def test_handoff_housekeeping_writes_original_execution_and_keeps_tombstone_mapping(
        store, monkeypatch, imported, completion, expected):
    from cron import jobs, executions
    monkeypatch.setattr(sched, 'run_job', lambda job, **kw: (True, 'raw', 'body', None))
    job = jobs.create_job(prompt='p', schedule='every 1h', deliver='bot-chat')
    sched.run_one_job(job)
    execution = executions.latest_execution(job['id'])['id']
    queue_key = execution
    if imported:
        queue_key = 'b' * 64
        with queue._transaction() as conn:
            conn.execute('UPDATE deliveries SET execution_id=? WHERE execution_id=?', (queue_key, execution))
    calls = []
    def admit(home, owner, message, **kwargs):
        key = kwargs['delivery_id']
        root = live._root(home); root.mkdir(parents=True, exist_ok=True)
        result = dict(id=key, delivery_id=key, status='queued', message=message)
        legacy.atomic_json_write(root / (key + '.json'), result)
        calls.append((home, key))
        return result
    monkeypatch.setattr(live, 'deliver_to_live_owner', admit)
    assert sched.drain_delivery_queue({}, None) == 1
    assert executions.latest_execution(job['id'])['delivery_outcome'] == 'queued'
    assert queue.get_status(queue_key)['status'] == 'pending'
    home, key = calls[0]
    legacy.atomic_json_write(live._root(home) / (key + '.json'),
                            dict(id=key, delivery_id=key, status=completion, error='failed' if completion == 'failed' else None))
    monkeypatch.setattr(queue, 'MAX_TERMINAL_DELIVERIES', 0)
    assert sched.drain_delivery_queue({}, None) == 0
    assert len(calls) == 1
    assert queue.get_status(queue_key)['status'] == expected
    assert executions.latest_execution(job['id'])['delivery_outcome'] == expected
    assert jobs.get_job(job['id'])['last_status'] == ('delivery_failed' if expected == 'failed' else 'ok')
    assert delivery_outcome.settled_outcome(execution) == expected
    if imported:
        assert delivery_outcome.settled_outcome(queue_key) is None


def test_fresh_process_recovers_post_receipt_pre_index_crash(store):
    key = 'a' * 64
    legacy.defer(key, {'id': 'a'}, 'body', store, for_failure=False)
    crash = '''
import os
from cron import bot_chat_legacy as m
key='a'*64
real=m.atomic_json_write
def fail(path,value,**kw):
    if path.name=='.terminal-index': os._exit(23)
    return real(path,value,**kw)
m.atomic_json_write=fail
with m.ownership_arbitration(key):
    r=m.read_pending(key); r['status']='settled'
    m._terminalize_unlocked(m._receipt_path(m._root(),key),r)
'''
    result = subprocess.run([sys.executable, '-c', crash], capture_output=True, text=True, timeout=20)
    assert result.returncode == 23, result.stderr
    record = legacy.read_pending(key)
    assert record['status'] == 'settled' and 'content' not in record
    recover = '''
import json
from pathlib import Path
from cron import bot_chat_legacy as m
m.MAX_TERMINAL_RECEIPTS=1
home=m._root().parent.parent
m.defer('b'*64,{'id':'b'},'body',home,for_failure=False,suppressed=True)
assert m.read_pending('a'*64) is None
assert m.read_pending('b'*64)['status']=='suppressed'
assert json.loads((m._root()/'.terminal-index').read_text())==['b'*64]
assert json.loads((m._root()/'.terminal-intent').read_text()) is None
print('fresh-process recovery passed')
'''
    result = subprocess.run([sys.executable, '-c', recover], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    print(result.stdout.strip())

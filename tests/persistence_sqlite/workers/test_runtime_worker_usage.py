import sqlite3
import pytest
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.sessions import create_session, get_session
from persistence.sqlite.conversation._store.usage import auxiliary_usage_by_task
from persistence.sqlite.workers._executions import register_worker_execution
from persistence.sqlite.workers._receipt import mutate_worker_execution


def test_usage_receipt_includes_route_rows_and_rolls_back_on_receipt_failure(backend, scope_a):
    holder = backend.open(scope_a).holder()
    create_session(holder, 's', 'cli')
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    scope = dict(epoch=epoch, execution_id='w', session_id='s', generation=0)
    register_worker_execution(holder, **scope, kind='compute', adoption_secret='private')
    payload = {'input_tokens': 13, 'output_tokens': 7, 'api_call_count': 1,
               'model': 'model-a', 'billing_provider': 'provider', 'estimated_cost_usd': 0.25}
    args = dict(**scope, sequence=1, operation='usage.main', payload=payload)
    receipt = mutate_worker_execution(holder, **args)
    assert mutate_worker_execution(holder, **args) == receipt
    assert get_session(holder, 's')['input_tokens'] == 13
    with holder._read_ctx() as conn:
        assert conn.execute('SELECT input_tokens FROM session_model_usage').fetchone()[0] == 13
    holder._execute_write(lambda c: c.execute("CREATE TRIGGER deny_receipt BEFORE INSERT ON worker_receipts BEGIN SELECT RAISE(ABORT, 'deny receipt'); END"))
    with pytest.raises(sqlite3.IntegrityError, match='deny receipt'):
        mutate_worker_execution(holder, **(args | {'sequence': 2}))
    assert get_session(holder, 's')['input_tokens'] == 13
    holder._execute_write(lambda c: c.execute('DROP TRIGGER deny_receipt'))
    aux = dict(**scope, sequence=2, operation='usage.auxiliary',
               payload={'task': 'compression', 'model': 'aux', 'input_tokens': 3})
    mutate_worker_execution(holder, **aux)
    mutate_worker_execution(holder, **aux)
    assert get_session(holder, 's')['input_tokens'] == 13
    with holder._read_ctx() as conn:
        assert conn.execute("SELECT input_tokens FROM session_model_usage WHERE task='compression'").fetchone()[0] == 3
    with pytest.raises(AuthorityRefusal, match='invalid_params'):
        mutate_worker_execution(holder, **(args | {'sequence': 3, 'payload': {'input_tokens': -1}}))


def test_worker_voice_turn_usage_counts_but_never_rewrites_the_session_route(backend, scope_a):
    """A managed worker's per-turn route (``task``) reaches the owner: totals count, the
    session's recorded model stays the one the user picked, and the per-task row is kept."""
    holder = backend.open(scope_a).holder()
    create_session(holder, 's', 'cli', model='main-model')
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    scope = dict(epoch=epoch, execution_id='w', session_id='s', generation=0)
    register_worker_execution(holder, **scope, kind='compute', adoption_secret='private')
    payload = {'input_tokens': 10, 'api_call_count': 1, 'model': 'voice-model',
               'billing_provider': 'voicep', 'task': 'voice_chat'}
    mutate_worker_execution(holder, **scope, sequence=1, operation='usage.main', payload=payload)
    row = get_session(holder, 's')
    assert (row['input_tokens'], row['model']) == (10, 'main-model')
    assert auxiliary_usage_by_task(holder, 's')['voice_chat']['input_tokens'] == 10

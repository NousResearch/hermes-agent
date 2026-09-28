import sqlite3
import pytest
from hermes_state import SessionDB
import hermes_state_runtime as rt


def test_usage_receipt_includes_route_rows_and_rolls_back_on_receipt_failure(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('s', source='cli')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        scope = dict(epoch=epoch, execution_id='w', session_id='s', generation=0)
        rt.register_worker_execution(db, **scope, kind='compute', adoption_secret='private')
        payload = {'input_tokens': 13, 'output_tokens': 7, 'api_call_count': 1,
                   'model': 'model-a', 'billing_provider': 'provider', 'estimated_cost_usd': 0.25}
        args = dict(**scope, sequence=1, operation='usage.main', payload=payload)
        receipt = rt.mutate_worker_execution(db, **args)
        assert rt.mutate_worker_execution(db, **args) == receipt
        assert db.get_session('s')['input_tokens'] == 13
        with db._read_ctx() as conn:
            assert conn.execute('SELECT input_tokens FROM session_model_usage').fetchone()[0] == 13
        db._execute_write(lambda c: c.execute("CREATE TRIGGER deny_receipt BEFORE INSERT ON worker_receipts BEGIN SELECT RAISE(ABORT, 'deny receipt'); END"))
        with pytest.raises(sqlite3.IntegrityError, match='deny receipt'):
            rt.mutate_worker_execution(db, **(args | {'sequence': 2}))
        assert db.get_session('s')['input_tokens'] == 13
        db._execute_write(lambda c: c.execute('DROP TRIGGER deny_receipt'))
        aux = dict(**scope, sequence=2, operation='usage.auxiliary',
                   payload={'task': 'compression', 'model': 'aux', 'input_tokens': 3})
        rt.mutate_worker_execution(db, **aux)
        rt.mutate_worker_execution(db, **aux)
        assert db.get_session('s')['input_tokens'] == 13
        with db._read_ctx() as conn:
            assert conn.execute("SELECT input_tokens FROM session_model_usage WHERE task='compression'").fetchone()[0] == 3
        with pytest.raises(rt.RuntimeStoreError, match='invalid_params'):
            rt.mutate_worker_execution(db, **(args | {'sequence': 3, 'payload': {'input_tokens': -1}}))
    finally:
        db.close()

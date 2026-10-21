"""A managed turn's receipt keeps its spend: the worker's result frame carries the bounded
accounting fields of ``run_conversation`` and the owner commits them with ``usage``, so
``-z --usage-file`` on a managed session reports tokens, cost, model and exit reason."""
import io

from agent.managed_worker import accept_result, encode_frame, read_frame, result_frame
from hermes_cli.oneshot import _USAGE_KEYS


def _run_conversation_result():
    """The keys agent/turn_finalizer.py returns (values from one two-call turn)."""
    return {'final_response': 'done', 'last_reasoning': None, 'messages': [{'role': 'user', 'content': 'go'}],
            'api_calls': 2, 'completed': True, 'turn_exit_reason': 'text_response(finish_reason=stop)',
            'failed': False, 'partial': False, 'interrupted': False, 'response_transformed': False,
            'pre_transform_response': None, 'response_previewed': False, 'response_reused': False,
            'model': 'm', 'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1',
            'input_tokens': 18, 'output_tokens': 10, 'cache_read_tokens': 2, 'cache_write_tokens': 0,
            'reasoning_tokens': 0, 'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30,
            'last_prompt_tokens': 10, 'estimated_cost_usd': 0.0125, 'cost_status': 'estimated',
            'cost_source': 'catalog', 'service_tier': None, 'session_id': 'worker-session',
            'error': 'x' * 10000}


def test_worker_result_frame_carries_the_usage_ledger():
    wire = read_frame(io.BytesIO(encode_frame({'type': 'result', 'result': result_frame(_run_conversation_result())})))
    result, usage = accept_result(wire['result'])
    # Same projection the in-process gateway turn commits (prompt/completion as input/output).
    assert usage == {'input_tokens': 20, 'output_tokens': 10, 'total_tokens': 30}
    # Every ledger key the usage file reads, except the session id the viewer stamps itself.
    assert {k for k in _USAGE_KEYS if k != 'session_id'} <= set(result), set(_USAGE_KEYS) - set(result)
    assert result['estimated_cost_usd'] == 0.0125 and result['turn_exit_reason'].startswith('text_response')
    assert len(result['error']) == 4096 and 'messages' not in result and 'base_url' not in result
    # Unknown or non-scalar fields are refused at the owner boundary.
    for bad in ({'final_response': 'x', 'messages': []}, {'final_response': 'x', 'model': ['m']}, {'failed': True},
                {'final_response': 'x', 'billing_block': {'provider': 'p', 'extra': 1}},
                {'final_response': 'x', 'billing_block': {'message': {'nested': 'no'}}},
                {'final_response': 'x', 'billing_block': 'not a dict'}):
        try:
            accept_result(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_failed_non_billing_turn_with_no_billing_block_is_accepted():
    """Turns that fail for any reason other than a billing wall return ``billing_block: None``; the owner accepts that frame instead of reading it as a lost worker."""
    raw = {**_run_conversation_result(), 'failed': True, 'completed': False, 'failure_reason': 'rate_limit', 'billing_block': None}
    wire = read_frame(io.BytesIO(encode_frame({'type': 'result', 'result': result_frame(raw)})))
    result, _usage = accept_result(wire['result'])
    assert result['failure_reason'] == 'rate_limit' and 'billing_block' not in result
    accepted, _ = accept_result({'final_response': 'x', 'billing_block': None})
    assert 'billing_block' not in accepted


def test_worker_result_frame_carries_the_billing_block():
    """A billing-blocked managed turn hands the owner the same structured block the in-process turn returns, so a local viewer's message.complete can carry ``billing``."""
    block = {'provider': 'custom', 'provider_label': 'Custom', 'model': 'm', 'billing_url': None,
             'is_nous': False, 'message': 'Add credits', 'unverified': True, 'internal': object()}
    raw = {**_run_conversation_result(), 'failed': True, 'failure_reason': 'billing', 'billing_block': block}
    wire = read_frame(io.BytesIO(encode_frame({'type': 'result', 'result': result_frame(raw)})))
    result, _usage = accept_result(wire['result'])
    assert result['billing_block'] == {k: v for k, v in block.items() if k != 'internal'}
    assert result['failure_reason'] == 'billing'


def _usage_db(tmp_path):
    from hermes_state import SessionDB
    return SessionDB(db_path=tmp_path / 'state.db')


def _settle(db, reservation, tip):
    """Settle as the owner does: the tip is whatever the owner's execution row names (compression moves it there)."""
    from types import SimpleNamespace
    from gateway.session_managed_worker import _settled_session_usage
    db._execute_write(lambda conn: conn.execute(
        "INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,generation,status,adoption_digest) "
        "VALUES('exec',?,'compute',1,1,'running','d')", (tip,)))
    try:
        return _settled_session_usage(SimpleNamespace(db=db), reservation, {'model': 'm'}, 'exec')['session_usage']
    finally:
        db._execute_write(lambda conn: conn.execute("DELETE FROM worker_executions WHERE execution_id='exec'"))


def test_settled_usage_sums_the_compression_lineage_a_worker_rotated_through(tmp_path):
    """A managed worker that compresses mid-turn moves its writes to the child session while the owner's reservation still names the parent; the settled turn-end usage covers the whole lineage."""
    db = _usage_db(tmp_path)
    with db:
        db.create_session('root', source='cli')
        db.update_token_counts('root', input_tokens=200, output_tokens=50, api_call_count=2, model='m')
        db.end_session('root', 'compression')
        db.create_session('tip', source='cli', parent_session_id='root')
        db.update_token_counts('tip', input_tokens=300, output_tokens=70, cache_read_tokens=100, api_call_count=1, model='m')
        settled = _settle(db, 'root', 'tip')
    assert (settled['input_tokens'], settled['output_tokens'], settled['cache_read_tokens'], settled['api_calls']) == (500, 120, 100, 3)
    assert settled['prompt_tokens'] == 600 and settled['total_tokens'] == 720


def test_settled_usage_follows_a_branch_through_its_own_compression(tmp_path):
    """A branched session that compresses mid-turn sums itself and its continuation, never the conversation it was branched from."""
    db = _usage_db(tmp_path)
    with db:
        db.create_session('origin', source='cli')
        db.update_token_counts('origin', input_tokens=9000, output_tokens=900, api_call_count=9, model='m')
        db.create_session('branch', source='cli', parent_session_id='origin', model_config={'_branched_from': 'origin'})
        db.update_token_counts('branch', input_tokens=200, output_tokens=50, api_call_count=2, model='m')
        db.end_session('branch', 'compression')
        db.create_session('tip', source='cli', parent_session_id='branch', model_config={'_branched_from': 'origin'})
        db.update_token_counts('tip', input_tokens=300, output_tokens=70, api_call_count=1, model='m')
        settled = _settle(db, 'branch', 'tip')
    assert (settled['input_tokens'], settled['output_tokens'], settled['api_calls']) == (500, 120, 3)


def test_settled_usage_ignores_a_stale_sibling_of_the_continuation(tmp_path):
    """A closed sibling under a compressed parent (an orphaned websocket session) is not part of the turn's lineage; only the path to the tip the execution ended on counts."""
    db = _usage_db(tmp_path)
    with db:
        db.create_session('root', source='cli')
        db.update_token_counts('root', input_tokens=10, api_call_count=1, model='m')
        db.end_session('root', 'compression')
        db.create_session('stale', source='cli', parent_session_id='root')
        db.update_token_counts('stale', input_tokens=1000, api_call_count=5, model='m')
        db.end_session('stale', 'ws_orphan_reap')
        db.create_session('live', source='cli', parent_session_id='root')
        db.update_token_counts('live', input_tokens=20, api_call_count=1, model='m')
        from_root = _settle(db, 'root', 'live')
        from_live = _settle(db, 'live', 'live')
    assert (from_root['input_tokens'], from_root['api_calls']) == (30, 2)
    assert (from_live['input_tokens'], from_live['api_calls']) == (30, 2)


def test_settled_usage_ignores_the_worker_reported_session_id(tmp_path):
    """The result frame's ``session_id`` is worker output: settlement reads the tip from the owner's execution row, so a frame naming a sibling or a foreign session changes nothing."""
    from types import SimpleNamespace
    from gateway.session_managed_worker import _settled_session_usage
    db = _usage_db(tmp_path)
    with db:
        db.create_session('mine', source='cli')
        db.update_token_counts('mine', input_tokens=10, api_call_count=1, model='m')
        db.end_session('mine', 'compression')
        db.create_session('sibling', source='cli', parent_session_id='mine')
        db.update_token_counts('sibling', input_tokens=5000, api_call_count=50, model='m')
        db.create_session('foreign', source='cli')
        db.update_token_counts('foreign', input_tokens=7000, api_call_count=70, model='m')
        for claimed in ('sibling', 'foreign'):
            settled = _settled_session_usage(SimpleNamespace(db=db), 'mine', {'model': 'm', 'session_id': claimed})['session_usage']
            assert (settled['input_tokens'], settled['api_calls']) == (10, 1)

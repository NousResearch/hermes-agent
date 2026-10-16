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
    for bad in ({'final_response': 'x', 'messages': []}, {'final_response': 'x', 'model': ['m']}, {'failed': True}):
        try:
            accept_result(bad)
        except ValueError:
            continue
        raise AssertionError(bad)

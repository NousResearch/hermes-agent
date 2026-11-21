"""Exact plugin tool batches inside the authoritative native conversation loop."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import json
import time

_current = ContextVar('plugin_tool_batch', default=None)


def validate(spec):
    if not isinstance(spec, dict) or not isinstance(spec.get('text'), str) or not spec['text'].strip():
        raise ValueError('A nonempty execution description is required')
    if type(spec.get('require_background', False)) is not bool:
        raise ValueError('require_background must be a boolean')
    calls = spec.get('calls')
    if not isinstance(calls, list) or not 1 <= len(calls) <= 32:
        raise ValueError('Expected between one and 32 tool calls')
    ids = set()
    for call in calls:
        if not isinstance(call, dict) or set(call) != {'call_id', 'name', 'arguments'}:
            raise ValueError('Invalid tool call')
        if any(not isinstance(call[key], str) or not call[key].strip() for key in ('call_id', 'name')):
            raise ValueError('Invalid tool identity')
        if call['call_id'] in ids or not isinstance(call['arguments'], dict):
            raise ValueError('Duplicate tool identity or invalid arguments')
        ids.add(call['call_id'])
    if len(json.dumps(spec)) > 128000:
        raise ValueError('Tool batch exceeds the input limit')
    return spec


@contextmanager
def _scope(spec, before):
    with before:
        token = _current.set(spec)
        try:
            yield
        finally:
            _current.reset(token)


def admission_scope(row, *, before=None):
    """Only a trusted automation ledger row can select exact execution."""
    if not str(row.get('principal_id', '')).startswith('automation:'):
        return None
    try:
        identity = json.loads(row.get('request_id') or 'null')
    except ValueError:
        return None
    if not isinstance(identity, list) or identity[:1] != ['plugin-tools']:
        return None
    # Invalid persisted requests fail closed, never fall back to model inference.
    spec = validate(json.loads(row['payload']['text']))
    return _scope(spec, before if before is not None else nullcontext())


def run(agent, message, kwargs):
    spec = _current.get()
    if spec is None:
        return agent.run_conversation(message, **kwargs)
    if agent.api_mode not in {'chat_completions', 'codex_responses'}:
        raise RuntimeError('Exact tools require Chat Completions or Codex Responses')
    # Native tool executor checks the session-pinned tool availability and approvals.
    from agent.turn_api_call import bind_completion
    from tools.delegate_tool_dispatch import require_background
    first = True
    def complete(parent, payload):
        nonlocal first
        calls = spec['calls'] if first else []
        first = False
        return _response(parent.api_mode, calls)
    policy = require_background(agent) if spec.get('require_background') else nullcontext()
    # The transport is agent-bound: nested subagents use their configured model.
    with bind_completion(agent, complete), policy:
        return agent.run_conversation(message, **kwargs)


def _response(api_mode, calls):
    final = '[SILENT]'
    if api_mode == 'codex_responses':
        from openai.types.responses import Response
        output = [{'type': 'function_call', 'id': 'fc_' + c['call_id'], 'call_id': c['call_id'],
                   'name': c['name'], 'arguments': json.dumps(c['arguments']), 'status': 'completed'} for c in calls]
        if not calls:
            output = [{'type': 'message', 'id': 'msg_plugin_tools', 'role': 'assistant', 'status': 'completed',
                       'content': [{'type': 'output_text', 'text': final, 'annotations': []}]}]
        return Response(id='resp_plugin_tools', created_at=time.time(), model='plugin-tools', object='response',
                        status='completed', output=output, parallel_tool_calls=True, tool_choice='auto', tools=[],
                        usage={'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
                               'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
                               'output_tokens_details': {'reasoning_tokens': 0}})
    from openai.types.chat import ChatCompletion
    message = {'role': 'assistant', 'content': None if calls else final}
    if calls:
        message['tool_calls'] = [{'id': c['call_id'], 'type': 'function', 'function': {
            'name': c['name'], 'arguments': json.dumps(c['arguments'])}} for c in calls]
    return ChatCompletion(id='plugin-tools', created=int(time.time()), model='plugin-tools', object='chat.completion',
        choices=[{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if calls else 'stop'}],
        usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})

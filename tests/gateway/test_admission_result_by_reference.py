"""A settled turn's stored result keeps the turn's heavy output by reference to its transcript rows.

The agent flushes every message of a turn to ``messages`` before settlement, so storing its tool
output, reasoning blobs and inline images again in ``gateway.admission.result.v1.*`` doubled them:
a reporter's store held 391 MB of these rows (p90 388 KB, max 1.6 MB) beside 471 MB of transcript.
The row now keeps the reply, flags and accounting by value and the rest by row id; every reader
(exact retry, Responses chaining, receipts) still gets the same dict back.
"""
import json

import pytest

from gateway.session_api_turn import _settled_api_result
from gateway.session_bot import _admission_outcome
from gateway.session_results import _redacted, admission_result, finish_result, retain_result
from hermes_state_runtime import _json, admit_session_input, claim_session_input, mutate_runtime_session
from hermes_state_terminal import RESULT_PREFIX, compact_result

TOOL_OUTPUT = 'line of tool output\n' * 20_000           # ~400 KB, the reporter's p90 row
REASONING = [{'type': 'thinking', 'thinking': 'step ' * 4_000, 'signature': 'sig' * 2_000},
             {'type': 'redacted_thinking', 'data': 'QUJD' * 10_000}]
REDACTED = 'QUJD' * 10_000                                # Bedrock's sidecar repeats the same base64
IMAGE = 'data:image/png;base64,' + 'iVBORw0KGgo' * 20_000
REPLY = 'the answer'
ROW_BOUND = 4_096                                         # bytes: a small row whatever the turn produced


def _flushed_turn(owner, sid, text='look at this'):
    """One turn as the agent hands it to settlement: its messages already flushed to ``messages``."""
    owner.db.create_session(sid, source='api_server')
    admit_session_input(owner.db, epoch=owner.epoch, principal_id='api', session_id=sid, request_id='r1',
                        payload={'text': text})
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id=sid)
    messages = [
        {'role': 'user', 'content': text},
        {'role': 'assistant', 'content': 'checking', 'reasoning': 'why ' * 500,
         'reasoning_details': REASONING,
         'bedrock_content_blocks': [{'reasoningContent': {'redactedContentBase64': REDACTED}}],
         'tool_calls': [{'id': 'c1', 'type': 'function',
                         'function': {'name': 'terminal', 'arguments': json.dumps({'command': 'x' * 2_000})}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'tool_name': 'terminal', 'content': TOOL_OUTPUT},
        {'role': 'tool', 'tool_call_id': 'c2', 'tool_name': 'vision',
         'content': [{'type': 'text', 'text': 'screenshot'}, {'type': 'image_url', 'image_url': {'url': IMAGE}}]},
        {'role': 'assistant', 'content': REPLY},
    ]
    from agent.session_persistence import _db_flush_row
    from agent.transcript_repair import sync_flushed_message_markers
    from types import SimpleNamespace
    agent = SimpleNamespace(_persist_user_message_idx=None, _persist_user_message_override=None)
    rows = [_db_flush_row(agent, m, False) for m in messages]
    owner.db.append_messages_batch(sid, rows)
    sync_flushed_message_markers(messages, rows)
    result = {'result': {'final_response': REPLY, 'messages': messages, 'completed': True,
                         'current_turn_user_idx': 0, 'input_tokens': 11, 'estimated_cost_usd': 0.25},
              'usage': {'input_tokens': 11, 'output_tokens': 7}}
    return row, result


def _stored_bytes(owner, admission_id):
    with owner.db._read_ctx() as conn:
        return conn.execute('SELECT length(value) FROM state_meta WHERE key=?',
                            (RESULT_PREFIX + admission_id,)).fetchone()[0]


def _by_value(result, text='look at this'):
    """What main stored for the same turn: the compacted, redacted copy, by value."""
    return len(_json(_redacted(compact_result(result, user_message=text))))


def test_a_heavy_turn_stores_a_small_row_and_every_reader_gets_it_back(owner):
    row, result = _flushed_turn(owner, 'heavy')
    before = _by_value(result)
    retain_result(owner.db, epoch=owner.epoch, row=row, result=result)
    stored = _stored_bytes(owner, row['admission_id'])
    assert stored < ROW_BOUND, f'{stored} bytes stored for a turn main stored in {before}'
    assert before > 100 * stored

    saved = admission_result(owner.db, row['admission_id'])
    value = saved['result']
    # Exact retry and accounting: by value, never through the transcript.
    assert value['final_response'] == REPLY and saved['usage'] == {'input_tokens': 11, 'output_tokens': 7}
    assert value['estimated_cost_usd'] == 0.25 and value['completed'] is True
    assert finish_result(owner.db, epoch=owner.epoch, row=row, response='other', outcome='completed')[1] == REPLY
    assert _admission_outcome(owner, row['admission_id']) == ('settled', REPLY)
    # Responses chaining / receipts: the turn suffix comes back as its transcript rows hold it.
    assistant, tool, image, final = value['messages']
    assert tool['content'] == TOOL_OUTPUT and tool['tool_call_id'] == 'c1'
    assert assistant['reasoning_details'] == REASONING and assistant['reasoning'] == 'why ' * 500
    assert assistant['tool_calls'][0]['function']['arguments'] == json.dumps({'command': 'x' * 2_000})
    assert final['content'] == REPLY
    # Documented degradation: the live-only Bedrock sidecar (same bytes as reasoning_details) and an
    # inline image's data URL are stored as the transcript keeps them.
    assert 'bedrock_content_blocks' not in assistant
    assert IMAGE not in json.dumps(value) and '[screenshot]' in image['content']
    assert _settled_api_result(owner, row['admission_id']) == (value, saved['usage'])


def test_a_referenced_field_is_redacted_on_reload(owner, monkeypatch):
    """The stored copy is redacted at the storage boundary; a reloaded field gets the same pass."""
    row, result = _flushed_turn(owner, 'secret')
    retain_result(owner.db, epoch=owner.epoch, row=row, result=result)
    import agent.redact
    monkeypatch.setattr(agent.redact, 'redact_sensitive_text', lambda text, **_: text.replace('tool output', 'REDACTED'))
    tool = admission_result(owner.db, row['admission_id'])['result']['messages'][1]
    assert 'tool output' not in tool['content'] and 'REDACTED' in tool['content']


def test_a_turn_that_was_never_flushed_is_stored_by_value(owner):
    """No transcript row names it (a mocked or refused turn): the row keeps everything, as before."""
    owner.db.create_session('loose', source='api_server')
    admit_session_input(owner.db, epoch=owner.epoch, principal_id='api', session_id='loose', request_id='r',
                        payload={'text': 'q'})
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id='loose')
    result = {'result': {'final_response': REPLY, 'current_turn_user_idx': 0, 'messages': [
        {'role': 'user', 'content': 'q'}, {'role': 'tool', 'tool_call_id': 'c', 'content': 'out' * 100},
        {'role': 'assistant', 'content': REPLY}]}, 'usage': {}}
    expected = json.loads(_json(_redacted(compact_result(result, user_message='q'))))
    retain_result(owner.db, epoch=owner.epoch, row=row, result=result)
    assert admission_result(owner.db, row['admission_id']) == expected


def test_a_deleted_session_keeps_the_exact_retry_reply_without_its_transcript(owner):
    row, result = _flushed_turn(owner, 'gone')
    retain_result(owner.db, epoch=owner.epoch, row=row, result=result)
    snap = owner.db.get_session('gone')
    mutate_runtime_session(owner.db, epoch=owner.epoch, principal_id='human', session_id='gone', request_id='del',
                           operation='delete', payload={}, expected_revision=snap['runtime_revision'],
                           expected_generation=snap['runtime_generation'])
    retired = admission_result(owner.db, row['admission_id'])
    assert retired['result']['final_response'] == REPLY and retired['result']['messages'] == []
    assert retired['usage'] == {'input_tokens': 11, 'output_tokens': 7}
    assert TOOL_OUTPUT not in json.dumps(retired)


@pytest.mark.parametrize('stream', [False, True], ids=['json', 'sse'])
@pytest.mark.asyncio
async def test_previous_response_chain_replays_the_referenced_tool_suffix(api, owner, stream):
    """A chained Responses turn rebuilds its history from the prior turn's stored result: the tool
    call and its output arrive intact although the row holds them by reference."""
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    seen = []

    async def handle(event):
        from gateway.session_api_turn import api_execution
        from gateway.session_results import execution_result
        from hermes_state import SessionDB
        history = api_execution.get()['history']
        seen.append(history)
        sid = owner.sessions and next(iter(owner.sessions))
        turn = [{'role': 'user', 'content': event.text},
                {'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': 'c1', 'type': 'function', 'function': {'name': 'terminal', 'arguments': '{"command":"ls"}'}}]},
                {'role': 'tool', 'tool_call_id': 'c1', 'content': TOOL_OUTPUT},
                {'role': 'assistant', 'content': 'a'}]
        assert isinstance(owner.db, SessionDB)
        from agent.session_persistence import _db_flush_row
        from agent.transcript_repair import sync_flushed_message_markers
        from types import SimpleNamespace
        agent = SimpleNamespace(_persist_user_message_idx=None, _persist_user_message_override=None)
        rows = [_db_flush_row(agent, m, False) for m in turn]
        owner.db.append_messages_batch(sid, rows)
        sync_flushed_message_markers(turn, rows)
        execution_result.get()['result'] = {'final_response': 'a', 'messages': [*history, *turn],
                                            'current_turn_user_idx': len(history)}
        return 'a'
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    async with TestClient(TestServer(app)) as client:
        first = await client.post('/v1/responses', json={'input': 'first'})
        rid = (await first.json())['id']
        second = await client.post('/v1/responses', json={
            'input': 'second', 'previous_response_id': rid, 'stream': stream})
        assert second.status == 200, await second.text()
        await second.text()
    tool = next(m for m in seen[1] if m['role'] == 'tool')
    assert tool['content'] == TOOL_OUTPUT
    with owner.db._read_ctx() as conn:
        sizes = [r[0] for r in conn.execute(
            "SELECT length(value) FROM state_meta WHERE key LIKE 'gateway.admission.result.v1.%'")]
    assert sizes and max(sizes) < ROW_BOUND, sizes

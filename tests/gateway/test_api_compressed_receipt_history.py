"""Responses chaining preserves a turn's compressed replacement, while ordinary receipts stay compact."""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
async def test_compressed_receipt_chains_its_snapshot_then_retires_history(api, owner):
    from gateway.session_api_turn import api_execution
    from gateway.session_results import admission_result, execution_result
    seen = []
    compressed = [
        {'role': 'assistant', 'content': 'compressed-summary-sentinel', '_compressed_summary': True},
        {'role': 'user', 'content': 'compress now'},
        {'role': 'assistant', 'content': 'compressed answer'},
    ]

    async def handle(event):
        history = api_execution.get()['history']
        seen.append(history)
        if event.text == 'compress now':
            result = {'final_response': 'compressed answer', '_compressed': True, 'messages': compressed}
        else:
            result = {'final_response': 'next answer', 'messages': [*history,
                {'role': 'user', 'content': event.text}, {'role': 'assistant', 'content': 'next answer'}]}
        execution_result.get()['result'] = result
        return result['final_response']

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    async with TestClient(TestServer(app)) as client:
        first = await client.post('/v1/responses', json={'input': 'compress now',
            'conversation_history': [{'role': 'user', 'content': 'old-uncompressed-history'}]},
            headers={'Idempotency-Key': 'compress-key'})
        assert first.status == 200
        first_result = await first.json()
        second = await client.post('/v1/responses', json={'input': 'next', 'previous_response_id': first_result['id']},
                                   headers={'Idempotency-Key': 'next-key'})
        assert second.status == 200
    assert seen[1] == compressed
    assert 'old-uncompressed-history' not in str(seen[1])
    rows = owner.db._read_all('SELECT admission_id,target_session_id FROM session_admissions ORDER BY seq')
    assert admission_result(owner.db, rows[0]['admission_id'])['result']['messages'] == compressed
    assert admission_result(owner.db, rows[1]['admission_id'])['result']['messages'] == [
        {'role': 'assistant', 'content': 'next answer'}]
    assert owner.db.delete_session(rows[0]['target_session_id'])
    for row in rows:
        saved = admission_result(owner.db, row['admission_id'])
        assert saved['result']['messages'] == []
        assert 'compressed-summary-sentinel' not in str(saved)

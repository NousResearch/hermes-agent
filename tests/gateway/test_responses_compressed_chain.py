"""``previous_response_id`` follows the logical conversation across an out-of-place compression."""
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


def _compressing_app(api, owner, calls):
    """The first turn compresses out of place (``compression.in_place: false``): a real
    compression child is published, the route advances to it and the turn reports the child's
    physical id, as the TurnRunner does."""
    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        root = event.source.chat_id
        result = {'final_response': 'answer ' + event.text, 'messages': []}
        if len(calls) == 1:
            child = root + '-child'
            owner.db.publish_compression_child(parent_session_id=root, child_session_id=child,
                source='api_server', messages=[{'role': 'assistant', 'content': 'summary'}],
                require_compression_lease=False)
            route = owner.sessions[root].route
            assert owner.runner.session_store.advance_compression_session(route, root, child) is not None
            result['session_id'] = child
        execution_result.get()['result'] = result
        return result['final_response']
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    return app


async def _post(client, body, key):
    response = await client.post('/v1/responses', json=body, headers={'Idempotency-Key': key})
    text = await response.text()
    if body.get('stream') and response.status == 200:
        events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: ')]
        return response.status, events[-1]['response']
    return response.status, json.loads(text)


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True], ids=['json', 'sse'])
async def test_previous_response_id_after_compression_child_admits_on_the_logical_owner(api, owner, stream):
    calls = []
    async with TestClient(TestServer(_compressing_app(api, owner, calls))) as client:
        status, first = await _post(client, {'input': 'one', 'stream': stream}, 'first')
        assert status == 200, first
        follow = {'input': 'two', 'previous_response_id': first['id'], 'stream': stream}
        status, second = await _post(client, follow, 'second')
        assert status == 200, second
        # The exact same-key retry replays the settled answer; it never admits a second turn.
        assert await _post(client, follow, 'second') == (200, second)
        # A keyless chained turn takes the same owner.
        third = await client.post('/v1/responses', json={'input': 'three', 'previous_response_id': second['id']})
        assert third.status == 200, await third.text()
    assert calls == ['one', 'two', 'three']
    roots = {row['target_session_id'] for row in owner.db._read_all(
        "SELECT target_session_id FROM session_admissions WHERE principal_id='api'", ())}
    # Both turns queued on one FIFO: the root of the compression lineage, never its child.
    assert len(roots) == 1 and not next(iter(roots)).endswith('-child')

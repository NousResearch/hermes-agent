"""The existing run status poll reports only its admission's live tool progress."""
import pytest

from gateway.session_authority import admission_row
from tests.gateway.test_api_unknown_resolution import _AUTH_HEADERS, _OTHER_AUTH_HEADERS, _runs_client


@pytest.mark.asyncio
async def test_run_progress_is_scoped_to_the_started_admission(api, owner):
    owner._schedule = lambda ref: None
    client = _runs_client(api)
    await client.start_server()
    try:
        paths = []
        for text in ('HEAD', 'FOLLOWER'):
            accepted = await client.post('/v1/runs', json={'input': text, 'session_id': 'api-session'},
                                         headers={**_AUTH_HEADERS, 'Idempotency-Key': text})
            assert accepted.status == 202, await accepted.text()
            paths.append('/v1/runs/' + (await accepted.json())['run_id'])
        row = admission_row(owner.persistence.admissions.claim(epoch=owner.epoch, session_id='api-session'))
        generation = row['generation']
        owner.executing['api-session'] = generation

        async def poll(path=paths[0]):
            response = await client.get(path, headers=_AUTH_HEADERS)
            assert response.status == 200, await response.text()
            return await response.json()

        assert 'progress_at' not in await poll()
        owner.publish_execution('api-session', generation, 'tool.start', {'tool_id': 'one', 'name': 'terminal'})
        active = await poll()
        assert active['tool'] == 'terminal' and active['progress_at'] > 0
        assert (await poll())['progress_at'] == active['progress_at']
        assert 'progress_at' not in await poll(paths[1])  # queued on the same session
        assert (await client.get(paths[0], headers=_OTHER_AUTH_HEADERS)).status == 404
        owner.publish_execution('api-session', generation + 1, 'tool.start', {'tool_id': 'late', 'name': 'wrong'})
        assert await poll() == active
        owner.publish_execution('api-session', generation, 'tool.complete', {'tool_id': 'one'})
        completed = await poll()
        assert 'tool' not in completed and completed['progress_at'] >= active['progress_at']
        owner.publish_execution('api-session', generation, 'tool.start', {'tool_id': 'two', 'name': 'terminal'})
        owner.persistence.admissions.settle(epoch=owner.epoch, admission_id=row['admission_id'],
                                            generation=generation, outcome='interrupted')
        assert 'progress_at' not in await poll()
        follower = admission_row(owner.persistence.admissions.claim(epoch=owner.epoch, session_id='api-session'))
        assert follower['generation'] != generation
        assert 'progress_at' not in await poll(paths[1])
    finally:
        for task in list(api._active_run_tasks.values()):
            task.cancel()
        await client.close()

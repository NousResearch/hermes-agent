"""A cancel landing while the terminal snapshot commits off the loop never overwrites it with
``incomplete``: the commit's worker keeps running after its awaiting handler is cancelled."""
import asyncio
import threading

import pytest

from gateway.platforms.api_server_openai_routes import _ResponsesStream


class _Sink:
    async def write(self, frame):
        pass


@pytest.mark.asyncio
async def test_cancel_during_the_terminal_commit_keeps_the_completed_snapshot(api):
    store = api._current_response_store()
    entered, release = threading.Event(), threading.Event()
    real_put = store.put

    def slow_put(response_id, data):
        if data.get('response', {}).get('status') == 'completed':
            entered.set()
            release.wait(5)
        return real_put(response_id, data)

    store.put = slow_put
    stream = _ResponsesStream(api, _Sink(), response_id='resp_cancel', model='m', created_at=0,
                              conversation_history=[], user_message='hi', instructions=None,
                              conversation=None, store=True, session_id='s')
    stream.result = {'final_response': 'done', 'messages': []}
    stream.final_response_text = 'done'
    completing = asyncio.create_task(stream.emit_completed())
    assert await asyncio.to_thread(entered.wait, 5)  # the completed snapshot is being written
    completing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await completing
    # _write_sse_responses' CancelledError branch.
    incomplete = asyncio.create_task(stream.persist_incomplete_if_needed())
    release.set()
    await incomplete
    assert store.get('resp_cancel')['response']['status'] == 'completed'

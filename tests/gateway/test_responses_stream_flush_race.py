"""Keyed Responses streaming: the end-of-stream flush waits for a timer flush still opening the message
item. The keyed output index is now read off the loop, so that open yields; an end-of-stream flush that
skipped the lock on an empty buffer closed the item before its index existed and failed the turn."""
import asyncio
import threading

import pytest

from gateway.platforms.api_server_openai_routes import _ResponsesStream


class _Sink:
    def __init__(self):
        self.frames = []

    async def write(self, frame):
        self.frames.append(frame)


@pytest.mark.asyncio
async def test_end_of_stream_flush_waits_for_a_timer_flush_awaiting_its_output_index(api):
    store = api._current_response_store()
    entered, release = threading.Event(), threading.Event()
    real_index = store.output_index

    def slow_index(*args):
        entered.set()
        release.wait(5)
        return real_index(*args)

    store.output_index = slow_index
    sink = _Sink()
    stream = _ResponsesStream(api, sink, response_id='resp_race', model='m', created_at=0,
                              conversation_history=[], user_message='hi', instructions=None,
                              conversation=None, store=False, session_id='s', durable_key=('race-key', 'fp'))
    await stream.dispatch('hello')
    timer = stream._batch_timer
    assert await asyncio.to_thread(entered.wait, 5)  # the timer flush is inside the off-loop index read

    async def end_of_stream():  # as _write_sse_responses runs it on the EOS sentinel
        stream.cancel_batch_timer()
        await stream.flush_batch()
        await stream.close_message_item()

    eos = asyncio.create_task(end_of_stream())
    for _ in range(5):
        await asyncio.sleep(0)
    release.set()
    await asyncio.gather(timer, eos)
    assert b'response.output_text.done' in b''.join(sink.frames)
    assert stream.final_response_text == 'hello'

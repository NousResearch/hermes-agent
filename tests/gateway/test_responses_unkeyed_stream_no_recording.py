"""An unkeyed streaming Responses request keeps no replay copy of its frames: only a keyed
request's replay record ever reads them."""
import pytest

from gateway.platforms.api_server_openai_routes import _ResponsesStream


class _Sink:
    def __init__(self):
        self.frames = []

    async def write(self, frame):
        self.frames.append(frame)


@pytest.mark.asyncio
@pytest.mark.parametrize('durable_key, recorded', [(None, 0), (('k', 'fp'), 1)], ids=['unkeyed', 'keyed'])
async def test_stream_records_frames_only_for_keyed_replay(api, durable_key, recorded):
    sink = _Sink()
    stream = _ResponsesStream(api, sink, response_id='resp_x', model='m', created_at=0,
                              conversation_history=[], user_message='hi', instructions=None,
                              conversation=None, store=False, session_id='s', durable_key=durable_key)
    await stream.write_event('response.created', {'type': 'response.created'})
    assert len(sink.frames) == 1 and len(stream.recorded_events) == recorded

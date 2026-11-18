"""``/v1/runs`` SSE carries the per-model-call meters the agent reports."""

import asyncio

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_runs import _RunStream


@pytest.mark.asyncio
async def test_model_call_events_reach_the_run_stream():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    loop = asyncio.get_running_loop()
    adapter._run_streams["run_meter"] = _RunStream()
    callback = adapter._make_run_event_callback("run_meter", loop)

    callback("model.rate_limited", None, None, None, model="m", provider="p", attempt=1,
             max_attempts=3, retry_after_ms=2000, wait_ms=2000, api_request_id=None)
    callback("usage.recorded", None, None, None, call_index=1, model="m", provider="p",
             input_tokens=20, output_tokens=7, cache_read_tokens=100, cache_write_tokens=0,
             reasoning_tokens=0, cost_usd=None, first_token_ms=310, duration_ms=900,
             used_tokens=120, window_tokens=128000)
    await asyncio.sleep(0)  # the callback hops onto the loop via call_soon_threadsafe
    events = [event for _, event in adapter._run_streams["run_meter"].backlog]

    assert [event["event"] for event in events] == ["model.rate_limited", "usage.recorded"]
    assert events[0]["retry_after_ms"] == 2000 and events[0]["attempt"] == 1
    assert "api_request_id" not in events[0]  # unknown values are omitted, not sent as null
    assert events[1]["first_token_ms"] == 310
    assert (events[1]["used_tokens"], events[1]["window_tokens"]) == (120, 128000)

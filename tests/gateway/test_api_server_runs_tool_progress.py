"""``tool.progress`` (live MCP progress ticks) on the /v1/runs SSE stream.

``agent/tool_progress_relay.py`` emits ``callback('tool.progress', name, message, None,
tool_call_id=..., progress=..., total=..., percent=...)``; this is the direct (non-admission)
callback path through ``_make_run_event_callback`` / ``_FIXED_EVENT_FIELDS`` in
``gateway/platforms/api_server_runs.py``. The admission-observer forwarding path is covered
separately in ``tests/gateway/test_api_tool_progress_observer.py``.

Split out of ``test_api_server_runs.py`` (at the 2000-line file-size ratchet) rather than grown
in place; reuses that module's ``_make_adapter`` fixture helper explicitly.
"""
import asyncio

import pytest

from gateway.platforms.api_server_runs import _RunStream
from tests.gateway.test_api_server_runs import _make_adapter


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.mark.parametrize(
    ("total", "percent"), [(2.0, 50), (None, None)], ids=["with_total", "missing_total"],
)
@pytest.mark.asyncio
async def test_tool_progress_event_forwards_mcp_fields_without_inferring_percent(adapter, total, percent):
    """The event reaches the stream with its own fields verbatim; a missing total/percent is
    never synthesized into a number. Routes only to ITS OWN run's stream, by exact run_id."""
    loop = asyncio.get_running_loop()
    adapter._run_streams["run_progress"] = _RunStream()
    adapter._run_streams["run_other"] = _RunStream()
    callback = adapter._make_run_event_callback("run_progress", loop)

    callback(
        "tool.progress", "fetch_url", "halfway", None,
        tool_call_id="call-1", progress=1.0, total=total, percent=percent,
    )
    await asyncio.sleep(0)  # the callback hops onto the loop via call_soon_threadsafe
    _, event = adapter._run_streams["run_progress"].backlog[-1]

    assert event["event"] == "tool.progress"
    assert event["run_id"] == "run_progress"
    assert event["name"] == "fetch_url"
    assert event["message"] == "halfway"
    assert event["progress"] == 1.0
    assert event["total"] == total
    assert event["percent"] == percent
    assert event["tool_call_id"] == "call-1"
    assert len(adapter._run_streams["run_other"].backlog) == 0


@pytest.mark.asyncio
async def test_tool_progress_message_is_secret_redacted_before_the_public_stream(adapter):
    """The ``message`` is free text an MCP server chose; it must be force-redacted the same way
    ``tool.completed``'s result preview and subagent free text are, before it reaches the public
    run SSE stream."""
    loop = asyncio.get_running_loop()
    adapter._run_streams["run_progress"] = _RunStream()
    callback = adapter._make_run_event_callback("run_progress", loop)

    callback(
        "tool.progress", "fetch_url", "token sk-abcdefghijklmnopqrstuvwxyz leaked", None,
        tool_call_id="call-1", progress=1.0, total=2.0, percent=50,
    )
    await asyncio.sleep(0)
    _, event = adapter._run_streams["run_progress"].backlog[-1]

    assert "abcdefghijklmnopqrstuvwxyz" not in event["message"]
    assert "leaked" in event["message"]

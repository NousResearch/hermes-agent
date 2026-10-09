"""Transport keepalives must not postpone the inference-progress watchdog."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletionChunk

from agent.chat_completion_helpers import _StreamingCall


def chunk(delta=None, finish_reason=None):
    return ChatCompletionChunk(
        id="test", object="chat.completion.chunk", created=0, model="test",
        choices=[] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    )


def test_keepalives_remain_diagnostic_without_resetting_progress(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("agent.chat_completion_helpers.time.time", lambda: clock[0])
    activity = Mock()
    call = _StreamingCall(NS(_touch_activity=activity), {}, None)
    diag = {}
    for event in [chunk(), chunk({}), chunk({"role": "assistant", "content": ""}),
                  chunk({"tool_calls": [{"index": 0}]}), NS(type="ping"),
                  NS(type="content_block_delta", delta=NS(type="text_delta", text=""))]:
        clock[0] += 60
        call._count_chunk(diag, event)
        assert call.last_chunk_time["t"] == 100.0
    assert clock[0] - call.last_chunk_time["t"] > 240
    assert diag["chunks"] == 6
    assert diag["first_chunk_at"] == 160.0
    activity.assert_not_called()

    # Exercise the same monitor that closes the request after a quiet period.
    done = [False]
    call.agent.base_url = "https://example.com"
    call.agent._interrupt_requested = False
    call._stream_stale_timeout = 240
    call._call_done = NS(is_set=lambda: done[0], wait=lambda **kwargs: None)
    kill = Mock(side_effect=lambda elapsed: done.__setitem__(0, True))
    monkeypatch.setattr(call, "_kill_stale_stream", kill)
    call._monitor_loop()
    kill.assert_called_once_with(clock[0] - 100.0)


@pytest.mark.parametrize("event", [
    chunk({"content": "answer"}), chunk({"reasoning_content": "thinking"}),
    chunk({"reasoning": "thinking"}), chunk({}, "stop"),
    chunk({"tool_calls": [{"index": 0, "id": "call_1"}]}),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]}),
    NS(type="content_block_delta", delta=NS(type="thinking_delta", thinking="reasoning")),
    NS(type="content_block_delta", delta=NS(type="input_json_delta", partial_json="{")),
    NS(type="message_start"), NS(type="message_stop"),
])
def test_output_and_protocol_progress_refresh_the_watchdog(monkeypatch, event):
    clock = [100.0]
    monkeypatch.setattr("agent.chat_completion_helpers.time.time", lambda: clock[0])
    activity = Mock()
    call = _StreamingCall(NS(_touch_activity=activity), {}, None)
    clock[0] = 200.0
    call._count_chunk({}, event)
    assert call.last_chunk_time["t"] == clock[0]
    activity.assert_called_once()

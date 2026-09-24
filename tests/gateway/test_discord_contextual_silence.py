"""Discord shared human turns may be intentionally quiet, including queued and crash delivery."""
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.response_filters import is_intentional_silence_response
from gateway.session import SessionSource
from tests.gateway.test_gateway_silence_tokens import _runner


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["thread", "group", "dm"])
@pytest.mark.parametrize("text,failed", [("NO_REPLY", False), ("", False), ("NO_REPLY", True)])
async def test_shared_silence_has_live_queued_and_crash_delivery_parity(
    monkeypatch, tmp_path, chat_type, text, failed,
):
    runner = _runner(monkeypatch, tmp_path)
    source = SessionSource(platform=Platform.DISCORD, chat_id="123", chat_type=chat_type, user_id="42")
    event = MessageEvent(text="FYI to another human", source=source, message_id="456")
    now = time.time()
    messages = [{"role": "user", "content": event.text, "timestamp": now},
                {"role": "assistant", "content": text, "timestamp": now}]
    result = {"final_response": text, "messages": messages, "tools": [], "history_offset": 0,
              "last_prompt_tokens": 0, "api_calls": 1, "failed": failed}
    runner._run_agent = AsyncMock(return_value=result)
    response = await runner._handle_message_with_agent(event, source, "agent:main:discord:thread:123", 1)
    quiet = chat_type != "dm" and text == "NO_REPLY" and not failed
    assert (response == "") is quiet
    if quiet:
        appended = [c.args[1] for c in runner.session_store.append_to_transcript.call_args_list]
        assert [m["role"] for m in appended if m.get("role") in {"user", "assistant"}] == ["user", "assistant"]
        assert appended[-1]["content"] == "NO_REPLY"

    if text == "NO_REPLY" and not failed:
        runner._deliver_queued_first_response = AsyncMock()
        ctx = SimpleNamespace(session_key="agent:main:discord:thread:123", source=source,
                              stream_consumer_holder=[None], mute_notification_reply=False,
                              persist_user_display_kind=None, _status_thread_metadata=None,
                              event_message_id="456", inbound_message_id="456", run_generation=1)
        await runner._run_agent_deliver_first_response(ctx, None, result, result, None)
        if quiet:
            runner._deliver_queued_first_response.assert_not_awaited()
        else:
            assert not is_intentional_silence_response(runner._deliver_queued_first_response.await_args.args[0])
        assert (runner._crash_left_reply(messages, now - 1, source) == "") is quiet

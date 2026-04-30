"""End-to-end integration test for the slow-LLM postback flow.

Drives `_dispatch_event → handle_message → _keep_typing → send → cache →
_handle_postback` as a single sequence, with the LLM stubbed via a slow
``_message_handler`` and the LINE API mocked via respx. Catches wiring gaps
between BasePlatformAdapter and the LineAdapter overrides — the kind of
production-only race that unit tests with hand-injected ``stop_event`` miss.
"""
import asyncio
import json
import time

import pytest
import respx
from httpx import Response

from gateway.platforms.line import LineAdapter, State
from tests.gateway.conftest import make_line_platform_config


@pytest.mark.asyncio
@respx.mock
async def test_slow_llm_postback_full_round_trip(monkeypatch):
    """Full slow-LLM round trip without manually injecting any internals.

    Sequence simulated:
      1. Webhook delivers an inbound message → _dispatch_event runs.
      2. base.handle_message spawns _keep_typing (real) AND a slow stubbed
         message handler (~0.3s) to force the threshold (0.1s) to elapse.
      3. _keep_typing sends the postback button via Reply API (mocked) and
         arms _pending_buttons.
      4. Slow handler returns its response → base calls send() → cached as
         READY under the button's request_id.
      5. Postback webhook (the user "tap") delivers the cached response via
         a fresh reply_token (mocked).

    Asserts every observable boundary: button POST shape, cache transitions,
    pending-button slot freed, postback delivery payload.
    """
    reply_route = respx.post("https://api.line.me/v2/bot/message/reply").mock(
        return_value=Response(200, json={})
    )
    respx.post("https://api.line.me/v2/bot/chat/loading/start").mock(
        return_value=Response(200, json={})
    )

    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
    monkeypatch.setenv("LINE_ALLOWED_USERS", "U_e2e")
    monkeypatch.setenv("LINE_SLOW_RESPONSE_THRESHOLD", "0.1")
    adapter = LineAdapter(make_line_platform_config(token="t"))

    final_response = "the final agent answer (cached for postback retrieval)"

    async def slow_handler(event):
        # Long enough to force threshold elapse (0.1s) so the button arms.
        await asyncio.sleep(0.4)
        return final_response

    adapter._message_handler = slow_handler

    inbound_event = {
        "type": "message",
        "replyToken": "rt-original",
        "source": {"type": "user", "userId": "U_e2e"},
        "timestamp": int(time.time() * 1000),
        "message": {"id": "m-1", "type": "text", "text": "slow prompt"},
    }
    # Drive the dispatch path. handle_message spawns _process_message_background
    # as a fire-and-forget task; await it explicitly so the slow handler +
    # _keep_typing + send() fully unwind before we assert.
    await adapter._dispatch_event(inbound_event)
    background_tasks = list(adapter._session_tasks.values())
    assert background_tasks, "handle_message should have spawned a background task"
    await asyncio.gather(*background_tasks, return_exceptions=True)

    # Button POST landed with Template Buttons + show_response postback.
    assert reply_route.called
    button_call = next(
        c for c in reply_route.calls
        if json.loads(c.request.content)["replyToken"] == "rt-original"
    )
    button_body = json.loads(button_call.request.content)
    template_msg = button_body["messages"][0]
    assert template_msg["type"] == "template"
    assert template_msg["template"]["type"] == "buttons"
    action = template_msg["template"]["actions"][0]
    payload = json.loads(action["data"])
    assert payload["action"] == "show_response"
    request_id = payload["request_id"]

    # send() (called by base after the slow handler returned) cached the
    # final response READY under the button's request_id and freed the slot.
    entry = adapter._cache.get(request_id)
    assert entry is not None
    assert entry.state is State.READY
    assert entry.payload == final_response
    assert "U_e2e" not in adapter._pending_buttons

    # Now simulate the user tapping the button.
    postback_event = {
        "type": "postback",
        "replyToken": "rt-postback-fresh",
        "source": {"type": "user", "userId": "U_e2e"},
        "timestamp": int(time.time() * 1000),
        "postback": {"data": json.dumps({"action": "show_response", "request_id": request_id})},
    }
    await adapter._dispatch_event(postback_event)

    # Postback delivered the cached payload via the fresh reply_token.
    postback_call = next(
        c for c in reply_route.calls
        if json.loads(c.request.content)["replyToken"] == "rt-postback-fresh"
    )
    postback_body = json.loads(postback_call.request.content)
    delivered_text = postback_body["messages"][0]["text"]
    assert delivered_text == final_response

    # Cache transitioned READY → DELIVERED (so a duplicate tap surfaces
    # "already replied" instead of re-delivering).
    entry = adapter._cache.get(request_id)
    assert entry.state is State.DELIVERED

    await adapter.disconnect()

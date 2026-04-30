"""LineAdapter send() routing + _keep_typing slow-LLM Quick Reply tests:

- ``send()`` smart routing (Reply API when token valid, Push fallback)
- ``send()`` pending-button cache (slow-LLM Quick Reply outcome)
- ``_keep_typing`` override (loading indicator + Quick Reply at threshold)
"""
import asyncio
import json
import time

import pytest
import respx
from httpx import Response

from gateway.platforms.line import LineAdapter, State
from tests.gateway.conftest import make_line_platform_config


def _adapter(monkeypatch, threshold: float = 45.0):
    """Build a LineAdapter for routing/keep-typing tests. Uses monkeypatch
    so env vars auto-restore (avoids cross-test pollution under xdist)."""
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
    monkeypatch.setenv("LINE_ALLOWED_USERS", "U1")
    monkeypatch.setenv("LINE_SLOW_RESPONSE_THRESHOLD", str(threshold))
    return LineAdapter(make_line_platform_config(token="t"))


# ---------------------------------------------------------------------------
# send() routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_send_uses_reply_api_when_token_valid(monkeypatch):
    """A cached, fresh reply_token must be consumed via the Reply API,
    avoiding Push API charges entirely."""
    reply_route = respx.post("https://api.line.me/v2/bot/message/reply").mock(
        return_value=Response(200, json={})
    )
    push_route = respx.post("https://api.line.me/v2/bot/message/push").mock(
        return_value=Response(200, json={})
    )
    adapter = _adapter(monkeypatch)
    adapter._reply_tokens["U1"] = ("rt-fresh", time.time() + 30)
    result = await adapter.send("U1", "hello")
    assert result.success is True
    assert reply_route.called
    assert not push_route.called
    # Reply token is single-use — must be consumed after success.
    assert "U1" not in adapter._reply_tokens


@pytest.mark.asyncio
@respx.mock
async def test_send_falls_back_to_push_when_token_expired(monkeypatch):
    """An expired reply_token must trigger Push API fallback so cron and
    operator-initiated deliveries still reach the user."""
    reply_route = respx.post("https://api.line.me/v2/bot/message/reply").mock(
        return_value=Response(200, json={})
    )
    push_route = respx.post("https://api.line.me/v2/bot/message/push").mock(
        return_value=Response(200, json={})
    )
    adapter = _adapter(monkeypatch)
    # Expired token (1s ago)
    adapter._reply_tokens["U1"] = ("rt-stale", time.time() - 1)
    result = await adapter.send("U1", "hello")
    assert result.success is True
    assert not reply_route.called
    assert push_route.called


@pytest.mark.asyncio
@respx.mock
async def test_send_falls_back_to_push_when_no_reply_token(monkeypatch):
    """No reply_token in cache (e.g. cron delivery) → Push API."""
    push_route = respx.post("https://api.line.me/v2/bot/message/push").mock(
        return_value=Response(200, json={})
    )
    adapter = _adapter(monkeypatch)
    result = await adapter.send("U_CRON_TARGET", "Cronjob Response: ...")
    assert result.success is True
    assert push_route.called


@pytest.mark.asyncio
@respx.mock
async def test_send_falls_back_to_push_when_reply_api_raises(monkeypatch):
    """If Reply API returns non-2xx (e.g. token already consumed by
    a race), send() must fall back to Push so the user still gets the
    message instead of a silent loss."""
    respx.post("https://api.line.me/v2/bot/message/reply").mock(
        return_value=Response(400, json={"message": "Invalid reply token"})
    )
    push_route = respx.post("https://api.line.me/v2/bot/message/push").mock(
        return_value=Response(200, json={})
    )
    adapter = _adapter(monkeypatch)
    adapter._reply_tokens["U1"] = ("rt-bad", time.time() + 30)
    result = await adapter.send("U1", "hello")
    assert result.success is True
    assert push_route.called


@pytest.mark.asyncio
@respx.mock
async def test_send_with_pending_button_caches_response_instead_of_push(monkeypatch):
    """When _keep_typing has already sent a slow-LLM Quick Reply button,
    send() must cache the response under the button's request_id (READY)
    and NOT invoke Push API — the user will fetch via postback tap."""
    push_route = respx.post("https://api.line.me/v2/bot/message/push").mock(
        return_value=Response(200, json={})
    )
    adapter = _adapter(monkeypatch)
    # Simulate _keep_typing having just sent a Quick Reply button.
    request_id = adapter._cache.register_pending()
    adapter._pending_buttons["U1"] = request_id

    result = await adapter.send("U1", "the slow answer")
    assert result.success is True
    assert result.message_id == request_id
    assert not push_route.called
    entry = adapter._cache.get(request_id)
    assert entry.state is State.READY
    assert entry.payload == "the slow answer"
    # Pending button slot is freed once cached so a second turn doesn't
    # accidentally inherit it.
    assert "U1" not in adapter._pending_buttons


@pytest.mark.asyncio
@respx.mock
async def test_send_does_not_cache_system_messages_into_pending_button(monkeypatch):
    """System messages from base/run.py (interrupt-ack, queue-ack, steer-ack)
    must NOT consume the pending-button slot — otherwise the orphan button
    would resolve to the wrong content and the user would never see the
    system message as a visible bubble.

    Regression for the cache-corruption bug observed during interrupt:
    busy-ack ('⚡ Interrupting current task...') was being silently swallowed
    into the pending-button cache, leaving the user with no feedback."""
    reply_route = respx.post("https://api.line.me/v2/bot/message/reply").mock(
        return_value=Response(200, json={})
    )
    push_route = respx.post("https://api.line.me/v2/bot/message/push").mock(
        return_value=Response(200, json={})
    )
    adapter = _adapter(monkeypatch)
    # Simulate _keep_typing having armed a button.
    request_id = adapter._cache.register_pending()
    adapter._pending_buttons["U1"] = request_id
    # AND a fresh reply_token from the interrupting message's webhook.
    adapter._reply_tokens["U1"] = ("rt-new", time.time() + 30)

    result = await adapter.send(
        "U1", "⚡ Interrupting current task. I'll respond shortly."
    )

    assert result.success is True
    # Routed via Reply API (the interrupting message's fresh token), NOT
    # cached into the orphan slot.
    assert reply_route.called
    assert not push_route.called
    # Cache entry untouched — still PENDING, waiting for the actual agent
    # response (or the orphan-cleanup in _keep_typing's finally block).
    entry = adapter._cache.get(request_id)
    assert entry.state is State.PENDING
    # Pending-button slot also untouched.
    assert adapter._pending_buttons["U1"] == request_id


@pytest.mark.asyncio
@respx.mock
async def test_send_extracts_markdown_image_to_native_bubble(monkeypatch):
    """Reply API delivery must extract ``![alt](https://...)`` markdown
    into a native LINE image message instead of sending raw text — same
    as v1's _build_reply_messages behavior."""
    reply_route = respx.post("https://api.line.me/v2/bot/message/reply").mock(
        return_value=Response(200, json={})
    )
    adapter = _adapter(monkeypatch)
    adapter._reply_tokens["U1"] = ("rt-fresh", time.time() + 30)
    await adapter.send(
        "U1",
        "Here is the chart:\n![chart](https://example.com/chart.png)\nDone.",
    )
    sent = json.loads(reply_route.calls.last.request.content)
    image_msgs = [m for m in sent["messages"] if m.get("type") == "image"]
    assert len(image_msgs) == 1
    assert image_msgs[0]["originalContentUrl"] == "https://example.com/chart.png"


# ---------------------------------------------------------------------------
# _keep_typing override — slow-LLM Quick Reply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_keep_typing_returns_quickly_when_agent_finishes_under_threshold(monkeypatch):
    """If stop_event fires before the slow-response threshold elapses,
    _keep_typing must NOT send a Quick Reply button — the response will
    be auto-delivered via reply_token in the normal send() path."""
    reply_route = respx.post("https://api.line.me/v2/bot/message/reply").mock(
        return_value=Response(200, json={})
    )
    loading_route = respx.post("https://api.line.me/v2/bot/chat/loading/start").mock(
        return_value=Response(200, json={})
    )
    adapter = _adapter(monkeypatch, threshold=10.0)
    adapter._reply_tokens["U1"] = ("rt-fresh", time.time() + 30)

    stop_event = asyncio.Event()
    # Signal "agent finished" almost immediately.
    asyncio.get_event_loop().call_soon(stop_event.set)

    await adapter._keep_typing("U1", stop_event=stop_event)

    # No Quick Reply button sent — token still cached for send() to use.
    assert not reply_route.called
    assert "U1" in adapter._reply_tokens
    assert "U1" not in adapter._pending_buttons
    # Loading indicator was triggered (DM only).
    assert loading_route.called


@pytest.mark.asyncio
@respx.mock
async def test_keep_typing_sends_quick_reply_button_when_threshold_elapses(monkeypatch):
    """At the threshold mark with the agent still running, _keep_typing
    must consume the cached reply_token to send a Quick Reply postback
    button, register the pending-button slot, and free the reply_token."""
    reply_route = respx.post("https://api.line.me/v2/bot/message/reply").mock(
        return_value=Response(200, json={})
    )
    respx.post("https://api.line.me/v2/bot/chat/loading/start").mock(
        return_value=Response(200, json={})
    )
    adapter = _adapter(monkeypatch, threshold=0.05)  # 50ms threshold for fast test
    adapter._reply_tokens["U1"] = ("rt-fresh", time.time() + 30)

    stop_event = asyncio.Event()  # never set — simulates LLM still running

    async def _keep_typing_in_background():
        await adapter._keep_typing("U1", stop_event=stop_event)

    keep_task = asyncio.create_task(_keep_typing_in_background())
    # Poll respx until the postback button POST lands (or timeout).
    # Standard pattern in other adapter tests for awaiting async-spawned
    # side effects.
    deadline = time.time() + 2.0
    while not reply_route.called and time.time() < deadline:
        await asyncio.sleep(0.02)

    stop_event.set()
    await keep_task

    # Button POST landed with the right shape.
    assert reply_route.called
    sent = json.loads(reply_route.calls.last.request.content)
    # Template Buttons message — persistent postback (not Quick Reply chip).
    template_msg = sent["messages"][0]
    assert template_msg["type"] == "template"
    action = template_msg["template"]["actions"][0]
    payload = json.loads(action["data"])
    assert payload["action"] == "show_response"
    request_id = payload["request_id"]
    # Reply token consumed by the button.
    assert "U1" not in adapter._reply_tokens

    # _keep_typing finished without a send() — orphan cleanup transitioned
    # the cache PENDING → ERROR (with interrupted-text) and popped the
    # pending-button slot. Tapping the persistent button later will surface
    # the interrupted message instead of looping in "still thinking".
    assert "U1" not in adapter._pending_buttons
    entry = adapter._cache.get(request_id)
    assert entry is not None
    assert entry.state is State.ERROR
    assert "interrupted" in entry.payload.lower()


@pytest.mark.asyncio
@respx.mock
async def test_keep_typing_no_token_no_button(monkeypatch):
    """If no reply_token is cached for the chat (e.g. cron-triggered run),
    _keep_typing must NOT send a Quick Reply button — there's nothing
    to send it on, and Push API would defeat the purpose."""
    reply_route = respx.post("https://api.line.me/v2/bot/message/reply").mock(
        return_value=Response(200, json={})
    )
    respx.post("https://api.line.me/v2/bot/chat/loading/start").mock(
        return_value=Response(200, json={})
    )
    adapter = _adapter(monkeypatch, threshold=0.05)
    # No reply_token registered — simulates cron path or stale chat.
    stop_event = asyncio.Event()  # never set

    async def runner():
        await adapter._keep_typing("U_NO_TOKEN", stop_event=stop_event)

    task = asyncio.create_task(runner())
    await asyncio.sleep(0.15)  # past threshold
    stop_event.set()
    await task
    assert not reply_route.called
    assert "U_NO_TOKEN" not in adapter._pending_buttons


# ---------------------------------------------------------------------------
# Outbound-only mode (no LINE_CHANNEL_SECRET)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_outbound_only_mode_can_push_with_no_secret(monkeypatch):
    """LINE_CHANNEL_ACCESS_TOKEN alone (no secret) must allow Push API
    deliveries — needed for setups that send notifications but don't
    receive webhooks (cron-only, broadcast bots, etc.)."""
    push_route = respx.post("https://api.line.me/v2/bot/message/push").mock(
        return_value=Response(200, json={})
    )
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
    adapter = LineAdapter(make_line_platform_config(token="t"))
    result = await adapter.send("U_HOME", "scheduled notice")
    assert result.success is True
    assert push_route.called

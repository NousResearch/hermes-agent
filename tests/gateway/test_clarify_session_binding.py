"""Native callbacks retain prompt ownership instead of deriving authority from an ID (#87780)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools import clarify_gateway as cm


async def _telegram_reply(clarify_id, session_key):
    from tests.gateway.test_telegram_clarify_buttons import _make_adapter

    adapter = _make_adapter()
    adapter._bot.send_message.return_value.message_id = 1
    result = await adapter.send_clarify("42", "Pick", ["A", "B"], clarify_id, session_key)
    assert result.success
    query = SimpleNamespace(
        data=f"cl:{clarify_id}:1", from_user=SimpleNamespace(id=42, first_name="Owner"),
        message=SimpleNamespace(chat_id=42, chat=SimpleNamespace(type="private"),
                                message_thread_id=None, text="Pick"),
        answer=AsyncMock(), edit_message_text=AsyncMock(),
    )
    await adapter._handle_callback_query(SimpleNamespace(callback_query=query), None)


async def _discord_reply(clarify_id, session_key):
    from tests.gateway.test_discord_clarify_buttons import _make_adapter, _make_interaction

    adapter = _make_adapter(allowed_users={"42"})
    channel = MagicMock()
    channel.send = AsyncMock(return_value=SimpleNamespace(id=1))
    adapter._client.get_channel.return_value = channel
    result = await adapter.send_clarify("42", "Pick", ["A", "B"], clarify_id, session_key)
    assert result.success
    view = channel.send.call_args.kwargs["view"]
    await view.children[1].callback(_make_interaction())


async def _slack_reply(clarify_id, session_key):
    from tests.gateway.test_slack_clarify_buttons import _make_adapter, _attach_auth_runner

    adapter = _make_adapter()
    _attach_auth_runner(adapter)
    client = adapter._team_clients["T1"]
    client.chat_postMessage = AsyncMock(return_value={"ts": "1.2"})
    client.chat_update = AsyncMock()
    result = await adapter.send_clarify("C1", "Pick", ["A", "B"], clarify_id, session_key)
    assert result.success
    body = {"message": {"ts": "1.2", "blocks": []}, "channel": {"id": "C1"},
            "user": {"name": "Owner", "id": "42"}}
    # A valid local message cannot lend its one-shot guard to an unrelated ID/channel.
    foreign = cm.register("foreign-id", "foreign-session", "Foreign", ["A", "B"])
    await adapter._handle_clarify_action(AsyncMock(), body, {
        "action_id": "hermes_clarify_choice_1", "value": "foreign-id|1"})
    action = {"action_id": "hermes_clarify_choice_1", "value": f"{clarify_id}|1"}
    await adapter._handle_clarify_action(AsyncMock(), {**body, "channel": {"id": "C2"}}, action)
    assert not foreign.event.is_set()
    assert adapter._clarify_resolved["1.2"] is False
    client.chat_update.assert_not_awaited()
    await adapter._handle_clarify_action(AsyncMock(), body, action)


async def _whatsapp_reply(clarify_id, session_key):
    from tests.gateway.test_whatsapp_cloud import _make_adapter

    adapter = _make_adapter()
    adapter._clarify_state[clarify_id] = session_key
    raw = {"from": "42", "type": "interactive", "interactive": {
        "type": "button_reply", "button_reply": {"id": f"cl:{clarify_id}:1", "title": "B"}}}
    await adapter._dispatch_interactive_reply(raw, {})


async def _relay_reply(clarify_id, session_key):
    from tests.gateway.relay.test_relay_interactive import _adapter, _event

    adapter, stub = _adapter()
    result = await adapter.send_clarify("c1", "Pick", ["A", "B"], clarify_id, session_key)
    assert result.success
    assert await adapter._consume_prompt_response(_event({
        "prompt_id": stub.sent[-1]["prompt_id"], "option_id": "c1"})) is True
    # Drain the real best-effort acknowledgements before leaving the test's loop.
    import asyncio
    if adapter._lifecycle_ack_tasks:
        await asyncio.gather(*adapter._lifecycle_ack_tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [_telegram_reply, _discord_reply, _slack_reply,
                                  _whatsapp_reply, _relay_reply])
async def test_native_callbacks_deliver_only_for_the_retained_session(reply, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()
    try:
        for index, profile in enumerate(("alpha", "beta", "alpha")):
            session_key = f"agent:{profile}:telegram:dm:42"
            clarify_id = f"owned-{index}"
            # A stale/mismatched prompt's stored key cannot authorize a different owner.
            foreign = cm.register(clarify_id, "other-owner", "Pick", ["A", "B"])
            await reply(clarify_id, session_key)
            assert foreign.response is None and not foreign.event.is_set()
            cm.clear_session("other-owner")
            owner = cm.register(clarify_id, session_key, "Pick", ["A", "B"])
            await reply(clarify_id, session_key)
            assert owner.event.is_set() and owner.response == "B"
            assert cm.wait_for_response(clarify_id, timeout=0) == "B"
    finally:
        with cm._lock:
            cm._entries.clear()
            cm._session_index.clear()
            cm._notify_cbs.clear()

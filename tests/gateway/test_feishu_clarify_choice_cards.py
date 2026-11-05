"""Feishu clarify cards and the choice-picker card, driven through the real card-action callback.

The SDK is optional, so callback-response types are stubbed and outbound sends are captured.
"""

import asyncio
import importlib.util
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

if importlib.util.find_spec("lark_oapi") is None and "lark_oapi" not in sys.modules:
    _mod = MagicMock()
    for _name in ("lark_oapi", "lark_oapi.api.im.v1", "lark_oapi.event", "lark_oapi.event.callback_type"):
        sys.modules.setdefault(_name, _mod)

from gateway.config import PlatformConfig
import plugins.platforms.feishu.adapter as feishu_module
from plugins.platforms.feishu.adapter import FeishuAdapter
from tools import clarify_gateway as cg

CHAT = "oc_chat"


class _Resp:
    def __init__(self):
        self.card = None


class _Card:
    type = data = None


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(feishu_module, "P2CardActionTriggerResponse", _Resp)
    monkeypatch.setattr(feishu_module, "CallBackCard", _Card)
    with cg._lock:
        cg._entries.clear()
        cg._session_index.clear()
    a = FeishuAdapter(PlatformConfig(enabled=True))
    a._client = MagicMock()
    a._allowed_group_users = {"ou_ok"}
    a._feishu_send_with_retry = AsyncMock(
        return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_card")))
    yield a
    with cg._lock:
        cg._entries.clear()
        cg._session_index.clear()


def _click(a, value, open_id="ou_ok"):
    data = SimpleNamespace(event=SimpleNamespace(
        token="tok", context=SimpleNamespace(open_chat_id=CHAT),
        operator=SimpleNamespace(open_id=open_id, user_id="", union_id=""),
        action=SimpleNamespace(tag="button", value=value)))
    response = a._on_card_action_trigger(data)
    return response.card.data if response is not None and response.card is not None else None


async def _ask(a, clarify_id="c1"):
    cg.register(clarify_id, "sess", "Pick a color", ["red", "blue"])
    result = await a.send_clarify(CHAT, "Pick a color", ["red", "blue"], clarify_id, "sess")
    assert result.success
    card = json.loads(a._feishu_send_with_retry.call_args.kwargs["payload"])
    return [b["value"] for b in card["elements"][1]["actions"]]


@pytest.mark.asyncio
async def test_clarify_tap_answers_once_and_only_for_an_authorized_operator(adapter):
    adapter._loop = asyncio.get_running_loop()
    values = await _ask(adapter)

    assert _click(adapter, values[1], open_id="ou_stranger") is None
    assert cg.get_pending("c1") is not None  # refused click left the question open

    settled = _click(adapter, values[1])
    assert "blue" in settled["elements"][0]["content"] and "action" not in json.dumps(settled["elements"])
    entry = cg._entries["c1"]
    assert entry.event.is_set() and entry.response == "blue"

    assert _click(adapter, values[0]) is None  # stale tap after resolution
    assert entry.response == "blue"


@pytest.mark.asyncio
async def test_other_tap_waits_for_text_and_retire_settles_the_card(adapter):
    adapter._loop = asyncio.get_running_loop()
    values = await _ask(adapter)

    _click(adapter, values[-1])
    assert cg._entries["c1"].awaiting_text and not cg._entries["c1"].event.is_set()

    await adapter.retire_clarify_card("c1", "✅ answered: green")
    request = adapter._client.im.v1.message.patch.call_args.args[0]
    assert request.message_id == "om_card"
    patched = json.loads(request.request_body.content)
    assert patched["config"]["update_multi"] is True
    assert "green" in patched["elements"][0]["content"] and len(patched["elements"]) == 1
    assert _click(adapter, values[0]) is None  # the retired card no longer answers


@pytest.mark.asyncio
async def test_choice_picker_applies_the_selection_once_for_an_authorized_operator(adapter):
    adapter._loop = asyncio.get_running_loop()
    applied = []

    async def on_choice(chat_id, value):
        applied.append((chat_id, value))
        return f"set to {value}"

    adapter.send = AsyncMock()
    choices = [{"value": "low", "label": "Low", "is_current": True}, {"value": "high", "label": "High"}]
    assert (await adapter.send_choice_picker(CHAT, "🧠 **Reasoning**\n\nPick:", choices, "sess", on_choice)).success
    card = json.loads(adapter._feishu_send_with_retry.call_args.kwargs["payload"])
    values = [b["value"] for b in card["elements"][1]["actions"]]

    assert _click(adapter, values[1], open_id="ou_stranger") is None
    settled = _click(adapter, values[1])
    assert "High" in settled["elements"][0]["content"]
    assert _click(adapter, values[0]) is None
    for _ in range(3):
        await asyncio.sleep(0)
    assert applied == [(CHAT, "high")]
    assert adapter.send.await_args.args == (CHAT, "set to high")


class _Body:
    def __init__(self, body: bytes):
        self.body = body

    async def readexactly(self, size: int) -> bytes:
        if len(self.body) < size:
            raise asyncio.IncompleteReadError(self.body, size)
        return self.body[:size]


async def _webhook_tap(a, value, open_id="ou_ok"):
    body = json.dumps({"schema": "2.0", "header": {"event_type": "card.action.trigger"}, "event": {
        "operator": {"open_id": open_id}, "context": {"open_chat_id": CHAT},
        "action": {"tag": "button", "value": value}}}).encode()
    response = await a._handle_webhook_request(
        SimpleNamespace(remote="127.0.0.1", content_length=None, headers={}, content=_Body(body)))
    assert response.status == 200
    return json.loads(response.body)


@pytest.mark.asyncio
async def test_webhook_tap_answers_the_clarify_and_returns_the_settled_card(adapter):
    adapter._loop = asyncio.get_running_loop()
    values = await _ask(adapter)

    assert await _webhook_tap(adapter, values[1], open_id="ou_stranger") == {}
    assert cg.get_pending("c1") is not None

    reply = await _webhook_tap(adapter, values[1])
    assert reply["card"]["type"] == "raw" and "blue" in reply["card"]["data"]["elements"][0]["content"]
    assert cg._entries["c1"].response == "blue"


@pytest.mark.asyncio
async def test_multi_select_card_accepts_a_typed_free_form_answer(adapter):
    """The multi-select hint offers "your own answer"; typed prose must answer, not cancel."""
    cg.register("m1", "sess", "Which envs?", ["staging", "prod"], multi_select=True)
    assert (await adapter.send_clarify(CHAT, "Which envs?", ["staging", "prod"], "m1", "sess")).success
    assert cg.attempt_text_response_for_session("sess", "use canary instead") == cg.TEXT_RESOLVED
    assert cg._entries["m1"].response == "use canary instead"


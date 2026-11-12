"""A native approval card answers only the request it was rendered for.

Two approvals can share one chat session: the first card's request ends (timeout,
``/approve`` from another client) while a newer request is queued behind it. A press on
the first card must not resolve the newer request, which the user has not seen yet.
"""

import importlib.util
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.gateway.platforms.base import SendResult

if importlib.util.find_spec("lark_oapi") is None and "lark_oapi" not in sys.modules:
    _lark = MagicMock()
    for _name in ("lark_oapi", "lark_oapi.api.im.v1", "lark_oapi.event", "lark_oapi.event.callback_type"):
        sys.modules.setdefault(_name, _lark)

from hermes_agent.tools import approval
from hermes_agent.tools.approval_gateway_wait import _ApprovalEntry

SESSION = "agent:main:test:dm:1"


@pytest.fixture
def newer_request_pending():
    """Request ``req-old`` already settled; ``req-new`` is the one still waiting."""
    entry = _ApprovalEntry({"command": "rm -rf build", "request_id": "req-new"})
    with approval._lock:
        approval._gateway_queues[SESSION] = [entry]
    yield entry
    with approval._lock:
        approval._gateway_queues.pop(SESSION, None)


async def _press_telegram_card(request_id):
    from plugins.platforms.telegram.adapter import TelegramAdapter
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t"))
    adapter._bot, adapter._app = AsyncMock(), MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
    await adapter.send_exec_approval(chat_id="1", command="rm -rf build", session_key=SESSION, request_id=request_id)
    query = AsyncMock()
    query.data = f"ea:once:{next(iter(adapter._approval_state))}"
    query.message = SimpleNamespace(chat_id=1, chat=SimpleNamespace(type="private"))
    query.from_user = SimpleNamespace(id=1, first_name="Alice")
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}):
        await adapter._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock())


async def _press_slack_card(request_id):
    from plugins.platforms.slack.adapter import SlackAdapter
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-t"))
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ts": "1.2"})
    adapter._app, adapter._bot_user_id = MagicMock(), "U_BOT"
    adapter._team_clients, adapter._team_bot_user_ids, adapter._channel_team = {"T1": client}, {"T1": "U_BOT"}, {"C1": "T1"}

    async def _allow(_event):
        return None
    adapter.set_message_handler(_allow)
    adapter._is_interactive_user_authorized = lambda *_a, **_k: True
    await adapter.send_exec_approval(chat_id="C1", command="rm -rf build", session_key=SESSION, request_id=request_id)
    body = {"message": {"ts": "1.2", "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": "x"}}]},
            "channel": {"id": "C1"}, "user": {"name": "alice", "id": "U1"}}
    await adapter._handle_approval_action(AsyncMock(), body, {"action_id": "hermes_approve_once", "value": SESSION})


async def _press_feishu_card(request_id):
    from plugins.platforms.feishu.adapter import FeishuAdapter
    adapter = FeishuAdapter(PlatformConfig(enabled=True))
    adapter._client = MagicMock()
    adapter.set_authorization_check(lambda *_a, **_k: True)
    sent = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="m1"))
    with patch.object(adapter, "_feishu_send_with_retry", new=AsyncMock(return_value=sent)), \
            patch.object(adapter, "send", new=AsyncMock()):
        await adapter.send_exec_approval(chat_id="oc_1", command="rm -rf build", session_key=SESSION, request_id=request_id)
        await adapter._resolve_approval(next(iter(adapter._approval_state)), "once", "Alice", open_id="ou_1", chat_id="oc_1")


async def _press_matrix_card(request_id):
    from plugins.platforms.matrix.adapter import MatrixAdapter
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
    adapter._user_id, adapter._client = "@bot:example.org", MagicMock()
    adapter.set_authorization_check(lambda *_a, **_k: True)
    with patch.object(adapter, "send", new=AsyncMock(return_value=SendResult(success=True, message_id="$card"))), \
            patch.object(adapter, "_send_reaction", new=AsyncMock(return_value=None)):
        await adapter.send_exec_approval(chat_id="!room:x", command="rm -rf build", session_key=SESSION, request_id=request_id)
        await adapter._on_reaction(SimpleNamespace(
            sender="@alice:x", event_id="$react", room_id="!room:x",
            content={"m.relates_to": {"event_id": "$card", "key": "✅"}}))


async def _press_whatsapp_card(request_id):
    from plugins.platforms.whatsapp_cloud.adapter import WhatsAppCloudAdapter
    adapter = WhatsAppCloudAdapter(PlatformConfig(enabled=True, extra={"phone_number_id": "1", "access_token": "t"}))
    with patch.object(adapter, "_post_message_result", new=AsyncMock(return_value=SendResult(success=True, message_id="w1"))), \
            patch.object(adapter, "send", new=AsyncMock()):
        await adapter.send_exec_approval(chat_id="15551234567", command="rm -rf build", session_key=SESSION, request_id=request_id)
        await adapter._handle_approval_tap("15551234567", {}, ["appr", next(iter(adapter._exec_approval_state)), "approve"])
    return "approve"  # Approve / Deny only: WhatsApp's 3-button cap leaves no room for the tiers


async def _press_teams_card(request_id):
    from tests.gateway.test_teams import TeamsAdapter  # installs the Teams SDK stand-in the adapter renders with
    adapter = TeamsAdapter(PlatformConfig(enabled=True, extra={"client_id": "bot", "client_secret": "s", "tenant_id": "t"}))
    adapter._app, adapter._send_card = object(), AsyncMock()
    adapter._card_action_denied, adapter._invoke_card = (lambda _from: None), (lambda body: body)
    buttons = []
    with patch.object(sys.modules[TeamsAdapter.__module__], "ExecuteAction", lambda **kw: buttons.append(kw)):
        await adapter.send_exec_approval(chat_id="c1", command="rm -rf build", session_key=SESSION, request_id=request_id)
    ctx = MagicMock()
    ctx.activity.value.action.data = next(b["data"] for b in buttons if b["data"]["hermes_action"] == "approve_once")
    await adapter._on_card_action(ctx)


async def _press_relay_card(request_id):
    from hermes_agent.gateway.platforms.event import MessageEvent, MessageType
    from hermes_agent.gateway.relay.adapter import RelayAdapter
    from hermes_agent.gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
    from hermes_agent.gateway.session import SessionSource
    from tests.gateway.relay.stub_connector import StubConnector
    desc = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION, platform="telegram", label="Telegram", max_message_length=4096,
        supports_draft_streaming=False, supports_edit=True, supports_threads=True, markdown_dialect="markdown_v2",
        len_unit="utf16", supported_ops=("send", "edit", "typing", "get_chat_info", "send_media", "prompt", "react"))
    stub = StubConnector(desc)
    adapter = RelayAdapter(PlatformConfig(), desc, transport=stub)
    await adapter.send_exec_approval("c1", "rm -rf build", SESSION, request_id=request_id)
    await adapter._consume_prompt_response(MessageEvent(
        text="/once", message_type=MessageType.COMMAND,
        source=SessionSource(platform="telegram", chat_id="c1", chat_type="dm", user_id="u1"),
        prompt_response={"prompt_id": stub.sent[-1]["prompt_id"], "option_id": "once"}))


_PRESSES = [_press_telegram_card, _press_slack_card, _press_feishu_card, _press_matrix_card, _press_whatsapp_card,
            _press_teams_card, _press_relay_card]


@pytest.mark.asyncio
@pytest.mark.parametrize("press", _PRESSES)
async def test_settled_card_does_not_resolve_the_next_request(newer_request_pending, press):
    await press("req-old")

    assert not newer_request_pending.event.is_set()
    assert approval._gateway_queues[SESSION] == [newer_request_pending]


@pytest.mark.asyncio
@pytest.mark.parametrize("press", _PRESSES)
async def test_card_resolves_its_own_request(newer_request_pending, press):
    pressed = await press("req-new") or "once"

    assert newer_request_pending.event.is_set()
    assert newer_request_pending.result == pressed


@pytest.mark.asyncio
async def test_matrix_settled_card_reports_expiry(newer_request_pending):
    """A reaction on a settled Matrix card says it expired and retires the card, as the other adapters do."""
    from agent.i18n import t
    from plugins.platforms.matrix.adapter import MatrixAdapter
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
    adapter._user_id, adapter._client = "@bot:example.org", MagicMock()
    adapter.set_authorization_check(lambda *_a, **_k: True)
    send = AsyncMock(return_value=SendResult(success=True, message_id="$card"))
    with patch.object(adapter, "send", new=send), patch.object(adapter, "_send_reaction", new=AsyncMock(return_value=None)):
        await adapter.send_exec_approval(chat_id="!room:x", command="rm -rf build", session_key=SESSION, request_id="req-old")
        await adapter._on_reaction(SimpleNamespace(
            sender="@alice:x", event_id="$react", room_id="!room:x",
            content={"m.relates_to": {"event_id": "$card", "key": "\u2705"}}))

    assert send.await_args_list[-1].args[:2] == ("!room:x", t("platform.matrix.approval.expired"))
    assert "$card" not in adapter._approval_prompts_by_event

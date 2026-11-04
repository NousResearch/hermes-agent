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

from gateway.config import PlatformConfig

if importlib.util.find_spec("lark_oapi") is None and "lark_oapi" not in sys.modules:
    _lark = MagicMock()
    for _name in ("lark_oapi", "lark_oapi.api.im.v1", "lark_oapi.event", "lark_oapi.event.callback_type"):
        sys.modules.setdefault(_name, _lark)

from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry

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


@pytest.mark.asyncio
@pytest.mark.parametrize("press", [_press_telegram_card, _press_slack_card, _press_feishu_card])
async def test_settled_card_does_not_resolve_the_next_request(newer_request_pending, press):
    await press("req-old")

    assert not newer_request_pending.event.is_set()
    assert approval._gateway_queues[SESSION] == [newer_request_pending]


@pytest.mark.asyncio
@pytest.mark.parametrize("press", [_press_telegram_card, _press_slack_card, _press_feishu_card])
async def test_card_resolves_its_own_request(newer_request_pending, press):
    await press("req-new")

    assert newer_request_pending.event.is_set()
    assert newer_request_pending.result == "once"

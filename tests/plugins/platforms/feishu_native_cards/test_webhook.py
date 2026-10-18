import asyncio
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock

from tests.plugins.platforms.feishu_native_cards.support import make_adapter
from tools import clarify_gateway as cg


class Body:
    def __init__(self, data):
        self.data = data

    async def readexactly(self, count):
        raise asyncio.IncompleteReadError(self.data, count)


def request(payload):
    raw = json.dumps(payload).encode()
    return NS(
        remote="127.0.0.1",
        headers={"Content-Type": "application/json"},
        content_length=len(raw),
        content=Body(raw),
    )


class WebhookTests(unittest.IsolatedAsyncioTestCase):
    async def test_authenticated_http_callback_returns_native_card_and_resolves_waiter(self):
        adapter, _hub, _ops = make_adapter()
        adapter._verification_token = "unit-verification"
        entry = cg.register("http", "http_session", "question", None)
        try:
            await adapter.send_clarify(
                "chat_a",
                "question",
                None,
                "http",
                "http_session",
                {"__hermes_prompt_user_ids": ("owner_a",)},
            )
            task = adapter._clarify_tasks["http"]
            payload = {
                "schema": "2.0",
                "header": {
                    "event_id": "http-event",
                    "event_type": "card.action.trigger",
                    "token": "unit-verification",
                },
                "event": {
                    "operator": {"open_id": "owner_a"},
                    "context": {"open_chat_id": "chat_a", "open_message_id": "im_test"},
                    "action": {
                        "value": {"hermes_clarify_action": "text_submit", "clarify_id": "http"},
                        "form_value": {"answer": "HTTP 原生回答"},
                    },
                },
            }
            result = await adapter._handle_webhook_request(request(payload))
            self.assertEqual(result.status, 200)
            self.assertIn("card", json.loads(result.text))
            self.assertEqual(entry.response, "HTTP 原生回答")
            await task
        finally:
            cg.clear_session("http_session")

    async def test_http_invalid_token_and_signature_never_dispatch(self):
        adapter, _hub, _ops = make_adapter()
        adapter._verification_token = "unit-verification"
        dispatch = adapter._build_event_handler
        adapter._build_event_handler = lambda: self.fail("未认证事件不能调用 SDK 分发器")
        payload = {
            "schema": "2.0",
            "header": {"event_type": "card.action.trigger", "token": "invalid"},
        }
        result = await adapter._handle_webhook_request(request(payload))
        self.assertEqual(result.status, 401)
        payload["header"]["token"] = "unit-verification"
        adapter._encrypt_key = "unit-signing"
        result = await adapter._handle_webhook_request(request(payload))
        self.assertEqual(result.status, 401)
        adapter._build_event_handler = dispatch

    async def test_generic_callback_uses_event_identity_instead_of_reusable_token(self):
        adapter, _hub, _ops = make_adapter()
        adapter._dispatch_synthetic_event = AsyncMock()
        for ident in ("event1", "event2", "event1"):
            await adapter._handle_card_action_event(
                NS(
                    header=NS(event_id=ident),
                    event=NS(
                        token="same-update-token",
                        operator=NS(open_id="owner"),
                        context=NS(open_chat_id="chat"),
                        action=NS(value={"custom": "test"}, tag="button"),
                    ),
                )
            )
        self.assertEqual(adapter._dispatch_synthetic_event.await_count, 2)
        self.assertEqual(
            [
                call.kwargs["message_id"]
                for call in adapter._dispatch_synthetic_event.await_args_list
            ],
            ["event1", "event2"],
        )

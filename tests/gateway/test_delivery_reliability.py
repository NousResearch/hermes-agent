"""Delivery reliability regression tests for gateway final replies."""

import asyncio
from types import SimpleNamespace
import unittest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource


class _SlowFinalSendAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.FEISHU)
        self.started = asyncio.Event()
        self.sent = asyncio.Event()
        self.metadata = None

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id: str):
        return {}

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None):
        return SendResult(True, message_id="om_visible")

    async def send_typing(self, chat_id: str, metadata=None):
        return None

    async def _send_with_retry(
        self,
        chat_id: str,
        content: str,
        reply_to=None,
        metadata=None,
        max_retries: int = 2,
        base_delay: float = 2.0,
    ):
        self.started.set()
        await asyncio.sleep(0.05)
        self.metadata = metadata
        self.sent.set()
        return SendResult(True, message_id="om_visible")


class _HangingFinalSendAdapter(_SlowFinalSendAdapter):
    async def _send_with_retry(
        self,
        chat_id: str,
        content: str,
        reply_to=None,
        metadata=None,
        max_retries: int = 2,
        base_delay: float = 2.0,
    ):
        self.started.set()
        self.metadata = metadata
        await asyncio.Event().wait()


class TestFinalResponseDeliveryReliability(unittest.IsolatedAsyncioTestCase):
    async def test_final_response_send_drains_after_cancellation_and_reraises(self):
        adapter = _SlowFinalSendAdapter()

        async def handler(event):
            return "final response"

        adapter.set_message_handler(handler)
        event = MessageEvent(
            text="hello",
            source=SessionSource(
                platform=Platform.FEISHU,
                chat_id="oc_chat",
                chat_type="group",
                user_id="ou_user",
                message_id="om_inbound",
            ),
            message_id="om_inbound",
        )

        task = asyncio.create_task(adapter._process_message_background(event, "feishu:oc_chat"))
        await adapter.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(adapter.sent.is_set())
        metadata = adapter.metadata
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertTrue(metadata.get("notify"))
        self.assertTrue(metadata.get("delivery_uuid"))

    async def test_cancelled_final_response_send_timeout_preserves_cancellation(self):
        adapter = _HangingFinalSendAdapter()
        adapter.final_send_cancel_drain_seconds = 0.01

        async def handler(event):
            return "final response"

        adapter.set_message_handler(handler)
        event = MessageEvent(
            text="hello",
            source=SessionSource(platform=Platform.FEISHU, chat_id="oc_chat", chat_type="group"),
            message_id="om_inbound",
        )

        task = asyncio.create_task(adapter._process_message_background(event, "feishu:oc_chat"))
        await adapter.started.wait()
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(task.cancelled())


class TestGatewayDeliveryConfig(unittest.TestCase):
    def test_final_send_cancel_drain_seconds_loads_from_nested_gateway_config(self):
        from gateway.config import GatewayConfig

        config = GatewayConfig.from_dict({"gateway": {"final_send_cancel_drain_seconds": 2.5}})

        self.assertEqual(config.final_send_cancel_drain_seconds, 2.5)

    def test_final_send_cancel_drain_seconds_rejects_negative_values(self):
        from gateway.config import GatewayConfig

        config = GatewayConfig.from_dict({"gateway": {"final_send_cancel_drain_seconds": -1}})

        self.assertEqual(config.final_send_cancel_drain_seconds, 5.0)


class TestFeishuDeliveryUuid(unittest.IsolatedAsyncioTestCase):
    async def test_send_reuses_stable_delivery_uuid_for_retry(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = object.__new__(FeishuAdapter)
        adapter._client = object()
        object.__setattr__(adapter, "MAX_MESSAGE_LENGTH", 4000)
        object.__setattr__(adapter, "format_message", lambda content: content)
        object.__setattr__(adapter, "truncate_message", lambda content, *args, **kwargs: [content])
        object.__setattr__(adapter, "_build_outbound_payload", lambda content: ("text", '{"text":"hello"}'))
        object.__setattr__(adapter, "_response_succeeded", lambda response: True)
        object.__setattr__(adapter, "_extract_response_field", lambda response, field_name: "om_visible")
        adapter._finalize_send_result = FeishuAdapter._finalize_send_result.__get__(adapter, FeishuAdapter)
        seen = []

        async def fake_send(**kwargs):
            seen.append(kwargs["metadata"]["delivery_uuid"])
            return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_visible"))

        adapter._feishu_send_with_retry = fake_send
        result = await adapter.send("oc_chat", "hello", metadata={"delivery_uuid": "stable-uuid"})

        self.assertTrue(result.success)
        self.assertEqual(seen, ["stable-uuid"])

    async def test_send_suffixes_delivery_uuid_for_split_chunks(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = object.__new__(FeishuAdapter)
        adapter._client = object()
        object.__setattr__(adapter, "MAX_MESSAGE_LENGTH", 5)
        object.__setattr__(adapter, "format_message", lambda content: content)
        object.__setattr__(adapter, "truncate_message", lambda content, *args, **kwargs: ["one", "two"])
        object.__setattr__(adapter, "_build_outbound_payload", lambda content: ("text", f'{{"text":"{content}"}}'))
        object.__setattr__(adapter, "_response_succeeded", lambda response: True)
        object.__setattr__(adapter, "_extract_response_field", lambda response, field_name: "om_visible")
        adapter._finalize_send_result = FeishuAdapter._finalize_send_result.__get__(adapter, FeishuAdapter)
        seen = []

        async def fake_send(**kwargs):
            seen.append(kwargs["metadata"]["delivery_uuid"])
            return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_visible"))

        adapter._feishu_send_with_retry = fake_send
        result = await adapter.send("oc_chat", "one two", metadata={"delivery_uuid": "stable-uuid"})

        self.assertTrue(result.success)
        self.assertEqual(seen, ["stable-uuid-0", "stable-uuid-1"])

    async def test_success_without_message_id_logs_warning(self):
        adapter = _SlowFinalSendAdapter()

        async def handler(event):
            return "final response"

        async def no_id_send(*args, **kwargs):
            return SendResult(True, message_id=None)

        adapter._send_with_retry = no_id_send
        adapter.set_message_handler(handler)
        event = MessageEvent(
            text="hello",
            source=SessionSource(platform=Platform.FEISHU, chat_id="oc_chat", chat_type="group"),
            message_id="om_inbound",
        )

        with self.assertLogs("gateway.platforms.base", level="WARNING") as logs:
            await adapter._process_message_background(event, "feishu:oc_chat")

        self.assertTrue(any("without message_id" in line for line in logs.output))

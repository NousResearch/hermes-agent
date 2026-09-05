import asyncio
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


class _InlineKeyboardButton:
    def __init__(self, text, callback_data):
        self.text = text
        self.callback_data = callback_data


class _InlineKeyboardMarkup:
    def __init__(self, rows):
        self.inline_keyboard = rows


adapter_stub = types.ModuleType("plugins.platforms.telegram.adapter")
adapter_stub.InlineKeyboardButton = _InlineKeyboardButton
adapter_stub.InlineKeyboardMarkup = _InlineKeyboardMarkup
adapter_stub.normalize_telegram_chat_id = lambda chat_id: int(chat_id)
adapter_stub.register = lambda registry: None
sys.modules.setdefault("plugins.platforms.telegram.adapter", adapter_stub)

from plugins.platforms.telegram.paperclip_bridge import (
    CALLBACK_PREFIX,
    dispatch_text_event,
    is_enabled,
    maybe_handle_callback,
    maybe_handle_command,
)


class FakeEvent:
    def __init__(self, text):
        self.text = text
        self.user_id = "123"
        self.user_name = "craig"
        self.message_id = "42"
        self.source = SimpleNamespace(chat_id="-100", chat_type="group", thread_id="91")
        self.metadata = {"thread_id": "91", "paperclip_company_id": "company-1", "paperclip_project_id": "project-9"}

    def get_command(self):
        return self.text.split()[0][1:]

    def get_command_args(self):
        parts = self.text.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""


class FakeAdapter:
    def __init__(self):
        self.config = SimpleNamespace(extra={
            "paperclip_bridge_url": "http://127.0.0.1:8787",
            "paperclip_bridge_bearer_token": "bridge-token",
            "paperclip_bridge_signing_secret": "bridge-secret",
            "paperclip_bridge_enabled": True,
        })
        self._reply_to_mode = None
        self._bot = AsyncMock()
        self._send_message_with_thread_fallback = AsyncMock()
        self.handle_message = AsyncMock()
        self._link_preview_kwargs = MagicMock(return_value={})
        self._thread_kwargs_for_send = MagicMock(return_value={"message_thread_id": 91})
        self._is_callback_user_authorized = MagicMock(return_value=True)


class TelegramPaperclipBridgeTests(unittest.TestCase):
    def test_log_command_calls_bridge_and_sends_buttons(self):
        adapter = FakeAdapter()
        event = FakeEvent("/log chase the Wazuh alert backlog")
        bridge_payload = {
            "status": "pending_approval",
            "approval_token": "ghlog_testtoken",
            "reply": {
                "text": "Log captured. Approve to create the Paperclip issue.",
                "buttons": [
                    {"text": "Approve", "action": "approve", "target": "ghlog_testtoken"},
                    {"text": "Reject", "action": "reject", "target": "ghlog_testtoken"},
                ],
            },
        }

        with patch("plugins.platforms.telegram.paperclip_bridge.post_json", return_value=(202, bridge_payload)):
            handled = asyncio.run(maybe_handle_command(adapter, event))

        self.assertTrue(handled)
        adapter._send_message_with_thread_fallback.assert_awaited_once()
        kwargs = adapter._send_message_with_thread_fallback.call_args.kwargs
        self.assertEqual(kwargs["chat_id"], -100)
        self.assertEqual(kwargs["text"], "Log captured. Approve to create the Paperclip issue.")
        buttons = kwargs["reply_markup"].inline_keyboard
        self.assertEqual(buttons[0][0].callback_data, f"{CALLBACK_PREFIX}approve:ghlog_testtoken")
        self.assertEqual(buttons[0][1].callback_data, f"{CALLBACK_PREFIX}reject:ghlog_testtoken")

    def test_natural_language_log_calls_bridge_and_preserves_task_text(self):
        adapter = FakeAdapter()
        event = FakeEvent("Please add this to Paperclip: draft a care-plan renewal post")
        bridge_payload = {
            "status": "pending_approval",
            "approval_token": "ghlog_natural",
            "reply": {
                "text": "Log captured. Approve to create the Paperclip issue.",
                "buttons": [
                    {"text": "Approve", "action": "approve", "target": "ghlog_natural"},
                    {"text": "Reject", "action": "reject", "target": "ghlog_natural"},
                ],
            },
        }

        with patch("plugins.platforms.telegram.paperclip_bridge.post_json", return_value=(202, bridge_payload)) as post:
            handled = asyncio.run(maybe_handle_command(adapter, event))

        self.assertTrue(handled)
        body = post.call_args.args[1]
        self.assertEqual(body["command"], "log")
        self.assertEqual(body["message"]["command_text"], "draft a care-plan renewal post")
        adapter._send_message_with_thread_fallback.assert_awaited_once()

    def test_log_this_in_paperclip_is_accepted_as_natural_language(self):
        adapter = FakeAdapter()
        event = FakeEvent("Log this in Paperclip: prepare the September content queue")
        bridge_payload = {"status": "pending_approval", "reply": {"text": "Pending", "buttons": []}}

        with patch("plugins.platforms.telegram.paperclip_bridge.post_json", return_value=(202, bridge_payload)) as post:
            handled = asyncio.run(maybe_handle_command(adapter, event))

        self.assertTrue(handled)
        self.assertEqual(post.call_args.args[1]["message"]["command_text"], "prepare the September content queue")

    def test_unrelated_natural_language_is_not_intercepted(self):
        adapter = FakeAdapter()
        event = FakeEvent("Tell me what Paperclip is doing")

        with patch("plugins.platforms.telegram.paperclip_bridge.post_json") as post:
            handled = asyncio.run(maybe_handle_command(adapter, event))

        self.assertFalse(handled)
        post.assert_not_called()
        adapter._send_message_with_thread_fallback.assert_not_awaited()

    def test_dispatch_routes_natural_language_without_agent_fallback(self):
        adapter = FakeAdapter()
        event = FakeEvent("Log this: prepare a five-post queue")
        bridge_payload = {
            "status": "pending_approval",
            "reply": {
                "text": "Pending",
                "buttons": [{"text": "Approve", "action": "approve", "target": "ghlog_dispatch"}],
            },
        }

        with patch("plugins.platforms.telegram.paperclip_bridge.post_json", return_value=(202, bridge_payload)):
            asyncio.run(dispatch_text_event(adapter, event))

        adapter.handle_message.assert_not_awaited()
        adapter._send_message_with_thread_fallback.assert_awaited_once()

    def test_native_approve_reaches_normal_hermes_dispatch(self):
        adapter = FakeAdapter()
        event = FakeEvent("/approve session")

        with patch("plugins.platforms.telegram.paperclip_bridge.post_json") as post:
            asyncio.run(dispatch_text_event(adapter, event))

        post.assert_not_called()
        adapter.handle_message.assert_awaited_once_with(event)

    def test_bridge_failure_is_consumed_and_raw_error_is_not_relayed(self):
        adapter = FakeAdapter()
        event = FakeEvent("Log this: draft a care-plan post")

        with patch(
            "plugins.platforms.telegram.paperclip_bridge.post_json",
            return_value=(500, {"error": "internal-token-value"}),
        ):
            handled = asyncio.run(maybe_handle_command(adapter, event))

        self.assertTrue(handled)
        sent_text = adapter._send_message_with_thread_fallback.call_args.kwargs["text"]
        self.assertNotIn("internal-token-value", sent_text)
        adapter.handle_message.assert_not_awaited()

    def test_bridge_credentials_can_come_from_process_environment(self):
        adapter = FakeAdapter()
        del adapter.config.extra["paperclip_bridge_bearer_token"]
        del adapter.config.extra["paperclip_bridge_signing_secret"]
        with patch.dict(
            os.environ,
            {
                "HERMES_PAPERCLIP_BRIDGE_BEARER_TOKEN": "env-token",
                "HERMES_PAPERCLIP_BRIDGE_SIGNING_SECRET": "env-secret",
            },
            clear=False,
        ):
            self.assertTrue(is_enabled(adapter))

    def test_insecure_remote_bridge_url_disables_integration(self):
        adapter = FakeAdapter()
        adapter.config.extra["paperclip_bridge_url"] = "http://bridge.example.test"
        self.assertFalse(is_enabled(adapter))

    def test_unknown_callback_action_is_rejected_without_bridge_call(self):
        adapter = FakeAdapter()
        query = AsyncMock()
        query.data = f"{CALLBACK_PREFIX}unexpected:ghlog_testtoken"
        query.from_user = MagicMock(id="123")

        with patch("plugins.platforms.telegram.paperclip_bridge.post_json") as post:
            handled = asyncio.run(
                maybe_handle_callback(
                    adapter,
                    query,
                    query.data,
                    query_chat_id=-100,
                    query_chat_type="group",
                    query_thread_id=91,
                    query_user_name="Craig",
                )
            )

        self.assertTrue(handled)
        post.assert_not_called()
        query.answer.assert_awaited_once_with(text="Invalid Paperclip action.")

    def test_reject_callback_acknowledges_rejection_and_uses_stable_key(self):
        adapter = FakeAdapter()
        payload = {"status": "rejected", "reply": {"text": "Rejected.", "edit_origin": True}}
        keys = []

        def fake_post(_url, _body, headers):
            keys.append(headers["X-Idempotency-Key"])
            return 200, payload

        for callback_id in ("cb-1", "cb-2"):
            query = AsyncMock()
            query.id = callback_id
            query.data = f"{CALLBACK_PREFIX}reject:ghlog_testtoken"
            query.message = MagicMock(chat_id=-100, message_id=55, message_thread_id=91)
            query.from_user = MagicMock(id="123", username="craig", first_name="Craig")
            with patch("plugins.platforms.telegram.paperclip_bridge.post_json", side_effect=fake_post):
                asyncio.run(
                    maybe_handle_callback(
                        adapter,
                        query,
                        query.data,
                        query_chat_id=-100,
                        query_chat_type="group",
                        query_thread_id=91,
                        query_user_name="Craig",
                    )
                )
            query.answer.assert_awaited_once_with(text="Rejected")

        self.assertEqual(keys[0], keys[1])

    def test_log_a_job_to_phrase_is_natural_intake(self):
        adapter = FakeAdapter()
        event = FakeEvent("log a job to get the data you need")
        bridge_payload = {
            "status": "pending_approval",
            "reply": {
                "text": "Pending",
                "buttons": [{"text": "Approve", "action": "approve", "target": "ghlog_jobphrase"}],
            },
        }

        with patch("plugins.platforms.telegram.paperclip_bridge.post_json", return_value=(202, bridge_payload)) as post:
            handled = asyncio.run(maybe_handle_command(adapter, event))

        self.assertTrue(handled)
        self.assertEqual(post.call_args.args[1]["message"]["command_text"], "get the data you need")

    def test_approve_callback_calls_bridge_and_edits_message(self):
        adapter = FakeAdapter()
        query = AsyncMock()
        query.id = "cb-1"
        query.data = f"{CALLBACK_PREFIX}approve:ghlog_testtoken"
        query.message = MagicMock()
        query.message.chat_id = -100
        query.message.message_id = 55
        query.message.message_thread_id = 91
        query.from_user = MagicMock()
        query.from_user.id = "123"
        query.from_user.username = "craig"
        query.from_user.first_name = "Craig"

        bridge_payload = {
            "status": "approved",
            "reply": {
                "text": "Approved. Created Paperclip issue GH-142.",
                "edit_origin": True,
            },
        }

        with patch("plugins.platforms.telegram.paperclip_bridge.post_json", return_value=(200, bridge_payload)):
            handled = asyncio.run(
                maybe_handle_callback(
                    adapter,
                    query,
                    query.data,
                    query_chat_id=-100,
                    query_chat_type="group",
                    query_thread_id=91,
                    query_user_name="Craig",
                )
            )

        self.assertTrue(handled)
        query.answer.assert_awaited()
        query.edit_message_text.assert_awaited_once()
        self.assertIn("GH-142", query.edit_message_text.call_args.kwargs["text"])


if __name__ == "__main__":
    unittest.main()

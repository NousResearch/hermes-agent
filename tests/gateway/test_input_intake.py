import asyncio
import json
import logging
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from gateway.config import Platform
from gateway.input_intake import (
    ACK_TEXT,
    INPUT_CHAT_ID,
    INPUT_THREAD_ID,
    IntakeError,
    accept_event,
    append_record_locked,
    build_record,
    default_buffer_path,
    is_input_intake_event,
)
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


class InputIntakeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.profile = Path(self.tmp.name) / "profile"
        self.profile.mkdir()
        self.old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(self.profile)
        self.buffer = default_buffer_path(self.profile)

    def tearDown(self):
        if self.old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self.old_home
        self.tmp.cleanup()

    def source(self, *, chat_id=INPUT_CHAT_ID, thread_id=INPUT_THREAD_ID, chat_type="group", user_id="754488239"):
        return SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=str(chat_id),
            chat_type=chat_type,
            user_id=user_id,
            user_name="Private Name",
            thread_id=thread_id,
        )

    def raw_message(self, *, text="hello", caption=None, message_id="123", photo=None, document=None, voice=None, audio=None):
        return SimpleNamespace(
            text=text,
            caption=caption,
            message_id=message_id,
            from_user=SimpleNamespace(id=754488239, username="oleks"),
            photo=photo,
            document=document,
            voice=voice,
            audio=audio,
            video=None,
            sticker=None,
        )

    def event(self, *, text="hello", message_id="123", source=None, msg_type=MessageType.TEXT, raw=None):
        if source is None:
            source = self.source()
        if raw is None:
            raw = self.raw_message(text=text, message_id=message_id)
        return MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            raw_message=raw,
            message_id=str(message_id),
        )

    def records(self):
        if not self.buffer.exists():
            return []
        return [json.loads(line) for line in self.buffer.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_01_input_text_creates_exactly_one_jsonl_record(self):
        result = accept_event(self.event(text="hello"), source_session_key="agent:main:telegram:group:-1004298945366:2664")
        self.assertEqual(result.status, "accepted")
        rows = self.records()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "hello")
        self.assertEqual(rows[0]["platform"], "telegram")
        self.assertEqual(rows[0]["status"], "received")

    def test_02_accepted_write_sends_exactly_one_ack(self):
        adapter = self.adapter_with_send()
        event = self.event(text="hello", message_id="200")
        handled = asyncio.run(adapter._maybe_handle_input_intake(event))
        self.assertTrue(handled)
        self.assertEqual(len(adapter.sent), 1)
        self.assertEqual(adapter.sent[0]["content"], ACK_TEXT)

    def test_03_ack_uses_thread_2664(self):
        adapter = self.adapter_with_send()
        asyncio.run(adapter._maybe_handle_input_intake(self.event(message_id="201")))
        self.assertEqual(adapter.sent[0]["metadata"]["thread_id"], INPUT_THREAD_ID)
        self.assertEqual(adapter.sent[0]["metadata"]["message_thread_id"], INPUT_THREAD_ID)

    def test_04_ack_uses_plain_text_metadata(self):
        adapter = self.adapter_with_send()
        asyncio.run(adapter._maybe_handle_input_intake(self.event(message_id="202")))
        self.assertIs(adapter.sent[0]["metadata"].get("telegram_plain_text_only"), True)

    def test_05_write_failure_sends_no_ack(self):
        adapter = self.adapter_with_send()
        with mock.patch("gateway.input_intake.accept_event", side_effect=IntakeError("disk_full")):
            handled = asyncio.run(adapter._maybe_handle_input_intake(self.event(message_id="203")))
        self.assertTrue(handled)
        self.assertEqual(adapter.sent, [])

    def test_06_duplicate_message_creates_no_second_record(self):
        event = self.event(message_id="204")
        self.assertEqual(accept_event(event, source_session_key="k").status, "accepted")
        self.assertEqual(accept_event(event, source_session_key="k").status, "duplicate")
        self.assertEqual(len(self.records()), 1)

    def test_07_duplicate_message_sends_no_second_ack(self):
        adapter = self.adapter_with_send()
        event = self.event(message_id="205")
        asyncio.run(adapter._maybe_handle_input_intake(event))
        asyncio.run(adapter._maybe_handle_input_intake(event))
        self.assertEqual(len(adapter.sent), 1)

    def test_08_non_input_topic_continues_normal_dispatch(self):
        adapter = self.adapter_for_text_handler(auth=True)
        update = SimpleNamespace(update_id=1, message=self.handler_msg(text="hello", thread_id="2654"), effective_message=None)
        asyncio.run(adapter._handle_text_message(update, None))
        self.assertEqual(adapter.enqueued, 1)
        self.assertEqual(len(adapter.sent), 0)

    def test_09_slash_command_in_input_continues_command_path(self):
        adapter = self.adapter_for_command_handler(auth=True)
        update = SimpleNamespace(update_id=2, message=self.handler_msg(text="/status", thread_id=INPUT_THREAD_ID), effective_message=None)
        asyncio.run(adapter._handle_command(update, None))
        self.assertEqual(adapter.handled, 1)
        self.assertEqual(len(self.records()), 0)

    def test_10_unauthorized_telegram_message_never_reaches_intake_helper(self):
        adapter = self.adapter_for_text_handler(auth=False)
        with mock.patch("gateway.input_intake.accept_event") as accept_mock:
            update = SimpleNamespace(update_id=3, message=self.handler_msg(text="secret", thread_id=INPUT_THREAD_ID), effective_message=None)
            asyncio.run(adapter._handle_text_message(update, None))
        accept_mock.assert_not_called()
        self.assertEqual(adapter.enqueued, 0)

    def test_11_general_root_temp_topics_are_not_intake(self):
        for thread_id in (None, "1", "766", "2654"):
            with self.subTest(thread_id=thread_id):
                ev = self.event(source=self.source(thread_id=thread_id), message_id=f"x{thread_id}")
                self.assertFalse(is_input_intake_event(ev))

    def test_12_ukrainian_utf8_is_preserved_exactly(self):
        text = "Привіт, тест ✓ ї ґ є"
        accept_event(self.event(text=text, message_id="206"), source_session_key="k")
        self.assertEqual(self.records()[0]["text"], text)

    def test_13_accepted_input_does_not_invoke_agent_dispatch(self):
        adapter = self.adapter_for_text_handler(auth=True)
        update = SimpleNamespace(update_id=4, message=self.handler_msg(text="hello", thread_id=INPUT_THREAD_ID), effective_message=None)
        asyncio.run(adapter._handle_text_message(update, None))
        self.assertEqual(adapter.enqueued, 0)
        self.assertEqual(len(adapter.sent), 1)

    def test_14_attachment_only_message_does_not_download_binary(self):
        photo = [SimpleNamespace(width=10, height=20, file_size=30, get_file=mock.Mock(side_effect=AssertionError("download")))]
        raw = self.raw_message(text=None, caption=None, message_id="207", photo=photo)
        ev = self.event(text="", message_id="207", msg_type=MessageType.PHOTO, raw=raw)
        accept_event(ev, source_session_key="k")
        row = self.records()[0]
        self.assertEqual(row["content_type"], "attachment_only")
        self.assertEqual(row["attachment"]["kind"], "photo")
        photo[0].get_file.assert_not_called()

    def test_15_existing_valid_jsonl_remains_valid_after_append(self):
        first = build_record(self.event(text="first", message_id="208"), "k")
        self.buffer.parent.mkdir(parents=True)
        self.buffer.write_text(json.dumps(first, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        accept_event(self.event(text="second", message_id="209"), source_session_key="k")
        rows = self.records()
        self.assertEqual([r["message_id"] for r in rows], ["208", "209"])

    def test_16_malformed_existing_jsonl_fails_closed_without_append_or_ack(self):
        self.buffer.parent.mkdir(parents=True)
        self.buffer.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")
        adapter = self.adapter_with_send()
        handled = asyncio.run(adapter._maybe_handle_input_intake(self.event(text="secret", message_id="210")))
        self.assertTrue(handled)
        self.assertEqual(adapter.sent, [])
        self.assertEqual(self.buffer.read_text(encoding="utf-8"), '{"ok": true}\nnot-json\n')

    def test_17_ack_failure_preserves_record_and_prevents_replay_duplication(self):
        adapter = self.adapter_with_send(success=False)
        ev = self.event(message_id="211")
        asyncio.run(adapter._maybe_handle_input_intake(ev))
        self.assertEqual(len(self.records()), 1)
        asyncio.run(adapter._maybe_handle_input_intake(ev))
        self.assertEqual(len(self.records()), 1)
        self.assertEqual(len(adapter.sent), 1)

    def test_18_message_content_absent_from_sanitized_logs(self):
        self.buffer.parent.mkdir(parents=True)
        self.buffer.write_text('not-json\n', encoding="utf-8")
        adapter = self.adapter_with_send()
        secret = "VERY_PRIVATE_SECRET_TEXT"
        with self.assertLogs("plugins.platforms.telegram.adapter", level="ERROR") as logs:
            asyncio.run(adapter._maybe_handle_input_intake(self.event(text=secret, message_id="212")))
        joined = "\n".join(logs.output)
        self.assertNotIn(secret, joined)
        self.assertIn("write_failed", joined)

    def adapter_with_send(self, *, success=True):
        adapter = object.__new__(TelegramAdapter)
        adapter._name = "telegram"
        adapter.platform = Platform.TELEGRAM
        adapter.config = SimpleNamespace(extra={"group_sessions_per_user": True, "thread_sessions_per_user": False})
        adapter.sent = []

        async def send(chat_id, content, metadata=None, **kwargs):
            adapter.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata or {}, "kwargs": kwargs})
            return SendResult(success=success, message_id="999" if success else None, error_kind=None if success else "unknown")

        adapter.send = send
        return adapter

    def handler_msg(self, *, text, thread_id):
        return SimpleNamespace(
            text=text,
            caption=None,
            message_id="300",
            date=datetime.now(timezone.utc),
            message_thread_id=thread_id,
            is_topic_message=thread_id is not None,
            chat=SimpleNamespace(id=int(INPUT_CHAT_ID), type="supergroup", is_forum=True, title="group"),
            from_user=SimpleNamespace(id=754488239, username="oleks", full_name="Oleks", is_bot=False),
            reply_to_message=None,
            photo=None,
            document=None,
            voice=None,
            audio=None,
            video=None,
            sticker=None,
        )

    def adapter_for_text_handler(self, *, auth):
        adapter = self.adapter_with_send()
        adapter._is_user_authorized_from_message = lambda msg: auth
        adapter._should_process_message = lambda msg, **kw: True
        adapter._should_observe_unmentioned_group_message = lambda msg: False
        adapter._ensure_forum_commands = mock.AsyncMock()
        adapter._clean_bot_trigger_text = lambda text: text
        adapter._cache_replied_media = mock.AsyncMock()
        adapter._apply_telegram_group_observe_attribution = lambda event: event
        adapter.enqueued = 0
        adapter._enqueue_text_event = lambda event: setattr(adapter, "enqueued", adapter.enqueued + 1)
        adapter._effective_message_thread_id = lambda msg: str(msg.message_thread_id) if msg.message_thread_id is not None else None
        adapter._get_dm_topic_info = lambda chat_id, thread_id: None
        return adapter

    def adapter_for_command_handler(self, *, auth):
        adapter = self.adapter_for_text_handler(auth=auth)
        adapter.handled = 0

        async def handle_message(event):
            adapter.handled += 1

        adapter.handle_message = handle_message
        return adapter


if __name__ == "__main__":
    unittest.main()

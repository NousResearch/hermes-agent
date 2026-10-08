"""Feishu inbound topic anchors and attachment routing regressions."""
import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

class TestTopicAnchors(unittest.TestCase):
    def test_inbound_thread_message_populates_source_message_id_anchor(self):
        """Detached replies use the current inbound message, not the topic root
        or the invalid omt_ thread id; quoted context still uses the root."""
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._dispatch_inbound_event = AsyncMock()
        adapter.get_chat_info = AsyncMock(
            return_value={"chat_id": "oc_chat", "name": "Group", "type": "group"}
        )
        adapter._resolve_sender_profile = AsyncMock(
            return_value={"user_id": "ou_user", "user_name": "张三", "user_id_alt": None}
        )
        adapter._fetch_message_text = AsyncMock(return_value=None)
        message = SimpleNamespace(
            chat_id="oc_chat",
            thread_id="omt_topic_abc",
            root_id="om_root_msg",
            parent_id=None,
            upper_message_id=None,
            message_type="text",
            content='{"text":"hi in topic"}',
            message_id="om_user_msg",
        )

        asyncio.run(
            adapter._process_inbound_message(
                data=SimpleNamespace(event=SimpleNamespace(message=message)),
                message=message,
                sender_id=SimpleNamespace(open_id="ou_user", user_id=None, union_id=None),
                is_bot=False,
                chat_type="group",
                message_id="om_user_msg",
            )
        )

        event = adapter._dispatch_inbound_event.await_args.args[0]
        # Match the event identity, including when a different topic root exists.
        self.assertEqual(event.source.thread_id, "omt_topic_abc")
        self.assertEqual(event.source.message_id, event.message_id)
        self.assertEqual(event.message_id, "om_user_msg")
        # event.reply_to_message_id is unchanged — still the root for context.
        self.assertEqual(event.reply_to_message_id, "om_root_msg")


    def test_inbound_thread_seed_message_populates_source_message_id_self(self):
        """A seed message (the first message of a new topic, no root_id yet)
        populates source.message_id with the message itself — it IS the root."""
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._dispatch_inbound_event = AsyncMock()
        adapter.get_chat_info = AsyncMock(
            return_value={"chat_id": "oc_chat", "name": "Group", "type": "group"}
        )
        adapter._resolve_sender_profile = AsyncMock(
            return_value={"user_id": "ou_user", "user_name": "张三", "user_id_alt": None}
        )
        adapter._fetch_message_text = AsyncMock(return_value=None)
        message = SimpleNamespace(
            chat_id="oc_chat",
            thread_id="omt_topic_new",
            root_id=None,
            parent_id=None,
            upper_message_id=None,
            message_type="text",
            content='{"text":"new topic"}',
            message_id="om_seed_msg",
        )

        asyncio.run(
            adapter._process_inbound_message(
                data=SimpleNamespace(event=SimpleNamespace(message=message)),
                message=message,
                sender_id=SimpleNamespace(open_id="ou_user", user_id=None, union_id=None),
                is_bot=False,
                chat_type="group",
                message_id="om_seed_msg",
            )
        )

        event = adapter._dispatch_inbound_event.await_args.args[0]
        self.assertEqual(event.source.message_id, "om_seed_msg")
        self.assertEqual(event.message_id, "om_seed_msg")


    def test_captioned_audio_preserves_post_routing(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._client = Mock()
        adapter._client.im.v1.file.create.return_value = SimpleNamespace(
            success=lambda: True, data=SimpleNamespace(file_key="file_key"),
        )
        adapter._client.im.v1.message.reply.return_value = SimpleNamespace(
            success=lambda: True, data=SimpleNamespace(message_id="om_sent"),
        )
        adapter._list_topic_reply_anchors = AsyncMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "voice.ogg"
            audio_path.write_bytes(b"opus")
            result = asyncio.run(adapter._send_uploaded_file_message(
                chat_id="oc_chat", file_path=str(audio_path), reply_to=None, caption="Voice caption",
                metadata={"thread_id": "omt_topic", "reply_to_message_id": "om_root"},
                outbound_message_type="audio",
            ))
        self.assertTrue(result.success)
        adapter._list_topic_reply_anchors.assert_not_awaited()
        adapter._client.im.v1.message.create.assert_not_called()
        request = adapter._client.im.v1.message.reply.call_args.args[0]
        self.assertEqual(request.message_id, "om_root")
        self.assertEqual(request.request_body.msg_type, "post")
        self.assertTrue(request.request_body.reply_in_thread)


    @patch.dict(os.environ, {}, clear=True)
    def test_extract_text_file_injects_content(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("hello from feishu")
            path = tmp.name

        try:
            text = asyncio.run(adapter._maybe_extract_text_document(path, "text/plain"))
        finally:
            os.unlink(path)

        self.assertIn("hello from feishu", text)
        self.assertIn("[Content of", text)

"""Group topic-reply toggle (FEISHU_THREAD_REPLY) behavior contracts.

The toggle upgrades a threadless group-chat quote reply to ``reply_in_thread=true``
(Feishu spawns the topic on the first threaded reply); DMs, open-id sends and
in-topic replies must be untouched. Lives apart from test_feishu.py per the
file-size ratchet (scripts/check FILE_LINES).
"""

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import lark_oapi
    _HAS_LARK_OAPI = True
except ImportError:
    _HAS_LARK_OAPI = False


@unittest.skipUnless(_HAS_LARK_OAPI, "lark-oapi not installed")
class TestFeishuThreadReplyToggle(unittest.TestCase):
    """FEISHU_THREAD_REPLY upgrades a group-chat quote reply into a topic reply.

    Feishu spawns the topic on the first ``reply_in_thread=true`` reply to a
    threadless group message; the toggle must not touch DMs, open-id sends,
    or replies already inside a topic.
    """

    def _reply_capture_adapter(self, *, thread_reply: bool, chat_type: str = "group"):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._thread_reply = thread_reply
        adapter._chat_info_cache = {"oc_chat": {"chat_id": "oc_chat", "name": "g", "type": chat_type}}
        captured = {}

        class _MessageAPI:
            def reply(self, request):
                captured["request"] = request
                return SimpleNamespace(
                    success=lambda: True, data=SimpleNamespace(message_id="om_reply"),
                )

            def create(self, request):  # pragma: no cover - must never be used here
                captured["create"] = request
                raise AssertionError("create must not be called when a reply anchor exists")

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(v1=SimpleNamespace(message=_MessageAPI())),
        )
        return adapter, captured

    @patch.dict(os.environ, {}, clear=True)
    def test_group_reply_forced_into_thread_when_enabled(self):
        adapter, captured = self._reply_capture_adapter(thread_reply=True)
        result = asyncio.run(adapter.send("oc_chat", "answer", reply_to="om_parent", metadata=None))
        self.assertTrue(result.success)
        self.assertTrue(captured["request"].request_body.reply_in_thread)

    @patch.dict(os.environ, {}, clear=True)
    def test_group_reply_stays_quote_when_disabled(self):
        adapter, captured = self._reply_capture_adapter(thread_reply=False)
        result = asyncio.run(adapter.send("oc_chat", "answer", reply_to="om_parent", metadata=None))
        self.assertTrue(result.success)
        self.assertFalse(captured["request"].request_body.reply_in_thread)

    @patch.dict(os.environ, {}, clear=True)
    def test_disabled_toggle_never_looks_up_chat_info(self):
        adapter, _ = self._reply_capture_adapter(thread_reply=False)

        async def _boom(*_a, **_k):
            raise AssertionError("get_chat_info must not be consulted while the toggle is off")

        adapter.get_chat_info = _boom
        asyncio.run(adapter.send("oc_chat", "answer", reply_to="om_parent"))

    @patch.dict(os.environ, {}, clear=True)
    def test_dm_chat_never_threaded(self):
        adapter, captured = self._reply_capture_adapter(thread_reply=True, chat_type="dm")
        asyncio.run(adapter.send("oc_chat", "answer", reply_to="om_parent"))
        self.assertFalse(captured["request"].request_body.reply_in_thread)

    @patch.dict(os.environ, {}, clear=True)
    def test_open_id_target_never_threaded(self):
        adapter, captured = self._reply_capture_adapter(thread_reply=True)
        asyncio.run(adapter.send("ou_user", "answer", reply_to="om_parent"))
        self.assertFalse(captured["request"].request_body.reply_in_thread)
        # and the shortcut must not have paid for a chat lookup
        self.assertNotIn("ou_user", adapter._chat_info_cache)

    @patch.dict(os.environ, {}, clear=True)
    def test_in_topic_behavior_unchanged_with_toggle_off(self):
        adapter, captured = self._reply_capture_adapter(thread_reply=False)
        asyncio.run(adapter.send(
            "oc_chat", "answer", reply_to="om_parent", metadata={"thread_id": "omt-1"},
        ))
        self.assertTrue(captured["request"].request_body.reply_in_thread)

    @patch.dict(os.environ, {}, clear=True)
    def test_settings_load_from_extra(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        self.assertTrue(FeishuAdapter(PlatformConfig(extra={"thread_reply": True}))._thread_reply)
        self.assertFalse(FeishuAdapter(PlatformConfig(extra={"thread_reply": False}))._thread_reply)

    @patch.dict(os.environ, {"FEISHU_THREAD_REPLY": "true"}, clear=True)
    def test_settings_load_from_env(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        self.assertTrue(FeishuAdapter(PlatformConfig())._thread_reply)

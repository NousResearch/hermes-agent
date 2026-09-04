"""Regression tests for Feishu adapter reply-in-thread routing.

These tests pin down the behaviour that:

1. When ``reply_in_thread`` is unset (default True), a reply to a message that
   lives in a Feishu topic still uses the reply API with ``reply_in_thread=True``.
2. When ``reply_in_thread`` is explicitly disabled via the platform ``extra``
   config, a reply to a top-level channel message skips the reply-in-thread
   path entirely — the response lands in the main chat (mirrors the Slack
   adapter's behaviour at ``plugins/platforms/slack/adapter.py`` lines 1435+).
3. When the target is a Feishu DM (``chat_type`` in metadata is ``"p2p"`` /
   ``"dm"``), the adapter must NEVER reply-in-thread even if ``reply_in_thread``
   is left at its default — the Feishu reply API renders reply_in_thread=true
   as a fresh discussion surface in p2p chats, which the client then shows
   as a "started a thread" UX.
4. A genuine root-topic reply (where ``thread_id`` equals
   ``reply_to_message_id`` because inbound maps ``root_id`` into both) still
   uses the reply API. The adapter must NOT try to detect "synthetic" threads
   via that equality check — it would suppress real topic replies.
5. The primary final-reply path (which routes through
   ``gateway.platforms.base._thread_metadata_for_source``) carries
   ``chat_type`` into the metadata dict, so a p2p source with a routing
   thread_id reaches the adapter as a DM and the rule in (3) fires.

The original bug is summarised in
``plugins/platforms/feishu/adapter.py::_send_raw_message``: the legacy
implementation read ``reply_in_thread = bool(metadata.get("thread_id"))``
without consulting ``self.config.extra`` and without considering the target's
chat class, so DM sends were silently downgraded into the Feishu reply API's
thread mode.
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class _ReplyCapturingClient:
    """Minimal Feishu lark-oapi stub.

    The adapter invokes ``self._client.im.v1.message.{reply,create}(request)``.
    We capture each request and return a successful stub response.
    """

    def __init__(self):
        self.replies = []
        self.creates = []
        reply_api = self
        create_api = self

        class _MessageAPI:
            def reply(self, request):
                reply_api._record_reply(request)
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_reply"),
                )

            def create(self, request):
                create_api._record_create(request)
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_create"),
                )

        self._message_api = _MessageAPI()

    @property
    def im(self):
        client = self

        class _ImNamespace:
            v1 = type("V1", (), {"message": client._message_api})()

        return _ImNamespace()

    def _record_reply(self, request):
        self.replies.append(request)

    def _record_create(self, request):
        self.creates.append(request)


class TestFeishuReplyInThread(unittest.TestCase):
    """Pin the reply-in-thread routing decisions in _send_raw_message."""

    def _build_adapter(self, extra=None):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        return FeishuAdapter(PlatformConfig(extra=extra or {}))

    def _attach_client(self, adapter, client):
        adapter._client = client

    @patch.dict("os.environ", {}, clear=True)
    def test_default_reply_in_thread_true_keeps_threaded_topic_behaviour(self):
        adapter = self._build_adapter()
        client = _ReplyCapturingClient()
        self._attach_client(adapter, client)

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="in-thread reply",
                    reply_to="om_parent",
                    metadata={"thread_id": "omt-topic", "chat_type": "group"},
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(len(client.replies), 1, "reply API should be used for group topics")
        body = client.replies[0].request_body
        self.assertEqual(client.replies[0].message_id, "om_parent")
        self.assertTrue(
            body.reply_in_thread,
            "threaded group topic must keep reply_in_thread=True",
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_extra_reply_in_thread_false_top_level_message_skips_topic(self):
        adapter = self._build_adapter(extra={"reply_in_thread": False})
        client = _ReplyCapturingClient()
        self._attach_client(adapter, client)

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="direct reply",
                    reply_to="om_parent",
                    metadata={"thread_id": "omt-topic", "chat_type": "group"},
                )
            )

        self.assertTrue(result.success)
        # reply_in_thread=False must route through message.create, not message.reply.
        self.assertEqual(
            len(client.replies),
            0,
            "reply_in_thread=False must not invoke the reply API",
        )
        self.assertGreater(
            len(client.creates),
            0,
            "reply_in_thread=False should fall back to message.create for routing",
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_dm_never_replies_in_thread_even_with_extra_enabled(self):
        adapter = self._build_adapter(extra={"reply_in_thread": True})
        client = _ReplyCapturingClient()
        self._attach_client(adapter, client)

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="hi there",
                    reply_to="om_parent",
                    metadata={
                        # Gateway stamps a routing thread_id even on DM sends —
                        # the adapter must ignore it and stay flat.
                        "thread_id": "omt-leaked",
                        "chat_type": "p2p",
                    },
                )
            )

        self.assertTrue(result.success)
        for req in client.replies:
            self.assertFalse(
                req.request_body.reply_in_thread,
                "DM replies must not set reply_in_thread=true (got request: %r)" % (req,),
            )
        for req in client.creates:
            # The lark-oapi SDK keeps the receiver id on request_body, not on
            # the request itself.
            body = getattr(req, "request_body", None)
            receive_id = getattr(body, "receive_id", None) if body else None
            self.assertNotEqual(
                receive_id,
                "omt-leaked",
                "DM create fallback must not address a leaked thread_id as receive_id",
            )

    @patch.dict("os.environ", {}, clear=True)
    def test_root_topic_reply_with_equality_still_uses_reply_api(self):
        """Inbound processing maps ``root_id`` into BOTH ``thread_id`` and
        ``reply_to_message_id`` (``plugins/platforms/feishu/adapter.py``
        ~3252). For a genuine root-topic reply those two fields are
        therefore equal. The adapter MUST treat this as a real topic
        reply, not a synthetic thread stamp, and call the reply API.

        This is the regression that nukes the previous
        ``thread_id == reply_to_message_id`` synthetic detector.
        """
        adapter = self._build_adapter()
        client = _ReplyCapturingClient()
        self._attach_client(adapter, client)

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="root-topic reply",
                    # Inbound lays these out identically for root-topic
                    # replies — adapter must NOT short-circuit this.
                    reply_to="omt-topic",
                    metadata={
                        "thread_id": "omt-topic",
                        "reply_to_message_id": "omt-topic",
                        "chat_type": "group",
                    },
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(
            len(client.replies),
            1,
            "root-topic reply (thread_id == reply_to_message_id) must still "
            "go through the reply API",
        )
        body = client.replies[0].request_body
        self.assertTrue(
            body.reply_in_thread,
            "root-topic reply must set reply_in_thread=True",
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_dm_metadata_via_base_helper_keeps_p2p_final_reply_flat(self):
        """End-to-end check that the *primary* final-reply path now feeds
        ``chat_type`` to the adapter. The base helper
        ``gateway.platforms.base._thread_metadata_for_source`` is what
        the final-reply path uses to build the metadata dict; a p2p
        source with a routing thread_id must produce metadata that the
        adapter recognises as DM and routes flat, with no reply API
        call and no ``thread_id`` as receive_id.
        """
        from types import SimpleNamespace
        from gateway.platforms.base import _thread_metadata_for_source

        adapter = self._build_adapter()
        client = _ReplyCapturingClient()
        self._attach_client(adapter, client)

        source = SimpleNamespace(
            platform="feishu",
            chat_type="p2p",
            thread_id="omt-leaked",
            chat_id="oc_chat",
        )
        metadata = _thread_metadata_for_source(source, reply_to_message_id="om_parent")
        # The helper must now carry chat_type through — that's the
        # primary-path signal the adapter needs to enforce the DM rule.
        self.assertEqual(
            metadata.get("chat_type"),
            "p2p",
            "base helper must forward chat_type on the primary final-reply path",
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="hi there",
                    reply_to="om_parent",
                    metadata=metadata,
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(
            len(client.replies),
            0,
            "p2p final reply must skip the reply API even with a routing thread_id",
        )
        self.assertGreater(
            len(client.creates),
            0,
            "p2p final reply must fall through to message.create",
        )
        for req in client.creates:
            body = getattr(req, "request_body", None)
            receive_id = getattr(body, "receive_id", None) if body else None
            self.assertNotEqual(
                receive_id,
                "omt-leaked",
                "p2p create fallback must not address a leaked thread_id",
            )
            self.assertEqual(
                receive_id,
                "oc_chat",
                "p2p final reply must address the chat_id, not the thread stamp",
            )


if __name__ == "__main__":
    unittest.main()

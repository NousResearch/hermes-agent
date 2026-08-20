"""Tests for Feishu adapter create_handoff_thread + thread routing.

Covers:
- create_handoff_thread: seed-anchor pattern (Feishu surfaces any reply
  chain as a "topic" in DMs and groups, so the anchor message_id is the
  thread handle — mirrors Slack).
- _send_raw_message anchorless thread routing: metadata["thread_id"] must
  go through the reply API with reply_in_thread=true, NOT message.create
  with receive_id_type="thread_id" (rejected by the Feishu API with
  99992402 — #78975).
- send() end-to-end: the real path scheduler/DeliveryRouter uses for
  continuable cron deliveries (metadata={"thread_id": <anchor>}).
- Reply-failure fallback semantics in _feishu_send_with_retry: with a
  thread anchor present, a failed reply must NOT fall back to a top-level
  create (that would spawn a new topic outside the thread); without a
  thread, the existing create fallback stays.
- Degradation contract: every failure path returns None so callers
  (scheduler ``_open_continuable_cron_thread``, gateway ``_process_handoff``)
  fall back to the origin-DM mirror without failing the delivery.

Real-data notes (verified 2026-08-05 from the live state.db):
- Feishu DM reply chains produce thread-keyed sessions keyed on the
  anchor (root) message id:
      session_key = agent:main:feishu:dm:oc_5449fd7cd1cab7e6df3e74cb81b4666e:om_x100b6ba5dc1a40a8c2acad0bace48b3
      thread_id   = om_x100b6ba5dc1a40a8c2acad0bace48b3   (om_ prefix = message id)
- Message ids are om_-prefixed; chat ids are oc_-prefixed (both p2p and
  group). These shapes are used in the fixtures below.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult

# Real-shape identifiers observed in production (see docstring).
_ANCHOR_MSG_ID = "om_x100b6ba5dc1a40a8c2acad0bace48b3"
_P2P_CHAT_ID = "oc_5449fd7cd1cab7e6df3e74cb81b4666e"
_GROUP_CHAT_ID = "oc_bb98679627588183ff7ab339347e6d62"


def _make_adapter():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    return FeishuAdapter(PlatformConfig())


def _install_message_api(adapter, captured, *, reply_returns=None, create_returns=None):
    """Wire a fake im.v1.message API capturing requests; each method returns
    a success SendResult-shaped response unless overridden."""

    def _ok(message_id):
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id=message_id))

    class _MessageAPI:
        def reply(self, request):
            captured["api"] = "reply"
            captured["request"] = request
            if callable(reply_returns):
                return reply_returns(request)
            return _ok("om_replied")

        def create(self, request):
            captured["api"] = "create"
            captured["request"] = request
            if callable(create_returns):
                return create_returns(request)
            return _ok("om_created")

    adapter._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=_MessageAPI()))
    )


def _run(coro):
    return asyncio.run(coro)


class TestCreateHandoffThread(unittest.TestCase):
    def test_anchor_message_and_returns_message_id(self):
        adapter = _make_adapter()
        adapter._client = Mock()

        async def _fake_send(chat_id, content):
            self.assertEqual(chat_id, _P2P_CHAT_ID)
            self.assertIn("Hermes —", content)
            return SendResult(success=True, message_id=_ANCHOR_MSG_ID)

        with patch.object(adapter, "send", side_effect=_fake_send):
            result = _run(adapter.create_handoff_thread(_P2P_CHAT_ID, "daily-review"))

        self.assertEqual(result, _ANCHOR_MSG_ID)

    def test_anchor_send_failure_returns_none(self):
        adapter = _make_adapter()
        adapter._client = Mock()

        async def _fake_send(chat_id, content):
            return SendResult(success=False, error="[230002] cannot send")

        with patch.object(adapter, "send", side_effect=_fake_send):
            result = _run(adapter.create_handoff_thread(_P2P_CHAT_ID, "x"))

        self.assertIsNone(result)

    def test_anchor_send_success_without_message_id_returns_none(self):
        """SendResult may report success but carry no message_id (defensive:
        never return a non-message-id as a thread handle)."""
        adapter = _make_adapter()
        adapter._client = Mock()

        async def _fake_send(chat_id, content):
            return SendResult(success=True, message_id=None)

        with patch.object(adapter, "send", side_effect=_fake_send):
            result = _run(adapter.create_handoff_thread(_P2P_CHAT_ID, "x"))

        self.assertIsNone(result)

    def test_no_client_returns_none(self):
        adapter = _make_adapter()
        adapter._client = None
        result = _run(adapter.create_handoff_thread(_P2P_CHAT_ID, "x"))
        self.assertIsNone(result)

    def test_exception_inside_is_swallowed_to_none(self):
        adapter = _make_adapter()
        adapter._client = Mock()

        async def _boom(chat_id, content):
            raise RuntimeError("network down")

        with patch.object(adapter, "send", side_effect=_boom):
            result = _run(adapter.create_handoff_thread(_P2P_CHAT_ID, "x"))

        self.assertIsNone(result)


class TestSendEndToEnd(unittest.TestCase):
    """send() — the real entry the scheduler/DeliveryRouter uses for
    continuable cron deliveries (metadata={"thread_id": <anchor>}, no
    reply_to)."""

    @staticmethod
    def _direct_patch():
        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        return patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct)

    def test_send_with_thread_id_metadata_routes_to_reply_in_thread(self):
        adapter = _make_adapter()
        captured = {}
        _install_message_api(adapter, captured)
        with self._direct_patch():
            result = _run(
                adapter.send(_P2P_CHAT_ID, "daily ledger", metadata={"thread_id": _ANCHOR_MSG_ID})
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["api"], "reply")
        self.assertEqual(captured["request"].message_id, _ANCHOR_MSG_ID)
        self.assertTrue(captured["request"].request_body.reply_in_thread)

    def test_send_with_reply_to_and_thread_id_metadata(self):
        """In-thread reply (user reply inside the topic): reply_to + thread_id
        both present → reply API, reply_in_thread still true (parity with the
        existing send_document test, test_feishu.py::test_send_document_reply_uses_thread_flag)."""
        adapter = _make_adapter()
        captured = {}
        _install_message_api(adapter, captured)
        with self._direct_patch():
            result = _run(
                adapter.send(
                    _GROUP_CHAT_ID,
                    "in-topic reply",
                    reply_to=_ANCHOR_MSG_ID,
                    metadata={"thread_id": _ANCHOR_MSG_ID},
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["api"], "reply")
        self.assertEqual(captured["request"].message_id, _ANCHOR_MSG_ID)
        self.assertTrue(captured["request"].request_body.reply_in_thread)

    def test_send_without_thread_uses_create_with_chat_id(self):
        adapter = _make_adapter()
        captured = {}
        _install_message_api(adapter, captured)
        with self._direct_patch():
            result = _run(adapter.send(_P2P_CHAT_ID, "plain message"))

        self.assertTrue(result.success)
        self.assertEqual(captured["api"], "create")
        self.assertEqual(captured["request"].receive_id_type, "chat_id")
        self.assertEqual(captured["request"].request_body.receive_id, _P2P_CHAT_ID)


class TestAnchorlessThreadRouting(unittest.TestCase):
    """#78975 regression at the _send_raw_message level: metadata['thread_id']
    must route via the reply API (reply_in_thread=true), never message.create
    with the invalid receive_id_type='thread_id'."""

    def test_thread_id_routes_via_reply_api_with_reply_in_thread(self):
        adapter = _make_adapter()
        captured = {}
        _install_message_api(adapter, captured)
        with TestSendEndToEnd._direct_patch():
            result = _run(
                adapter._send_raw_message(
                    chat_id=_P2P_CHAT_ID,
                    msg_type="text",
                    payload='{"text":"hi"}',
                    reply_to=None,
                    metadata={"thread_id": _ANCHOR_MSG_ID},
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["api"], "reply")
        self.assertEqual(captured["request"].message_id, _ANCHOR_MSG_ID)
        self.assertTrue(captured["request"].request_body.reply_in_thread)

    def test_empty_thread_id_metadata_falls_through_to_create(self):
        """An empty/None thread_id value is falsy — must not attempt a
        reply against it; falls through to the normal create path."""
        adapter = _make_adapter()
        captured = {}
        _install_message_api(adapter, captured)
        with TestSendEndToEnd._direct_patch():
            result = _run(
                adapter._send_raw_message(
                    chat_id=_P2P_CHAT_ID,
                    msg_type="text",
                    payload='{"text":"hi"}',
                    reply_to=None,
                    metadata={"thread_id": ""},
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["api"], "create")
        self.assertEqual(captured["request"].request_body.receive_id, _P2P_CHAT_ID)

    def test_no_thread_metadata_uses_create_with_chat_id(self):
        adapter = _make_adapter()
        captured = {}
        _install_message_api(adapter, captured)
        with TestSendEndToEnd._direct_patch():
            result = _run(
                adapter._send_raw_message(
                    chat_id=_P2P_CHAT_ID,
                    msg_type="text",
                    payload='{"text":"hi"}',
                    reply_to=None,
                    metadata=None,
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["api"], "create")
        self.assertEqual(captured["request"].receive_id_type, "chat_id")
        self.assertEqual(captured["request"].request_body.receive_id, _P2P_CHAT_ID)


class TestReplyFallback(unittest.TestCase):
    """_feishu_send_with_retry reply-failure semantics (#78975 interaction)."""

    def test_reply_failure_with_thread_anchor_skips_top_level_fallback(self):
        """Reply target withdrawn/missing (code 230011) inside a thread must
        NOT fall back to a top-level create — that would spawn a NEW message
        outside the thread (or, worse, a new topic). The failure propagates."""
        adapter = _make_adapter()
        calls = []

        async def _fake_send_raw(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                success=lambda: False,
                code=230011,
                msg="reply target withdrawn",
                raw=SimpleNamespace(content=b"{}"),
            )

        with patch.object(adapter, "_send_raw_message", side_effect=_fake_send_raw):
            result = _run(
                adapter._feishu_send_with_retry(
                    chat_id=_P2P_CHAT_ID,
                    msg_type="text",
                    payload='{"text":"hi"}',
                    reply_to=_ANCHOR_MSG_ID,
                    metadata={"thread_id": _ANCHOR_MSG_ID},
                )
            )

        self.assertFalse(result.success())
        # Exactly one attempt — no top-level fallback send.
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["reply_to"], _ANCHOR_MSG_ID)

    def test_reply_failure_without_thread_falls_back_to_create(self):
        """Without a thread anchor, a failed reply (230011) degrades to a
        fresh message in the chat — existing behavior preserved."""
        adapter = _make_adapter()
        calls = []

        async def _fake_send_raw(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(
                    success=lambda: False,
                    code=230011,
                    msg="reply target withdrawn",
                    raw=SimpleNamespace(content=b"{}"),
                )
            return SimpleNamespace(
                success=lambda: True,
                data=SimpleNamespace(message_id="om_fallback"),
            )

        with patch.object(adapter, "_send_raw_message", side_effect=_fake_send_raw):
            result = _run(
                adapter._feishu_send_with_retry(
                    chat_id=_P2P_CHAT_ID,
                    msg_type="text",
                    payload='{"text":"hi"}',
                    reply_to="om_withdrawn",
                    metadata=None,
                )
            )

        self.assertTrue(result.success())
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[1]["reply_to"])  # second attempt is a bare create


if __name__ == "__main__":
    unittest.main()

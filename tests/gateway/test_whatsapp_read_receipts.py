"""Tests for WhatsApp read-receipts (blue tick) support.

Two layers are covered:

1. Config bridging — the opt-in `whatsapp.read_receipts` config.yaml key is
   translated into the `WHATSAPP_READ_RECEIPTS` env var (consumed by both the
   Node bridge and the adapter) by the `_apply_yaml_config` hook.

2. Cross-layer admission ordering — the receipt is sent only *after* the
   adapter admits the message (`_build_message_event` returns non-None, i.e.
   `_should_process_message` passed), so a group message rejected for group
   policy or a missing mention never gets a blue tick. The receipt payload
   carries the sender's participant JID for groups and omits it for DMs.

The receipt *skip rules* themselves (fromMe, status/broadcast/newsletter, DM
vs group participant on the Baileys key) live in the bridge and are unit-tested
in `scripts/whatsapp-bridge/bridge.native.test.mjs`; the bridge starts a socket
and HTTP server at import, so it can't be exercised from pytest.
"""

import asyncio
from unittest.mock import patch, MagicMock

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from plugins.platforms.whatsapp.adapter import _apply_yaml_config, WhatsAppAdapter


# ---------------------------------------------------------------------------
# 1. config.yaml -> WHATSAPP_READ_RECEIPTS env bridging
# ---------------------------------------------------------------------------


class TestReadReceiptsYamlBridging:
    def test_true_sets_env(self):
        with patch.dict("os.environ", {}, clear=True):
            _apply_yaml_config({}, {"read_receipts": True})
            import os
            assert os.environ.get("WHATSAPP_READ_RECEIPTS") == "true"

    def test_false_sets_env_off(self):
        with patch.dict("os.environ", {}, clear=True):
            _apply_yaml_config({}, {"read_receipts": False})
            import os
            # The bridge treats anything not in {1,true,yes,on} as off.
            assert os.environ.get("WHATSAPP_READ_RECEIPTS") == "false"

    def test_absent_key_leaves_env_unset(self):
        with patch.dict("os.environ", {}, clear=True):
            _apply_yaml_config({}, {"reply_prefix": "x"})
            import os
            assert "WHATSAPP_READ_RECEIPTS" not in os.environ

    def test_env_var_takes_precedence_over_yaml(self):
        with patch.dict("os.environ", {"WHATSAPP_READ_RECEIPTS": "1"}, clear=True):
            _apply_yaml_config({}, {"read_receipts": False})
            import os
            assert os.environ.get("WHATSAPP_READ_RECEIPTS") == "1"


# ---------------------------------------------------------------------------
# 2. adapter admission ordering + receipt payload
# ---------------------------------------------------------------------------


class _CapturingSession:
    """Minimal aiohttp-like session that records the last POST."""

    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json})
        session = self

        class _Ctx:
            async def __aenter__(self):
                return MagicMock(status=200)

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def _make_adapter(read_receipts=True):
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True))
    adapter._read_receipts = read_receipts
    adapter._bridge_port = 3999
    adapter._http_session = _CapturingSession()
    return adapter


class TestDispatchOrdering:
    """`_dispatch_incoming` marks read only after admission, then dispatches."""

    def test_rejected_message_is_not_marked_read(self):
        adapter = _make_adapter()
        marked = []

        async def fake_build(_data):
            return None  # _should_process_message rejected it

        async def fake_mark(data):
            marked.append(data)

        async def fake_handle(_event):
            raise AssertionError("rejected message must not be dispatched")

        adapter._build_message_event = fake_build
        adapter._mark_read_if_enabled = fake_mark
        adapter.handle_message = fake_handle

        asyncio.run(adapter._dispatch_incoming(
            {"chatId": "g@g.us", "messageId": "1", "isGroup": True}
        ))
        assert marked == [], "policy/mention-rejected message got a read receipt"

    def test_admitted_message_is_marked_read_then_dispatched(self):
        adapter = _make_adapter()
        order = []

        event = MagicMock()
        event.message_type = MessageType.PHOTO  # non-text -> handle_message

        async def fake_build(_data):
            return event

        async def fake_mark(_data):
            order.append("mark")

        async def fake_handle(_event):
            order.append("dispatch")

        adapter._build_message_event = fake_build
        adapter._mark_read_if_enabled = fake_mark
        adapter.handle_message = fake_handle

        asyncio.run(adapter._dispatch_incoming(
            {"chatId": "g@g.us", "messageId": "1", "isGroup": True,
             "senderId": "u@s.whatsapp.net"}
        ))
        assert order == ["mark", "dispatch"], order

    def test_admitted_text_is_batched_after_mark(self):
        adapter = _make_adapter()
        order = []

        event = MagicMock()
        event.message_type = MessageType.TEXT

        async def fake_build(_data):
            return event

        async def fake_mark(_data):
            order.append("mark")

        adapter._build_message_event = fake_build
        adapter._mark_read_if_enabled = fake_mark
        adapter._enqueue_text_event = lambda _e: order.append("enqueue")

        asyncio.run(adapter._dispatch_incoming(
            {"chatId": "u@s.whatsapp.net", "messageId": "1", "isGroup": False}
        ))
        assert order == ["mark", "enqueue"], order


class TestMarkReadPayload:
    """`_mark_read_if_enabled` posts the right key, and only when enabled."""

    def test_group_payload_includes_participant(self):
        adapter = _make_adapter(read_receipts=True)
        asyncio.run(adapter._mark_read_if_enabled(
            {"chatId": "g@g.us", "messageId": "m1", "isGroup": True,
             "senderId": "u@s.whatsapp.net"}
        ))
        call = adapter._http_session.calls[-1]
        assert call["url"].endswith("/mark-read")
        assert call["json"] == {
            "chatId": "g@g.us", "messageId": "m1",
            "participant": "u@s.whatsapp.net",
        }

    def test_dm_payload_omits_participant(self):
        adapter = _make_adapter(read_receipts=True)
        asyncio.run(adapter._mark_read_if_enabled(
            {"chatId": "u@s.whatsapp.net", "messageId": "m1", "isGroup": False,
             "senderId": "u@s.whatsapp.net"}
        ))
        assert adapter._http_session.calls[-1]["json"] == {
            "chatId": "u@s.whatsapp.net", "messageId": "m1",
        }

    def test_disabled_makes_no_request(self):
        adapter = _make_adapter(read_receipts=False)
        asyncio.run(adapter._mark_read_if_enabled(
            {"chatId": "g@g.us", "messageId": "m1", "isGroup": True,
             "senderId": "u@s.whatsapp.net"}
        ))
        assert adapter._http_session.calls == []

    def test_missing_message_id_makes_no_request(self):
        adapter = _make_adapter(read_receipts=True)
        asyncio.run(adapter._mark_read_if_enabled({"chatId": "g@g.us"}))
        assert adapter._http_session.calls == []

"""A non-numeric Telegram reply anchor sends unanchored instead of failing the send (#139611).

Route wakes carry a synthetic producer identity as ``message_id``
(``["kanban-wake", <session_key>, <uuid>]``); before v0.22.3 the source layer could let it (or a
copy persisted in delivery-ledger metadata / queued sends across a restart) reach the adapter's
raw ``int()`` coercions, and every retry of the send died on ``ValueError``. The adapters now
convert defensively: a numeric anchor threads as before, anything else warns once and sends
without a quote — a reply with no quote is strictly better than a failed send.
"""

import logging

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter

WAKE = '["kanban-wake","agent:main:telegram:dm:123:456","0f1e2d3c4b5a"]'


@pytest.fixture(autouse=True)
def _reset_anchor_warning():
    TelegramAdapter._NON_NUMERIC_ANCHOR_WARNED = False
    yield
    TelegramAdapter._NON_NUMERIC_ANCHOR_WARNED = False


@pytest.fixture()
def adapter():
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


class TestMetadataAnchor:
    def test_a_wake_identity_in_metadata_sends_unanchored(self, adapter):
        # The exact regression shape: raw int() raised ValueError on the JSON array.
        assert adapter._metadata_reply_to_message_id({"telegram_reply_to_message_id": WAKE}) is None

    def test_a_numeric_metadata_anchor_threads(self, adapter):
        assert adapter._metadata_reply_to_message_id({"telegram_reply_to_message_id": "123"}) == 123

    def test_empty_and_absent_metadata_stay_none(self, adapter):
        assert adapter._metadata_reply_to_message_id({"telegram_reply_to_message_id": ""}) is None
        assert adapter._metadata_reply_to_message_id(None) is None

    @pytest.mark.parametrize("bad", ["-1", "1.5", "msg-123", 'auth:v1:["basic","","u"]'])
    def test_other_non_numeric_anchors_send_unanchored(self, adapter, bad):
        assert adapter._metadata_reply_to_message_id({"telegram_reply_to_message_id": bad}) is None


class TestSendAnchor:
    def test_a_wake_identity_reply_to_sends_unanchored(self, adapter):
        assert adapter._reply_to_message_id_for_send(WAKE) is None

    def test_a_numeric_reply_to_threads(self, adapter):
        assert adapter._reply_to_message_id_for_send("123") == 123

    def test_chunk_routing_survives_a_wake_identity(self, adapter):
        # index 0 with the default "first" mode threads; the poisoned anchor must not raise.
        _, _, reply_to_id = adapter._chunk_reply_routing("123", WAKE, None, None, 0)
        assert reply_to_id is None

    def test_chunk_routing_threads_a_numeric_anchor(self, adapter):
        _, _, reply_to_id = adapter._chunk_reply_routing("123", "456", None, None, 0)
        assert reply_to_id == 456


def test_the_non_numeric_anchor_warning_fires_once(adapter, caplog):
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.telegram.adapter"):
        adapter._reply_to_message_id_for_send(WAKE)
        adapter._metadata_reply_to_message_id({"telegram_reply_to_message_id": WAKE})
        adapter._chunk_reply_routing("123", WAKE, None, None, 0)
    dropped = [r for r in caplog.records if "non-numeric reply anchor" in r.message]
    assert len(dropped) == 1

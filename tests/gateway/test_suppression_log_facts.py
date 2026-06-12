"""Tests for the delivery facts appended to the gateway's final-send
suppression log line (#27942, #29200).

The suppression branch in gateway/run.py is not practically reachable from
a unit test (it lives deep inside the agent-response flow with the stream
consumer as a closure local), so the composed suffix is produced by a pure
module-level helper that these tests exercise directly."""

import hashlib
from unittest.mock import MagicMock

from gateway.run import _delivery_log_facts
from gateway.stream_consumer import GatewayStreamConsumer


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:8]


class TestDeliveryLogFacts:
    def test_consumer_facts_present_and_content_free(self):
        """All fields appear with correct values; message text never leaks."""
        consumer = GatewayStreamConsumer(MagicMock(), "chat_123")
        consumer._accumulated = "delivered text"
        consumer._last_sent_text = "delivered text"
        consumer._message_id = "msg_9"

        suffix = _delivery_log_facts("delivered text", consumer)

        assert "final_len=14" in suffix
        assert f"final_digest={_digest('delivered text')}" in suffix
        assert "acc_len=14" in suffix
        assert f"acc_digest={_digest('delivered text')}" in suffix
        assert "last_sent_len=14" in suffix
        assert "sc_msg_id=msg_9" in suffix
        assert "last_edit_overflowed=False" in suffix
        assert "delivered text" not in suffix

    def test_divergence_between_final_and_accumulated_is_visible(self):
        """When the agent's final response differs from what the consumer
        delivered, both lengths and both digests expose the divergence."""
        consumer = GatewayStreamConsumer(MagicMock(), "chat_123")
        consumer._accumulated = "short"
        consumer._last_sent_text = "short"
        consumer._message_id = "msg_9"

        suffix = _delivery_log_facts("a longer final response", consumer)

        assert "final_len=23" in suffix
        assert "acc_len=5" in suffix
        digest_final = _digest("a longer final response")
        digest_acc = _digest("short")
        assert digest_final != digest_acc
        assert f"final_digest={digest_final}" in suffix
        assert f"acc_digest={digest_acc}" in suffix

    def test_no_consumer_logs_sentinels_without_raising(self):
        """previewed-only suppressions have no stream consumer; the suffix
        still composes with sentinel fields."""
        suffix = _delivery_log_facts("final text", None)
        assert "final_len=10" in suffix
        assert "sc_msg_id=?" in suffix
        assert "acc_len=?" in suffix

    def test_poisoned_summary_never_raises(self):
        """Log composition must never be able to abort the suppression
        decision: a raising summary degrades to sentinels."""

        class Poisoned:
            def delivery_summary(self):
                raise RuntimeError("boom")

        suffix = _delivery_log_facts("final", Poisoned())
        assert "final_len=5" in suffix
        assert "sc_msg_id=?" in suffix

    def test_non_string_final_text_degrades_to_zero(self):
        suffix = _delivery_log_facts(None, None)
        assert "final_len=0" in suffix

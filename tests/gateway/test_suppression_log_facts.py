"""Tests for the delivery facts appended to the gateway's final-send
suppression log line (#27942, #29200).

The suppression branch in gateway/run.py is not practically reachable from
a unit test (it lives deep inside the agent-response flow with the stream
consumer as a closure local), so the composed suffix is produced by a pure
module-level helper that these tests exercise directly."""

from unittest.mock import MagicMock

from gateway.run import _SUPPRESS_FINAL_SEND_LOG_FMT, _delivery_log_facts
from gateway.stream_consumer import GatewayStreamConsumer
from gateway.stream_consumer import _short_digest as _digest


def _fields(suffix: str) -> list[str]:
    """Split the composed suffix into whole key=value tokens so length
    assertions cannot false-pass on digit prefixes (acc_len=14 vs 140)."""
    return suffix.split()


class TestDeliveryLogFacts:
    def test_consumer_facts_present_and_content_free(self):
        """All fields appear with correct values; message text never leaks."""
        consumer = GatewayStreamConsumer(MagicMock(), "chat_123")
        consumer._accumulated = "delivered text"
        consumer._last_sent_text = "delivered text"
        consumer._message_id = "msg_9"

        suffix = _delivery_log_facts("delivered text", consumer)

        fields = _fields(suffix)
        assert "final_len=14" in fields
        assert f"final_digest={_digest('delivered text')}" in fields
        assert "acc_len=14" in fields
        assert f"acc_digest={_digest('delivered text')}" in fields
        assert "last_sent_len=14" in fields
        assert "sc_msg_id=msg_9" in fields
        assert "last_edit_overflowed=False" in fields
        assert "delivered text" not in suffix

    def test_divergence_between_final_and_accumulated_is_visible(self):
        """When the agent's final response differs from what the consumer
        delivered, both lengths and both digests expose the divergence."""
        consumer = GatewayStreamConsumer(MagicMock(), "chat_123")
        consumer._accumulated = "short"
        consumer._last_sent_text = "short"
        consumer._message_id = "msg_9"

        suffix = _delivery_log_facts("a longer final response", consumer)

        fields = _fields(suffix)
        assert "final_len=23" in fields
        assert "acc_len=5" in fields
        digest_final = _digest("a longer final response")
        digest_acc = _digest("short")
        assert f"final_digest={digest_final}" in fields
        assert f"acc_digest={digest_acc}" in fields

    def test_no_consumer_logs_sentinels_without_raising(self):
        """previewed-only suppressions have no stream consumer; the suffix
        still composes with sentinel fields."""
        suffix = _delivery_log_facts("final text", None)
        fields = _fields(suffix)
        assert "final_len=10" in fields
        assert "sc_msg_id=?" in fields
        assert "acc_len=?" in fields

    def test_live_consumer_without_message_id_logs_sentinel_id(self):
        """A consumer that never sent anything reports message_id=None; the
        composed field falls back to the sentinel."""
        consumer = GatewayStreamConsumer(MagicMock(), "chat_x")
        suffix = _delivery_log_facts("text", consumer)
        fields = _fields(suffix)
        assert "sc_msg_id=?" in fields
        assert "acc_len=0" in fields

    def test_raising_digest_degrades_final_fields_to_sentinels(self, monkeypatch):
        """No statement in the helper may raise: a failing digest (or a
        failing lazy import) degrades the final fields to sentinels instead
        of propagating into the suppression decision."""
        import gateway.stream_consumer as sc_module

        def _boom(text):
            raise RuntimeError("digest unavailable")

        monkeypatch.setattr(sc_module, "_short_digest", _boom)
        suffix = _delivery_log_facts("final text", None)
        fields = _fields(suffix)
        assert "final_len=?" in fields
        assert "final_digest=?" in fields
        assert "sc_msg_id=?" in fields

    def test_suppression_log_format_matches_arg_count(self):
        """The suppression logger.info call passes exactly five arguments;
        logging swallows placeholder mismatches silently, so the format
        string is pinned here instead."""
        assert _SUPPRESS_FINAL_SEND_LOG_FMT.count("%s") == 5
        assert _SUPPRESS_FINAL_SEND_LOG_FMT.startswith(
            "Suppressing normal final send for session %s"
        )

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

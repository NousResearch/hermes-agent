"""Regression test for #81052 — duplicate message delivery on slow turns.

When the gateway takes the queued-follow-up fallback path (``adapter.send()``
of ``first_response`` because streaming confirmation did not arrive in time),
the normal completion pipeline that runs after the fallback was unaware the
response had already been delivered. The user saw the same reply twice.

The fix introduces ``GatewayStreamConsumer.mark_out_of_band_delivery`` so the
gateway can record the non-streaming delivery and the subsequent
``_stream_confirmed_final_delivery`` predicate returns ``True``, suppressing
the second send.
"""

from unittest.mock import MagicMock

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _make_consumer() -> GatewayStreamConsumer:
    """Build a bare GatewayStreamConsumer for the new flag tests."""
    return GatewayStreamConsumer(
        adapter=MagicMock(),
        chat_id="123",
        config=StreamConsumerConfig(cursor=" ▉"),
        metadata=None,
        run_still_current=lambda: True,
    )


class TestMarkOutOfBandDelivery:
    def test_sets_final_response_sent_flag(self):
        consumer = _make_consumer()
        assert consumer.final_response_sent is False

        consumer.mark_out_of_band_delivery("hello world")

        assert consumer.final_response_sent is True

    def test_sets_final_content_delivered_flag(self):
        consumer = _make_consumer()
        assert consumer.final_content_delivered is False

        consumer.mark_out_of_band_delivery("hello world")

        assert consumer.final_content_delivered is True

    def test_records_payload_for_delivered_final_matches(self):
        """``delivered_final_matches`` must return ``True`` for the recorded
        text so ``_stream_confirmed_final_delivery`` recognises the message
        as delivered when the normal completion pipeline runs."""
        consumer = _make_consumer()
        consumer.mark_out_of_band_delivery("queued follow-up reply")

        assert consumer.delivered_final_matches("queued follow-up reply") is True

    def test_mismatch_returns_false(self):
        """If the recorded text differs from the completed ``final_response``,
        ``delivered_final_matches`` returns ``False`` (mirroring the streaming
        path's behaviour for stale preview snapshots — #71643). The caller
        then keeps the normal send so the user gets the correct answer
        instead of a phantom duplicate.
        """
        consumer = _make_consumer()
        consumer.mark_out_of_band_delivery("first guess")

        assert consumer.delivered_final_matches("completed response") is False

    def test_empty_text_is_no_op(self):
        """An empty ``text`` must not poison the flags — without this guard a
        caller that hands ``first_response = ""`` would mark the consumer as
        delivered even though nothing was sent."""
        consumer = _make_consumer()

        consumer.mark_out_of_band_delivery("")

        assert consumer.final_response_sent is False
        assert consumer.final_content_delivered is False
"""Regression coverage for #105341 — interim-only stream consumers must not
fire the duplicate-send diagnostic.

With ``streaming.enabled: false`` and ``display.interim_assistant_messages:
true`` the gateway still builds a ``GatewayStreamConsumer`` (interim
commentary is relayed through it), but the consumer is never fed the final
reply's stream deltas. ``_run_agent_mark_streamed_delivery`` could not tell
\"consumer built only for interim messages\" from \"consumer that streamed but
lost its delivery confirmation\", so it logged a guaranteed-false-positive
``possible duplicate send`` warning once per turn.
"""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.run_turn import GatewayTurnMixin


class _InterimOnlyHarness(GatewayTurnMixin):
    """Mixin with the delivery-confirmation helper stubbed to False so the
    turn always reaches the duplicate-risk diagnostic branch."""

    def _run_agent_stream_confirmed_final_delivery(self, _sc, _final, *, previewed=False):
        return False


def _run_mark_streamed_delivery(consumer, caplog):
    harness = _InterimOnlyHarness()
    turn_ctx = SimpleNamespace(
        stream_consumer_holder=[consumer],
        source=SimpleNamespace(platform="telegram"),
        session_key="sess-105341",
    )
    with caplog.at_level("WARNING", logger="gateway.run_turn"):
        asyncio.run(
            harness._run_agent_mark_streamed_delivery(
                {"final_response": "olá"}, turn_ctx
            )
        )
    return caplog


def _consumer(stream_deltas_enabled):
    return SimpleNamespace(
        final_content_delivered=False,
        delivered_final_matches=None,
        message_id=None,
        stream_deltas_enabled=stream_deltas_enabled,
    )


def test_stream_capable_consumer_still_warns(caplog):
    """Control: a consumer that CAN receive final deltas keeps the diagnostic
    (the wecom ack-timeout case it was written for)."""
    caplog = _run_mark_streamed_delivery(_consumer(True), caplog)
    assert any("possible duplicate send" in r.message for r in caplog.records)


def test_interim_only_consumer_skips_duplicate_warning(caplog):
    """#105341: consumer created only for interim commentary (streaming off,
    interim messages on) is never fed the final's deltas — no false positive."""
    caplog = _run_mark_streamed_delivery(_consumer(False), caplog)
    assert not any("possible duplicate send" in r.message for r in caplog.records)


def test_consumer_flag_defaults_true_on_real_consumer():
    """Every construction site that can race the final send keeps the warning:
    the flag defaults to True on the real consumer."""
    from gateway.stream_consumer import GatewayStreamConsumer

    consumer = GatewayStreamConsumer(adapter=SimpleNamespace(), chat_id="c")
    assert consumer.stream_deltas_enabled is True

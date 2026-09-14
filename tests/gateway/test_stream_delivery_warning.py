"""Regression coverage for stream-delivery duplicate-send diagnostics (#110590)."""

from types import SimpleNamespace

import pytest

from gateway.run_turn import GatewayTurnMixin
from gateway.turn_context import TurnContext


async def _mark_delivery(response, consumer, caplog):
    """Run the production delivery marker with a minimal stream-consumer context."""
    context = TurnContext(
        session_key="agent:main:telegram:dm:test",
        stream_consumer_holder=[consumer],
    )
    with caplog.at_level("WARNING", logger="gateway.run"):
        await GatewayTurnMixin()._run_agent_mark_streamed_delivery(response, context)


@pytest.mark.asyncio
async def test_idle_stream_consumer_does_not_warn_about_a_duplicate_send(caplog):
    """An idle Telegram-like consumer is not evidence of a possible duplicate."""
    consumer = SimpleNamespace(
        final_response_sent=False,
        final_content_delivered=False,
    )

    await _mark_delivery(
        {"final_response": "normal final reply", "response_previewed": False},
        consumer,
        caplog,
    )

    assert not any("possible duplicate send" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_unconfirmed_preview_keeps_duplicate_send_warning(caplog):
    """Keep the diagnostic when a preview was reported but cannot be confirmed."""
    consumer = SimpleNamespace(
        final_response_sent=False,
        final_content_delivered=False,
    )

    await _mark_delivery(
        {"final_response": "normal final reply", "response_previewed": True},
        consumer,
        caplog,
    )

    assert any("possible duplicate send" in record.message for record in caplog.records)

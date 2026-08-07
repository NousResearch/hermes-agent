"""Tests for the stream-consumer flood guards.

When the model API fails (rate-limit / 429 / connection drop) the
``GatewayStreamConsumer`` may hold a partial, possibly huge, accumulated
buffer (e.g. an echoed system prompt / skill context).  Historically that
buffer was flushed to the user as a flood of split Telegram messages.

These tests pin the guard that prevents that:

* ``api_error_fn`` — when the agent's model call failed, the consumer
  suppresses the accumulated buffer and delivers ONE short clean error.

They also pin the **end-to-end wiring** the gateway relies on: the agent
exposes ``api_failed_summary`` after a terminal API failure, and the
lambda ``gateway/run.py`` uses to bridge that to the consumer returns
the summary at the moment the consumer checks it.  This guards against
regressions where the agent-state lifecycle or the gateway wiring is
broken (a direct unit test on the consumer that injects a static
``api_error_fn`` would miss those).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _make_adapter() -> MagicMock:
    """Minimal MagicMock adapter wired for send/edit/delete."""
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SimpleNamespace(
        success=True, message_id="preview_1",
    ))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(
        success=True, message_id="preview_1",
    ))
    adapter.delete_message = AsyncMock(return_value=True)
    return adapter


def _sent_texts(adapter) -> list[str]:
    texts = []
    for call in adapter.send.call_args_list:
        texts.append(call.kwargs.get("content", ""))
    for call in adapter.edit_message.call_args_list:
        texts.append(call.kwargs.get("content", ""))
    return texts


def _gateway_api_error_lambda(agent_holder):
    """Mirror the lambda wired by ``gateway/run.py`` into the consumer.

    ``gateway/run.py`` stores the agent in a one-element list and exposes
    ``api_failed_summary`` via ``getattr(agent, "api_failed_summary", None)``
    so the consumer reads a *live* view of the agent state at the moment
    of the final flush — not a snapshot taken at consumer construction.
    """
    return lambda: (
        getattr(agent_holder[0], "api_failed_summary", None)
        if agent_holder and agent_holder[0] is not None
        else None
    )


class TestApiErrorSuppression:
    @pytest.mark.asyncio
    async def test_api_error_suppresses_accumulated_buffer(self):
        """On API failure the consumer sends the clean error, not the buffer.

        The buffer is held until ``got_done`` (large buffer_threshold) to
        mimic the real fallback scenario: the consumer accumulates the whole
        streamed response and would otherwise dump it as a flood of split
        messages at finalization.
        """
        adapter = _make_adapter()
        # A big buffer that looks like an echoed system prompt + skills.
        big_buffer = "SYSTEM PROMPT ... " + "skill content " * 5000
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=10_000_000),
            api_error_fn=lambda: "HTTP 429: Weekly usage limit reached.",
        )
        consumer.on_delta(big_buffer)
        consumer.finish()
        await consumer.run()

        sent = _sent_texts(adapter)
        # Exactly one message, and it is the short clean error — never the
        # raw buffer.
        assert len(sent) == 1
        assert "HTTP 429" in sent[0]
        assert big_buffer[:20] not in sent[0]

        # Delivery flags set so the gateway skips re-delivering the buffer.
        assert consumer.final_response_sent is True
        assert consumer.final_content_delivered is True
        assert consumer.already_sent is True

    @pytest.mark.asyncio
    async def test_no_api_error_keeps_legacy_behaviour(self):
        """Without an api_error_fn the consumer still delivers the buffer."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=10_000_000),
        )
        consumer.on_delta("Hello from the model")
        consumer.finish()
        await consumer.run()

        sent = _sent_texts(adapter)
        assert any("Hello from the model" in t for t in sent)
        # Delivery flags set (legacy behaviour) — no API-error short-circuit.
        assert consumer.final_response_sent is True


class TestApiErrorEndToEndWiring:
    """End-to-end coverage of the agent-state → consumer bridge.

    The agent records ``api_failed_summary`` on a terminal model failure,
    and the gateway's ``api_error_fn`` lambda reads it lazily.  This
    exercises the full chain through the real lambda shape (not a static
    callable) so a regression in either the agent-state lifecycle OR the
    gateway wiring surfaces here.
    """

    @pytest.mark.asyncio
    async def test_terminal_api_failure_is_suppressed_at_final_flush(self):
        """A large buffer accumulated BEFORE the agent reports failure is suppressed.

        Reproduces the real Telegram flood symptom: the model returns
        partial content (often echoing the system prompt / skill context)
        before the conversation loop records the terminal API error.  When
        ``got_done`` arrives, the consumer must consult the live
        ``api_failed_summary`` and suppress the buffer.
        """
        agent_holder: list = [None]

        class _StubAgent:
            # The agent's __init__ signature is irrelevant — the gateway
            # only ever reads ``api_failed_summary`` off the live object.
            api_failed_summary = None

        agent_holder[0] = _StubAgent()

        adapter = _make_adapter()
        big_buffer = "SYSTEM PROMPT ... " + "skill content " * 5000
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=10_000_000),
            api_error_fn=_gateway_api_error_lambda(agent_holder),
        )
        # Stream partial content first (mimics a model that returns some
        # output then errors mid-stream).
        consumer.on_delta(big_buffer)
        # The conversation loop's terminal error path records the summary
        # *between* the last delta and the consumer's final flush — exactly
        # when ``gateway/run.py`` is calling the lambda.
        agent_holder[0].api_failed_summary = "HTTP 429: Weekly usage limit reached."
        consumer.finish()
        await consumer.run()

        sent = _sent_texts(adapter)
        # Exactly one message, the clean error — never the raw buffer.
        assert len(sent) == 1
        assert "HTTP 429" in sent[0]
        assert big_buffer[:20] not in sent[0]
        assert consumer.final_response_sent is True
        assert consumer.final_content_delivered is True
        assert consumer.already_sent is True

    @pytest.mark.asyncio
    async def test_no_failure_does_not_suppress_streamed_content(self):
        """When the agent never reports failure, the guard does not fire.

        Guards against an over-eager guard that would suppress legitimate
        large replies: ``_send_api_error_final`` must not run, so no
        ``"⚠️ Model API error"`` text reaches the user and the consumer
        does not pre-emptively set ``_final_content_delivered`` (which
        would otherwise tell the gateway to skip its own final send).
        """
        agent_holder: list = [None]

        class _StubAgent:
            api_failed_summary = None

        agent_holder[0] = _StubAgent()

        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=10_000_000),
            api_error_fn=_gateway_api_error_lambda(agent_holder),
        )
        consumer.on_delta("Here is a complete, well-formed reply.")
        consumer.finish()
        await consumer.run()

        sent = _sent_texts(adapter)
        # The clean error must never appear — only the legitimate content.
        assert not any("Model API error" in t for t in sent)
        # And the actual reply must reach the user.
        assert any("complete, well-formed reply" in t for t in sent)

    @pytest.mark.asyncio
    async def test_summary_set_after_buffer_but_before_flush_is_seen(self):
        """A late ``api_failed_summary`` (set after the last delta) is honored.

        In the real flow, the conversation loop records the terminal error
        *during* the API call and the consumer hasn't run its final flush
        yet.  A summary that appears between the last ``on_delta`` and
        ``run()`` must still suppress the buffer.
        """
        agent_holder: list = [None]

        class _StubAgent:
            api_failed_summary = None

        agent_holder[0] = _StubAgent()

        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=10_000_000),
            api_error_fn=_gateway_api_error_lambda(agent_holder),
        )
        # Buffer accumulated cleanly; only later does the agent learn it failed.
        consumer.on_delta("looks like a normal response ... " * 500)
        agent_holder[0].api_failed_summary = (
            "All API retries exhausted with no successful response."
        )
        consumer.finish()
        await consumer.run()

        sent = _sent_texts(adapter)
        assert len(sent) == 1
        assert "retries exhausted" in sent[0]
        assert "looks like a normal response" not in sent[0]

    @pytest.mark.asyncio
    async def test_lambda_handles_missing_agent_holder(self):
        """If the agent slot is empty, the lambda returns None (no suppression).

        Defends against a regression where the gateway wires the lambda
        before the agent is created.  The consumer must fall back to its
        legacy (non-suppressing) behaviour.
        """
        agent_holder: list = [None]
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=10_000_000),
            api_error_fn=_gateway_api_error_lambda(agent_holder),
        )
        consumer.on_delta("normal response")
        consumer.finish()
        await consumer.run()

        sent = _sent_texts(adapter)
        assert any("normal response" in t for t in sent)

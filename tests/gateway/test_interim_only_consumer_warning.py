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


def _make_edit_adapter():
    """Editable adapter with no draft or native streaming (Feishu's shape)."""
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    A = type("EditOnlyAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 8000})
    A.__abstractmethods__ = frozenset()
    a = A.__new__(A)
    a._typing_paused = set()
    a._fatal_error_message = None
    a.sent = []

    async def _send(chat_id, content, reply_to=None, metadata=None, **kw):
        a.sent.append(content)
        return SendResult(success=True, message_id="m1")
    a.send = _send

    async def _edit(chat_id, message_id, content, **kw):
        a.sent.append(content)
        return SendResult(success=True, message_id=message_id)
    a.edit_message = _edit
    return a


def test_consumer_that_streamed_nothing_skips_duplicate_warning(caplog):
    """#127395: a stream-capable consumer that received no deltas this turn never
    sends, so the gateway's normal final send is the only send — no false positive."""
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    adapter = _make_edit_adapter()
    sc = GatewayStreamConsumer(
        adapter, "oc_chat", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
    )
    final = "Here is the answer you asked for."

    async def _turn():
        task = asyncio.create_task(sc.run())
        sc.finish(final)
        await asyncio.wait_for(task, 2)

    asyncio.run(_turn())
    assert adapter.sent == []
    assert sc.stream_deltas_enabled is True

    turn_ctx = SimpleNamespace(
        stream_consumer_holder=[sc],
        source=SimpleNamespace(platform="feishu"),
        session_key="sess-127395",
    )
    response = {"final_response": final}
    with caplog.at_level("WARNING", logger="gateway.run_turn"):
        asyncio.run(GatewayTurnMixin()._run_agent_mark_streamed_delivery(response, turn_ctx))
    assert "already_sent" not in response
    assert not any("possible duplicate send" in r.message for r in caplog.records)

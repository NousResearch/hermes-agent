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
from gateway.stream_consumer import GatewayStreamConsumer


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


def _surface_consumer(has_surface):
    """REAL GatewayStreamConsumer fed the final's deltas whose delivery signals
    stayed unset; only its visible-delivery surface varies (#127395). A real
    consumer — not a constant — so the property body itself executes."""
    sc = GatewayStreamConsumer(adapter=SimpleNamespace(), chat_id="chat-127395")
    if has_surface:
        # Enough1122's frozen-preview teardown: a landed preview keeps its id in
        # _preview_message_ids while _message_id is cleared (the _split_first_send /
        # _enter_fallback_mode pattern) — the preview is still on screen.
        sc._track_preview_id("om_123")
        sc._message_id = None
    return sc


def test_no_visible_surface_skips_duplicate_warning(caplog):
    """#127395: deltas were fed but NOTHING was ever shown (no preview message, no
    send, no open native bubble, no finalized segment) — e.g. an adapter whose
    streaming path has no ack semantics. The normal final send is the only
    delivery, so the diagnostic fires once per turn as a guaranteed false alarm."""
    caplog = _run_mark_streamed_delivery(_surface_consumer(False), caplog)
    assert not any("possible duplicate send" in r.message for r in caplog.records)


def test_open_preview_surface_still_warns(caplog):
    """Control (#127395): a consumer whose preview is still on screen after teardown
    keeps the diagnostic — the frozen preview next to the normal final send is
    exactly the duplicate risk it was written for."""
    caplog = _run_mark_streamed_delivery(_surface_consumer(True), caplog)
    assert any("possible duplicate send" in r.message for r in caplog.records)


# --- The property itself, against a real consumer (a SimpleNamespace constant
# would leave the body unexecuted: replacing it with `return False` stayed green).


def test_property_fresh_consumer_reads_no_surface():
    """A consumer that never showed anything reads as no visible surface."""
    assert _surface_consumer(False).has_visible_delivery_surface is False


def test_property_frozen_preview_after_teardown_reads_surface():
    """A landed preview whose _message_id was cleared by teardown stays on screen:
    _preview_message_ids is the record of visible previews (what _stale_preview_ids
    acts on), so the property must still report a visible delivery surface."""
    sc = _surface_consumer(True)
    assert sc._stale_preview_ids() == {"om_123"}  # the preview IS still on screen
    assert sc.has_visible_delivery_surface is True


def test_property_silence_marker_teardown_reads_no_surface():
    """_suppress_silence_marker deletes the previews AND clears _preview_message_ids
    with them — that teardown must keep reading as no surface."""
    sc = _surface_consumer(True)
    sc._preview_message_ids = set()  # _suppress_silence_marker's cleanup
    assert sc.has_visible_delivery_surface is False


def _async_stub(result):
    async def _call(*_args, **_kwargs):
        return result
    return _call


def test_boundary_send_fallback_records_landed_preview():
    """#127409 (Enough1122): the boundary send() fallback lands the pre-prompt text
    on screen but recorded nothing, so the duplicate-risk diagnostic read the
    consumer as 'no visible surface' — the same class of gap as the frozen preview.
    The landed message is now tracked like any other preview."""
    adapter = SimpleNamespace(
        send_stream_frame=_async_stub(None),  # finalize not confirmed (falsy frame)
        send=_async_stub(SimpleNamespace(success=True, message_id="om_boundary")),
    )
    sc = GatewayStreamConsumer(adapter=adapter, chat_id="chat-127409")

    async def scenario():
        return await sc._finalize_boundary_stream("Approval")

    assert asyncio.run(scenario()) is True
    assert sc._preview_message_ids == {"om_boundary"}
    assert sc.has_visible_delivery_surface is True

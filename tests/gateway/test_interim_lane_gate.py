"""Regression coverage for #117272 — a platform that supports neither message
editing nor native streaming never gets a stream consumer, so the interim lane's
plain-send fallback delivered completed agent messages (including the turn's
final answer) as status sends the turn-final dedup cannot see: every long reply
arrived twice on Weixin.

The gate: with no consumer there is no interim lane at all, and the fallback
inside ``interim_assistant_cb`` is removed — a completed message must never be
sent outside the delivery ledger.
"""

import logging
from types import SimpleNamespace

from gateway.config import StreamingConfig
from gateway.run_turn import GatewayTurnMixin
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext


class _NonEditableAdapter:
    """Weixin-shaped: no message editing, no native streaming transport."""

    SUPPORTS_MESSAGE_EDITING = False
    SUPPORTS_NATIVE_STREAMING = False


class _EditableAdapter:
    """Telegram-shaped control: editable, so the interim-only consumer (#105341) still builds."""

    SUPPORTS_MESSAGE_EDITING = True


class _Runner(GatewayTurnMixin):
    """Minimal runner driving the real ``_build_stream_consumer_config`` logic —
    the non-editable RuntimeError path is exactly what the gate protects."""

    def __init__(self, adapter):
        self.config = SimpleNamespace(streaming=StreamingConfig())  # streaming off (default)
        self._adapter = adapter

    def _delivery_adapter_for(self, source):
        return self._adapter


def _turn_runner(adapter):
    ctx = TurnContext(
        source=SimpleNamespace(platform="weixin", chat_id="chat-117272"),
        interim_assistant_messages_enabled=True,
        resolve_display_setting=lambda *args, **kwargs: None,  # follow the global default (off)
        _run_still_current=lambda: True,
        _status_adapter=object(),  # present — the fallback send would have a target
        _status_thread_metadata={},
    )
    return TurnRunner(_Runner(adapter), ctx)


def test_non_editable_platform_disables_interim_lane(caplog):
    """No consumer (non-editable platform, streaming off) => no interim lane: the
    callback would otherwise send the final answer invisibly to the turn-final
    dedup (#117272). The gate leaves a debug trace for duplicate-delivery
    investigations."""
    tr = _turn_runner(_NonEditableAdapter())
    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        stream_consumer, _, _, want_interim = tr._setup_stream_consumer("weixin")
    assert stream_consumer is None
    assert want_interim is False
    assert any("interim lane gated off" in record.message for record in caplog.records)


def test_interim_callback_never_plain_sends_without_consumer():
    """The interim callback must have no plain-send escape hatch: any text it
    receives without a consumer would bypass the delivery ledger (#117272)."""
    tr = _turn_runner(_NonEditableAdapter())
    _, _, interim_cb, _ = tr._setup_stream_consumer("weixin")
    sends = []
    tr._send_status_text = lambda text, metadata, log_message: sends.append((text, log_message))
    interim_cb("a completed agent message", already_streamed=False)
    assert sends == []


def test_editable_platform_keeps_interim_only_lane():
    """Control (#105341): an editable platform with streaming off still builds the
    interim-only consumer and keeps the interim lane enabled."""
    tr = _turn_runner(_EditableAdapter())
    stream_consumer, _, _, want_interim = tr._setup_stream_consumer("telegram")
    assert stream_consumer is not None
    assert want_interim is True

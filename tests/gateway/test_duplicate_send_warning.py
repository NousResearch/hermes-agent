"""Regression for #111281: the 'possible duplicate send' diagnostic warning.

``GatewayTurnMixin._run_agent_mark_streamed_delivery`` must warn only when a preview
actually happened without a send. A turn with a live stream consumer but no preview and
no send is the NORMAL case — warning there fires on every turn and is pure noise.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from gateway.run_turn import GatewayTurnMixin
from gateway.turn_context import TurnContext


def _mixin() -> GatewayTurnMixin:
    return GatewayTurnMixin.__new__(GatewayTurnMixin)


def _quiet_consumer() -> SimpleNamespace:
    """A stream consumer that delivered nothing: no suppression flags set."""
    return SimpleNamespace(
        final_content_delivered=False,
        final_response_sent=False,
        has_delivered_text=lambda text: False,
        delivered_final_matches=None,
        message_id="m1",
        adapter=None,
        _turn_split_delivery=False,
    )


def _ctx(consumer: SimpleNamespace) -> TurnContext:
    return TurnContext(
        source=SimpleNamespace(chat_id="c1"),
        session_key="s1",
        stream_consumer_holder=[consumer],
    )


def _run_mark(response: dict, consumer: SimpleNamespace) -> None:
    asyncio.run(_mixin()._run_agent_mark_streamed_delivery(response, _ctx(consumer)))


def test_no_warning_when_nothing_previewed_or_sent(caplog):
    """A stream consumer with no preview and no send is the normal case: silent."""
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        _run_mark({"final_response": "hello"}, _quiet_consumer())
    assert "possible duplicate send" not in caplog.text


def test_warning_fires_when_previewed_but_not_sent(caplog):
    """A preview that never got a confirming send keeps the duplicate-risk warning."""
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        _run_mark(
            {"final_response": "hello", "response_previewed": True},
            _quiet_consumer(),
        )
    assert "possible duplicate send" in caplog.text

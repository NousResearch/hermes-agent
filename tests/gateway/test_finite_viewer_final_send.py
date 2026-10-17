"""Regression coverage for #139666 — a finite viewer's turn must not mark the
reply ``already_sent`` off the stream consumer's transport confirmation.

``hermes -z`` runs through the gateway transport with ``finite: true``: the
client drops every ``message.delta`` frame and prints only the terminal reply
(the completion frame's text). The local route's adapter acknowledges every
send, so the stream consumer finalizes with ``final_response_sent=True`` even
though the deltas reached nobody. Marking the turn ``already_sent`` on that
signal makes ``_hmwa_deliver_turn_response`` return None, the settlement
publishes ``text: None``, and one-shot consumers get an empty stdout with
exit 1 while the answer sits in the transcript.
"""

import asyncio
from types import SimpleNamespace

from gateway.run_turn import GatewayTurnMixin
from gateway.session_finite import finite_turn_scope


class _Harness(GatewayTurnMixin):
    """Only the delivery decision under test; the real confirmation predicate runs."""


def _consumer():
    # The local route's reality after a finalized stream: transport-confirmed flags
    # with no payload mismatch record (LocalSessionAdapter.send always succeeds).
    return SimpleNamespace(
        final_response_sent=True,
        final_content_delivered=True,
        delivered_final_matches=None,
        message_id="msg-139666",
        stream_deltas_enabled=True,
    )


def _turn_ctx():
    return SimpleNamespace(
        stream_consumer_holder=[_consumer()],
        source=SimpleNamespace(platform="local"),
        session_key="agent:main:local:dm:local-139666",
    )


def _mark_streamed_delivery():
    response = {"final_response": "MARCADOR-Z-9F3A"}
    asyncio.run(_Harness()._run_agent_mark_streamed_delivery(response, _turn_ctx()))
    return response


def test_finite_turn_keeps_the_terminal_reply():
    with finite_turn_scope(True):
        response = _mark_streamed_delivery()
    assert "already_sent" not in response


def test_nonfinite_turn_still_suppresses():
    response = _mark_streamed_delivery()
    assert response.get("already_sent") is True

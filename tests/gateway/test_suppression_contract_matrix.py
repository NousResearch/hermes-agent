"""Contract matrix for the gateway's final-send suppression (#82656).

The gateway skips its own final send when the stream consumer claims the turn
final already reached the user (``gateway/run.py``: ``final_response_sent`` /
``final_content_delivered``, reconciled through ``delivered_final_matches``).
Every incident in this family — #71643 (stale finalize snapshot), #78541
(payload-less multi-message split), #82656 (frozen preview left with a visible
cursor) — is the same failure: the consumer claimed delivery for text the
platform never rendered, so the corrective send was suppressed and the answer
was lost with no retry.

Each of those was fixed with a scenario test pinned to one branch of
``GatewayStreamConsumer.run()``.  ``run()``'s ``got_done`` handler now has five
sibling branches that each set the suppression flags and record a turn-final
payload, and nothing checks them as a group — a new branch (or a new early
return in ``_send_or_edit``) can reintroduce the class without failing a test.

This module pins the invariant instead of the branch:

    If the consumer offers the gateway any signal it would trust, the COMPLETE
    final text must have reached the wire.

It drives the real consumer against a matrix of adapter behaviours and asserts
the invariant for every combination, so the guarantee holds no matter which
branch a given scenario happens to take.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

CURSOR = " ▉"
PREFIX = "Ack received. Starting the deploy now,"
TAIL = " and here is the rest of the answer, generated after the last preview edit."
FULL = PREFIX + TAIL


# ---------------------------------------------------------------------------
# Adapter behaviours
# ---------------------------------------------------------------------------
#
# A behaviour maps "how many frames have rendered so far" to one of:
#   True    — the call succeeds and the frame renders
#   False   — the call fails (flood control, transport error)
#   "lie"   — the call is ACKed but the frame never renders
#
# The "lie" mode is the transport failure the #82656 report describes: an edit
# the platform accepts and then drops.  The consumer advances its bookkeeping
# from the ACK, so it has no way to know.

ALWAYS = lambda rendered: True                                   # noqa: E731
NEVER = lambda rendered: False                                   # noqa: E731
DIES_AFTER_2 = lambda rendered: rendered < 2                     # noqa: E731
LIES_AFTER_2 = lambda rendered: True if rendered < 2 else "lie"  # noqa: E731
LIES_ALWAYS = lambda rendered: "lie"                             # noqa: E731

EDIT_BEHAVIOURS = {
    "edit_always": ALWAYS,
    "edit_dies_after_2": DIES_AFTER_2,
    "edit_never": NEVER,
    "edit_lies_after_2": LIES_AFTER_2,
    "edit_lies_always": LIES_ALWAYS,
}
SEND_BEHAVIOURS = {
    "send_always": ALWAYS,
    "send_never": NEVER,
}

# Only a lying edit transport can put a claim on the wire that the consumer
# cannot audit.  Those combinations are tracked separately (see
# ``test_lying_edit_transport_is_the_open_gap``) so the honest-transport matrix
# stays a hard guarantee.
LYING_EDITS = {"edit_lies_after_2", "edit_lies_always"}


class WireAdapter(BasePlatformAdapter):
    """Adapter that records only the frames a user could actually see."""

    def __init__(self, *, edit_behaviour, send_behaviour, prefers_fresh_final):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self._edit_behaviour = edit_behaviour
        self._send_behaviour = send_behaviour
        self._prefers_fresh_final = prefers_fresh_final
        self.wire = []  # (kind, payload) for every frame that rendered
        self._next_id = 0

    def prefers_fresh_final_streaming(self, text=None) -> bool:
        return self._prefers_fresh_final

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {}

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        if self._send_behaviour(len(self.wire)) is not True:
            return SendResult(success=False, error="send rejected")
        self._next_id += 1
        self.wire.append(("send", content))
        return SendResult(success=True, message_id=f"m-{self._next_id}")

    async def edit_message(
        self, chat_id, message_id, content, *, finalize: bool = False, metadata=None
    ) -> SendResult:
        verdict = self._edit_behaviour(len(self.wire))
        if verdict is True:
            self.wire.append(("edit", content))
            return SendResult(success=True, message_id=message_id)
        if verdict == "lie":
            return SendResult(success=True, message_id=message_id)
        return SendResult(success=False, error="flood control")

    async def delete_message(self, chat_id, message_id) -> bool:
        self.wire.append(("delete", message_id))
        return True

    def rendered_complete_answer(self, final_text: str) -> bool:
        """True when some frame the platform rendered carried *final_text*."""
        return any(
            kind in ("send", "edit") and final_text.strip() in payload
            for kind, payload in self.wire
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _drive(adapter, *, interrupt: bool):
    """Stream PREFIX then TAIL, then either finish or cancel the consumer."""
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(cursor=CURSOR, edit_interval=0.0)
    )
    task = asyncio.create_task(consumer.run())
    for delta in (PREFIX, TAIL):
        consumer.on_delta(delta)
        await asyncio.sleep(0.01)
    if interrupt:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    else:
        consumer.finish()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
    return consumer


def _consumer_claims_final_delivery(consumer, final_text: str) -> bool:
    """Whether the gateway would suppress its normal final send.

    Mirrors the decision in ``gateway/run.py`` (``_stream_confirmed_final_delivery``
    plus the ``_stale_finalized`` reconciliation), which lives inside
    ``_run_agent`` and cannot be imported.  Kept deliberately small: a
    ``False`` verdict from ``delivered_final_matches`` vetoes both flags,
    anything else lets them through.
    """
    verdict = consumer.delivered_final_matches(final_text)
    if verdict is False:
        return False
    return bool(consumer.final_response_sent or consumer.final_content_delivered)


def _scenarios(*, lying_edits: bool):
    for edit_name, edit_behaviour in EDIT_BEHAVIOURS.items():
        if (edit_name in LYING_EDITS) is not lying_edits:
            continue
        for send_name, send_behaviour in SEND_BEHAVIOURS.items():
            for prefers_fresh_final in (False, True):
                for interrupt in (False, True):
                    yield pytest.param(
                        edit_behaviour,
                        send_behaviour,
                        prefers_fresh_final,
                        interrupt,
                        id=f"{edit_name}-{send_name}"
                        f"-fresh{int(prefers_fresh_final)}"
                        f"-{'interrupted' if interrupt else 'clean'}",
                    )


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "edit_behaviour,send_behaviour,prefers_fresh_final,interrupt",
    list(_scenarios(lying_edits=False)),
)
@pytest.mark.asyncio
async def test_suppression_requires_the_complete_answer_on_the_wire(
    edit_behaviour, send_behaviour, prefers_fresh_final, interrupt
):
    """No honest-transport scenario may claim delivery it cannot back up.

    This is the guarantee the #71643 / #78541 / #82656 fixes each established
    for one branch.  Asserting it across the matrix means a new ``got_done``
    branch, or a new early ``return True`` in ``_send_or_edit``, cannot
    reintroduce the class unnoticed.
    """
    adapter = WireAdapter(
        edit_behaviour=edit_behaviour,
        send_behaviour=send_behaviour,
        prefers_fresh_final=prefers_fresh_final,
    )
    consumer = await _drive(adapter, interrupt=interrupt)

    if _consumer_claims_final_delivery(consumer, FULL):
        assert adapter.rendered_complete_answer(FULL), (
            "consumer claims the turn final was delivered, but no rendered frame "
            "carried the complete answer — the gateway would suppress its final "
            f"send and lose it. flags=(response_sent={consumer.final_response_sent}, "
            f"content_delivered={consumer.final_content_delivered}) "
            f"verdict={consumer.delivered_final_matches(FULL)!r} wire={adapter.wire!r}"
        )


@pytest.mark.asyncio
async def test_no_delivery_signal_is_not_reported_as_a_duplicate_risk(caplog):
    """A consumer that delivered no final content requires the normal final send.

    Its existence alone cannot produce a duplicate: both delivery signals are
    false. The diagnostic should retain that fact without raising a duplicate
    alert for the health-check monitor.
    """
    from gateway.run import GatewayRunner

    consumer = SimpleNamespace(
        final_response_sent=False,
        final_content_delivered=False,
        delivered_final_matches=lambda final_text: False,
    )
    turn_ctx = SimpleNamespace(
        stream_consumer_holder=[consumer],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1"),
        session_key="agent:main:telegram:dm:chat-1",
    )
    response = {"final_response": "Complete final answer."}

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        await GatewayRunner._run_agent_mark_streamed_delivery(
            object.__new__(GatewayRunner), response, turn_ctx
        )

    assert response.get("already_sent") is not True
    assert "possible duplicate send" not in caplog.text
    assert "normal final send required" in caplog.text


@pytest.mark.parametrize(
    "edit_behaviour,send_behaviour,prefers_fresh_final,interrupt",
    list(_scenarios(lying_edits=True)),
)
@pytest.mark.asyncio
async def test_lying_edit_transport_is_the_open_gap(
    edit_behaviour, send_behaviour, prefers_fresh_final, interrupt
):
    """An edit ACKed but never rendered can still suppress the final send.

    ``_send_or_edit`` advances ``_last_sent_text`` from the call's return value,
    and every ``got_done`` branch records its turn-final payload from that same
    (or an even more optimistic) source.  When the transport ACKs a frame it
    drops, both the recorded payload and the acked text hold the complete
    answer while the screen still shows the cursor-suffixed preview — exactly
    the #82656 report.

    This test documents the remaining exposure rather than asserting it away:
    the invariant is checked, and the scenarios that still violate it are
    reported as expected failures.  Closing the gap turns them into passes,
    at which point this test's ``xfail`` branch stops being reached and the
    marker can be dropped along with the fix.
    """
    adapter = WireAdapter(
        edit_behaviour=edit_behaviour,
        send_behaviour=send_behaviour,
        prefers_fresh_final=prefers_fresh_final,
    )
    consumer = await _drive(adapter, interrupt=interrupt)

    claims = _consumer_claims_final_delivery(consumer, FULL)
    rendered = adapter.rendered_complete_answer(FULL)
    if claims and not rendered:
        pytest.xfail(
            "known gap (#82656): claim rests on an ACK the platform dropped; "
            f"wire={adapter.wire!r}"
        )
    assert not claims or rendered

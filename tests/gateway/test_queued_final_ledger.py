"""A queued-lane final is ledger-bracketed like the normal final send.

When a follow-up is queued behind a turn (a message typed while the agent worked, a
subagent batch reporting back), the first response is delivered by the queued lane in
``run_turn.py`` before the follow-up runs. That lane called ``adapter.send`` bare and
discarded the result: no delivery-ledger row was recorded, so a final refused there
(flood control, a transport that had just died) was lost for good. Neither the boot
sweep nor the runtime redelivery could see it, and the follow-up ran as if the answer
had landed. On 5 Sep 2026 two long replies were lost this way within an hour, both
refused by Telegram flood control while a delegation batch arrived the same second.

Now the queued lane runs the same bracket as the normal lane: the obligation is
recorded before the send, the send goes through the adapter's retrying transport, and
the result finalizes the row. The obligation id is keyed on the raw inbound message id
exactly as the normal lane keys it; the reply anchor is only the reply target and is
None wherever replies are not used (Telegram forum topics), so it cannot identify the
turn. The reply is marked notify-worthy like every other final. The reconcile-by-edit
path is unchanged. Adapters without the base contract and sends without a session key
keep the plain send.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult

SESSION_KEY = "agent:main:telegram:dm:5230977008"
TOPIC_SESSION_KEY = "agent:main:telegram:group:-1001:topic:7"
CHAT = "5230977008"
INBOUND_ID = "5301"
TEXT = "**Yes.** I would make room for one island trip and one Balkan trip."


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (os.getpid(), 202))
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)
    yield


_COLUMNS = ("obligation_id", "session_key", "state", "attempts", "last_error", "content", "chat_id",
            "platform")


def _rows():
    with dl._connect() as conn:
        cur = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM delivery_obligations")
        return [dict(zip(_COLUMNS, r)) for r in cur.fetchall()]


def _source(*, chat_id=CHAT, thread_id=None, chat_type="dm"):
    return SimpleNamespace(platform=Platform.TELEGRAM, chat_id=chat_id, thread_id=thread_id,
                           chat_type=chat_type)


def _telegram_adapter(send_result=None):
    """A real Telegram adapter (the base contract) whose transport send is a mock."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    runner = MagicMock()
    runner._adapter_for_source = MagicMock(return_value=adapter)
    runner._schedule_flood_redelivery = MagicMock(return_value=187.0)
    adapter.gateway_runner = runner
    adapter.send = AsyncMock(return_value=send_result or SendResult(success=True, message_id="900"))
    adapter.edit_message = AsyncMock(
        return_value=SendResult(success=False, error="message to edit not found"))
    return adapter


def _plain_adapter():
    """A relay-style double without the base contract: only ``send`` and the media helpers."""
    return SimpleNamespace(
        name="relay", extract_media=BasePlatformAdapter.extract_media,
        send=AsyncMock(return_value=SendResult(success=True, message_id="p1")))


def _runner():
    from gateway.run import GatewayRunner

    return object.__new__(GatewayRunner)


async def _deliver(adapter, *, session_key=SESSION_KEY, stream_consumer=None, metadata=None,
                   text=TEXT, source=None, anchor=INBOUND_ID, inbound_id=INBOUND_ID):
    from gateway.run import GatewayRunner

    await GatewayRunner._deliver_queued_first_response(
        _runner(), text, source=source or _source(), adapter=adapter, metadata=metadata,
        event_message_id=anchor, text_already_delivered=False, deliver_media=False,
        stream_consumer=stream_consumer, session_key=session_key, inbound_message_id=inbound_id)


# ---------------------------------------------------------------------------
# The bracket: record before the send, finalize from the result.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_delivered_queued_final_is_recorded_and_marked_delivered():
    adapter = _telegram_adapter()

    await _deliver(adapter)

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["state"] == "delivered"
    assert rows[0]["session_key"] == SESSION_KEY
    assert rows[0]["content"] == TEXT
    assert (rows[0]["platform"], rows[0]["chat_id"]) == ("telegram", CHAT)
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_row_is_already_attempting_while_the_send_is_in_flight():
    """Record BEFORE the send: a crash mid-send must leave a row the next boot can redeliver."""
    adapter = _telegram_adapter()
    seen: list = []

    async def _send(**kwargs):
        seen.append([(r["state"], r["content"]) for r in _rows()])
        return SendResult(success=True, message_id="900")

    adapter.send = AsyncMock(side_effect=_send)

    await _deliver(adapter)

    assert seen == [[("attempting", TEXT)]]


@pytest.mark.asyncio
async def test_the_obligation_id_is_the_one_the_normal_lane_would_use():
    """Same turn, same text: re-recording from either lane is idempotent, never a second row."""
    adapter = _telegram_adapter()

    await _deliver(adapter)

    assert _rows()[0]["obligation_id"] == dl.compute_obligation_id(SESSION_KEY, INBOUND_ID, TEXT)


@pytest.mark.asyncio
async def test_forum_topic_turns_are_identified_by_their_inbound_id_not_the_reply_anchor():
    """Telegram forum topics route by topic metadata and never reply, so the anchor is None for
    every message in the topic. Two turns answering with the same text must still get two rows,
    or the second would overwrite the first's outstanding obligation and retry state."""
    adapter = _telegram_adapter()
    topic = _source(chat_id="-1001", thread_id="7", chat_type="supergroup")

    await _deliver(adapter, session_key=TOPIC_SESSION_KEY, source=topic, anchor=None,
                   inbound_id="5301")
    await _deliver(adapter, session_key=TOPIC_SESSION_KEY, source=topic, anchor=None,
                   inbound_id="5302")

    ids = sorted(r["obligation_id"] for r in _rows())
    assert ids == sorted([dl.compute_obligation_id(TOPIC_SESSION_KEY, "5301", TEXT),
                          dl.compute_obligation_id(TOPIC_SESSION_KEY, "5302", TEXT)])
    assert [c.kwargs["reply_to"] for c in adapter.send.await_args_list] == [None, None]


@pytest.mark.asyncio
async def test_a_flood_refused_queued_final_stays_in_the_ledger_as_failed(caplog):
    """The row survives the refusal, so the sweeps (and a flood timer, where present) can redeliver
    it."""
    adapter = _telegram_adapter(SendResult(success=False, error="flood_control:185.0"))

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await _deliver(adapter)

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["state"] == "failed"
    assert rows[0]["last_error"] == "flood_control:185.0"
    assert any("Queued-lane final send" in r.getMessage()
               and "flood_control:185.0" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_any_other_refusal_is_recorded_as_failed_too():
    adapter = _telegram_adapter(
        SendResult(success=False, error="Forbidden: bot was blocked by the user"))

    await _deliver(adapter)

    assert _rows()[0]["state"] == "failed"
    assert "blocked" in _rows()[0]["last_error"]


@pytest.mark.asyncio
async def test_the_queued_final_is_sent_like_the_normal_final():
    """Reply anchor on the inbound message and a notify-worthy copy of the thread metadata."""
    adapter = _telegram_adapter()
    metadata = {"thread_id": "topic-7"}

    await _deliver(adapter, metadata=metadata)

    kwargs = adapter.send.await_args.kwargs
    assert kwargs["chat_id"] == CHAT
    assert kwargs["content"] == TEXT
    assert kwargs["reply_to"] == INBOUND_ID
    assert kwargs["metadata"]["notify"] is True
    assert kwargs["metadata"]["thread_id"] == "topic-7"
    assert metadata == {"thread_id": "topic-7"}  # the caller's dict is cloned, not mutated


# ---------------------------------------------------------------------------
# What is unchanged: reconcile-by-edit, plain adapters, no session key.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reconcile_by_edit_records_no_obligation():
    adapter = _telegram_adapter()
    adapter.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="sealed-9"))
    sc = SimpleNamespace(message_id="sealed-9", _turn_split_delivery=False)

    await _deliver(adapter, stream_consumer=sc)

    adapter.edit_message.assert_awaited_once()
    adapter.send.assert_not_awaited()
    assert _rows() == []


@pytest.mark.asyncio
async def test_a_failed_reconcile_edit_falls_through_to_the_bracketed_send():
    adapter = _telegram_adapter()  # edit_message fails by default
    sc = SimpleNamespace(message_id="sealed-9", _turn_split_delivery=False)

    await _deliver(adapter, stream_consumer=sc)

    adapter.send.assert_awaited_once()
    assert _rows()[0]["state"] == "delivered"


@pytest.mark.asyncio
async def test_an_adapter_without_the_base_contract_keeps_the_plain_send():
    adapter = _plain_adapter()

    await _deliver(adapter, metadata={"thread_id": "t"})

    adapter.send.assert_awaited_once_with(CHAT, TEXT, metadata={"thread_id": "t"})
    assert _rows() == []


@pytest.mark.asyncio
async def test_without_a_session_key_the_plain_send_is_kept():
    """No key, no ledger row possible: the send still happens, unbracketed, as before."""
    adapter = _telegram_adapter()

    await _deliver(adapter, session_key=None, metadata={"thread_id": "t"})

    adapter.send.assert_awaited_once_with(CHAT, TEXT, metadata={"thread_id": "t"})
    assert _rows() == []


@pytest.mark.asyncio
async def test_a_plain_send_failure_is_still_logged(caplog):
    adapter = _plain_adapter()
    adapter.send = AsyncMock(return_value=SendResult(success=False, error="flood_control:30.0"))

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await _deliver(adapter)

    assert any("Queued-lane final send" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# The call site hands the session key and the raw inbound id over.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_queued_lane_receives_the_session_key_and_inbound_id_from_the_turn():
    from gateway.run import GatewayRunner

    runner = _runner()
    runner._run_agent_stream_confirmed_final_delivery = MagicMock(return_value=False)
    runner._is_intentional_silence = MagicMock(return_value=False)
    runner._pop_post_delivery_callback = MagicMock(return_value=None)
    runner._deliver_queued_first_response = AsyncMock()
    # A forum-topic turn: no reply anchor, but a raw inbound id the ledger must be keyed on.
    turn_ctx = SimpleNamespace(
        session_key=TOPIC_SESSION_KEY, stream_consumer_holder=[None],
        source=_source(chat_id="-1001", thread_id="7", chat_type="supergroup"),
        _status_thread_metadata={"thread_id": "7"}, event_message_id=None,
        inbound_message_id=INBOUND_ID, run_generation=3)
    adapter = _telegram_adapter()

    await GatewayRunner._run_agent_deliver_first_response(
        runner, turn_ctx, adapter, {"final_response": TEXT}, {}, None)

    runner._deliver_queued_first_response.assert_awaited_once()
    kwargs = runner._deliver_queued_first_response.await_args.kwargs
    assert kwargs["session_key"] == TOPIC_SESSION_KEY
    assert kwargs["inbound_message_id"] == INBOUND_ID
    assert kwargs["event_message_id"] is None
    assert kwargs["text_already_delivered"] is False

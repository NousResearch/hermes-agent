"""A frozen streamed bubble is deleted after the ledger redelivers the final the platform refused.

Two leftovers the abandoned-preview cleanup (the parent commit) did not reach, both seen on Telegram
on 8 and 9 Sep 2026:

* the stale-finalize branch: the consumer's finalize edit had succeeded on a stale snapshot, the
  gateway's reconciliation edit was refused inside a flood window, and the normal final send put a
  second copy below the frozen bubble. That branch registered no cleanup at all.
* the redelivery path: the gateway's own final was refused too (penalties of 102 s and 269 s, both
  over the 60 s inline cap), the ledger redelivered it minutes later, and the cleanup that WAS
  registered had already stood down because the post-delivery stamp said the text did not land.
  The redelivery runs outside the turn, so nothing fired it.

Now the stale-finalize branch registers the same cleanup when its edit is refused, and a cleanup that
stands down on a refused send hands its delete to the ledger row carrying the final: base.py stamps
the obligation id on the session event beside the delivery stamp, the runner keeps a small
process-local registry keyed by obligation id, and the redelivery loop fires the registered
follow-ups once the redelivered send has landed untruncated. The guards are unchanged: a follow-up
fires only for its own obligation and only on a landed redelivery; with no ledger row (the ledger
off, a slash command, an ephemeral notice) the bubble is the reader's only copy and stays; and a
redelivery after a restart finds an empty registry, because the consumer it would clean up for is
gone with the old process.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import delivery_ledger as dl
from gateway import run as run_mod
from gateway import run_startup as run_startup_mod
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key

SESSION_KEY = "agent:main:telegram:dm:5230977008"
CHAT = "5230977008"
FINAL = "**For you, I'd start with Hostelworld Social Pass and free Meetup.** Try TripBFF too."
REFUSED = SendResult(success=False, error="flood_control:269.0")


@pytest.fixture
def scheduled(monkeypatch):
    """Capture coroutines handed to safe_schedule_threadsafe so the test awaits them deterministically."""
    captured: list = []
    monkeypatch.setattr(run_mod, "safe_schedule_threadsafe",
                        lambda coro, loop, **kw: captured.append(coro))
    return captured


def _abandoned_consumer(*, stale=("901",)):
    """The parent commit's case: every edit refused, nothing here delivered the answer."""
    consumer = SimpleNamespace(final_content_delivered=False, message_id="901", adapter=MagicMock())
    consumer.abandoned_preview_ids = MagicMock(return_value=set(stale))
    consumer.delete_abandoned_previews = AsyncMock()
    return consumer


def _stale_consumer(*, edit_result=REFUSED, edit_raises=None, message_id="901", split=False):
    """The stale-finalize case: the finalize edit landed a stale snapshot, so the recorded payload does
    not match the completed response and the gateway tries a reconciliation edit."""
    consumer = SimpleNamespace(final_content_delivered=True, message_id=message_id, adapter=MagicMock())
    consumer.delivered_final_matches = MagicMock(return_value=False)
    consumer._turn_split_delivery = split
    if edit_raises is not None:
        consumer.adapter.edit_message = AsyncMock(side_effect=edit_raises)
    else:
        consumer.adapter.edit_message = AsyncMock(return_value=edit_result)
    consumer.abandoned_preview_ids = MagicMock(return_value={"901"})
    consumer.delete_abandoned_previews = AsyncMock()
    return consumer


def _session_event(*, delivered, obligation_id=None, stamp_obligation=True):
    event = asyncio.Event()
    event._hermes_final_delivered = delivered
    if stamp_obligation:
        event._hermes_final_obligation_id = obligation_id
    return event


def _adapter(*, session_event=None):
    adapter = SimpleNamespace()
    adapter.register_post_delivery_callback = MagicMock()
    adapter._active_sessions = {}
    if session_event is not None:
        adapter._active_sessions[SESSION_KEY] = session_event
    return adapter


def _runner(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = MagicMock(return_value=adapter)
    runner._run_agent_stream_confirmed_final_delivery = MagicMock(return_value=False)
    return runner


def _turn_ctx(consumer, session_key=SESSION_KEY):
    return SimpleNamespace(
        stream_consumer_holder=[consumer],
        source=SimpleNamespace(chat_id=CHAT, thread_id=None, platform="telegram"),
        session_key=session_key,
        run_generation=3,
    )


async def _drive(runner, turn_ctx):
    """Run the real decision function; returns the response dict it decided on."""
    from gateway.run import GatewayRunner

    response = {"final_response": FINAL}
    await GatewayRunner._run_agent_mark_streamed_delivery(runner, response, turn_ctx)
    return response


def _registered_callback(adapter):
    adapter.register_post_delivery_callback.assert_called_once()
    args, kwargs = adapter.register_post_delivery_callback.call_args
    assert args[0] == SESSION_KEY
    assert kwargs["generation"] == 3
    return args[1]


def _followups(runner):
    return getattr(runner, "_redelivery_followups", {}) or {}


# ---------------------------------------------------------------------------
# The stale-finalize branch registers the cleanup when its reconciliation edit is refused.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_refused_reconciliation_edit_registers_the_cleanup(scheduled):
    consumer = _stale_consumer()
    adapter = _adapter(session_event=_session_event(delivered=True))

    response = await _drive(_runner(adapter), _turn_ctx(consumer))

    # The edit was tried and refused, so the normal final send is NOT suppressed.
    consumer.adapter.edit_message.assert_awaited_once()
    assert not response.get("already_sent")
    callback = _registered_callback(adapter)
    consumer.delete_abandoned_previews.assert_not_awaited()

    callback()
    assert len(scheduled) == 1
    await scheduled[0]
    consumer.delete_abandoned_previews.assert_awaited_once_with({"901"})


@pytest.mark.asyncio
async def test_a_reconciliation_edit_that_raises_registers_the_cleanup(scheduled):
    consumer = _stale_consumer(edit_raises=RuntimeError("socket closed"))
    adapter = _adapter(session_event=_session_event(delivered=True))

    response = await _drive(_runner(adapter), _turn_ctx(consumer))

    assert not response.get("already_sent")
    _registered_callback(adapter)()
    assert len(scheduled) == 1
    await scheduled[0]
    consumer.delete_abandoned_previews.assert_awaited_once_with({"901"})


@pytest.mark.asyncio
async def test_a_landed_reconciliation_edit_registers_nothing():
    """The edit brought the bubble up to the complete reply: it IS the answer now."""
    consumer = _stale_consumer(edit_result=SendResult(success=True, message_id="901"))
    adapter = _adapter(session_event=_session_event(delivered=True))

    response = await _drive(_runner(adapter), _turn_ctx(consumer))

    assert response.get("already_sent") is True
    adapter.register_post_delivery_callback.assert_not_called()


@pytest.mark.asyncio
async def test_a_stale_finalize_on_a_split_chain_registers_nothing():
    consumer = _stale_consumer(split=True)
    adapter = _adapter(session_event=_session_event(delivered=True))

    await _drive(_runner(adapter), _turn_ctx(consumer))

    consumer.adapter.edit_message.assert_not_awaited()
    adapter.register_post_delivery_callback.assert_not_called()


@pytest.mark.parametrize("message_id", [None, "__no_edit__"])
@pytest.mark.asyncio
async def test_a_stale_finalize_without_an_editable_message_registers_nothing(message_id):
    consumer = _stale_consumer(message_id=message_id)
    adapter = _adapter(session_event=_session_event(delivered=True))

    await _drive(_runner(adapter), _turn_ctx(consumer))

    consumer.adapter.edit_message.assert_not_awaited()
    adapter.register_post_delivery_callback.assert_not_called()


# ---------------------------------------------------------------------------
# A cleanup that stands down on a refused send hands its delete to the ledger redelivery.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_refused_final_with_a_ledger_row_defers_the_delete_to_the_redelivery(scheduled):
    from gateway.run import GatewayRunner

    consumer = _abandoned_consumer()
    adapter = _adapter(session_event=_session_event(delivered=False, obligation_id="ob-1"))
    runner = _runner(adapter)

    await _drive(runner, _turn_ctx(consumer))
    _registered_callback(adapter)()

    # Nothing is deleted while the reader has only the frozen bubble.
    assert scheduled == []
    consumer.delete_abandoned_previews.assert_not_awaited()
    assert set(_followups(runner)) == {"ob-1"}

    fired = GatewayRunner._fire_redelivery_followups(runner, "ob-1")

    assert fired == 1
    assert len(scheduled) == 1
    await scheduled[0]
    consumer.delete_abandoned_previews.assert_awaited_once_with({"901"})
    # One-shot: the row is gone from the registry once fired.
    assert _followups(runner) == {}


@pytest.mark.parametrize("session_event", [
    _session_event(delivered=False, obligation_id=None),
    _session_event(delivered=False, obligation_id=""),
    _session_event(delivered=False, stamp_obligation=False),
], ids=["stamp-none", "stamp-empty", "stamp-absent"])
@pytest.mark.asyncio
async def test_a_refused_final_without_a_ledger_row_stands_down(scheduled, session_event):
    """No row means nothing will ever replace the bubble: it is the reader's only copy and stays."""
    consumer = _abandoned_consumer()
    adapter = _adapter(session_event=session_event)
    runner = _runner(adapter)

    await _drive(runner, _turn_ctx(consumer))
    _registered_callback(adapter)()

    assert scheduled == []
    assert _followups(runner) == {}
    consumer.delete_abandoned_previews.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_follow_up_fires_only_for_its_own_obligation(scheduled):
    from gateway.run import GatewayRunner

    consumer = _abandoned_consumer()
    adapter = _adapter(session_event=_session_event(delivered=False, obligation_id="ob-1"))
    runner = _runner(adapter)
    await _drive(runner, _turn_ctx(consumer))
    _registered_callback(adapter)()

    assert GatewayRunner._fire_redelivery_followups(runner, "ob-2") == 0
    assert GatewayRunner._fire_redelivery_followups(runner, "") == 0

    assert scheduled == []
    assert set(_followups(runner)) == {"ob-1"}


@pytest.mark.asyncio
async def test_the_stale_finalize_branch_also_defers_on_a_refused_final(scheduled):
    """The 8 Sep shape end to end at the decision layer: reconciliation edit refused, then the normal
    final refused as well, then the ledger redelivers."""
    from gateway.run import GatewayRunner

    consumer = _stale_consumer()
    adapter = _adapter(session_event=_session_event(delivered=False, obligation_id="ob-1"))
    runner = _runner(adapter)

    response = await _drive(runner, _turn_ctx(consumer))
    assert not response.get("already_sent")
    _registered_callback(adapter)()
    assert scheduled == []

    assert GatewayRunner._fire_redelivery_followups(runner, "ob-1") == 1
    await scheduled[0]
    consumer.delete_abandoned_previews.assert_awaited_once_with({"901"})


def test_the_registry_is_bounded(monkeypatch):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    clock = [1000.0]
    monkeypatch.setattr(run_startup_mod.time, "monotonic", lambda: clock[0])
    cap = run_startup_mod._REDELIVERY_FOLLOWUP_CAP

    for i in range(cap + 1):
        assert GatewayRunner._register_redelivery_followup(runner, f"ob-{i}", MagicMock()) is True
    # Past the cap the oldest registration goes first.
    assert len(_followups(runner)) == cap
    assert "ob-0" not in _followups(runner)
    assert f"ob-{cap}" in _followups(runner)

    # A second follow-up for the same row joins it rather than displacing anything.
    GatewayRunner._register_redelivery_followup(runner, f"ob-{cap}", MagicMock())
    assert len(_followups(runner)) == cap
    assert len(_followups(runner)[f"ob-{cap}"][1]) == 2

    # Entries older than the ledger's own stale cutoff are dropped on the next registration.
    clock[0] += run_startup_mod._REDELIVERY_FOLLOWUP_TTL_SECONDS + 1
    GatewayRunner._register_redelivery_followup(runner, "ob-fresh", MagicMock())
    assert set(_followups(runner)) == {"ob-fresh"}

    # Nothing registers without an id or a callable.
    assert GatewayRunner._register_redelivery_followup(runner, "", MagicMock()) is False
    assert GatewayRunner._register_redelivery_followup(runner, "ob-x", None) is False


def test_a_failing_follow_up_never_breaks_the_redelivery():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    ok = MagicMock()
    GatewayRunner._register_redelivery_followup(runner, "ob-1", MagicMock(side_effect=RuntimeError("boom")))
    GatewayRunner._register_redelivery_followup(runner, "ob-1", ok)

    assert GatewayRunner._fire_redelivery_followups(runner, "ob-1") == 1
    ok.assert_called_once()


# ---------------------------------------------------------------------------
# The redelivery loop fires follow-ups only for a landed, untruncated redelivery.
# ---------------------------------------------------------------------------

def _redelivery_runner(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: "default"
    return runner


def _claimed_row(oid, *, content="the final answer"):
    return {"obligation_id": oid, "session_key": SESSION_KEY, "platform": "telegram", "chat_id": CHAT,
            "thread_id": None, "content": content, "needs_marker": True, "marker": dl.FLOOD_MARKER,
            "attempts": 1}


def _record(oid, *, content="the final answer"):
    dl.record_obligation(obligation_id=oid, session_key=SESSION_KEY, platform="telegram", chat_id=CHAT,
                         thread_id=None, content=content, adapter_profile=None)
    dl.mark_attempting(oid)
    dl.mark_failed(oid, "flood_control:102.0")


@pytest.mark.parametrize("result, fires", [
    (SendResult(success=True, message_id="55"), True),
    (SendResult(success=False, error="flood_control:34.0"), False),
    (SendResult(success=True, message_id="55", truncated=True), False),
], ids=["landed", "refused-again", "landed-truncated"])
@pytest.mark.asyncio
async def test_the_redelivery_loop_fires_the_follow_up_only_when_the_send_landed(result, fires):
    from gateway.run import GatewayRunner

    _record("ob-1")
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=result)
    runner = _redelivery_runner(adapter)
    followup = MagicMock()
    GatewayRunner._register_redelivery_followup(runner, "ob-1", followup)

    redelivered = await runner._redeliver_claimed_obligations([_claimed_row("ob-1")])

    assert adapter.send.call_args.kwargs["content"].startswith(dl.FLOOD_MARKER)
    if fires:
        assert redelivered == 1
        followup.assert_called_once()
        assert _followups(runner) == {}
    else:
        followup.assert_not_called()
        # Still registered: a later attempt may land, and a truncated one is not a replacement.
        assert set(_followups(runner)) == {"ob-1"}


@pytest.mark.asyncio
async def test_a_redelivery_with_no_follow_up_registered_is_unchanged():
    _record("ob-1")
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="55"))
    runner = _redelivery_runner(adapter)

    assert await runner._redeliver_claimed_obligations([_claimed_row("ob-1")]) == 1
    assert _followups(runner) == {}


# ---------------------------------------------------------------------------
# base.py: the ledger bracket stamps the row on the session event; the hook clears it.
# ---------------------------------------------------------------------------

class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _stub_adapter():
    return _StubAdapter(PlatformConfig(enabled=True, token="t", typing_indicator=False), Platform.TELEGRAM)


def _session(adapter):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=CHAT, chat_type="dm")
    session_key = build_session_key(source)
    session_event = asyncio.Event()
    session_event._hermes_run_generation = 3
    adapter._active_sessions[session_key] = session_event
    return source, session_key, session_event


@pytest.mark.parametrize("recorded", ["ob-1", None], ids=["ledgered", "not-ledgered"])
@pytest.mark.asyncio
async def test_the_ledger_bracket_stamps_the_row_and_the_hook_clears_it(recorded):
    adapter = _stub_adapter()
    adapter._record_delivery_obligation = AsyncMock(return_value=recorded)
    adapter._finalize_delivery_obligation = AsyncMock()
    adapter._send_with_retry = AsyncMock(return_value=REFUSED)
    source, session_key, session_event = _session(adapter)
    # A stale value from an earlier turn on the same event must never survive into this one.
    session_event._hermes_final_obligation_id = "ob-stale"
    seen = []
    adapter.register_post_delivery_callback(
        session_key, lambda: seen.append(getattr(session_event, "_hermes_final_obligation_id", "unset")),
        generation=3)
    adapter._message_handler = AsyncMock(return_value=FINAL)

    await adapter._process_message_background(
        MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="123"), session_key)

    assert seen == [recorded]
    assert session_event._hermes_final_obligation_id is None


@pytest.mark.asyncio
async def test_end_to_end_a_refused_final_is_cleaned_up_once_the_ledger_redelivers_it(scheduled):
    """The 9 Sep incident in miniature, on the real ledger: the consumer gave up, the gateway's own
    final was refused with a penalty over the inline cap, the ledger redelivered it later."""
    from gateway.run import GatewayRunner

    adapter = _stub_adapter()
    adapter._send_with_retry = AsyncMock(return_value=SendResult(
        success=False, error="Flood control exceeded. Retry in 102 seconds"))
    source, session_key, session_event = _session(adapter)
    consumer = _abandoned_consumer()
    turn_ctx = _turn_ctx(consumer, session_key=session_key)
    runner = _redelivery_runner(adapter)
    runner._adapter_for_source = MagicMock(return_value=adapter)

    async def _handler(_event):
        GatewayRunner._run_agent_schedule_abandoned_preview_cleanup(runner, consumer, source, turn_ctx, False)
        return FINAL

    adapter._message_handler = _handler
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="123")

    await adapter._process_message_background(event, session_key)

    # Refused: the ledger holds the final as failed, the cleanup stood down and deferred to that row.
    oid = dl.compute_obligation_id(session_key, "123", FINAL)
    with dl._connect() as conn:
        state = conn.execute("SELECT state FROM delivery_obligations WHERE obligation_id=?", (oid,)).fetchone()[0]
    assert state == "failed"
    assert scheduled == []
    consumer.delete_abandoned_previews.assert_not_awaited()
    assert set(_followups(runner)) == {oid}

    # The penalty passes and the ledger redelivers through the runner's loop.
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="56"))
    row = _claimed_row(oid, content=FINAL)
    assert await runner._redeliver_claimed_obligations([row]) == 1

    assert adapter.send.call_args.kwargs["content"] == dl.FLOOD_MARKER + FINAL
    assert len(scheduled) == 1
    await scheduled[0]
    consumer.delete_abandoned_previews.assert_awaited_once_with({"901"})
    assert _followups(runner) == {}


# ---------------------------------------------------------------------------
# The row stamp is bound to the turn's OWN final send, never inherited.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_turn_whose_own_final_never_went_out_reads_no_row():
    """The stamp comes from THIS task's own final send and nowhere else. A queued-lane send inside the
    handler (an earlier message's reply, through the public bracket) and a stale value left on the
    shared event must not become this turn's row: the reply to message A must never be allowed to
    delete message B's only preview when B's own final was suppressed or never produced."""
    adapter = _stub_adapter()
    adapter._record_delivery_obligation = AsyncMock(return_value="ob-A")
    adapter._finalize_delivery_obligation = AsyncMock()
    adapter._send_with_retry = AsyncMock(return_value=REFUSED)
    source, session_key, session_event = _session(adapter)
    session_event._hermes_final_obligation_id = "ob-stale"
    seen = []
    adapter.register_post_delivery_callback(
        session_key, lambda: seen.append(getattr(session_event, "_hermes_final_obligation_id", "unset")),
        generation=3)

    async def _handler(event):
        # The queued lane delivers an EARLIER message's reply mid-turn, and it is refused...
        result, _sender = await adapter.send_final_ledgered(
            event, session_key, "the reply to A", {}, reply_to=None, is_ephemeral_response=False)
        assert result is REFUSED
        assert adapter._record_delivery_obligation.await_count == 1
        # ...while this turn's own final never goes out (interrupted, empty, suppressed).
        return None

    adapter._message_handler = _handler
    await adapter._process_message_background(
        MessageEvent(text="B", message_type=MessageType.TEXT, source=source, message_id="124"), session_key)

    assert seen == [None]
    assert session_event._hermes_final_obligation_id is None


@pytest.mark.asyncio
async def test_a_cancelled_hook_still_clears_the_row_stamp():
    """A callback cancelled mid-await must not leave this firing's row on the shared event."""
    adapter = _stub_adapter()
    _source, session_key, session_event = _session(adapter)
    started = asyncio.Event()

    async def _hang():
        started.set()
        await asyncio.sleep(3600)

    adapter.register_post_delivery_callback(session_key, _hang, generation=3)
    task = asyncio.create_task(adapter._fire_post_delivery_callback(
        session_key, session_event, delivered=False, obligation_id="ob-1"))
    await started.wait()
    assert session_event._hermes_final_obligation_id == "ob-1"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert session_event._hermes_final_obligation_id is None


@pytest.mark.asyncio
async def test_the_public_bracket_keeps_its_result_and_adapter_contract():
    """The queued lane unpacks ``(result, adapter)`` from ``send_final_ledgered``; the row id travels
    only on the private path the normal lane uses."""
    adapter = _stub_adapter()
    adapter._record_delivery_obligation = AsyncMock(return_value="ob-1")
    adapter._finalize_delivery_obligation = AsyncMock()
    sent = SendResult(success=True, message_id="9")
    adapter._send_with_retry = AsyncMock(return_value=sent)
    source, session_key, _ = _session(adapter)
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="123")

    assert await adapter.send_final_ledgered(
        event, session_key, FINAL, {}, reply_to=None, is_ephemeral_response=False) == (sent, adapter)
    assert await adapter._send_final_text(event, session_key, FINAL, {}, False, 0, lambda _r: None) == "ob-1"


# ---------------------------------------------------------------------------
# Registry timing and expiry.
# ---------------------------------------------------------------------------

def test_a_follow_up_registered_after_the_row_already_landed_runs_at_once():
    """The reconnect sweep redelivers inside the refused send's own finalization, before the turn's
    post-delivery callback registers: by the time the follow-up arrives the row is delivered."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    assert GatewayRunner._fire_redelivery_followups(runner, "ob-1") == 0
    late = MagicMock()

    assert GatewayRunner._register_redelivery_followup(runner, "ob-1", late) is True

    late.assert_called_once()
    assert _followups(runner) == {}
    # Another row is unaffected.
    other = MagicMock()
    GatewayRunner._register_redelivery_followup(runner, "ob-2", other)
    other.assert_not_called()
    assert set(_followups(runner)) == {"ob-2"}


def test_expiry_releases_follow_ups_without_another_registration(monkeypatch):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    clock = [5000.0]
    monkeypatch.setattr(run_startup_mod.time, "monotonic", lambda: clock[0])
    GatewayRunner._register_redelivery_followup(runner, "ob-1", MagicMock())
    assert set(_followups(runner)) == {"ob-1"}

    clock[0] += run_startup_mod._REDELIVERY_FOLLOWUP_TTL_SECONDS + 1
    GatewayRunner._prune_redelivery_followups(runner)

    assert _followups(runner) == {}


def test_a_new_row_arms_its_own_expiry_timer(monkeypatch):
    from gateway.run import GatewayRunner

    loop = MagicMock()
    monkeypatch.setattr(run_startup_mod.asyncio, "get_running_loop", lambda: loop)
    runner = object.__new__(GatewayRunner)

    GatewayRunner._register_redelivery_followup(runner, "ob-1", MagicMock())
    GatewayRunner._register_redelivery_followup(runner, "ob-1", MagicMock())  # same row: one timer

    loop.call_later.assert_called_once()
    delay, fn = loop.call_later.call_args.args[:2]
    assert delay > run_startup_mod._REDELIVERY_FOLLOWUP_TTL_SECONDS
    assert fn == runner._prune_redelivery_followups

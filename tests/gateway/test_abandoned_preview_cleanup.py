"""A preview the stream consumer abandoned is deleted once the gateway's own final lands.

The consumer deletes the previews it replaces on its fresh-final and fallback paths. It does not
own the case where it gives up entirely and the GATEWAY sends the final instead, and nothing
cleaned up behind that: on 6 Sep 2026 four preview edits were refused inside a Telegram flood
window (9s waits against a 5s inline cap, so each failed closed), the preview froze mid-render,
and the complete 2888-character reply arrived six seconds later as a separate message. The reader
was left with a truncated preview above the real answer, showing raw MarkdownV2 markers and the
streaming cursor.

Three guards keep the cleanup from ever removing the reader's only copy of the answer:

* skipped when the stream did deliver the content (those previews ARE the reply);
* segment-only ids, so finalized earlier segments (delivered preambles) survive;
* the delete runs from the post-delivery callback, which fires whether or not the final send
  succeeded, so it reads the outcome base.py stamps on the session event (did the final TEXT land,
  by its own send or as a TTS caption; audio alone does not count) and stands down otherwise. That
  stamp is bound to the finishing turn: a queued follow-up shares the session event, so the hook
  fires before the drain hand-off rather than from ``finally``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import run as run_mod
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key

SESSION_KEY = "agent:main:telegram:dm:5230977008"
CHAT = "5230977008"
FINAL = "**No, the 9 euro is admission to the whole fair, not just the one-hour lesson.**"


@pytest.fixture
def scheduled(monkeypatch):
    """Capture coroutines handed to safe_schedule_threadsafe so the test awaits them deterministically."""
    captured: list = []
    monkeypatch.setattr(run_mod, "safe_schedule_threadsafe",
                        lambda coro, loop, **kw: captured.append(coro))
    return captured


def _consumer(*, stale=("901",), delivered=False, with_seam=True):
    consumer = SimpleNamespace(
        final_content_delivered=delivered,
        message_id="901",
        adapter=MagicMock(),
    )
    if with_seam:
        consumer.abandoned_preview_ids = MagicMock(return_value=set(stale))
        consumer.delete_abandoned_previews = AsyncMock()
    return consumer


def _session_event(*, delivered=True, stamped=True):
    event = asyncio.Event()
    if stamped:
        event._hermes_final_delivered = delivered
    return event


def _adapter(*, with_hook=True, session_event=None):
    adapter = SimpleNamespace()
    if with_hook:
        adapter.register_post_delivery_callback = MagicMock()
    adapter._active_sessions = {}
    if session_event is not None:
        adapter._active_sessions[SESSION_KEY] = session_event
    return adapter


def _runner(adapter, *, streamed=False):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = MagicMock(return_value=adapter)
    runner._run_agent_stream_confirmed_final_delivery = MagicMock(return_value=streamed)
    return runner


def _turn_ctx(consumer, session_key=SESSION_KEY):
    return SimpleNamespace(
        stream_consumer_holder=[consumer],
        source=SimpleNamespace(chat_id=CHAT, thread_id=None, platform="telegram"),
        session_key=session_key,
        run_generation=3,
    )


async def _drive(runner, turn_ctx):
    """Run the real decision function down to the abandoned-preview branch."""
    from gateway.run import GatewayRunner

    await GatewayRunner._run_agent_mark_streamed_delivery(
        runner, {"final_response": FINAL}, turn_ctx)


def _registered_callback(adapter):
    adapter.register_post_delivery_callback.assert_called_once()
    args, kwargs = adapter.register_post_delivery_callback.call_args
    assert args[0] == SESSION_KEY
    assert kwargs["generation"] == 3
    return args[1]


# ---------------------------------------------------------------------------
# The branch registers the cleanup; a landed final lets it delete the stale previews.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_abandoned_preview_is_deleted_after_the_replacement_lands(scheduled):
    consumer = _consumer()
    adapter = _adapter(session_event=_session_event(delivered=True))
    turn_ctx = _turn_ctx(consumer)

    await _drive(_runner(adapter), turn_ctx)

    callback = _registered_callback(adapter)
    # Nothing is deleted until the callback fires: it fires after the final send.
    consumer.delete_abandoned_previews.assert_not_awaited()

    callback()
    assert len(scheduled) == 1
    await scheduled[0]

    consumer.delete_abandoned_previews.assert_awaited_once_with({"901"})


@pytest.mark.asyncio
async def test_nothing_is_deleted_inline_before_the_final_send(scheduled):
    """The branch runs before the normal final send, so an inline delete could leave the reader with
    neither the preview nor a replacement."""
    consumer = _consumer()
    adapter = _adapter(session_event=_session_event(delivered=True))

    await _drive(_runner(adapter), _turn_ctx(consumer))

    assert scheduled == []
    consumer.delete_abandoned_previews.assert_not_awaited()


@pytest.mark.parametrize("session_event", [
    pytest.param(_session_event(delivered=False), id="final-send-failed"),
    pytest.param(_session_event(stamped=False), id="no-stamp"),
    pytest.param(None, id="no-active-session"),
])
@pytest.mark.asyncio
async def test_a_final_that_did_not_land_keeps_the_preview(scheduled, session_event):
    """The post-delivery hook fires from ``finally`` even when the final send failed. Without the
    delivered stamp the frozen preview is all the reader has, so the callback must stand down."""
    consumer = _consumer()
    adapter = _adapter(session_event=session_event)

    await _drive(_runner(adapter), _turn_ctx(consumer))
    _registered_callback(adapter)()

    assert scheduled == []
    consumer.delete_abandoned_previews.assert_not_awaited()


# ---------------------------------------------------------------------------
# Guards.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_previews_that_hold_the_delivered_answer_are_kept():
    """When the stream did deliver the content, those previews ARE the reply."""
    from gateway.run import GatewayRunner

    consumer = _consumer(delivered=True)
    adapter = _adapter(session_event=_session_event(delivered=True))
    turn_ctx = _turn_ctx(consumer)

    GatewayRunner._run_agent_schedule_abandoned_preview_cleanup(
        _runner(adapter), consumer, turn_ctx.source, turn_ctx, True)

    adapter.register_post_delivery_callback.assert_not_called()


@pytest.mark.asyncio
async def test_a_consumer_with_no_previews_registers_nothing():
    consumer = _consumer(stale=())
    adapter = _adapter()

    await _drive(_runner(adapter), _turn_ctx(consumer))

    adapter.register_post_delivery_callback.assert_not_called()


@pytest.mark.asyncio
async def test_a_consumer_without_the_cleanup_seam_is_ignored():
    """Relay-style consumers without the transport mixin must not raise here."""
    consumer = _consumer(with_seam=False)
    adapter = _adapter()

    await _drive(_runner(adapter), _turn_ctx(consumer))

    adapter.register_post_delivery_callback.assert_not_called()


@pytest.mark.asyncio
async def test_an_adapter_without_the_post_delivery_hook_is_ignored():
    consumer = _consumer()
    adapter = _adapter(with_hook=False)

    await _drive(_runner(adapter), _turn_ctx(consumer))

    consumer.delete_abandoned_previews.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_registration_never_breaks_the_turn():
    consumer = _consumer()
    adapter = _adapter()
    adapter.register_post_delivery_callback = MagicMock(side_effect=RuntimeError("adapter gone"))

    await _drive(_runner(adapter), _turn_ctx(consumer))

    consumer.delete_abandoned_previews.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_refused_delete_never_propagates(scheduled):
    """The cleanup is best effort: a delete that raises must not surface from the callback."""
    consumer = _consumer()
    consumer.delete_abandoned_previews = AsyncMock(side_effect=RuntimeError("flood"))
    adapter = _adapter(session_event=_session_event(delivered=True))

    await _drive(_runner(adapter), _turn_ctx(consumer))
    _registered_callback(adapter)()
    await scheduled[0]

    consumer.delete_abandoned_previews.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_stream_that_delivered_the_final_takes_the_suppression_path(scheduled):
    """Sanity check on the branch we must NOT disturb: a confirmed streamed delivery suppresses the
    normal send and registers no cleanup."""
    consumer = _consumer()
    adapter = _adapter(session_event=_session_event(delivered=True))
    response = {"final_response": FINAL}

    from gateway.run import GatewayRunner
    await GatewayRunner._run_agent_mark_streamed_delivery(
        _runner(adapter, streamed=True), response, _turn_ctx(consumer))

    assert response.get("already_sent") is True
    adapter.register_post_delivery_callback.assert_not_called()
    assert scheduled == []


# ---------------------------------------------------------------------------
# The consumer's seam: segment-only ids, and the delete goes through the retrying helper.
# ---------------------------------------------------------------------------

def _real_consumer():
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    adapter = MagicMock(spec=BasePlatformAdapter)
    adapter.delete_message = AsyncMock(return_value=True)
    return GatewayStreamConsumer(adapter=adapter, chat_id=CHAT, config=StreamConsumerConfig())


def test_finalized_earlier_segments_are_not_abandoned():
    """A tool boundary finalizes the segment before it and resets the per-segment state, but the
    turn-wide preview set keeps those ids. Handing that set over would delete delivered preambles
    whose text is absent from the final answer."""
    consumer = _real_consumer()

    # Segment 1 streamed, landed and was finalized at a tool boundary.
    consumer._message_id = "801"
    consumer._preview_message_ids.add("801")
    consumer._segment_preview_message_ids.add("801")
    consumer._reset_segment_state()
    assert consumer._message_id is None and "801" in consumer._preview_message_ids

    # Segment 2 is the one the consumer gave up on.
    consumer._message_id = "901"
    consumer._preview_message_ids.add("901")
    consumer._segment_preview_message_ids.add("901")

    assert consumer.abandoned_preview_ids() == {"901"}


def test_a_split_chain_is_never_handed_over():
    """After an oversized split the sealed head chunks hold delivered text, and a long final may be
    capped by the adapter (message count or length) while still reporting success. Nothing is deleted."""
    consumer = _real_consumer()
    consumer._message_id = "901"
    consumer._preview_message_ids.add("901")
    consumer._segment_preview_message_ids.add("901")
    consumer._turn_split_delivery = True

    assert consumer.abandoned_preview_ids() == set()


def test_several_bubbles_in_one_segment_are_never_handed_over():
    """Several bubbles can hold more text than a capped final delivered; only a single frozen bubble
    is provably covered by any successful final send."""
    consumer = _real_consumer()
    consumer._message_id = "902"
    for mid in ("901", "902"):
        consumer._preview_message_ids.add(mid)
        consumer._segment_preview_message_ids.add(mid)

    assert consumer.abandoned_preview_ids() == set()


def test_the_no_edit_sentinel_is_never_an_abandoned_preview():
    consumer = _real_consumer()
    consumer._message_id = "__no_edit__"

    assert consumer.abandoned_preview_ids() == set()


@pytest.mark.asyncio
async def test_delete_abandoned_previews_requests_the_bounded_retry():
    """Telegram reports a refused delete by returning False, and the flood window that stranded the
    preview can refuse the delete too, so the existing bounded retry must be requested."""
    consumer = _real_consumer()
    consumer._delete_previews = AsyncMock()

    await consumer.delete_abandoned_previews({"901", "902"})

    consumer._delete_previews.assert_awaited_once_with(
        {"901", "902"}, label="Abandoned preview", retry_on_false=True)


# ---------------------------------------------------------------------------
# End to end through the real background processor: the delete waits for a LANDED final.
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


@pytest.mark.parametrize("final_landed", [True, False], ids=["final-landed", "final-refused"])
@pytest.mark.asyncio
async def test_end_to_end_the_delete_waits_for_a_landed_final(scheduled, final_landed):
    from gateway.run import GatewayRunner

    adapter = _StubAdapter(
        PlatformConfig(enabled=True, token="t", typing_indicator=False), Platform.TELEGRAM)
    adapter._send_with_retry = AsyncMock(return_value=SendResult(
        success=final_landed, message_id="777" if final_landed else None,
        error=None if final_landed else "Flood control exceeded. Retry in 9 seconds"))

    source = SessionSource(platform=Platform.TELEGRAM, chat_id=CHAT, chat_type="dm")
    session_key = build_session_key(source)
    session_event = asyncio.Event()
    session_event._hermes_run_generation = 3
    adapter._active_sessions[session_key] = session_event

    consumer = _consumer()
    turn_ctx = _turn_ctx(consumer, session_key=session_key)
    runner = _runner(adapter)

    async def _handler(_event):
        # What run_turn does on the abandoned-preview branch, BEFORE base.py sends the final.
        GatewayRunner._run_agent_schedule_abandoned_preview_cleanup(
            runner, consumer, source, turn_ctx, False)
        return FINAL

    adapter._message_handler = _handler
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="123")

    await adapter._process_message_background(event, session_key)

    adapter._send_with_retry.assert_awaited_once()
    if final_landed:
        assert len(scheduled) == 1
        await scheduled[0]
        consumer.delete_abandoned_previews.assert_awaited_once_with({"901"})
    else:
        assert scheduled == []
        consumer.delete_abandoned_previews.assert_not_awaited()


# ---------------------------------------------------------------------------
# Voice replies: only the TEXT landing counts.
# ---------------------------------------------------------------------------

async def _run_voice_turn(tmp_path, *, final_text, text_send_ok):
    """A turn with auto-TTS on: the audio always lands; the text send outcome is the parameter."""
    from gateway.run import GatewayRunner

    adapter = _StubAdapter(
        PlatformConfig(enabled=True, token="t", typing_indicator=False), Platform.TELEGRAM)
    audio = tmp_path / "reply.ogg"
    audio.write_bytes(b"ogg")
    adapter._wants_auto_tts = MagicMock(return_value=True)
    adapter._synthesize_auto_tts = AsyncMock(return_value=([str(audio)], None))
    adapter.play_tts = AsyncMock(return_value=SendResult(success=True, message_id="a1"))
    adapter._send_with_retry = AsyncMock(return_value=SendResult(
        success=text_send_ok, message_id="777" if text_send_ok else None,
        error=None if text_send_ok else "Flood control exceeded. Retry in 9 seconds"))

    source = SessionSource(platform=Platform.TELEGRAM, chat_id=CHAT, chat_type="dm")
    session_key = build_session_key(source)
    session_event = asyncio.Event()
    session_event._hermes_run_generation = 3
    adapter._active_sessions[session_key] = session_event

    consumer = _consumer()
    turn_ctx = _turn_ctx(consumer, session_key=session_key)
    runner = _runner(adapter)

    async def _handler(_event):
        GatewayRunner._run_agent_schedule_abandoned_preview_cleanup(
            runner, consumer, source, turn_ctx, False)
        return final_text

    adapter._message_handler = _handler
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="123")
    await adapter._process_message_background(event, session_key)
    return adapter, consumer


@pytest.mark.asyncio
async def test_audio_that_landed_without_its_text_keeps_the_preview(scheduled, tmp_path):
    """A reply over Telegram's caption limit sends the audio bare and the text separately. If that text
    send is refused, the audio's success must not authorise deleting the reader's only text copy."""
    adapter, consumer = await _run_voice_turn(
        tmp_path, final_text="x" * 1500, text_send_ok=False)

    adapter.play_tts.assert_awaited_once()
    adapter._send_with_retry.assert_awaited_once()
    assert scheduled == []
    consumer.delete_abandoned_previews.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_delivered_tts_caption_counts_as_the_text_landing(scheduled, tmp_path):
    """A short reply rides along as the audio's caption and the text send is skipped: the reader has
    the text, so the abandoned preview goes."""
    adapter, consumer = await _run_voice_turn(tmp_path, final_text=FINAL, text_send_ok=True)

    adapter.play_tts.assert_awaited_once()
    adapter._send_with_retry.assert_not_awaited()
    assert len(scheduled) == 1
    await scheduled[0]
    consumer.delete_abandoned_previews.assert_awaited_once_with({"901"})


# ---------------------------------------------------------------------------
# Queued follow-up: the finishing turn fires ITS callback, with ITS outcome, before the hand-off.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_draining_turn_never_fires_the_next_turns_cleanup(scheduled):
    """A follow-up queued mid-turn is drained by a new task that shares the session's interrupt Event
    and re-stamps its run generation as soon as it starts. The finishing turn must pop and fire its own
    callback before that hand-off; popped from ``finally`` instead, it could be the next turn's cleanup,
    fired with this turn's delivery outcome while that turn's final send is still in flight."""
    from gateway.run import GatewayRunner

    adapter = _StubAdapter(
        PlatformConfig(enabled=True, token="t", typing_indicator=False), Platform.TELEGRAM)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=CHAT, chat_type="dm")
    session_key = build_session_key(source)
    session_event = asyncio.Event()
    session_event._hermes_run_generation = 1
    adapter._active_sessions[session_key] = session_event
    runner = _runner(adapter)

    first_consumer, second_consumer = _consumer(stale=("901",)), _consumer(stale=("902",))
    second_turn_started, release_second_send = asyncio.Event(), asyncio.Event()
    event1 = MessageEvent(text="first", message_type=MessageType.TEXT, source=source, message_id="1")
    event2 = MessageEvent(text="second", message_type=MessageType.TEXT, source=source, message_id="2")

    sends: list = []

    async def _send_with_retry(**kwargs):
        sends.append(kwargs["content"])
        if len(sends) == 1:
            return SendResult(success=True, message_id="11")  # the first turn's final lands
        await release_second_send.wait()  # the second turn's final stays in flight, then is refused
        return SendResult(success=False, error="Flood control exceeded. Retry in 9 seconds")

    adapter._send_with_retry = _send_with_retry

    # The real typing stop awaits the platform; that yield is what lets the drain task run first.
    real_stop = adapter._stop_typing_refresh

    async def _yielding_stop(*args, **kwargs):
        await asyncio.sleep(0)
        return await real_stop(*args, **kwargs)

    adapter._stop_typing_refresh = _yielding_stop

    turns: list = []

    async def _handler(event):
        turns.append(event.message_id)
        if len(turns) == 1:
            adapter._pending_messages[session_key] = event2  # a follow-up arrives mid-turn
            ctx = _turn_ctx(first_consumer, session_key=session_key)
            ctx.run_generation = 1
            GatewayRunner._run_agent_schedule_abandoned_preview_cleanup(
                runner, first_consumer, source, ctx, False)
            return "first answer"
        session_event._hermes_run_generation = 2  # what the gateway's generation binding does
        ctx = _turn_ctx(second_consumer, session_key=session_key)
        ctx.run_generation = 2
        GatewayRunner._run_agent_schedule_abandoned_preview_cleanup(
            runner, second_consumer, source, ctx, False)
        second_turn_started.set()
        return "second answer"

    adapter._message_handler = _handler

    await adapter._process_message_background(event1, session_key)
    await asyncio.wait_for(second_turn_started.wait(), 5)
    await asyncio.sleep(0)

    # The first turn's own cleanup fired, with the first turn's outcome ...
    assert len(scheduled) == 1
    await scheduled[0]
    first_consumer.delete_abandoned_previews.assert_awaited_once_with({"901"})
    # ... and the second turn's did not while its final was still in flight.
    second_consumer.delete_abandoned_previews.assert_not_awaited()

    drain_task = adapter._session_tasks.get(session_key)
    release_second_send.set()
    if drain_task is not None:
        await asyncio.wait_for(drain_task, 5)
    assert turns == ["1", "2"]
    # The second turn's final was refused, so its preview stays.
    assert len(scheduled) == 1
    second_consumer.delete_abandoned_previews.assert_not_awaited()


# ---------------------------------------------------------------------------
# The plain-text fallback in _send_with_retry: a success that carried only a prefix is not delivery.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("final_text, expect_delete", [
    pytest.param("x" * 4933, False, id="fallback-truncated-keeps-preview"),
    pytest.param("y" * 800, True, id="fallback-complete-deletes-preview"),
])
@pytest.mark.asyncio
async def test_a_truncated_plain_text_fallback_is_not_text_delivery(scheduled, final_text, expect_delete):
    """After a formatting failure, ``_send_with_retry`` falls back to plain text capped at 3500 chars and
    returns the fallback's success. For a long reply that is a prefix, and the frozen preview may hold
    text the replacement does not, so it must stay. A complete fallback is the whole reply and the
    preview goes."""
    from gateway.run import GatewayRunner

    adapter = _StubAdapter(
        PlatformConfig(enabled=True, token="t", typing_indicator=False), Platform.TELEGRAM)
    adapter.send = AsyncMock(side_effect=[
        SendResult(success=False, error="Bad Request: can't parse entities: unmatched '*'"),
        SendResult(success=True, message_id="778"),
    ])

    source = SessionSource(platform=Platform.TELEGRAM, chat_id=CHAT, chat_type="dm")
    session_key = build_session_key(source)
    session_event = asyncio.Event()
    session_event._hermes_run_generation = 3
    adapter._active_sessions[session_key] = session_event

    consumer = _consumer()
    turn_ctx = _turn_ctx(consumer, session_key=session_key)
    runner = _runner(adapter)

    async def _handler(_event):
        GatewayRunner._run_agent_schedule_abandoned_preview_cleanup(
            runner, consumer, source, turn_ctx, False)
        return final_text

    adapter._message_handler = _handler
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="123")
    await adapter._process_message_background(event, session_key)

    # The real retry helper ran: the formatted send failed, the plain-text fallback landed.
    assert adapter.send.await_count == 2
    fallback_text = adapter.send.await_args_list[1].kwargs["content"]
    assert fallback_text.startswith("(Response formatting failed, plain text:)")
    assert (len(fallback_text) < len(final_text)) is (not expect_delete)

    if expect_delete:
        assert len(scheduled) == 1
        await scheduled[0]
        consumer.delete_abandoned_previews.assert_awaited_once_with({"901"})
    else:
        assert scheduled == []
        consumer.delete_abandoned_previews.assert_not_awaited()

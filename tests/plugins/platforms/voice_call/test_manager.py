"""CallManager state machine, timers, and mock end-to-end flows."""

import asyncio

import pytest

from plugins.platforms.voice_call.config import TimeoutsConfig
from plugins.platforms.voice_call.events import (
    CallState,
    EventType,
    NormalizedEvent,
)
from plugins.platforms.voice_call.manager import CallManager, CallNotFoundError
from plugins.platforms.voice_call.providers.mock import MockProvider



async def wait_for_state(manager, call_id, state, timeout=1.0):
    """Poll until the call reaches ``state`` (events flow through tasks)."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        record = manager.get_call(call_id)
        if record is not None and record.state == state:
            return record
        await asyncio.sleep(0.01)
    record = manager.get_call(call_id)
    raise AssertionError(
        f"call {call_id} never reached {state}; now: "
        f"{record.state if record else 'gone'}"
    )


@pytest.mark.asyncio
async def test_outbound_conversation_reaches_listening(manager, provider):
    record = await manager.initiate_call("+15555550001", message="hello there")
    assert record.provider_call_id.startswith("mock-")
    await wait_for_state(manager, record.call_id, CallState.LISTENING)
    # Initial message was spoken after answer.
    assert provider.spoken == [(record.provider_call_id, "hello there")]
    assert provider.listening == [record.provider_call_id]
    assert [t.speaker for t in record.transcript] == ["bot"]


@pytest.mark.asyncio
async def test_outbound_notify_speaks_and_hangs_up(vc_config, provider, store):
    vc_config.outbound.default_mode = "notify"
    manager = CallManager(vc_config, provider, store)
    provider.event_sink = manager.process_event
    record = await manager.initiate_call("+15555550001", message="your build is done")
    deadline = asyncio.get_running_loop().time() + 1.0
    while manager.get_call(record.call_id) and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)
    assert manager.get_call(record.call_id) is None  # finalized
    assert provider.spoken == [(record.provider_call_id, "your build is done")]
    assert provider.hangups == [record.provider_call_id]
    assert record.state == CallState.HANGUP_BOT


@pytest.mark.asyncio
async def test_continue_call_upgrades_notify_to_conversation(
    vc_config, provider, store
):
    """continue_call on a notify call cancels the auto-hangup, starts
    carrier transcription, and waits for the caller's reply."""
    vc_config.outbound.default_mode = "notify"
    vc_config.outbound.notify_hangup_delay_s = 0.3
    manager = CallManager(vc_config, provider, store)
    provider.event_sink = manager.process_event
    record = await manager.initiate_call("+15555550001", message="deploy is ready")
    deadline = asyncio.get_running_loop().time() + 1.0
    while not record.answered_at and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)

    async def caller_replies():
        await asyncio.sleep(0.1)
        await manager.process_event(NormalizedEvent(
            type=EventType.CALL_SPEECH, provider="mock",
            provider_call_id=record.provider_call_id, text="yes ship it",
        ))

    asyncio.get_running_loop().create_task(caller_replies())
    reply = await manager.continue_call(record.call_id, "shall I proceed?")
    assert reply == "yes ship it"
    assert record.mode == "conversation"          # upgraded
    assert provider.listening == [record.provider_call_id]  # transcription on
    await asyncio.sleep(0.5)
    assert manager.get_call(record.call_id) is not None  # no auto-hangup
    await manager.shutdown()


@pytest.mark.asyncio
async def test_speak_on_notify_call_rearms_hangup(vc_config, provider, store):
    """speak_to_user on a notify call holds the auto-hangup while the new
    message plays, then re-arms it — hang up after the LAST message."""
    vc_config.outbound.default_mode = "notify"
    vc_config.outbound.notify_hangup_delay_s = 0.25
    manager = CallManager(vc_config, provider, store)
    provider.event_sink = manager.process_event
    record = await manager.initiate_call("+15555550001", message="first part")
    deadline = asyncio.get_running_loop().time() + 1.0
    while not record.answered_at and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.1)
    await manager.speak(record.call_id, "and one more thing")
    await asyncio.sleep(0.15)
    # Original timer (would have fired at 0.25 after answer) was re-armed.
    assert manager.get_call(record.call_id) is not None
    await asyncio.sleep(0.25)
    assert manager.get_call(record.call_id) is None  # then it hung up
    assert record.end_reason == "notify-complete"
    assert [t.text for t in record.transcript if t.speaker == "bot"] == [
        "first part", "and one more thing",
    ]


@pytest.mark.asyncio
async def test_ring_timeout_marks_no_answer(store, make_config):
    cfg = make_config(timeouts=TimeoutsConfig(max_call_s=5, ring_s=0.05, silence_s=0,
                                              transcript_wait_s=0.5))
    cfg.provider_extra = {"mock": {"auto_answer": False}}
    provider = MockProvider(cfg)
    manager = CallManager(cfg, provider, store)
    provider.event_sink = manager.process_event
    record = await manager.initiate_call("+15555550001")
    await asyncio.sleep(0.15)
    assert record.state == CallState.NO_ANSWER
    assert record.end_reason == "ring-timeout"
    assert provider.hangups == [record.provider_call_id]


@pytest.mark.asyncio
async def test_initiate_failure_finalizes_failed(store, make_config):
    cfg = make_config()
    cfg.provider_extra = {"mock": {"fail_initiate": True}}
    provider = MockProvider(cfg)
    manager = CallManager(cfg, provider, store)
    with pytest.raises(RuntimeError):
        await manager.initiate_call("+15555550001")
    assert manager.get_active_calls() == []


@pytest.mark.asyncio
async def test_initiate_validates_numbers(manager):
    with pytest.raises(ValueError):
        await manager.initiate_call("not-a-number")
    manager.config.from_number = None
    with pytest.raises(ValueError):
        await manager.initiate_call("+15555550001")


@pytest.mark.asyncio
async def test_inbound_call_answers_greets_and_listens(manager, provider):
    await manager.process_event(
        NormalizedEvent(
            type=EventType.CALL_INITIATED,
            provider="mock",
            provider_call_id="mock-in-1",
            direction="inbound",
            from_number="+15555550009",
            to_number="+15555550000",
        )
    )
    record = manager.call_for_chat("+15555550009")
    assert record is not None
    assert provider.answered == ["mock-in-1"]
    await manager.process_event(
        NormalizedEvent(type=EventType.CALL_ANSWERED, provider="mock",
                        provider_call_id="mock-in-1")
    )
    assert record.state == CallState.LISTENING
    # Greeting spoken
    assert provider.spoken[0][1] == manager.config.inbound_greeting
    assert record.session_key == "+15555550009"  # per-phone scope


@pytest.mark.asyncio
async def test_session_key_per_call_scope(vc_config, provider, store):
    vc_config.session_scope = "per-call"
    manager = CallManager(vc_config, provider, store)
    provider.event_sink = manager.process_event
    record = await manager.initiate_call("+15555550001")
    assert record.session_key == f"+15555550001:{record.call_id}"


@pytest.mark.asyncio
async def test_speech_resolves_continue_call_waiter(manager, provider):
    record = await manager.initiate_call("+15555550001")
    await wait_for_state(manager, record.call_id, CallState.LISTENING)

    async def deliver_speech():
        await asyncio.sleep(0.05)
        await manager.process_event(
            NormalizedEvent(
                type=EventType.CALL_SPEECH,
                provider="mock",
                provider_call_id=record.provider_call_id,
                text="yes please",
            )
        )

    asyncio.get_running_loop().create_task(deliver_speech())
    reply = await manager.continue_call(record.call_id, "shall I proceed?")
    assert reply == "yes please"
    speakers = [t.speaker for t in record.transcript]
    assert speakers.count("user") == 1


@pytest.mark.asyncio
async def test_continue_call_ignores_speech_during_question_playback(
    vc_config, store
):
    """The caller reacting to the PREVIOUS message ('thanks') while the
    question is still playing must not be captured as the answer — the real
    answer comes after call.speak.ended."""

    class SlowSpeakProvider(MockProvider):
        async def speak(self, call, text):
            self.spoken.append((call.provider_call_id, text))
            if self.event_sink is not None:
                async def _ended(pid=call.provider_call_id, cid=call.call_id):
                    await asyncio.sleep(0.3)  # playback takes a while
                    await self.event_sink(NormalizedEvent(
                        type=EventType.CALL_SPEAK_ENDED, provider=self.name,
                        provider_call_id=pid, call_id=cid,
                    ))
                asyncio.get_running_loop().create_task(_ended())

    provider = SlowSpeakProvider(vc_config)
    manager = CallManager(vc_config, provider, store)
    provider.event_sink = manager.process_event
    record = await manager.initiate_call("+15555550001")
    deadline = asyncio.get_running_loop().time() + 1.0
    while record.state != CallState.LISTENING and \
            asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)

    async def caller_talks():
        await asyncio.sleep(0.1)   # mid-playback: reaction to part two
        await manager.process_event(NormalizedEvent(
            type=EventType.CALL_SPEECH, provider="mock",
            provider_call_id=record.provider_call_id, text="thanks",
        ))
        await asyncio.sleep(0.4)   # after speak.ended: the real answer
        await manager.process_event(NormalizedEvent(
            type=EventType.CALL_SPEECH, provider="mock",
            provider_call_id=record.provider_call_id, text="eggs and toast",
        ))

    asyncio.get_running_loop().create_task(caller_talks())
    reply = await manager.continue_call(record.call_id, "what was breakfast?")
    assert reply == "eggs and toast"   # not "thanks"
    # The discarded reaction never entered the transcript either.
    user_lines = [t.text for t in record.transcript if t.speaker == "user"]
    assert user_lines == ["eggs and toast"]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_continue_call_times_out(manager):
    record = await manager.initiate_call("+15555550001")
    await wait_for_state(manager, record.call_id, CallState.LISTENING)
    with pytest.raises(asyncio.TimeoutError):
        await manager.continue_call(record.call_id, "anyone there?")


@pytest.mark.asyncio
async def test_transcript_callback_fires_without_waiter(vc_config, provider, store):
    received = []

    async def on_transcript(record, text):
        received.append((record.call_id, text))

    manager = CallManager(vc_config, provider, store, on_final_transcript=on_transcript)
    provider.event_sink = manager.process_event
    record = await manager.initiate_call("+15555550001")
    await wait_for_state(manager, record.call_id, CallState.LISTENING)
    await manager.process_event(
        NormalizedEvent(
            type=EventType.CALL_SPEECH, provider="mock",
            provider_call_id=record.provider_call_id, text="hello bot",
        )
    )
    await asyncio.sleep(0.05)
    assert received == [(record.call_id, "hello bot")]
    # Partial transcripts never dispatch.
    await manager.process_event(
        NormalizedEvent(
            type=EventType.CALL_SPEECH, provider="mock",
            provider_call_id=record.provider_call_id, text="par", is_final=False,
        )
    )
    await asyncio.sleep(0.05)
    assert len(received) == 1


@pytest.mark.asyncio
async def test_user_hangup_finalizes_and_breaks_waiter(manager, provider):
    record = await manager.initiate_call("+15555550001")
    await wait_for_state(manager, record.call_id, CallState.LISTENING)

    async def hang_up():
        await asyncio.sleep(0.05)
        await manager.process_event(
            NormalizedEvent(
                type=EventType.CALL_ENDED, provider="mock",
                provider_call_id=record.provider_call_id, reason="hangup",
            )
        )

    asyncio.get_running_loop().create_task(hang_up())
    with pytest.raises(RuntimeError, match="call ended"):
        await manager.continue_call(record.call_id, "still there?")
    assert record.state == CallState.HANGUP_USER
    assert manager.get_call(record.call_id) is None
    assert manager.call_for_chat("+15555550001") is None


@pytest.mark.asyncio
async def test_event_dedupe(manager, provider):
    record = await manager.initiate_call("+15555550001")
    await wait_for_state(manager, record.call_id, CallState.LISTENING)
    event = NormalizedEvent(
        type=EventType.CALL_SPEECH, provider="mock",
        provider_call_id=record.provider_call_id, text="once",
        dedupe_key="evt-1",
    )
    await manager.process_event(event)
    await manager.process_event(event)
    user_lines = [t for t in record.transcript if t.speaker == "user"]
    assert len(user_lines) == 1


@pytest.mark.asyncio
async def test_orphan_outbound_event_replayed_after_initiate(manager, provider):
    """A webhook racing initiate_call() is stashed and drained, not lost."""
    provider.auto_answer = False
    orig_initiate = provider.initiate_call

    async def slow_initiate(call):
        pid = await orig_initiate(call)
        # Event arrives while initiate is still in flight (before mapping).
        await manager.process_event(
            NormalizedEvent(
                type=EventType.CALL_ANSWERED, provider="mock", provider_call_id=pid,
            )
        )
        return pid

    provider.initiate_call = slow_initiate
    record = await manager.initiate_call("+15555550001")
    # Drained orphan advanced the call: answered → listening (conversation).
    await wait_for_state(manager, record.call_id, CallState.LISTENING)
    assert record.answered_at is not None


@pytest.mark.asyncio
async def test_end_call_and_unknown_call_errors(manager, provider):
    record = await manager.initiate_call("+15555550001")
    await wait_for_state(manager, record.call_id, CallState.LISTENING)
    await manager.end_call(record.call_id)
    assert record.state == CallState.HANGUP_BOT
    with pytest.raises(CallNotFoundError):
        await manager.end_call(record.call_id)
    with pytest.raises(CallNotFoundError):
        await manager.speak("nope", "hi")


@pytest.mark.asyncio
async def test_dtmf_send_and_receive(manager, provider):
    record = await manager.initiate_call("+15555550001")
    await wait_for_state(manager, record.call_id, CallState.LISTENING)
    await manager.send_dtmf(record.call_id, "123#")
    assert provider.dtmf_sent == [(record.provider_call_id, "123#")]
    await manager.process_event(
        NormalizedEvent(
            type=EventType.CALL_DTMF, provider="mock",
            provider_call_id=record.provider_call_id, digits="9",
        )
    )
    assert record.metadata["dtmf_sent"] == ["123#"]
    assert record.metadata["dtmf_received"] == ["9"]


@pytest.mark.asyncio
async def test_max_duration_hangs_up(store, provider, make_config):
    cfg = make_config(
        timeouts=TimeoutsConfig(max_call_s=0.1, ring_s=5, silence_s=0,
                                transcript_wait_s=0.5)
    )
    manager = CallManager(cfg, provider, store)
    provider.event_sink = manager.process_event
    record = await manager.initiate_call("+15555550001")
    await wait_for_state(manager, record.call_id, CallState.LISTENING)
    await asyncio.sleep(0.2)
    assert record.state == CallState.TIMEOUT
    assert record.end_reason == "max-duration"


@pytest.mark.asyncio
async def test_silence_timeout_ends_conversation(store, provider, make_config):
    cfg = make_config(
        timeouts=TimeoutsConfig(max_call_s=5, ring_s=5, silence_s=0.1,
                                transcript_wait_s=0.5)
    )
    manager = CallManager(cfg, provider, store)
    provider.event_sink = manager.process_event
    record = await manager.initiate_call("+15555550001")
    await wait_for_state(manager, record.call_id, CallState.LISTENING)
    await asyncio.sleep(0.25)
    assert record.state == CallState.TIMEOUT
    assert record.end_reason == "silence-timeout"


@pytest.mark.asyncio
async def test_shutdown_cancels_timers_and_waiters(manager):
    record = await manager.initiate_call("+15555550001")
    await wait_for_state(manager, record.call_id, CallState.LISTENING)
    await manager.shutdown()
    assert manager._timers == {}
    # Call record is still active (it survives restarts via the store).
    assert manager.get_call(record.call_id) is not None

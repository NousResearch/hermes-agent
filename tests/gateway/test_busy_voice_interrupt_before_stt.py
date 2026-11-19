"""A busy voice note interrupts the running turn (and gets its busy ack) before STT finishes.

Interrupt mode used to await the clip's whole transcription before ``agent.interrupt()`` and the
ack, so a long voice note left the running turn going for as long as STT took (#143961). Every
interrupt entry point must signal at once and leave the clip, still queued with its audio, to the
follow-up turn, which transcribes it exactly once.
"""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _Agent:
    def __init__(self):
        self.interrupts = []

    def interrupt(self, message=None, **_):
        self.interrupts.append(message)

    def get_activity_summary(self):
        return {}


class _Adapter:
    def __init__(self):
        self._pending_messages, self._active_sessions, self.sent = {}, {}, []

    async def _send_with_retry(self, *, chat_id, content, **_):
        self.sent.append(content)


async def _interrupt_via(route, runner, event, source, adapter, agent, key):
    if route == "busy_handler":
        runner._queue_or_replace_pending_event(key, event)  # the busy handler queues before it acks
        outcome = runner._BusySteerOutcome("interrupt", False, False, False, False)
        await runner._busy_interrupt_and_ack(event, key, adapter, runner._session_state(key), agent, outcome)
    elif route == "priority":
        await runner._hm_busy_interrupt(event, source, agent, key)
    else:  # the running turn's interrupt monitor, for a voice note the adapter queued
        adapter._pending_messages[key] = event
        await runner._run_agent_fire_pending_interrupt(
            adapter, agent, source, key, asyncio.Event(), [None], log_context="Voice-interrupt", log=lambda: None)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["busy_handler", "priority", "monitor"])
async def test_busy_voice_note_interrupts_before_its_transcription_finishes(route, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    runner = GatewayRunner(config=GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", chat_type="dm")
    key = runner._session_key_for_source(source)
    adapter, agent = _Adapter(), _Agent()
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda _source: adapter)
    monkeypatch.setattr(runner, "_should_echo_stt_transcripts", lambda: False)
    runner._session_state(key).turn.agent = agent
    voice = tmp_path / "note.ogg"
    voice.write_bytes(b"OggS")
    event = MessageEvent(text="", message_type=MessageType.VOICE, source=source, message_id="v1",
                         media_urls=[str(voice)], media_types=["audio/ogg"])

    stt_calls, release = [], asyncio.Event()

    async def slow_clip(path, *_stt):  # a long voice note: STT holds until the test releases it
        stt_calls.append(path)
        await release.wait()
        return "stop that", f'"stop that"\n[User sent audio: {path}]'

    monkeypatch.setattr(runner, "_transcribe_one_clip", slow_clip)
    signal = asyncio.ensure_future(_interrupt_via(route, runner, event, source, adapter, agent, key))
    try:
        await asyncio.wait({signal}, timeout=2)
        assert agent.interrupts == [f"[User sent audio: {voice}]"], (
            f"interrupt waited for STT (calls={stt_calls}, interrupts={agent.interrupts})")
        if route == "busy_handler":
            assert adapter.sent, "the busy ack waited for STT"
    finally:
        release.set()
        await asyncio.wait_for(signal, timeout=5)

    # The follow-up still gets the transcript: the queued event kept its audio and the drain's
    # transcription runs STT once for it.
    assert adapter._pending_messages.get(key) is event
    text, transcripts = await runner._transcribe_and_echo_pending_voice(
        event, adapter, source, "", log_context="Voice-drain")
    assert transcripts == ["stop that"] and text.startswith('"stop that"')
    assert stt_calls == [str(voice)]

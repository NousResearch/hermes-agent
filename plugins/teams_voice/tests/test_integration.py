"""Integration-ish tests: handler wiring with fake sessions/realtime."""

from __future__ import annotations

import asyncio

from plugins.teams_voice import handlers, meeting, protocol
from plugins.teams_voice.config import resolve_config
from plugins.teams_voice.realtime.openai_client import RealtimeConfig


class FakeWS:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, human_count=1):
        self._ws = FakeWS()
        self.recording_active = True
        self.human_count = human_count
        self.call_id = "call-1"
        self.cancels: list[int] = []

    async def send_expression(self, e):
        ...

    async def send_audio_frame(self, *a):
        ...

    async def send_speech_marks(self, *a, **k):
        ...

    async def send_assistant_cancel(self, t):
        self.cancels.append(t)


class FakeRealtime:
    def __init__(self):
        self.auto: list[bool] = []
        self.created = 0
        self.cancelled = 0

    async def set_auto_response(self, enabled):
        self.auto.append(enabled)

    async def create_response(self):
        self.created += 1

    async def cancel_response(self):
        self.cancelled += 1


def _start(aad="aad-x", direction="inbound"):
    return protocol.SessionStart(
        type="session.start", call_id="call-1", thread_id="thread-1",
        caller=protocol.CallerInfo(aad_id=aad, display_name="X"),
        recording_status="inactive", direction=direction,
    )


def test_realtime_allowlist_rejects_and_closes_ws():
    cfg = resolve_config(extra={"shared_secret": "s", "allowlist": ["aad-allowed"]})
    h = handlers.RealtimeCallSessionHandler(RealtimeConfig(api_key="x"), bridge_config=cfg)
    sess = FakeSession()
    asyncio.run(h.on_session_start(sess, _start(aad="aad-OTHER")))
    assert sess._ws.closed is True
    assert h._rt is None  # rejected before connecting


def test_streaming_allowlist_rejects_and_closes_ws():
    cfg = resolve_config(extra={"shared_secret": "s", "allowlist": ["aad-allowed"]})
    h = handlers.StreamingCallSessionHandler(bridge_config=cfg)
    sess = FakeSession()
    asyncio.run(h.on_session_start(sess, _start(aad="nope")))
    assert sess._ws.closed is True


def test_realtime_group_gate_manual_response_no_leak():
    cfg = resolve_config(extra={"shared_secret": "s", "wake_phrases": ["aria"]})
    h = handlers.RealtimeCallSessionHandler(RealtimeConfig(api_key="x"), bridge_config=cfg)
    h._rt = FakeRealtime()
    sess = FakeSession(human_count=2)
    h._session = sess

    # 2 humans → auto-response turned OFF (race-free gate)
    asyncio.run(h.on_participants(sess, protocol.Participants(type="participants", count=2)))
    assert h._rt.auto[-1] is False

    # unaddressed turn → drop + cancel, NO response created
    asyncio.run(h._on_input_transcript("just chatting with my colleague"))
    assert h._drop_response is True and h._rt.cancelled >= 1 and h._rt.created == 0

    # addressed turn → we explicitly create the response
    asyncio.run(h._on_input_transcript("aria, what do you think?"))
    assert h._rt.created == 1


def test_streaming_session_end_cancels_inflight_task():
    h = handlers.StreamingCallSessionHandler(bridge_config=resolve_config(extra={"shared_secret": "s"}))
    h._session = FakeSession()

    async def run():
        async def _slow():
            await asyncio.sleep(5)

        task = asyncio.create_task(_slow())
        h._utterance_task = task
        await asyncio.sleep(0)
        await h.on_session_end(h._session, protocol.SessionEnd(type="session.end", reason="x"))
        try:
            await task
        except asyncio.CancelledError:
            pass
        return task

    task = asyncio.run(run())
    assert task.cancelled()


def test_meeting_post_minutes_injected_deliver():
    class FakeConsult:
        async def ask(self, q, *, timeout_s=120.0):
            return "Key Points: discussed X. Decisions: ship it."

    delivered = {}

    async def fake_deliver(conv, text):
        delivered["conv"] = conv
        delivered["text"] = text
        return True

    t = meeting.MeetingTranscript()
    t.add("Sara", "let's ship")
    out = asyncio.run(meeting.post_minutes(FakeConsult(), t, "19:abc@thread.v2", deliver=fake_deliver))
    assert "posted the minutes" in out.lower()
    assert delivered["conv"] == "19:abc@thread.v2"
    assert "Key Points" in delivered["text"]

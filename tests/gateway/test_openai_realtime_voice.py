"""Tests for the Discord OpenAI Realtime voice bridge."""

from __future__ import annotations

import base64
import json
import logging


class _FakeWS:
    def __init__(self, recv_frames=None):
        self.sent = []
        self.closed = False
        self._recv_frames = list(recv_frames or [])

    def send(self, payload):
        self.sent.append(json.loads(payload))

    def recv(self):
        if not self._recv_frames:
            raise RuntimeError("fake ws empty")
        frame = self._recv_frames.pop(0)
        return json.dumps(frame) if isinstance(frame, dict) else frame

    def close(self):
        self.closed = True


def test_load_realtime_voice_config_supports_discord_overrides():
    from gateway.openai_realtime_voice import load_realtime_voice_config

    cfg = {
        "voice": {
            "realtime": {
                "enabled": True,
                "model": "gpt-realtime-2",
                "voice": "marin",
                "discord": {"voice": "cedar", "reasoning_effort": "medium"},
            },
        },
    }

    rt = load_realtime_voice_config(cfg)

    assert rt.enabled is True
    assert rt.model == "gpt-realtime-2"
    assert rt.voice == "cedar"
    assert rt.reasoning_effort == "medium"


def test_load_realtime_voice_config_supports_input_silence_threshold():
    from gateway.openai_realtime_voice import load_realtime_voice_config

    cfg = {
        "voice": {
            "silence_threshold": 200,
            "realtime": {
                "enabled": True,
                "input_silence_threshold": 120,
            },
        },
    }

    rt = load_realtime_voice_config(cfg)

    assert rt.input_silence_threshold == 120


def test_load_realtime_voice_config_supports_response_start_timeout():
    from gateway.openai_realtime_voice import load_realtime_voice_config

    cfg = {
        "voice": {
            "realtime": {
                "enabled": True,
                "response_start_timeout_ms": 2500,
            },
        },
    }

    rt = load_realtime_voice_config(cfg)

    assert rt.response_start_timeout_ms == 2500


def test_load_realtime_voice_config_aliases_codex_auth_to_managed():
    from gateway.openai_realtime_voice import load_realtime_voice_config

    cfg = {"voice": {"realtime": {"enabled": True, "auth_mode": "codex"}}}

    rt = load_realtime_voice_config(cfg)

    assert rt.auth_mode == "managed"


def test_resolve_realtime_auth_direct(monkeypatch):
    import gateway.openai_realtime_voice as rt

    monkeypatch.setattr(rt, "resolve_openai_audio_api_key", lambda: "sk-direct")

    auth = rt.resolve_realtime_auth(rt.RealtimeVoiceConfig(auth_mode="auto"))

    assert auth.mode == "direct"
    assert auth.api_key == "sk-direct"
    assert auth.websocket_url == rt.DIRECT_REALTIME_WS_URL


def test_resolve_realtime_auth_managed_uses_codex_oauth(monkeypatch):
    import gateway.openai_realtime_voice as rt

    monkeypatch.setattr(rt, "resolve_openai_audio_api_key", lambda: "sk-direct")
    import hermes_cli.auth as auth_mod
    monkeypatch.setattr(
        auth_mod,
        "resolve_codex_runtime_credentials",
        lambda **_kw: {"api_key": "codex-oauth-token"},
    )

    auth = rt.resolve_realtime_auth(rt.RealtimeVoiceConfig(auth_mode="managed"))

    assert auth.mode == "managed"
    assert auth.api_key == "codex-oauth-token"
    assert auth.websocket_url == rt.DIRECT_REALTIME_WS_URL


def test_resolve_realtime_auth_auto_falls_back_to_codex_oauth(monkeypatch):
    import gateway.openai_realtime_voice as rt
    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(rt, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(
        auth_mod,
        "resolve_codex_runtime_credentials",
        lambda **_kw: {"api_key": "codex-oauth-token"},
    )

    auth = rt.resolve_realtime_auth(rt.RealtimeVoiceConfig(auth_mode="auto"))

    assert auth.mode == "managed"
    assert auth.api_key == "codex-oauth-token"


def test_discord_pcm_to_realtime_pcm_downmixes_and_decimates():
    from gateway.openai_realtime_voice import discord_pcm_to_realtime_pcm

    frame_1 = (1000).to_bytes(2, "little", signed=True) + (3000).to_bytes(2, "little", signed=True)
    frame_2 = (5000).to_bytes(2, "little", signed=True) + (7000).to_bytes(2, "little", signed=True)

    converted = discord_pcm_to_realtime_pcm(frame_1 + frame_2)

    assert converted == (2000).to_bytes(2, "little", signed=True)


def test_discord_pcm_to_realtime_pcm_resamples_non_default_rate():
    from gateway.openai_realtime_voice import discord_pcm_to_realtime_pcm

    converted = discord_pcm_to_realtime_pcm(
        b"\x01\x00" * 160,
        src_rate=16000,
        src_channels=1,
        dst_rate=24000,
    )

    assert len(converted) >= 470


def test_pcm_rms_reads_s16le_samples():
    from gateway.openai_realtime_voice import pcm_rms

    pcm = (300).to_bytes(2, "little", signed=True) + (-300).to_bytes(2, "little", signed=True)

    assert pcm_rms(pcm) == 300


def test_realtime_session_instructions_uses_default_for_empty_config():
    import gateway.openai_realtime_voice as rt

    instructions = rt.realtime_session_instructions(rt.RealtimeVoiceConfig())

    assert "Discord voice channel" in instructions
    assert "Hermes tools" in instructions
    assert "use terminal" in instructions
    assert "Do not answer those requests from memory" in instructions


def test_realtime_session_instructions_preserves_custom_config():
    import gateway.openai_realtime_voice as rt

    instructions = rt.realtime_session_instructions(
        rt.RealtimeVoiceConfig(instructions="Be very brief.")
    )

    assert instructions == "Be very brief."


def test_session_start_sends_realtime_update(monkeypatch):
    import gateway.openai_realtime_voice as rt

    ws = _FakeWS()
    captured = {}

    def connect(url, **kwargs):
        captured["url"] = url
        captured["headers"] = dict(kwargs.get("additional_headers") or kwargs.get("extra_headers") or [])
        return ws

    monkeypatch.setattr(rt, "_require_websocket_connect", lambda: connect)

    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True, instructions="Be concise."),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run a command",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        on_audio_response=lambda _pcm: None,
    )

    session.start()
    session.close()

    assert "model=gpt-realtime-2" in captured["url"]
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert "OpenAI-Beta" not in captured["headers"]
    update = ws.sent[0]
    assert update["type"] == "session.update"
    assert update["session"]["model"] == "gpt-realtime-2"
    assert update["session"]["output_modalities"] == ["audio"]
    assert update["session"]["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
    }
    assert update["session"]["audio"]["input"]["turn_detection"] == {
        "type": "server_vad",
        "threshold": 0.55,
        "prefix_padding_ms": 250,
        "silence_duration_ms": 350,
        "create_response": True,
        "interrupt_response": True,
    }
    assert update["session"]["audio"]["output"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
    }
    assert update["session"]["audio"]["output"]["voice"] == "marin"
    assert update["session"]["instructions"] == "Be concise."
    assert update["session"]["tools"][0]["name"] == "terminal"


def test_session_start_sends_default_voice_tool_instructions(monkeypatch):
    import gateway.openai_realtime_voice as rt

    ws = _FakeWS()
    monkeypatch.setattr(rt, "_require_websocket_connect", lambda: (lambda *_a, **_kw: ws))

    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
    )

    session.start()
    session.close()

    assert "Discord voice channel" in ws.sent[0]["session"]["instructions"]
    assert "Hermes tools" in ws.sent[0]["session"]["instructions"]


def test_send_discord_pcm_appends_base64_audio():
    import gateway.openai_realtime_voice as rt

    ws = _FakeWS()
    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
    )
    session._ws = ws
    session._running.set()

    frame = (1000).to_bytes(2, "little", signed=True) + (3000).to_bytes(2, "little", signed=True)
    session.send_discord_pcm(frame * 2)

    payload = ws.sent[0]
    assert payload["type"] == "input_audio_buffer.append"
    assert base64.b64decode(payload["audio"]) == (2000).to_bytes(2, "little", signed=True)
    session.close()


def test_send_discord_pcm_ignores_local_silence():
    import gateway.openai_realtime_voice as rt

    ws = _FakeWS()
    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True, input_silence_threshold=120),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
    )
    session._ws = ws
    session._running.set()

    session.send_discord_pcm(b"\x00\x00" * 480)

    assert ws.sent == []
    assert session._turn_timer is None


def test_send_discord_pcm_schedules_turn_for_local_speech():
    import gateway.openai_realtime_voice as rt

    ws = _FakeWS()
    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True, input_silence_threshold=120),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
    )
    session._ws = ws
    session._running.set()

    frame = (1000).to_bytes(2, "little", signed=True) + (1000).to_bytes(2, "little", signed=True)
    session.send_discord_pcm(frame * 2)

    try:
        assert ws.sent[0]["type"] == "input_audio_buffer.append"
        assert session._turn_timer is not None
    finally:
        session.close()


def test_manual_turn_finalize_commits_and_creates_response():
    import gateway.openai_realtime_voice as rt

    ws = _FakeWS()
    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True, manual_turn_timeout_ms=100),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
    )
    session._ws = ws
    session._running.set()
    session._turn_has_audio = True
    session._last_input_audio_at = rt.time.monotonic() - 1

    session._finalize_input_turn()

    assert [payload["type"] for payload in ws.sent] == [
        "input_audio_buffer.commit",
        "response.create",
    ]
    assert session._turn_has_audio is False


def test_manual_turn_finalize_arms_response_start_watchdog(monkeypatch):
    import gateway.openai_realtime_voice as rt

    class FakeTimer:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.started = False
            self.cancelled = False

        def start(self):
            self.started = True

        def cancel(self):
            self.cancelled = True

    timers = []

    def fake_timer(delay, callback):
        timer = FakeTimer(delay, callback)
        timers.append(timer)
        return timer

    monkeypatch.setattr(rt.threading, "Timer", fake_timer)
    ws = _FakeWS()
    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True, manual_turn_timeout_ms=100, response_start_timeout_ms=2500),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
    )
    session._ws = ws
    session._running.set()
    session._turn_has_audio = True
    session._last_input_audio_at = rt.time.monotonic() - 1

    session._finalize_input_turn()

    assert [payload["type"] for payload in ws.sent] == [
        "input_audio_buffer.commit",
        "response.create",
    ]
    assert session._awaiting_response_start is True
    assert session._response_create_attempts == 1
    assert timers[-1].delay == 2.5
    assert timers[-1].started is True


def test_server_vad_speech_stopped_keeps_manual_turn_timer_armed():
    import gateway.openai_realtime_voice as rt

    class FakeTimer:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
    )
    timer = FakeTimer()
    session._turn_has_audio = True
    session._turn_timer = timer

    session._handle_frame({"type": "input_audio_buffer.speech_stopped"})

    assert session._turn_has_audio is True
    assert session._turn_timer is timer
    assert timer.cancelled is False


def test_response_created_cancels_manual_turn_timer():
    import gateway.openai_realtime_voice as rt

    class FakeTimer:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
    )
    timer = FakeTimer()
    session._turn_has_audio = True
    session._turn_timer = timer

    session._handle_frame({"type": "response.created"})

    assert session._response_in_progress is True
    assert session._turn_has_audio is False
    assert session._turn_timer is None
    assert timer.cancelled is True


def test_response_created_cancels_response_start_watchdog():
    import gateway.openai_realtime_voice as rt

    class FakeTimer:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
    )
    timer = FakeTimer()
    session._awaiting_response_start = True
    session._response_create_attempts = 1
    session._response_watchdog_timer = timer

    session._handle_frame({"type": "response.created"})

    assert session._awaiting_response_start is False
    assert session._response_create_attempts == 0
    assert session._response_watchdog_timer is None
    assert timer.cancelled is True


def test_response_start_watchdog_retries_once_then_gives_up(monkeypatch):
    import gateway.openai_realtime_voice as rt

    class FakeTimer:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.started = False

        def start(self):
            self.started = True

        def cancel(self):
            pass

    timers = []

    def fake_timer(delay, callback):
        timer = FakeTimer(delay, callback)
        timers.append(timer)
        return timer

    monkeypatch.setattr(rt.threading, "Timer", fake_timer)
    ws = _FakeWS()
    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True, response_start_timeout_ms=2500),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
    )
    session._ws = ws
    session._running.set()
    session._awaiting_response_start = True
    session._response_create_attempts = 1

    session._response_start_timed_out()

    assert [payload["type"] for payload in ws.sent] == ["response.create"]
    assert session._awaiting_response_start is True
    assert session._response_create_attempts == 2
    assert timers[-1].delay == 2.5
    assert timers[-1].started is True

    session._response_start_timed_out()

    assert [payload["type"] for payload in ws.sent] == ["response.create"]
    assert session._awaiting_response_start is False
    assert session._response_create_attempts == 2


def test_manual_turn_finalize_rearms_while_response_in_progress(monkeypatch):
    import gateway.openai_realtime_voice as rt

    class FakeTimer:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.started = False
            self.cancelled = False

        def start(self):
            self.started = True

        def cancel(self):
            self.cancelled = True

    timers = []

    def fake_timer(delay, callback):
        timer = FakeTimer(delay, callback)
        timers.append(timer)
        return timer

    monkeypatch.setattr(rt.threading, "Timer", fake_timer)
    ws = _FakeWS()
    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True, manual_turn_timeout_ms=100),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
    )
    session._ws = ws
    session._running.set()
    session._turn_has_audio = True
    session._last_input_audio_at = rt.time.monotonic() - 1
    session._response_in_progress = True

    session._finalize_input_turn()

    assert ws.sent == []
    assert len(timers) == 1
    assert timers[0].delay == 0.1
    assert timers[0].started is True


def test_response_done_rearms_pending_second_turn(monkeypatch):
    import gateway.openai_realtime_voice as rt

    class FakeTimer:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.started = False

        def start(self):
            self.started = True

        def cancel(self):
            pass

    timers = []

    def fake_timer(delay, callback):
        timer = FakeTimer(delay, callback)
        timers.append(timer)
        return timer

    monkeypatch.setattr(rt.threading, "Timer", fake_timer)
    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True, manual_turn_timeout_ms=100),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
    )
    session._turn_has_audio = True
    session._last_input_audio_at = rt.time.monotonic() - 1
    session._response_in_progress = True

    session._handle_frame({"type": "response.done"})

    assert session._response_in_progress is False
    assert len(timers) == 1
    assert timers[0].delay == 0.01
    assert timers[0].started is True


def test_late_response_item_does_not_clear_pending_barge_in(monkeypatch):
    import gateway.openai_realtime_voice as rt

    class FakeTimer:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.started = False
            self.cancelled = False

        def start(self):
            self.started = True

        def cancel(self):
            self.cancelled = True

    timers = []

    def fake_timer(delay, callback):
        timer = FakeTimer(delay, callback)
        timers.append(timer)
        return timer

    monkeypatch.setattr(rt.threading, "Timer", fake_timer)
    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True, manual_turn_timeout_ms=100),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
    )
    session._response_in_progress = True
    session._turn_has_audio = True
    session._pending_barge_in = True
    session._last_input_audio_at = rt.time.monotonic() - 1

    session._handle_frame({"type": "response.output_item.added"})

    assert session._turn_has_audio is True
    assert session._pending_barge_in is True
    assert len(timers) == 1
    assert timers[0].started is True


def test_barge_in_cancels_response_and_ignores_stale_audio_delta():
    import gateway.openai_realtime_voice as rt

    ws = _FakeWS()
    flushed = []
    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=flushed.append,
    )
    session._ws = ws
    session._running.set()
    session._response_in_progress = True
    session._response_audio.extend(b"stale")

    frame = (1000).to_bytes(2, "little", signed=True) + (1000).to_bytes(2, "little", signed=True)
    session.send_discord_pcm(frame * 2)
    session.send_discord_pcm(frame * 2)

    try:
        assert [payload["type"] for payload in ws.sent[:3]] == [
            "response.cancel",
            "input_audio_buffer.append",
            "input_audio_buffer.append",
        ]
        assert session._pending_barge_in is True
        assert session._turn_has_audio is True
        assert session._response_audio == bytearray()

        stale_chunk = b"\x01\x02" * (rt.DEFAULT_OUTPUT_FLUSH_BYTES // 2)
        session._handle_frame({
            "type": "response.audio.delta",
            "delta": base64.b64encode(stale_chunk).decode("ascii"),
        })

        assert flushed == []
        assert session._response_audio == bytearray()
        assert session._turn_has_audio is True
    finally:
        session.close()


def test_user_audio_start_callback_fires_once_per_turn():
    import gateway.openai_realtime_voice as rt

    ws = _FakeWS()
    starts = []
    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
        on_user_audio_start=lambda: starts.append("start"),
    )
    session._ws = ws
    session._running.set()

    frame = (1000).to_bytes(2, "little", signed=True) + (1000).to_bytes(2, "little", signed=True)
    session.send_discord_pcm(frame * 2)
    session.send_discord_pcm(frame * 2)

    try:
        assert starts == ["start"]
    finally:
        session.close()


def test_audio_delta_flushes_incrementally_at_stream_threshold():
    import gateway.openai_realtime_voice as rt

    flushed = []
    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=flushed.append,
    )

    chunk = b"\x01\x02" * (rt.DEFAULT_OUTPUT_FLUSH_BYTES // 2)
    session._handle_frame({
        "type": "response.audio.delta",
        "delta": base64.b64encode(chunk).decode("ascii"),
    })

    assert flushed == [chunk]
    assert session._response_audio == bytearray()


def test_audio_done_flushes_remainder_below_stream_threshold():
    import gateway.openai_realtime_voice as rt

    flushed = []
    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=flushed.append,
    )

    chunk = b"\x01\x02" * 100
    session._handle_frame({
        "type": "response.audio.delta",
        "delta": base64.b64encode(chunk).decode("ascii"),
    })
    assert flushed == []

    session._handle_frame({"type": "response.audio.done"})

    assert flushed == [chunk]
    assert session._response_audio == bytearray()


def test_function_call_executes_hermes_tool_and_resumes(monkeypatch, caplog):
    import gateway.openai_realtime_voice as rt
    import model_tools

    ws = _FakeWS()
    monkeypatch.setattr(
        model_tools,
        "handle_function_call",
        lambda name, args, **_kw: json.dumps({"name": name, "args": args}),
    )

    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
    )
    session._ws = ws

    caplog.set_level(logging.INFO, logger="gateway.openai_realtime_voice")
    session._handle_frame({
        "type": "response.function_call_arguments.done",
        "name": "terminal",
        "call_id": "call_1",
        "arguments": "{\"cmd\":\"pwd\"}",
    })

    assert ws.sent[0]["type"] == "conversation.item.create"
    assert ws.sent[0]["item"]["call_id"] == "call_1"
    assert json.loads(ws.sent[0]["item"]["output"]) == {"name": "terminal", "args": {"cmd": "pwd"}}
    assert ws.sent[1]["type"] == "response.create"
    assert "OpenAI Realtime voice completed Hermes tool: name=terminal call_id=call_1" in caplog.text


def test_function_call_duplicate_call_id_executes_once(monkeypatch):
    import gateway.openai_realtime_voice as rt
    import model_tools

    ws = _FakeWS()
    calls = []
    monkeypatch.setattr(
        model_tools,
        "handle_function_call",
        lambda name, args, **_kw: calls.append((name, args)) or "{}",
    )

    session = rt.OpenAIRealtimeVoiceSession(
        config=rt.RealtimeVoiceConfig(enabled=True),
        auth=rt.RealtimeAuthConfig(api_key="sk-test", websocket_url=rt.DIRECT_REALTIME_WS_URL, mode="direct"),
        tool_schemas=[],
        on_audio_response=lambda _pcm: None,
    )
    session._ws = ws

    frame = {
        "type": "response.function_call_arguments.done",
        "name": "terminal",
        "call_id": "call_1",
        "arguments": "{}",
    }
    session._handle_frame(frame)
    session._handle_frame(frame)

    assert calls == [("terminal", {})]
    assert [payload["type"] for payload in ws.sent] == [
        "conversation.item.create",
        "response.create",
    ]

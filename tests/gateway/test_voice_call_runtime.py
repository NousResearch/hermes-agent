import pytest

from gateway.voice_call_runtime import (
    CallRecord,
    RealtimeVoiceBridge,
    SaluteSpeechTranscriber,
    VoiceCallRuntime,
    _voice_realtime_model,
    _voice_realtime_voice,
    _voice_vad_eagerness,
)


def test_voice_transcript_filters_realtime_junk(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runtime = VoiceCallRuntime()
    call = CallRecord(call_id="call_test", to="+70000000000", task="Позвонить")

    runtime._add_transcript(call, "callee", "😎", source="openai_realtime")
    runtime._add_transcript(call, "callee", " Алло. ", source="openai_realtime")
    runtime._add_transcript(call, "callee", "Алло.", source="openai_realtime")

    assert len(call.raw_transcript) == 3
    assert call.transcript == [
        {
            "role": "callee",
            "text": "Алло.",
            "source": "openai_realtime",
            "timestamp": call.transcript[0]["timestamp"],
        }
    ]


def test_salute_transcript_extractor_handles_common_payloads():
    assert SaluteSpeechTranscriber._extract_transcript('{"result":["Алло, ресторан."]}') == (
        "Алло, ресторан."
    )
    assert SaluteSpeechTranscriber._extract_transcript(
        '{"results":[{"text":"Добрый день."},{"normalized_text":"Слушаю вас."}]}'
    ) == "Добрый день. Слушаю вас."


def test_salute_oauth_parser_accepts_legacy_and_giga_chat_payloads():
    now = 1_778_000_000.0

    legacy_token, legacy_expires_at = SaluteSpeechTranscriber._parse_oauth_payload(
        {"access_token": "legacy-token", "expires_at": 1_778_001_000_000},
        now,
    )
    giga_token, giga_expires_at = SaluteSpeechTranscriber._parse_oauth_payload(
        {"tok": "giga-token", "exp": 1_778_002_000},
        now,
    )

    assert legacy_token == "legacy-token"
    assert legacy_expires_at == 1_778_001_000
    assert giga_token == "giga-token"
    assert giga_expires_at == 1_778_002_000


def test_voice_vad_defaults_to_labota_fast_turn_taking(monkeypatch):
    monkeypatch.delenv("VOICE_CALL_VAD_EAGERNESS", raising=False)
    assert _voice_vad_eagerness() == "high"

    monkeypatch.setenv("VOICE_CALL_VAD_EAGERNESS", "low")
    assert _voice_vad_eagerness() == "low"

    monkeypatch.setenv("VOICE_CALL_VAD_EAGERNESS", "unexpected")
    assert _voice_vad_eagerness() == "high"


def test_openai_realtime_defaults_to_gpt_realtime_2(monkeypatch):
    monkeypatch.delenv("VOICE_CALL_REALTIME_MODEL", raising=False)
    monkeypatch.delenv("VOICE_CALL_ASSISTANT_VOICE", raising=False)

    assert _voice_realtime_model("openai") == "gpt-realtime-2"
    assert _voice_realtime_voice("openai") == "marin"

    monkeypatch.setenv("VOICE_CALL_REALTIME_MODEL", "gpt-realtime-1.5")
    monkeypatch.setenv("VOICE_CALL_ASSISTANT_VOICE", "cedar")

    assert _voice_realtime_model("openai") == "gpt-realtime-1.5"
    assert _voice_realtime_voice("openai") == "cedar"


def test_openai_realtime_session_uses_ga_audio_schema(monkeypatch):
    monkeypatch.delenv("VOICE_CALL_REALTIME_TRANSCRIPTION_MODEL", raising=False)
    monkeypatch.setenv("VOICE_CALL_VAD_EAGERNESS", "high")
    call = CallRecord(call_id="call_test", to="+70000000000", task="Позвонить")
    bridge = RealtimeVoiceBridge(
        provider="openai",
        api_key="test-key",
        model="gpt-realtime-2",
        voice="marin",
        instructions="Говори кратко.",
        language="ru",
        call=call,
        on_assistant_audio=lambda audio: None,
        on_transcript=lambda role, text: None,
    )

    event = bridge._openai_session_update_event()
    session = event["session"]

    assert event["type"] == "session.update"
    assert session["type"] == "realtime"
    assert session["model"] == "gpt-realtime-2"
    assert session["output_modalities"] == ["audio"]
    assert session["audio"]["input"]["format"] == {"type": "audio/pcmu"}
    assert session["audio"]["output"]["format"] == {"type": "audio/pcmu"}
    assert session["audio"]["output"]["voice"] == "marin"
    assert session["audio"]["input"]["transcription"] == {
        "model": "whisper-1",
        "language": "ru",
    }
    assert session["audio"]["input"]["turn_detection"] == {
        "type": "semantic_vad",
        "eagerness": "high",
        "create_response": True,
    }
    assert "modalities" not in session
    assert "input_audio_format" not in session
    assert "output_audio_format" not in session
    assert "input_audio_transcription" not in session


def test_openai_realtime_initial_response_uses_ga_audio_schema():
    call = CallRecord(call_id="call_test", to="+70000000000", task="Позвонить")
    bridge = RealtimeVoiceBridge(
        provider="openai",
        api_key="test-key",
        model="gpt-realtime-2",
        voice="marin",
        instructions="Говори кратко.",
        language="ru",
        call=call,
        on_assistant_audio=lambda audio: None,
        on_transcript=lambda role, text: None,
    )

    event = bridge._openai_response_create_event("Поздоровайся.")

    assert event == {
        "type": "response.create",
        "response": {
            "output_modalities": ["audio"],
            "audio": {
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": "marin",
                },
            },
            "instructions": "Поздоровайся.",
        },
    }


@pytest.mark.asyncio
async def test_openai_realtime_connect_omits_beta_header():
    class FakeWebSocket:
        closed = False

        def __init__(self):
            self.sent = []

        async def send_json(self, payload):
            self.sent.append(payload)

    class FakeSession:
        def __init__(self):
            self.ws = FakeWebSocket()
            self.url = ""
            self.headers = {}
            self.heartbeat = None

        async def ws_connect(self, url, *, headers, heartbeat):
            self.url = url
            self.headers = headers
            self.heartbeat = heartbeat
            return self.ws

    call = CallRecord(call_id="call_test", to="+70000000000", task="Позвонить")
    bridge = RealtimeVoiceBridge(
        provider="openai",
        api_key="test-key",
        model="gpt-realtime-2",
        voice="marin",
        instructions="Говори кратко.",
        language="ru",
        call=call,
        on_assistant_audio=lambda audio: None,
        on_transcript=lambda role, text: None,
    )
    fake_session = FakeSession()
    bridge.session = fake_session

    await bridge._connect_openai()

    assert fake_session.url == "wss://api.openai.com/v1/realtime?model=gpt-realtime-2"
    assert fake_session.headers == {"Authorization": "Bearer test-key"}
    assert fake_session.heartbeat == 20
    assert fake_session.ws.sent == [bridge._openai_session_update_event()]


def test_telegram_calendar_link_is_hidden_behind_text_entity():
    text = (
        "Бронь подтверждена.\n\n"
        "📅 Добавить в календарь: "
        "https://calendar.google.com/calendar/render?action=TEMPLATE&text=%D0%A2%D0%B5%D1%81%D1%82"
    )

    visible, entities = VoiceCallRuntime._telegram_calendar_link_entities(text)

    assert visible == "Бронь подтверждена.\n\n📅 Добавить в календарь"
    assert entities == [
        {
            "type": "text_link",
            "offset": len("Бронь подтверждена.\n\n"),
            "length": VoiceCallRuntime._telegram_utf16_len("📅 Добавить в календарь"),
            "url": "https://calendar.google.com/calendar/render?action=TEMPLATE&text=%D0%A2%D0%B5%D1%81%D1%82",
        }
    ]


def test_russian_greeting_name_transliterates_first_latin_token():
    from gateway.run import GatewayRunner

    assert GatewayRunner._russian_greeting_name("Tsevdn Kanduev") == "Цевдн"
    assert GatewayRunner._russian_greeting_name("Павел Богомолов") == "Павел"

"""Tests for the bundled OpenAI Realtime provider against a fake WebSocket.

No network, no OpenAI key: the connector is injected and every frame is
scripted. The GA wire shapes asserted here are the ones the hermes-talk
plugin exercised live (session payload, event names, command encodings).
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

import plugins.realtime_voice.openai as openai_plugin
from agent.realtime_voice_orchestrator import RealtimeVoiceHost, RealtimeVoiceOrchestrator
from agent.realtime_voice_provider import (
    PCM16_24K,
    InputAudioCommitted,
    InputSpeechStarted,
    InputSpeechStopped,
    InputTranscript,
    OutputAudio,
    OutputTranscript,
    RealtimeAudioFormat,
    RealtimeCapability,
    RealtimeTool,
    RealtimeToolResult,
    ResponseCompleted,
    ResponseStarted,
    SessionClosed,
    SessionFailure,
    SessionReady,
    ToolCall,
)
from plugins.realtime_voice.openai.provider import (
    CAPABILITIES,
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    REALTIME_WS_URL,
    OpenAIRealtimeProvider,
    OpenAIRealtimeSession,
    build_session_update,
    decode_event,
)


class FakeWebSocket:
    """The subset of ``websockets.asyncio.client.ClientConnection`` we use."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.closed = 0
        self.failure: Exception | None = None

    async def send(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def close(self) -> None:
        self.closed += 1

    def feed(self, *frames) -> None:
        for frame in frames:
            self.incoming.put_nowait(frame)

    def __aiter__(self):
        return self

    async def __anext__(self):
        frame = await self.incoming.get()
        if isinstance(frame, Exception):
            raise frame
        return frame


def _frame(payload: dict) -> str:
    return json.dumps(payload)


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-realtime-key")
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    return "sk-test-realtime-key"


@pytest.fixture
def socket():
    return FakeWebSocket()


@pytest.fixture
def provider(socket):
    dials: list[tuple[str, dict]] = []

    async def connector(url, headers):
        dials.append((url, dict(headers)))
        return socket

    instance = OpenAIRealtimeProvider(connector=connector)
    instance.dials = dials  # type: ignore[attr-defined]
    return instance


# -- metadata ----------------------------------------------------------------


class TestProviderMetadata:
    def test_identity_and_catalog(self, provider):
        assert provider.name == "openai"
        assert provider.display_name == "OpenAI Realtime"
        assert provider.default_model() == DEFAULT_MODEL == "gpt-realtime-2.1"
        assert provider.default_voice() == DEFAULT_VOICE == "marin"
        assert provider.capabilities == CAPABILITIES
        assert RealtimeCapability.OUTPUT_TRUNCATION in provider.capabilities
        assert RealtimeCapability.SESSION_RESUMPTION not in provider.capabilities
        assert [voice["id"] for voice in provider.list_voices()][:2] == ["marin", "cedar"]
        assert provider.get_setup_schema()["env_vars"][0]["key"] == "OPENAI_API_KEY"

    def test_availability_follows_the_audio_key(self, provider, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        assert provider.is_available() is False
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-voice")
        assert provider.is_available() is True

    def test_plugin_registers_through_the_context_hook(self):
        registered = []

        class Ctx:
            def register_realtime_voice_provider(self, provider):
                registered.append(provider)

        openai_plugin.register(Ctx())

        assert len(registered) == 1
        assert isinstance(registered[0], OpenAIRealtimeProvider)

    def test_bundled_backend_autoloads_into_the_registry(self):
        from hermes_cli.plugins import PluginManager

        from agent import realtime_voice_registry

        realtime_voice_registry._reset_for_tests()
        try:
            manager = PluginManager()
            manager.discover_and_load()
            loaded = manager._plugins.get("realtime_voice/openai")
            assert loaded is not None and loaded.enabled, getattr(loaded, "error", None)
            provider = realtime_voice_registry.get_provider("openai", scope=manager.scope_key)
            assert isinstance(provider, OpenAIRealtimeProvider)
        finally:
            realtime_voice_registry._reset_for_tests()


# -- session setup -----------------------------------------------------------


class TestOpenSession:
    @pytest.mark.asyncio
    async def test_connects_with_bearer_header_and_sends_session_update(
        self, provider, socket, api_key
    ):
        setup = openai_plugin.provider.RealtimeVoiceSetup(
            model="gpt-realtime",
            voice="cedar",
            instructions="Be brief.",
            tools=(RealtimeTool("lookup", "Look things up", {"type": "object"}),),
            automatic_response=False,
        )

        session = await provider.open_session(setup)

        assert provider.dials == [
            (f"{REALTIME_WS_URL}?model=gpt-realtime", {"Authorization": f"Bearer {api_key}"})
        ]
        assert isinstance(session, OpenAIRealtimeSession)
        assert session.input_audio_format == PCM16_24K
        assert session.output_audio_format == PCM16_24K
        assert socket.sent == [
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": "Be brief.",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "noise_reduction": {"type": "near_field"},
                            "turn_detection": {
                                "type": "server_vad",
                                "create_response": False,
                                "interrupt_response": True,
                            },
                            "transcription": {"model": "gpt-4o-mini-transcribe"},
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "voice": "cedar",
                        },
                    },
                    "tools": [
                        {
                            "type": "function",
                            "name": "lookup",
                            "description": "Look things up",
                            "parameters": {"type": "object"},
                        }
                    ],
                    "tool_choice": "auto",
                },
            }
        ]
        assert api_key not in json.dumps(socket.sent)

    @pytest.mark.asyncio
    async def test_defaults_fill_model_and_voice_and_omit_tools(self, provider, socket, api_key):
        await provider.open_session(openai_plugin.provider.RealtimeVoiceSetup())

        assert provider.dials[0][0] == f"{REALTIME_WS_URL}?model={DEFAULT_MODEL}"
        session_payload = socket.sent[0]["session"]
        assert session_payload["audio"]["output"]["voice"] == DEFAULT_VOICE
        assert "tools" not in session_payload
        assert session_payload["audio"]["input"]["turn_detection"]["create_response"] is True

    @pytest.mark.asyncio
    async def test_refuses_without_a_key_before_dialing(self, provider, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            await provider.open_session(openai_plugin.provider.RealtimeVoiceSetup())
        assert provider.dials == []

    @pytest.mark.asyncio
    async def test_refuses_unsupported_audio_formats_before_dialing(self, provider, api_key):
        narrow = RealtimeAudioFormat(sample_rate_hz=16_000)

        with pytest.raises(ValueError, match="input audio must be audio/pcm 24000 Hz"):
            await provider.open_session(
                openai_plugin.provider.RealtimeVoiceSetup(input_audio=narrow)
            )
        assert provider.dials == []

    @pytest.mark.asyncio
    async def test_failed_session_update_closes_the_socket(self, provider, socket, api_key):
        async def broken_send(text):
            raise OSError("write failed")

        socket.send = broken_send  # type: ignore[assignment]

        with pytest.raises(OSError, match="write failed"):
            await provider.open_session(openai_plugin.provider.RealtimeVoiceSetup())
        assert socket.closed == 1


# -- commands ----------------------------------------------------------------


class TestCommands:
    @pytest.mark.asyncio
    async def test_audio_commit_response_cancel_truncate_and_context(self, socket):
        session = OpenAIRealtimeSession(socket)

        await session.send_audio(b"\x01\x02")
        await session.send_audio(b"")
        await session.commit_audio()
        await session.create_response()
        await session.create_response(metadata={"correlation": "x"})
        await session.cancel_response()
        await session.cancel_response("resp_1")
        await session.truncate_output("item_1", 1500)
        await session.add_context("ctx_1", "The user is Pedro.")
        await session.remove_context("ctx_1")

        assert socket.sent == [
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x01\x02").decode()},
            {"type": "input_audio_buffer.commit"},
            {"type": "response.create"},
            {"type": "response.create", "response": {"metadata": {"correlation": "x"}}},
            {"type": "response.cancel"},
            {"type": "response.cancel", "response_id": "resp_1"},
            {
                "type": "conversation.item.truncate",
                "item_id": "item_1",
                "content_index": 0,
                "audio_end_ms": 1500,
            },
            {
                "type": "conversation.item.create",
                "item": {
                    "id": "ctx_1",
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "The user is Pedro."}],
                },
            },
            {"type": "conversation.item.delete", "item_id": "ctx_1"},
        ]

    @pytest.mark.asyncio
    async def test_tool_results_go_out_as_one_batch_then_one_continuation(self, socket):
        session = OpenAIRealtimeSession(socket)
        results = [RealtimeToolResult("call_a", "A"), RealtimeToolResult("call_b", "B")]

        await session.submit_tool_results(results)
        await session.submit_tool_results(results[:1], continue_response=False)

        assert socket.sent == [
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_a", "output": "A"},
            },
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_b", "output": "B"},
            },
            {"type": "response.create"},
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_a", "output": "A"},
            },
        ]

    @pytest.mark.asyncio
    async def test_close_closes_the_socket_once(self, socket):
        session = OpenAIRealtimeSession(socket)
        await session.close()
        await session.close()
        assert socket.closed == 1


# -- decoding ----------------------------------------------------------------


class TestDecodeEvent:
    @pytest.mark.parametrize(
        ("wire", "expected"),
        [
            ({"type": "session.created", "session": {"id": "sess_1"}}, SessionReady("sess_1")),
            ({"type": "session.created", "session": {}}, SessionReady("session")),
            (
                {"type": "input_audio_buffer.speech_started", "item_id": "in_1", "audio_start_ms": 120},
                InputSpeechStarted(item_id="in_1", audio_start_ms=120),
            ),
            (
                {"type": "input_audio_buffer.speech_stopped", "item_id": "in_1", "audio_end_ms": 900},
                InputSpeechStopped(item_id="in_1", audio_end_ms=900),
            ),
            (
                {"type": "input_audio_buffer.committed", "item_id": "in_1"},
                InputAudioCommitted(item_id="in_1"),
            ),
            (
                {"type": "response.created", "response": {"id": "resp_1", "metadata": {"k": 1}}},
                ResponseStarted(response_id="resp_1", metadata={"k": "1"}),
            ),
            (
                {
                    "type": "response.output_audio.delta",
                    "response_id": "resp_1",
                    "item_id": "item_1",
                    "delta": base64.b64encode(b"pcm").decode(),
                },
                OutputAudio(data=b"pcm", item_id="item_1", response_id="resp_1"),
            ),
            (
                {
                    "type": "response.output_audio_transcript.delta",
                    "response_id": "resp_1",
                    "item_id": "item_1",
                    "delta": "Hel",
                },
                OutputTranscript(text="Hel", final=False, item_id="item_1", response_id="resp_1"),
            ),
            (
                {
                    "type": "response.output_audio_transcript.done",
                    "response_id": "resp_1",
                    "transcript": "Hello.",
                },
                OutputTranscript(text="Hello.", final=True, response_id="resp_1"),
            ),
            (
                {"type": "response.output_text.delta", "response_id": "resp_1", "delta": "txt"},
                OutputTranscript(text="txt", final=False, response_id="resp_1"),
            ),
            (
                {"type": "response.output_text.done", "response_id": "resp_1", "text": "txt."},
                OutputTranscript(text="txt.", final=True, response_id="resp_1"),
            ),
            (
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": "in_1",
                    "delta": "hi",
                },
                InputTranscript(text="hi", final=False, item_id="in_1"),
            ),
            (
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "in_1",
                    "transcript": "  hi there \n",
                },
                InputTranscript(text="hi there", final=True, item_id="in_1"),
            ),
            (
                {
                    "type": "response.function_call_arguments.done",
                    "response_id": "resp_1",
                    "item_id": "item_2",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"q": "x"}',
                },
                ToolCall(
                    call_id="call_1",
                    name="lookup",
                    arguments='{"q": "x"}',
                    response_id="resp_1",
                    item_id="item_2",
                ),
            ),
            (
                {"type": "response.done", "response": {"id": "resp_1", "status": "completed"}},
                ResponseCompleted(response_id="resp_1", status="completed"),
            ),
            (
                {"type": "error", "error": {"type": "invalid_request_error", "code": "bad", "message": "no"}},
                SessionFailure(code="bad", message="no", terminal=False),
            ),
        ],
    )
    def test_maps_ga_events_to_contract_events(self, wire, expected):
        assert decode_event(wire) == expected

    @pytest.mark.parametrize(
        "wire",
        [
            {"type": "rate_limits.updated"},
            {"type": "session.updated", "session": {"id": "sess_1"}},
            {"type": "response.output_item.added"},
            {"type": "response.output_audio_transcript.delta", "delta": ""},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "  "},
            {"type": "error", "error": {"message": "Cancellation failed: no active response found"}},
            {},
        ],
    )
    def test_ignores_noise_and_the_benign_cancel_race(self, wire):
        assert decode_event(wire) is None

    def test_malformed_audio_is_a_non_terminal_failure(self):
        empty = decode_event({"type": "response.output_audio.delta", "delta": ""})
        bad = decode_event({"type": "response.output_audio.delta", "delta": "@@not-b64@@"})

        assert isinstance(empty, SessionFailure) and empty.terminal is False
        assert isinstance(bad, SessionFailure) and bad.terminal is False

    def test_malformed_identifier_is_a_non_terminal_protocol_failure(self):
        event = decode_event(
            {"type": "response.function_call_arguments.done", "call_id": "", "name": "lookup"}
        )

        assert isinstance(event, SessionFailure)
        assert event.terminal is False
        assert event.code == "protocol"
        assert "response.function_call_arguments.done" in event.message


# -- receive loop ------------------------------------------------------------


class TestEventStream:
    @pytest.mark.asyncio
    async def test_frames_flow_and_a_clean_close_ends_the_stream(self, socket):
        session = OpenAIRealtimeSession(socket)
        socket.feed(
            _frame({"type": "session.created", "session": {"id": "sess_1"}}),
            b'{"type": "response.created", "response": {"id": "resp_1"}}',
            "not json at all",
            _frame([1, 2, 3]),
            _frame({"type": "response.done", "response": {"id": "resp_1"}}),
            ConnectionClosedOK(Close(1000, "bye"), None),
        )

        events = [event async for event in session.events()]

        assert events == [
            SessionReady("sess_1"),
            ResponseStarted(response_id="resp_1"),
            SessionFailure(code="protocol", message="provider sent malformed JSON", terminal=False),
            SessionFailure(code="protocol", message="provider sent a non-object event", terminal=False),
            ResponseCompleted(response_id="resp_1"),
            SessionClosed(reason="code 1000: bye"),
        ]

    @pytest.mark.asyncio
    async def test_abnormal_close_is_a_terminal_failure(self, socket):
        session = OpenAIRealtimeSession(socket)
        socket.feed(ConnectionClosedError(None, Close(1011, "server error")))

        events = [event async for event in session.events()]

        assert len(events) == 1
        failure = events[0]
        assert isinstance(failure, SessionFailure)
        assert failure.terminal is True
        assert failure.code == "connection_closed"
        assert "code 1011: server error" in failure.message

    @pytest.mark.asyncio
    async def test_transport_exception_is_a_terminal_failure(self, socket):
        session = OpenAIRealtimeSession(socket)
        socket.feed(OSError("network unreachable"))

        events = [event async for event in session.events()]

        assert events == [
            SessionFailure(code="transport", message="OSError: network unreachable")
        ]

    @pytest.mark.asyncio
    async def test_end_of_iteration_without_close_frame_is_a_clean_close(self):
        class EndingSocket(FakeWebSocket):
            async def __anext__(self):
                raise StopAsyncIteration

        session = OpenAIRealtimeSession(EndingSocket())

        assert [event async for event in session.events()] == [
            SessionClosed(reason="end of stream")
        ]


# -- end to end through the orchestrator -------------------------------------


class _Host(RealtimeVoiceHost):
    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.transcripts: list[tuple[str, bool]] = []
        self.barge_ins = 0

    def on_output_audio(self, pcm):
        self.audio.append(pcm)

    def on_output_transcript(self, text, final):
        self.transcripts.append((text, final))

    def on_barge_in(self):
        self.barge_ins += 1
        return 420


@pytest.mark.asyncio
async def test_orchestrator_drives_a_tool_turn_and_a_barge_in_over_the_wire(socket):
    async def executor(name, arguments):
        return f"{name}={arguments['city']}"

    session = OpenAIRealtimeSession(socket)
    host = _Host()
    orchestrator = RealtimeVoiceOrchestrator(session, host, tool_executor=executor)
    socket.feed(
        _frame({"type": "session.created", "session": {"id": "sess_1"}}),
        _frame({"type": "response.created", "response": {"id": "resp_1"}}),
        _frame(
            {
                "type": "response.function_call_arguments.done",
                "response_id": "resp_1",
                "item_id": "item_1",
                "call_id": "call_1",
                "name": "weather",
                "arguments": '{"city": "Lisbon"}',
            }
        ),
        _frame({"type": "response.done", "response": {"id": "resp_1", "status": "completed"}}),
    )
    run_task = asyncio.create_task(orchestrator.run())
    for _ in range(20):
        await asyncio.sleep(0)
    assert socket.sent == [
        {
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": "call_1", "output": "weather=Lisbon"},
        },
        {"type": "response.create"},
    ]

    socket.sent.clear()
    socket.feed(
        _frame({"type": "response.created", "response": {"id": "resp_2"}}),
        _frame(
            {
                "type": "response.output_audio.delta",
                "response_id": "resp_2",
                "item_id": "item_2",
                "delta": base64.b64encode(b"speech").decode(),
            }
        ),
        _frame({"type": "input_audio_buffer.speech_started", "item_id": "in_2", "audio_start_ms": 5}),
        _frame(
            {
                "type": "response.output_audio.delta",
                "response_id": "resp_2",
                "item_id": "item_2",
                "delta": base64.b64encode(b"tail").decode(),
            }
        ),
        _frame({"type": "response.done", "response": {"id": "resp_2", "status": "cancelled"}}),
        ConnectionClosedOK(Close(1000, ""), None),
    )
    await asyncio.wait_for(run_task, timeout=2)

    assert host.audio == [b"speech"]
    assert host.barge_ins == 1
    assert socket.sent == [
        {"type": "response.cancel", "response_id": "resp_2"},
        {
            "type": "conversation.item.truncate",
            "item_id": "item_2",
            "content_index": 0,
            "audio_end_ms": 420,
        },
    ]
    assert socket.closed == 1

"""Tests for ``hermes realtime`` (hermes_cli.realtime_voice) without devices or network."""

from __future__ import annotations

import argparse
import asyncio

import pytest

from agent import realtime_voice_registry
from agent.realtime_voice_provider import (
    PCM16_24K,
    InputTranscript,
    OutputAudio,
    OutputTranscript,
    RealtimeAudioFormat,
    RealtimeCapability,
    RealtimeVoiceProvider,
    RealtimeVoiceSession,
    ResponseCompleted,
    ResponseStarted,
    SessionClosed,
    SessionFailure,
    SessionReady,
    ToolCall,
)
from hermes_cli import realtime_voice
from hermes_cli.realtime_voice import (
    DEFAULT_INSTRUCTIONS,
    RealtimeAudioError,
    TerminalVoiceHost,
    cmd_realtime,
    make_tool_executor,
    run_session,
    to_realtime_tools,
)


class FakeAudio:
    def __init__(self, chunks=()) -> None:
        self.chunks = list(chunks)
        self.started: tuple | None = None
        self.stopped = 0
        self.played: list[bytes] = []
        self.resets = 0
        self.played_ms = 0
        self.pending = False
        self.fail_start: Exception | None = None

    def start(self, input_format, output_format) -> None:
        if self.fail_start is not None:
            raise self.fail_start
        self.started = (input_format, output_format)

    def stop(self) -> None:
        self.stopped += 1

    def read_input_chunk(self):
        return self.chunks.pop(0) if self.chunks else None

    def queue_playback(self, pcm: bytes) -> None:
        self.played.append(pcm)

    def drain_playback(self) -> bool:
        dropped = self.pending
        self.pending = False
        return dropped

    def reset_played_ms(self) -> None:
        self.resets += 1
        self.played_ms = 0


class FakeSession(RealtimeVoiceSession):
    def __init__(self, script) -> None:
        super().__init__(
            {RealtimeCapability.TOOL_CALLING},
            input_audio=RealtimeAudioFormat(sample_rate_hz=16_000),
            output_audio=PCM16_24K,
        )
        self.script = list(script)
        self.sent_audio: list[bytes] = []
        self.submissions = []
        self.close_count = 0
        self.gate = asyncio.Event()

    async def send_audio(self, audio: bytes) -> None:
        self.sent_audio.append(audio)
        if len(self.sent_audio) >= 2:
            self.gate.set()

    async def _submit_tool_results(self, results, continue_response) -> None:
        self.submissions.append((results, continue_response))

    def _events(self):
        async def stream():
            for event in self.script:
                if event == "wait-for-audio":
                    await self.gate.wait()
                    continue
                if event == "yield":
                    for _ in range(10):
                        await asyncio.sleep(0)
                    continue
                yield event

        return stream()

    async def _close(self) -> None:
        self.close_count += 1


class FakeProvider(RealtimeVoiceProvider):
    capabilities = frozenset({RealtimeCapability.TOOL_CALLING})

    def __init__(self, script=(), *, available=True, name="fake") -> None:
        self._script = script
        self._available = available
        self._name = name
        self.setups = []
        self.session: FakeSession | None = None

    @property
    def name(self):
        return self._name

    def is_available(self):
        return self._available

    def list_models(self):
        return ({"id": "fake-model"},)

    def list_voices(self):
        return ({"id": "fake-voice"},)

    def get_setup_schema(self):
        return {"name": "Fake", "badge": "", "tag": "", "env_vars": ({"key": "FAKE_KEY"},)}

    async def open_session(self, setup):
        self.setups.append(setup)
        self.session = FakeSession(self._script)
        return self.session


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    realtime_voice_registry._reset_for_tests()
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda force=False: None)
    yield
    realtime_voice_registry._reset_for_tests()


@pytest.fixture
def out():
    lines: list[str] = []
    return lines


# -- tool bridge -------------------------------------------------------------


def test_to_realtime_tools_converts_chat_schemas_and_skips_junk() -> None:
    definitions = [
        {
            "type": "function",
            "function": {
                "name": " web_search ",
                "description": "Search",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        },
        {"type": "function", "function": {"name": "bare"}},
        {"type": "function", "function": {"name": ""}},
        {"type": "function"},
        "garbage",
    ]

    tools = to_realtime_tools(definitions)

    assert [tool.name for tool in tools] == ["web_search", "bare"]
    assert tools[0].parameters == {"type": "object", "properties": {"q": {"type": "string"}}}
    assert tools[1].parameters == {"type": "object"}
    assert tools[1].description == ""


@pytest.mark.asyncio
async def test_tool_executor_runs_handle_function_call_off_the_loop(monkeypatch) -> None:
    import model_tools

    seen = {}

    def fake_handle(name, args, **kwargs):
        import threading

        seen["name"] = name
        seen["args"] = args
        seen["kwargs"] = kwargs
        seen["thread"] = threading.current_thread()
        return '{"ok": true}'

    monkeypatch.setattr(model_tools, "handle_function_call", fake_handle)
    execute = make_tool_executor(task_id="realtime-abc", toolset="hermes-cli")

    result = await execute("terminal", {"command": "ls"})

    assert result == '{"ok": true}'
    assert seen["name"] == "terminal"
    assert seen["args"] == {"command": "ls"}
    assert seen["kwargs"]["task_id"] == "realtime-abc"
    assert seen["kwargs"]["enabled_toolsets"] == ["hermes-cli"]
    assert seen["kwargs"]["tool_call_id"].startswith("realtime-abc-")
    assert seen["thread"].daemon is True


# -- terminal host -----------------------------------------------------------


def test_terminal_host_barge_in_reports_played_ms_only_when_mid_playback(out) -> None:
    audio = FakeAudio()
    host = TerminalVoiceHost(audio, out=out.append)

    assert host.on_barge_in() is None  # nothing ever played

    host.on_response_started("r1")
    host.on_output_item_started("item1")
    host.on_output_audio(b"pcm")
    audio.played_ms = 340
    assert audio.resets == 1
    assert host.on_barge_in() == 340  # response still open: truncate at 340 ms

    host.on_output_item_started("item2")
    host.on_response_completed("r1")
    audio.played_ms = 900
    audio.pending = True
    assert host.on_barge_in() == 900  # response done but audio still queued

    host.on_output_item_started("item3")
    audio.pending = False
    assert host.on_barge_in() is None  # fully played: nothing to truncate


def test_terminal_host_prints_transcripts_and_tool_lines(out, capsys) -> None:
    host = TerminalVoiceHost(FakeAudio(), out=out.append)
    call = ToolCall(call_id="c1", name="web_search", arguments="{}")

    host.on_session_ready("sess")
    host.on_input_transcript("partial", False)
    host.on_input_transcript("hello there", True)
    host.on_output_transcript("Hi ", False)
    host.on_output_transcript("there.", False)
    host.on_output_transcript("Hi there.", True)
    host.on_tool_call(call)
    host.on_tool_result(call, "…", "ok")
    host.on_tool_result(call, "…", "timeout")
    host.on_error("wobble", terminal=False)
    host.on_session_closed("")

    assert out == [
        "realtime: connected (session sess). Speak; Ctrl+C hangs up.",
        "You: hello there",
        "→ tool: web_search",
        "← tool web_search: timeout",
        "realtime: warning: wobble",
        "realtime: session closed (no reason given)",
    ]
    assert capsys.readouterr().out == "Hermes: Hi there.\n"


# -- session runner ----------------------------------------------------------


@pytest.mark.asyncio
async def test_run_session_reports_unknown_and_unconfigured_providers(out) -> None:
    realtime_voice_registry.register_provider(FakeProvider(available=False, name="offline"))

    assert await run_session(provider_name="missing", out=out.append) == 2
    assert out[0] == "realtime: unknown provider 'missing'. Registered providers:"
    assert any("offline" in line and "needs setup" in line for line in out[1:])

    out.clear()
    assert await run_session(provider_name="offline", out=out.append) == 2
    assert out == ["realtime: provider 'offline' is not configured (set FAKE_KEY)"]


@pytest.mark.asyncio
async def test_run_session_streams_mic_audio_playback_and_transcripts(out, monkeypatch) -> None:
    monkeypatch.setattr(
        realtime_voice,
        "build_tool_catalog",
        lambda toolset: to_realtime_tools(
            [{"type": "function", "function": {"name": f"tool_from_{toolset}"}}]
        ),
    )
    provider = FakeProvider(
        [
            SessionReady(session_id="sess"),
            "wait-for-audio",
            InputTranscript(text="what time is it", final=True),
            ResponseStarted(response_id="r1"),
            OutputAudio(data=b"pcm-1", item_id="i1", response_id="r1"),
            OutputTranscript(text="It is noon.", final=True, response_id="r1"),
            ResponseCompleted(response_id="r1"),
            SessionClosed(reason="hangup"),
        ]
    )
    realtime_voice_registry.register_provider(provider)
    audio = FakeAudio(chunks=[b"mic-1", b"mic-2"])

    code = await run_session(
        provider_name="fake",
        voice="fake-voice",
        toolset="voice-set",
        audio_factory=lambda: audio,
        out=out.append,
    )

    assert code == 0
    setup = provider.setups[0]
    assert setup.model == "fake-model"
    assert setup.voice == "fake-voice"
    assert setup.instructions == DEFAULT_INSTRUCTIONS
    assert [tool.name for tool in setup.tools] == ["tool_from_voice-set"]
    assert audio.started == (RealtimeAudioFormat(sample_rate_hz=16_000), PCM16_24K)
    assert provider.session.sent_audio == [b"mic-1", b"mic-2"]
    assert audio.played == [b"pcm-1"]
    assert audio.stopped == 1
    assert provider.session.close_count == 1
    assert "You: what time is it" in out
    assert "Hermes: It is noon." in out
    assert out[-1] == "realtime: session closed (hangup)"


@pytest.mark.asyncio
async def test_run_session_without_tools_opens_a_toolless_session(out, monkeypatch) -> None:
    def explode(toolset):
        raise AssertionError("tool catalog must not be built with --no-tools")

    monkeypatch.setattr(realtime_voice, "build_tool_catalog", explode)
    provider = FakeProvider([SessionClosed()])
    realtime_voice_registry.register_provider(provider)

    code = await run_session(
        provider_name="fake",
        tools_enabled=False,
        audio_factory=lambda: FakeAudio(),
        out=out.append,
    )

    assert code == 0
    assert provider.setups[0].tools == ()


@pytest.mark.asyncio
async def test_run_session_returns_one_on_terminal_failure(out, monkeypatch) -> None:
    monkeypatch.setattr(realtime_voice, "build_tool_catalog", lambda toolset: ())
    provider = FakeProvider([SessionFailure(code="connection_closed", message="gone")])
    realtime_voice_registry.register_provider(provider)
    audio = FakeAudio()

    code = await run_session(provider_name="fake", audio_factory=lambda: audio, out=out.append)

    assert code == 1
    assert out[-1] == "realtime: session failed: connection_closed: gone"
    assert audio.stopped == 1
    assert provider.session.close_count == 1


@pytest.mark.asyncio
async def test_run_session_closes_session_when_audio_devices_fail(out, monkeypatch) -> None:
    monkeypatch.setattr(realtime_voice, "build_tool_catalog", lambda toolset: ())
    provider = FakeProvider([SessionClosed()])
    realtime_voice_registry.register_provider(provider)
    audio = FakeAudio()
    audio.fail_start = RealtimeAudioError("no microphone")

    code = await run_session(provider_name="fake", audio_factory=lambda: audio, out=out.append)

    assert code == 1
    assert out[-1] == "realtime: no microphone"
    assert provider.session.close_count == 1


@pytest.mark.asyncio
async def test_cancelling_the_session_stops_the_mic_and_closes_cleanly(out, monkeypatch) -> None:
    monkeypatch.setattr(realtime_voice, "build_tool_catalog", lambda toolset: ())
    provider = FakeProvider([SessionReady(session_id="sess"), "wait-for-audio"])
    realtime_voice_registry.register_provider(provider)
    audio = FakeAudio()

    task = asyncio.create_task(
        run_session(provider_name="fake", audio_factory=lambda: audio, out=out.append)
    )
    for _ in range(20):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert audio.stopped == 1
    assert provider.session.close_count == 1
    leftovers = [
        t for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and t.get_name().startswith("realtime-voice-")
    ]
    assert leftovers == []


# -- argparse entry ----------------------------------------------------------


def test_cmd_realtime_list_prints_providers(capsys) -> None:
    realtime_voice_registry.register_provider(FakeProvider(name="fake"))
    realtime_voice_registry.register_provider(FakeProvider(available=False, name="other"))

    assert cmd_realtime(argparse.Namespace(list=True)) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("fake") and "ready" in lines[0]
    assert lines[1].startswith("other") and "needs setup" in lines[1]


def test_cmd_realtime_list_with_no_providers(capsys) -> None:
    assert cmd_realtime(argparse.Namespace(list=True)) == 0
    assert "No realtime voice providers" in capsys.readouterr().out


def test_cmd_realtime_passes_flags_and_normalizes_provider(monkeypatch) -> None:
    seen = {}

    async def fake_run(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(realtime_voice, "run_session", fake_run)
    args = argparse.Namespace(
        list=False,
        provider=" OpenAI ",
        model="gpt-realtime",
        voice="cedar",
        toolset=None,
        no_tools=True,
        tool_timeout=12.5,
    )

    assert cmd_realtime(args) == 0
    assert seen == {
        "provider_name": "openai",
        "model": "gpt-realtime",
        "voice": "cedar",
        "toolset": "hermes-cli",
        "tools_enabled": False,
        "tool_timeout_s": 12.5,
    }


def test_cmd_realtime_reports_failures_as_one_line(monkeypatch, capsys) -> None:
    async def boom(**kwargs):
        raise OSError("dial failed")

    monkeypatch.setattr(realtime_voice, "run_session", boom)
    args = argparse.Namespace(list=False, provider=None, model=None, voice=None,
                              toolset=None, no_tools=False, tool_timeout=None)

    assert cmd_realtime(args) == 1
    assert "realtime: OSError: dial failed" in capsys.readouterr().err


def test_realtime_is_a_builtin_subcommand() -> None:
    from hermes_cli.main import _BUILTIN_SUBCOMMANDS

    assert "realtime" in _BUILTIN_SUBCOMMANDS

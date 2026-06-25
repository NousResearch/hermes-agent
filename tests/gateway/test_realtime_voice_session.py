"""Tests for provider-neutral realtime voice session contracts."""

from __future__ import annotations

import asyncio
import json

import pytest

from gateway.realtime_voice.config import DEFAULT_REALTIME_TOOLS, RealtimeVoiceConfig
from gateway.realtime_voice.session import (
    RealtimeAudioDelta,
    RealtimeToolCall,
    RealtimeTranscriptDelta,
    RealtimeVoiceSession,
)
from gateway.realtime_voice.xai import XAIRealtimeVoiceSession


def test_realtime_voice_config_defaults_are_safe_for_experimental_discord_voice():
    config = RealtimeVoiceConfig()

    assert config.provider == "xai"
    assert config.model == "grok-voice-latest"
    assert config.voice == "ara"
    assert config.max_session_minutes == 20
    assert config.max_background_tasks == 3
    assert config.transcript_to_text_channel is True
    assert config.allow_tools == DEFAULT_REALTIME_TOOLS


def test_realtime_voice_config_from_dict_coerces_limits_and_tools():
    config = RealtimeVoiceConfig.from_dict(
        {
            "provider": "XAI",
            "model": "grok-voice-latest",
            "voice": "alloy",
            "max_session_minutes": "0",
            "max_background_tasks": "-2",
            "transcript_to_text_channel": "false",
            "allow_tools": ["ask_agent", 123, "get_agent_task_status"],
            "providers": {"xai": {"model": "grok-voice-latest"}},
        }
    )

    assert config.provider == "xai"
    assert config.model == "grok-voice-latest"
    assert config.voice == "alloy"
    assert config.max_session_minutes == 1
    assert config.max_background_tasks == 0
    assert config.transcript_to_text_channel is False
    assert config.allow_tools == ("ask_agent", "get_agent_task_status")
    assert config.providers == {"xai": {"model": "grok-voice-latest"}}


def test_realtime_voice_config_empty_allow_tools_disables_all_tools():
    """Promise: an explicit empty allow_tools list disables provider tool exposure.

    Real failure caught: profile config cannot turn realtime tools off because an empty
    list is silently replaced with defaults.
    """
    config = RealtimeVoiceConfig.from_dict({"allow_tools": []})

    assert config.allow_tools == ()


def test_realtime_provider_factory_creates_xai_session_from_neutral_config():
    """Promise: gateway code can ask the neutral factory for the configured provider.

    Real failure caught: Discord hard-codes xAI construction and future providers require
    gateway command rewrites.
    """
    from gateway.realtime_voice.providers import create_realtime_voice_session

    session = create_realtime_voice_session(
        RealtimeVoiceConfig.from_dict({"provider": "xai"}),
        instructions="be brief",
        on_event=lambda _event: None,
    )

    assert isinstance(session, XAIRealtimeVoiceSession)
    assert session.instructions == "be brief"


def test_realtime_provider_factory_rejects_unknown_provider_with_clear_error():
    """Promise: unsupported providers fail visibly instead of crashing at Discord join.

    Real failure caught: provider swapping silently falls through to xAI or raises an
    opaque import/runtime error.
    """
    from gateway.realtime_voice.providers import (
        RealtimeVoiceProviderError,
        create_realtime_voice_session,
    )

    with pytest.raises(RealtimeVoiceProviderError, match="openai.*xai"):
        create_realtime_voice_session(
            RealtimeVoiceConfig.from_dict({"provider": "openai"}),
            instructions="be brief",
            on_event=lambda _event: None,
        )


def test_realtime_provider_factory_allows_registered_future_provider_without_gateway_code():
    """Promise: future realtime providers plug into the registry/factory seam.

    Real failure caught: adding OpenAI/GPT realtime later requires rewriting Discord
    gateway command handling instead of only registering a provider adapter.
    """
    from gateway.realtime_voice.providers import (
        create_realtime_voice_session,
        register_realtime_voice_provider,
    )
    from gateway.realtime_voice.session import RealtimeVoiceSession

    class FakeProviderSession(RealtimeVoiceSession):
        def __init__(self, config, *, instructions="", on_event=None):
            self.config = config
            self.instructions = instructions
            self.on_event = on_event

        async def start(self):
            pass

        async def stop(self):
            pass

        async def send_audio_pcm16(self, data: bytes, sample_rate: int):
            pass

        async def interrupt(self):
            pass

        async def update_instructions(self, instructions: str):
            self.instructions = instructions

    register_realtime_voice_provider("fake-provider", FakeProviderSession)

    session = create_realtime_voice_session(
        RealtimeVoiceConfig.from_dict({"provider": "fake-provider"}),
        instructions="future seam",
        on_event=lambda _event: None,
    )

    assert isinstance(session, FakeProviderSession)
    assert session.instructions == "future seam"


def test_xai_realtime_session_payload_exposes_allowed_hermes_tools_and_provider_builtins():
    """Promise: xAI receives both provider-native tools and Hermes custom tools.

    Real failure caught: the realtime model can hear speech but cannot call the Hermes
    bridge tools requested by config.
    """
    config = RealtimeVoiceConfig.from_dict(
        {
            "allow_tools": ["ask_agent", "web_search", "x_search"],
            "providers": {"xai": {"sample_rate": 16000}},
        }
    )
    session = XAIRealtimeVoiceSession(config, instructions="be brief")

    payload = session._session_update_payload()

    assert payload["type"] == "session.update"
    assert payload["session"]["model"] == "grok-voice-latest"
    assert payload["session"]["voice"] == "ara"
    assert payload["session"]["instructions"] == "be brief"
    assert payload["session"]["audio"]["input"]["format"]["rate"] == 16000
    assert {tool["type"] for tool in payload["session"]["tools"]} == {
        "function",
        "web_search",
        "x_search",
    }
    ask_tool = next(tool for tool in payload["session"]["tools"] if tool.get("name") == "ask_agent")
    assert ask_tool["description"]
    assert ask_tool["parameters"]["type"] == "object"
    assert session._endpoint_for_base_url("https://api.x.ai/v1") == "wss://api.x.ai/v1/realtime"


def test_xai_realtime_session_payload_filters_disallowed_hermes_tools():
    """Promise: config allow_tools is the source of truth for exposed custom tools.

    Real failure caught: the model gets tool names that profile config explicitly did not
    allow, creating surprising realtime side effects.
    """
    session = XAIRealtimeVoiceSession(
        RealtimeVoiceConfig.from_dict({"allow_tools": ["web_search"]}),
        instructions="be brief",
    )

    tools = session._session_update_payload()["session"]["tools"]

    assert tools == [{"type": "web_search"}]


class _ToolResultSession:
    def __init__(self):
        self.results = []

    async def submit_tool_result(self, call_id: str, output: str) -> None:
        self.results.append((call_id, output))


@pytest.mark.asyncio
async def test_realtime_tool_bridge_executes_ask_agent_and_returns_provider_result():
    """Promise: provider tool calls execute Hermes bridge behavior, not just logging.

    Real failure caught: RealtimeToolCall(name="ask_agent") is observed but the provider
    never receives a tool result with the original call_id.
    """
    from gateway.realtime_voice.tool_bridge import RealtimeToolBridge

    async def ask_agent(prompt: str) -> str:
        assert prompt == "What changed?"
        return "The realtime tool bridge is now executing."

    provider_session = _ToolResultSession()
    bridge = RealtimeToolBridge(RealtimeVoiceConfig(), ask_agent=ask_agent)

    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall("ask_agent", {"prompt": "What changed?"}, "call_ask"),
    )

    assert provider_session.results == [
        ("call_ask", "The realtime tool bridge is now executing.")
    ]


@pytest.mark.asyncio
async def test_realtime_tool_bridge_ask_context_uses_read_only_context_provider():
    """Realtime voice can ask for prior context without exposing broad tools."""
    from gateway.realtime_voice.tool_bridge import RealtimeToolBridge, hermes_realtime_tool_definitions

    definitions = hermes_realtime_tool_definitions(["ask_context"])
    assert [definition.name for definition in definitions] == ["ask_context"]
    assert definitions[0].parameters["required"] == ["question"]

    async def ask_context(question: str) -> str:
        return f"Context says: {question} -> Monday plan"

    provider_session = _ToolResultSession()
    bridge = RealtimeToolBridge(
        RealtimeVoiceConfig(allow_tools=("ask_context",)),
        ask_agent=lambda _prompt: "unused",
        ask_context=ask_context,
    )

    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall("ask_context", {"question": "what did we decide Monday?"}, "call_context"),
    )

    assert provider_session.results == [
        ("call_context", "Context says: what did we decide Monday? -> Monday plan")
    ]


@pytest.mark.asyncio
async def test_realtime_tool_bridge_splits_spoken_result_from_display_artifacts():
    """Provider audio gets speak-safe text while Discord text can receive artifacts.

    Real failure caught: Hazel reads raw URLs aloud instead of saying a clean spoken
    sentence while the text channel gets the link/artifact payload.
    """
    from gateway.realtime_voice.tool_bridge import RealtimeToolBridge

    display_updates = []

    async def ask_agent(_prompt: str):
        return {
            "speak": "I found the reference and posted it in chat.",
            "display": "Reference: https://example.com/reference",
            "artifacts": [{"type": "url", "url": "https://example.com/reference"}],
        }

    async def on_tool_display(update):
        display_updates.append(update)

    provider_session = _ToolResultSession()
    bridge = RealtimeToolBridge(
        RealtimeVoiceConfig(),
        ask_agent=ask_agent,
        on_tool_display=on_tool_display,
    )

    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall("ask_agent", {"prompt": "find the ref"}, "call_display"),
    )

    assert provider_session.results == [
        ("call_display", "I found the reference and posted it in chat.")
    ]
    assert "https://" not in provider_session.results[0][1]
    assert display_updates == [
        {
            "tool": "ask_agent",
            "display": "Reference: https://example.com/reference",
            "artifacts": [{"type": "url", "url": "https://example.com/reference"}],
        }
    ]


@pytest.mark.asyncio
async def test_realtime_tool_bridge_denies_disallowed_tool_without_crashing_session():
    """Promise: disallowed tools are denied safely and visibly.

    Real failure caught: a provider can invoke an unconfigured tool and either crash the
    voice session or perform an unintended side effect.
    """
    from gateway.realtime_voice.tool_bridge import RealtimeToolBridge

    provider_session = _ToolResultSession()
    bridge = RealtimeToolBridge(
        RealtimeVoiceConfig.from_dict({"allow_tools": ["ask_agent"]}),
        ask_agent=lambda _prompt: "unused",
    )

    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall("terminal", {"command": "rm -rf /"}, "call_bad"),
    )

    assert provider_session.results
    assert provider_session.results[0][0] == "call_bad"
    assert "not allowed" in provider_session.results[0][1].lower()


@pytest.mark.asyncio
async def test_realtime_tool_bridge_reports_tool_failure_without_killing_session():
    """Promise: allowed tool failures become safe provider-visible results.

    Real failure caught: one failing Hermes call tears down the whole realtime voice
    session or leaves the provider waiting forever.
    """
    from gateway.realtime_voice.tool_bridge import RealtimeToolBridge

    async def broken_ask(_prompt: str) -> str:
        raise RuntimeError("secret stack detail")

    provider_session = _ToolResultSession()
    bridge = RealtimeToolBridge(RealtimeVoiceConfig(), ask_agent=broken_ask)

    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall("ask_agent", {"prompt": "fail"}, "call_fail"),
    )

    assert provider_session.results[0][0] == "call_fail"
    assert "failed" in provider_session.results[0][1].lower()
    assert "secret stack detail" not in provider_session.results[0][1]


@pytest.mark.asyncio
async def test_realtime_tool_bridge_runs_bounded_background_task_lifecycle():
    """Promise: realtime background tasks can be started, polled, and summarized.

    Real failure caught: voice-triggered long work either blocks audio, runs unbounded,
    or cannot be retrieved after completion.
    """
    from gateway.realtime_voice.tool_bridge import RealtimeToolBridge

    release = asyncio.Event()

    async def ask_agent(prompt: str) -> str:
        await release.wait()
        return f"finished {prompt}"

    provider_session = _ToolResultSession()
    bridge = RealtimeToolBridge(
        RealtimeVoiceConfig.from_dict({"max_background_tasks": 1}),
        ask_agent=ask_agent,
    )

    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall("start_agent_task", {"prompt": "deep work"}, "call_start"),
    )
    start_output = provider_session.results[-1][1]
    assert "task_id" in start_output
    task_id = json.loads(start_output)["task_id"]

    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall("start_agent_task", {"prompt": "too much"}, "call_overflow"),
    )
    assert "maximum" in provider_session.results[-1][1].lower()

    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall("get_agent_task_status", {"task_id": task_id}, "call_status"),
    )
    assert "running" in provider_session.results[-1][1].lower()

    release.set()
    await asyncio.sleep(0)
    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall("summarize_agent_task", {"task_id": task_id}, "call_summary"),
    )
    assert "finished deep work" in provider_session.results[-1][1]


@pytest.mark.asyncio
async def test_realtime_tool_bridge_notifies_when_background_task_completes():
    """Promise: voice-started background work pushes completion state without polling.

    Real failure caught: Kamell has to keep asking for status because realtime tasks
    finish silently after the voice model already acknowledged the start.
    """
    from gateway.realtime_voice.tool_bridge import RealtimeToolBridge

    release = asyncio.Event()
    updates = []

    async def ask_agent(prompt: str) -> str:
        await release.wait()
        return f"finished {prompt}"

    async def on_task_update(update):
        updates.append(update)

    provider_session = _ToolResultSession()
    bridge = RealtimeToolBridge(
        RealtimeVoiceConfig.from_dict({"max_background_tasks": 1}),
        ask_agent=ask_agent,
        on_task_update=on_task_update,
    )

    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall("start_agent_task", {"prompt": "cleanup"}, "call_start"),
    )
    task_id = json.loads(provider_session.results[-1][1])["task_id"]

    release.set()
    for _ in range(20):
        if updates:
            break
        await asyncio.sleep(0)

    assert updates == [
        {"task_id": task_id, "title": "cleanup", "status": "completed", "summary": "finished cleanup"}
    ]


@pytest.mark.asyncio
async def test_realtime_tool_bridge_start_task_returns_title_and_internal_id():
    """User-facing realtime tasks should have human titles while keeping machine ids."""
    from gateway.realtime_voice.tool_bridge import RealtimeToolBridge

    async def ask_agent(prompt: str) -> str:
        return f"finished {prompt}"

    provider_session = _ToolResultSession()
    bridge = RealtimeToolBridge(RealtimeVoiceConfig(), ask_agent=ask_agent)

    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall(
            "start_agent_task",
            {"prompt": "Build a website for the bakery", "title": "Bakery website"},
            "call_start",
        ),
    )

    payload = json.loads(provider_session.results[-1][1])
    assert payload["task_id"].startswith("rt_")
    assert payload["title"] == "Bakery website"
    assert payload["status"] == "running"


@pytest.mark.asyncio
async def test_realtime_tool_bridge_status_can_be_requested_by_title():
    """Voice users should be able to ask about a task by its title, not raw id."""
    from gateway.realtime_voice.tool_bridge import RealtimeToolBridge

    release = asyncio.Event()

    async def ask_agent(prompt: str) -> str:
        await release.wait()
        return f"finished {prompt}"

    provider_session = _ToolResultSession()
    bridge = RealtimeToolBridge(RealtimeVoiceConfig(), ask_agent=ask_agent)

    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall(
            "start_agent_task",
            {"prompt": "Clean stale files", "title": "Office cleanup"},
            "call_start",
        ),
    )
    started = json.loads(provider_session.results[-1][1])

    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall("get_agent_task_status", {"title": "office"}, "call_status"),
    )

    status = json.loads(provider_session.results[-1][1])
    assert status["task_id"] == started["task_id"]
    assert status["title"] == "Office cleanup"
    assert status["status"] == "running"


@pytest.mark.asyncio
async def test_realtime_tool_bridge_title_lookup_reports_ambiguity_instead_of_guessing():
    """Ambiguous spoken title lookups must not return the wrong task."""
    from gateway.realtime_voice.tool_bridge import RealtimeToolBridge

    release = asyncio.Event()

    async def ask_agent(prompt: str) -> str:
        await release.wait()
        return f"finished {prompt}"

    provider_session = _ToolResultSession()
    bridge = RealtimeToolBridge(RealtimeVoiceConfig(), ask_agent=ask_agent)

    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall("start_agent_task", {"prompt": "Build site A", "title": "Website draft"}, "call_a"),
    )
    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall("start_agent_task", {"prompt": "Build site B", "title": "Website polish"}, "call_b"),
    )

    await bridge.handle_tool_call(
        provider_session,
        RealtimeToolCall("summarize_agent_task", {"title": "website"}, "call_summary"),
    )

    output = provider_session.results[-1][1].lower()
    assert "multiple" in output or "ambiguous" in output
    assert "website draft" in output
    assert "website polish" in output


@pytest.mark.asyncio
async def test_xai_realtime_session_parses_provider_events():
    events = []

    async def on_event(event):
        events.append(event)

    session = XAIRealtimeVoiceSession(RealtimeVoiceConfig(), on_event=on_event)
    await session._handle_server_event(
        {"type": "response.output_audio.delta", "delta": "AAE="}
    )
    await session._handle_server_event(
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello"}
    )
    await session._handle_server_event(
        {"type": "response.function_call_arguments.done", "name": "ask_agent", "arguments": '{"prompt":"x"}', "call_id": "c1"}
    )

    assert isinstance(events[0], RealtimeAudioDelta)
    assert events[0].pcm16 == b"\x00\x01"
    assert isinstance(events[1], RealtimeTranscriptDelta)
    assert events[1].role == "user"
    assert isinstance(events[2], RealtimeToolCall)
    assert events[2].arguments == {"prompt": "x"}


def test_realtime_events_are_plain_data_objects():
    audio = RealtimeAudioDelta(pcm16=b"\x00\x01", sample_rate=24000)
    transcript = RealtimeTranscriptDelta(role="assistant", text="hello", final=True)
    tool_call = RealtimeToolCall(
        name="ask_agent",
        arguments={"prompt": "answer briefly"},
        call_id="call_123",
    )

    assert audio.pcm16 == b"\x00\x01"
    assert audio.sample_rate == 24000
    assert transcript.role == "assistant"
    assert transcript.final is True
    assert tool_call.arguments["prompt"] == "answer briefly"


@pytest.mark.asyncio
async def test_realtime_voice_session_contract_can_be_implemented():
    class FakeRealtimeSession(RealtimeVoiceSession):
        def __init__(self):
            self.calls = []

        async def start(self) -> None:
            self.calls.append(("start",))

        async def stop(self) -> None:
            self.calls.append(("stop",))

        async def send_audio_pcm16(self, data: bytes, sample_rate: int) -> None:
            self.calls.append(("audio", data, sample_rate))

        async def interrupt(self) -> None:
            self.calls.append(("interrupt",))

        async def update_instructions(self, instructions: str) -> None:
            self.calls.append(("instructions", instructions))

    session = FakeRealtimeSession()

    await session.start()
    await session.send_audio_pcm16(b"abc", 48000)
    await session.interrupt()
    await session.update_instructions("be brief")
    await session.stop()

    assert session.calls == [
        ("start",),
        ("audio", b"abc", 48000),
        ("interrupt",),
        ("instructions", "be brief"),
        ("stop",),
    ]

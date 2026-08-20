"""Observed photos must survive the queued-follow-up path too (#47415).

`_build_gateway_agent_history` lifts observed group rows out of the replayed
history into a text-only context prefix, so `result["messages"]` — what the
queued path reuses as history — never contains them.  A photo observed while a
previous turn was still running would therefore be invisible when the queued
`@bot` follow-up is prepared, even though the idle path handles it.
"""

import base64
import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource


_ONE_BY_ONE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6L2ioAAAAASUVORK5CYII="
)


class CaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="sent-1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class CaptureAgent:
    calls = []

    def __init__(self, **kwargs):
        self.tools = []
        self.tool_progress_callback = kwargs.get("tool_progress_callback")

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls.append(message)
        # The agent returns its own message list, which never carries observed
        # rows — this is the condition that loses the photo.
        return {
            "final_response": f"done-{len(type(self).calls)}",
            "messages": [],
            "api_calls": 1,
        }


class _FakeAsyncStore:
    """Minimal async session store exposing the persisted observed transcript."""

    def __init__(self, transcript):
        self._transcript = transcript

    async def get_or_create_session(self, source):
        return SimpleNamespace(session_id="sess-observed")

    async def load_transcript(self, session_id):
        return list(self._transcript)


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._decide_image_input_mode = lambda **_kw: "native"
    return runner


@pytest.mark.asyncio
async def test_queued_followup_attaches_photo_observed_during_previous_turn(monkeypatch, tmp_path):
    CaptureAgent.calls = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    observed_image = tmp_path / "observed.png"
    observed_image.write_bytes(_ONE_BY_ONE_PNG)

    # Persisted while the first turn was mid-flight: an unmentioned group photo.
    transcript = [
        {
            "role": "user",
            "observed": True,
            "content": (
                "[Alice Example|111]\nveja esta foto\n\n"
                f"[image 'observed.png' saved at: {observed_image}]"
            ),
            "media_urls": [str(observed_image)],
            "media_types": ["image/png"],
        },
    ]

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    # ``async_session_store`` is a read-only property that builds a facade over
    # the real SessionStore; swap it for the fake transcript.
    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "async_session_store",
        property(lambda self: _FakeAsyncStore(transcript)),
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
    )

    # The addressed follow-up carries no media of its own.
    adapter._pending_messages["agent:main:telegram:group:-1001"] = MessageEvent(
        text="@bot what was in that photo?",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-1",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-observed",
        session_key="agent:main:telegram:group:-1001",
    )

    assert result["final_response"] == "done-2"
    assert len(CaptureAgent.calls) == 2

    queued_message = CaptureAgent.calls[1]
    assert isinstance(queued_message, list), (
        "queued follow-up should be multimodal once the observed photo is attached"
    )
    assert any(part.get("type") == "image_url" for part in queued_message)

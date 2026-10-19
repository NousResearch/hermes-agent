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
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
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
    on_first_turn = None

    def __init__(self, **kwargs):
        self.tools = []
        self.tool_progress_callback = kwargs.get("tool_progress_callback")

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        type(self).calls.append(message)
        if len(type(self).calls) == 1 and type(self).on_first_turn:
            type(self).on_first_turn()
        # The agent returns its own message list, which never carries observed
        # rows — this is the condition that loses the photo.
        return {
            "final_response": f"done-{len(type(self).calls)}",
            "messages": [],
            "api_calls": 1,
        }


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

    from gateway.config import GatewayConfig
    from gateway.platforms.base import get_image_cache_dir
    from gateway.session import SessionStore, build_session_key
    from tests.gateway.test_observed_group_image_attachment import _observe_photo

    observed_image = get_image_cache_dir() / "observed.png"
    observed_image.write_bytes(_ONE_BY_ONE_PNG)
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    runner.session_store = store
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group")
    entry = store.get_or_create_session(source)
    session_key = build_session_key(source)

    def observe_during_active_turn():
        observed_source = _observe_photo(store, observed_image)
        assert build_session_key(observed_source) == session_key
        adapter._pending_messages[session_key] = MessageEvent(
            text="@bot what was in that photo?", message_type=MessageType.TEXT,
            source=source, message_id="queued-1",
        )

    CaptureAgent.on_first_turn = observe_during_active_turn

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id=entry.session_id,
        session_key=session_key,
    )

    assert result["final_response"] == "done-2"
    assert len(CaptureAgent.calls) == 2

    queued_message = CaptureAgent.calls[1]
    assert isinstance(queued_message, list), (
        "queued follow-up should be multimodal once the observed photo is attached"
    )
    assert any(part.get("type") == "image_url" for part in queued_message)

    store.close_all_db_handles()


@pytest.mark.asyncio
async def test_observed_store_failure_keeps_queued_agent_history():
    from gateway.observed_images import history_with_observed_tail

    class UnavailableStore:
        async def get_or_create_session(self, source):
            raise OSError("store unavailable")

    history = [{"role": "assistant", "content": "previous turn"}]
    runner = SimpleNamespace(async_session_store=UnavailableStore())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group")
    assert await history_with_observed_tail(runner, history, source) is history


@pytest.mark.asyncio
async def test_queued_observation_reload_runs_inside_owning_profile(monkeypatch, tmp_path):
    from gateway.observed_images import observed_image_content
    from gateway.session import build_session_key
    from hermes_constants import get_hermes_home

    owner = tmp_path / "owner"
    image = owner / "cache" / "images" / "observed.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(_ONE_BY_ONE_PNG)
    reads = []

    class ScopedStore:
        async def get_or_create_session(self, source):
            reads.append(get_hermes_home())
            return SimpleNamespace(session_id="owner-session")

        async def load_transcript(self, session_id):
            reads.append(get_hermes_home())
            return [{"role": "user", "observed": True,
                     "content": observed_image_content("photo", [str(image)], ["image/png"])}]

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run.GatewayRunner, "async_session_store", property(lambda self: ScopedStore()))
    runner = _make_runner(CaptureAdapter())
    runner.config.multiplex_profiles = True
    runner._resolve_profile_home_for_source = lambda source: owner
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group", profile="owner")
    key = build_session_key(source)
    await runner._prepare_profile_scoped_inbound_message_text(
        event=MessageEvent(text="describe it", source=source), source=source,
        history=[], session_key=key, reload_observed_tail=True,
    )
    assert reads == [owner, owner]
    assert runner._consume_pending_native_image_paths(key) == [str(image)]

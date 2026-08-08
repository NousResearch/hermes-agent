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
        self.typing = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="sent-1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class CaptureQueuedNativeImageAgent:
    calls = []
    session_ids = []
    rotate_to_session_id: str | None = None
    omit_first_session_id = False

    def __init__(self, **kwargs):
        self.tools = []
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.session_id = kwargs.get("session_id")
        type(self).session_ids.append(self.session_id)

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls.append(message)
        if len(type(self).calls) == 1 and type(self).rotate_to_session_id:
            self.session_id = type(self).rotate_to_session_id
        result = {
            "final_response": f"done-{len(type(self).calls)}",
            "messages": [],
            "api_calls": 1,
        }
        if not (len(type(self).calls) == 1 and type(self).omit_first_session_id):
            result["session_id"] = self.session_id
        return result


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
    runner.session_store = SimpleNamespace(_entries={})
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
async def test_queued_followup_uses_pending_event_session_key_for_native_images(monkeypatch, tmp_path):
    CaptureQueuedNativeImageAgent.calls = []
    CaptureQueuedNativeImageAgent.session_ids = []
    CaptureQueuedNativeImageAgent.rotate_to_session_id = None
    CaptureQueuedNativeImageAgent.omit_first_session_id = False

    fake_dotenv = types.ModuleType("dotenv")
    setattr(fake_dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    setattr(fake_run_agent, "AIAgent", CaptureQueuedNativeImageAgent)
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)

    image_path = tmp_path / "queued-image.png"
    image_path.write_bytes(_ONE_BY_ONE_PNG)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
    )
    pending_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )

    adapter._pending_messages["agent:main:telegram:group:-1001"] = MessageEvent(
        text="describe this",
        message_type=MessageType.PHOTO,
        source=pending_source,
        media_urls=[str(image_path)],
        media_types=["image/png"],
        message_id="queued-1",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-native-image-followup",
        session_key="agent:main:telegram:group:-1001",
    )

    assert result["final_response"] == "done-2"
    assert len(CaptureQueuedNativeImageAgent.calls) == 2
    queued_message = CaptureQueuedNativeImageAgent.calls[1]
    assert isinstance(queued_message, list)
    assert queued_message[0]["type"] == "text"
    assert queued_message[0]["text"].startswith("describe this")
    assert any(part.get("type") == "image_url" for part in queued_message)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "initial_session_id",
        "rotated_session_id",
        "omit_first_session_id",
        "expected_session_id",
        "expected_session_ids",
    ),
    [
        (
            "parent-session",
            "continuation-session",
            False,
            "continuation-session",
            ["parent-session", "continuation-session"],
        ),
        (
            "parent-session",
            None,
            True,
            "parent-session",
            ["parent-session", "parent-session"],
        ),
    ],
    ids=["rotated-session", "missing-session-id"],
)
async def test_queued_followup_uses_session_id_returned_by_prior_turn(
    monkeypatch,
    tmp_path,
    initial_session_id,
    rotated_session_id,
    omit_first_session_id,
    expected_session_id,
    expected_session_ids,
):
    CaptureQueuedNativeImageAgent.calls = []
    CaptureQueuedNativeImageAgent.session_ids = []
    CaptureQueuedNativeImageAgent.rotate_to_session_id = rotated_session_id
    CaptureQueuedNativeImageAgent.omit_first_session_id = omit_first_session_id

    fake_dotenv = types.ModuleType("dotenv")
    setattr(fake_dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    setattr(fake_run_agent, "AIAgent", CaptureQueuedNativeImageAgent)
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    session_key = "agent:main:telegram:dm:42"
    cache_refreshes = []

    async def capture_cache_refresh(
        refreshed_session_key,
        refreshed_session_id,
        *,
        previous_session_id=None,
        expected_agent=None,
    ):
        cache_refreshes.append(
            (
                refreshed_session_key,
                refreshed_session_id,
                previous_session_id,
                getattr(expected_agent, "session_id", None),
            )
        )

    runner._refresh_agent_cache_message_count = capture_cache_refresh
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        chat_type="dm",
    )
    adapter._pending_messages[session_key] = MessageEvent(
        text="follow up",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-2",
    )

    result = await runner._run_agent(
        message="first turn",
        context_prompt="",
        history=[],
        source=source,
        session_id=initial_session_id,
        session_key=session_key,
    )

    assert result["session_id"] == expected_session_id
    assert CaptureQueuedNativeImageAgent.session_ids == expected_session_ids
    assert cache_refreshes == [
        (
            session_key,
            expected_session_id,
            initial_session_id,
            expected_session_id,
        )
    ]


def test_goal_recheck_recovers_active_goal_from_rotated_parent(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    try:
        goals.GoalManager(session_id="parent-session").set("ship the fix")
        runner = _make_runner(CaptureAdapter())

        assert runner._goal_still_active_for_session(
            "continuation-session",
            previous_session_id="parent-session",
        )
        assert goals.GoalManager(session_id="continuation-session").is_active()
        assert not goals.GoalManager(session_id="parent-session").is_active()
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_queued_goal_recheck_uses_rotated_and_parent_session_ids(
    monkeypatch,
    tmp_path,
):
    CaptureQueuedNativeImageAgent.calls = []
    CaptureQueuedNativeImageAgent.session_ids = []
    CaptureQueuedNativeImageAgent.rotate_to_session_id = "continuation-session"
    CaptureQueuedNativeImageAgent.omit_first_session_id = False

    fake_dotenv = types.ModuleType("dotenv")
    setattr(fake_dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    setattr(fake_run_agent, "AIAgent", CaptureQueuedNativeImageAgent)
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    checked_session_ids = []

    def capture_goal_recheck(session_id, *, previous_session_id=None):
        checked_session_ids.append((session_id, previous_session_id))
        return True

    runner._goal_still_active_for_session = capture_goal_recheck
    session_key = "agent:main:telegram:dm:42"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        chat_type="dm",
    )
    adapter._pending_messages[session_key] = MessageEvent(
        text="[Continuing toward your standing goal]\nGoal: ship the fix",
        message_type=MessageType.TEXT,
        source=source,
        message_id="goal-queued-1",
    )

    await runner._run_agent(
        message="first turn",
        context_prompt="",
        history=[],
        source=source,
        session_id="parent-session",
        session_key=session_key,
    )

    assert checked_session_ids == [("continuation-session", "parent-session")]
    assert CaptureQueuedNativeImageAgent.session_ids == [
        "parent-session",
        "continuation-session",
    ]


def test_unknown_goal_state_is_requeued_once_then_parked():
    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    session_key = "agent:main:telegram:dm:42"
    event = MessageEvent(
        text="[Continuing toward your standing goal]\nGoal: ship the fix",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="42",
            chat_type="dm",
        ),
    )

    runner._defer_goal_continuation_recheck(session_key, adapter, event)
    assert adapter._pending_messages.pop(session_key) is event
    assert event.metadata["_goal_state_rechecks"] == 1

    runner._defer_goal_continuation_recheck(session_key, adapter, event)
    assert session_key not in adapter._pending_messages
    assert runner._session_state(session_key).conversation.queued_events == [event]

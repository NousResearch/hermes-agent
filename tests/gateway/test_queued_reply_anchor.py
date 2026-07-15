import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    AnchoredReply,
    BasePlatformAdapter,
    MessageEvent,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class CaptureFeishuAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.FEISHU)
        self.sent = []

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
        return SendResult(success=True, message_id=f"sent-{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class TwoTurnAgent:
    calls = []
    on_turn = None

    def __init__(self, **kwargs):
        self.tools = []
        self.tool_progress_callback = kwargs.get("tool_progress_callback")

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls.append(message)
        turn = len(type(self).calls)
        if type(self).on_turn is not None:
            type(self).on_turn(turn)
        prior = list(conversation_history or [])
        return {
            "final_response": f"answer-{turn}",
            "messages": prior
            + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"answer-{turn}"},
            ],
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
        multiplex_profiles=False,
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


@pytest.mark.asyncio
async def test_queued_feishu_turns_keep_their_own_reply_anchors(monkeypatch, tmp_path):
    TwoTurnAgent.calls = []
    TwoTurnAgent.on_turn = None

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = TwoTurnAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***"},
    )

    adapter = CaptureFeishuAdapter()
    runner = _make_runner(adapter)
    source_a = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_group",
        chat_type="group",
        user_id="ou_a",
        user_name="A",
    )
    source_b = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_group",
        chat_type="group",
        user_id="ou_b",
        user_name="B",
    )
    session_key = build_session_key(source_a, group_sessions_per_user=False)
    pending_event = MessageEvent(
        text="question-b",
        source=source_b,
        message_id="om_b",
    )
    adapter._pending_messages[session_key] = pending_event

    result = await runner._run_agent(
        message="question-a",
        context_prompt="",
        history=[],
        source=source_a,
        session_id="sess-feishu-shared",
        session_key=session_key,
        event_message_id="om_a",
    )

    assert [call["content"] for call in adapter.sent] == ["answer-1"]
    assert adapter.sent[0]["reply_to"] == "om_a"
    assert result["final_response"] == "answer-2"
    assert result["delivery_reply_to_message_id"] == "om_b"
    assert result["delivery_event"] is pending_event


@pytest.mark.asyncio
async def test_three_queued_feishu_turns_keep_each_reply_anchor(monkeypatch, tmp_path):
    TwoTurnAgent.calls = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = TwoTurnAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***"},
    )

    adapter = CaptureFeishuAdapter()
    runner = _make_runner(adapter)
    source_a = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_group",
        chat_type="group",
        user_id="ou_a",
    )
    source_b = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_group",
        chat_type="group",
        user_id="ou_b",
    )
    source_c = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_group",
        chat_type="group",
        user_id="ou_c",
    )
    session_key = build_session_key(source_a, group_sessions_per_user=False)
    event_b = MessageEvent(text="question-b", source=source_b, message_id="om_b")
    event_c = MessageEvent(text="question-c", source=source_c, message_id="om_c")
    adapter._pending_messages[session_key] = event_b

    def queue_third_turn(turn):
        if turn == 2:
            adapter._pending_messages[session_key] = event_c

    TwoTurnAgent.on_turn = queue_third_turn
    try:
        result = await runner._run_agent(
            message="question-a",
            context_prompt="",
            history=[],
            source=source_a,
            session_id="sess-feishu-three-turns",
            session_key=session_key,
            event_message_id="om_a",
        )
    finally:
        TwoTurnAgent.on_turn = None

    assert [(call["content"], call["reply_to"]) for call in adapter.sent] == [
        ("answer-1", "om_a"),
        ("answer-2", "om_b"),
    ]
    assert result["final_response"] == "answer-3"
    assert result["delivery_reply_to_message_id"] == "om_c"
    assert result["delivery_event"] is event_c


@pytest.mark.asyncio
async def test_base_delivery_uses_queued_turn_reply_anchor():
    adapter = CaptureFeishuAdapter()
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_group",
        chat_type="group",
        user_id="ou_a",
    )
    event = MessageEvent(
        text="question-a",
        source=source,
        message_id="om_a",
    )

    async def handler(_event):
        return AnchoredReply("answer-b", "om_b")

    adapter.set_message_handler(handler)
    await adapter._process_message_background(
        event,
        build_session_key(source, group_sessions_per_user=False),
    )

    assert adapter.sent == [
        {
            "chat_id": "oc_group",
            "content": "answer-b",
            "reply_to": "om_b",
            "metadata": {"notify": True},
        }
    ]


@pytest.mark.asyncio
async def test_base_media_delivery_uses_queued_turn_event_and_anchor():
    adapter = CaptureFeishuAdapter()
    source_a = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_group",
        chat_type="group",
        user_id="ou_a",
    )
    source_b = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_group",
        chat_type="group",
        user_id="ou_b",
    )
    event_a = MessageEvent(text="question-a", source=source_a, message_id="om_a")
    event_b = MessageEvent(text="question-b", source=source_b, message_id="om_b")

    async def handler(_event):
        return AnchoredReply(
            "![plot](https://example.com/plot.png)",
            "om_b",
            event_b,
        )

    adapter.set_message_handler(handler)
    await adapter._process_message_background(
        event_a,
        build_session_key(source_a, group_sessions_per_user=False),
    )

    assert adapter.sent == [
        {
            "chat_id": "oc_group",
            "content": "plot\nhttps://example.com/plot.png",
            "reply_to": "om_b",
            "metadata": {"notify": True},
        }
    ]


@pytest.mark.asyncio
async def test_text_claims_anchor_before_following_image():
    adapter = CaptureFeishuAdapter()
    source_a = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_group",
        chat_type="group",
        user_id="ou_a",
    )
    event_a = MessageEvent(text="question-a", source=source_a, message_id="om_a")
    event_b = MessageEvent(
        text="question-b",
        source=SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_group",
            chat_type="group",
            user_id="ou_b",
        ),
        message_id="om_b",
    )

    async def handler(_event):
        return AnchoredReply(
            "answer-b\n![plot](https://example.com/plot.png)",
            "om_b",
            event_b,
        )

    adapter.set_message_handler(handler)
    await adapter._process_message_background(
        event_a,
        build_session_key(source_a, group_sessions_per_user=False),
    )

    assert [sent["reply_to"] for sent in adapter.sent] == ["om_b", None]


@pytest.mark.asyncio
async def test_legacy_batch_override_remains_compatible():
    adapter = CaptureFeishuAdapter()
    batch_calls = []

    async def legacy_send_multiple_images(
        chat_id,
        images,
        metadata=None,
        human_delay=0.0,
    ):
        batch_calls.append((chat_id, images, metadata, human_delay))

    adapter.send_multiple_images = legacy_send_multiple_images
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_group",
        chat_type="group",
        user_id="ou_a",
    )
    event = MessageEvent(text="question-a", source=source, message_id="om_a")

    async def handler(_event):
        return AnchoredReply(
            "![plot](https://example.com/plot.png)",
            "om_b",
            event,
        )

    adapter.set_message_handler(handler)
    await adapter._process_message_background(
        event,
        build_session_key(source, group_sessions_per_user=False),
    )

    assert len(batch_calls) == 1
    assert batch_calls[0][0] == "oc_group"


@pytest.mark.asyncio
async def test_gateway_handler_routes_final_text_to_last_queued_turn(
    monkeypatch,
    tmp_path,
):
    from tests.gateway.test_gateway_silence_tokens import _runner

    runner = _runner(monkeypatch, tmp_path)
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_group",
        chat_type="group",
        user_id="ou_a",
    )
    event = MessageEvent(
        text="question-a",
        source=source,
        message_id="om_a",
    )
    delivery_event = MessageEvent(
        text="question-b",
        source=SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_group",
            chat_type="group",
            user_id="ou_b",
        ),
        message_id="om_b",
    )
    runner._reply_anchor_for_event = lambda evt: evt.message_id
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "answer-b",
            "messages": [
                {"role": "user", "content": "question-a"},
                {"role": "assistant", "content": "answer-a"},
                {"role": "user", "content": "question-b"},
                {"role": "assistant", "content": "answer-b"},
            ],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "api_calls": 2,
            "failed": False,
            "delivery_reply_to_message_id": "om_b",
            "delivery_event": delivery_event,
        }
    )

    response = await runner._handle_message_with_agent(
        event,
        source,
        "agent:main:feishu:group:oc_group",
        1,
    )

    assert isinstance(response, AnchoredReply)
    assert response == "answer-b"
    assert response.reply_to_message_id == "om_b"
    assert response.event is delivery_event


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply_mode", "expected_ancillary_reply_to"),
    [("first", None), ("all", "om_b")],
)
async def test_streaming_ancillary_delivery_uses_last_queued_event(
    monkeypatch,
    tmp_path,
    reply_mode,
    expected_ancillary_reply_to,
):
    from tests.gateway.test_gateway_silence_tokens import _runner

    runner = _runner(monkeypatch, tmp_path)
    adapter = CaptureFeishuAdapter()
    adapter._reply_to_mode = reply_mode
    runner.adapters = {Platform.FEISHU: adapter}
    runner._adapter_for_source = lambda _source: adapter
    runner._should_send_voice_reply = MagicMock(return_value=True)
    runner._send_voice_reply = AsyncMock()
    runner._deliver_media_from_response = AsyncMock()
    monkeypatch.setenv("FEISHU_HOME_CHAT", "oc_group")
    monkeypatch.setattr(
        "gateway.runtime_footer.build_footer_line",
        lambda **_kwargs: "runtime-footer",
    )

    source_a = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_group",
        chat_type="group",
        user_id="ou_a",
    )
    source_b = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_group",
        chat_type="group",
        user_id="ou_b",
    )
    event_a = MessageEvent(text="question-a", source=source_a, message_id="om_a")
    event_b = MessageEvent(text="question-b", source=source_b, message_id="om_b")
    runner._reply_anchor_for_event = lambda evt: evt.message_id
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "answer-b",
            "messages": [
                {"role": "user", "content": "question-b"},
                {"role": "assistant", "content": "answer-b"},
            ],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "api_calls": 1,
            "failed": False,
            "already_sent": True,
            "delivery_reply_to_message_id": "om_b",
            "delivery_event": event_b,
        }
    )

    response = await runner._handle_message_with_agent(
        event_a,
        source_a,
        "agent:main:feishu:group:oc_group",
        1,
    )

    assert response is None
    runner._send_voice_reply.assert_awaited_once_with(event_b, "answer-b")
    expected_media_kwargs = (
        {"reply_to": expected_ancillary_reply_to}
        if expected_ancillary_reply_to is not None
        else {}
    )
    runner._deliver_media_from_response.assert_awaited_once_with(
        "answer-b", event_b, adapter, **expected_media_kwargs
    )
    assert adapter.sent[-1]["content"] == "runtime-footer"
    assert adapter.sent[-1]["reply_to"] == expected_ancillary_reply_to

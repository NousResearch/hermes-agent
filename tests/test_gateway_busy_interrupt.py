import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session import SessionSource


class DummyAdapter:
    def __init__(self):
        self._pending_messages = {}
        self.sent = []

    async def _send_with_retry(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })


class DummyAgent:
    def __init__(self):
        self.interrupts = []

    def interrupt(self, text):
        self.interrupts.append(text)

    def get_activity_summary(self):
        return {
            "api_call_count": 2,
            "max_iterations": 10,
            "current_tool": "terminal",
        }


def make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    return runner


def make_event(text="继续", message_id="m1"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.WEIXIN,
            chat_id="chat-1",
            chat_type="dm",
            thread_id=None,
            user_id="user-1",
            user_name="User",
        ),
        message_id=message_id,
    )


@pytest.mark.asyncio
async def test_busy_interrupt_ack_is_sent_for_every_user_message():
    runner = make_runner()
    adapter = DummyAdapter()
    agent = DummyAgent()
    session_key = "weixin:dm:chat-1"
    runner.adapters[Platform.WEIXIN] = adapter
    runner._running_agents[session_key] = agent

    first = make_event("第一条", "m1")
    second = make_event("第二条", "m2")

    assert await runner._handle_active_session_busy_message(first, session_key) is True
    assert await runner._handle_active_session_busy_message(second, session_key) is True

    assert agent.interrupts == ["第一条", "第二条"]
    assert len(adapter.sent) == 2
    assert adapter.sent[0]["reply_to"] == "m1"
    assert adapter.sent[1]["reply_to"] == "m2"
    assert "Interrupting current task" in adapter.sent[0]["content"]
    assert "Interrupting current task" in adapter.sent[1]["content"]


@pytest.mark.asyncio
async def test_busy_pending_sentinel_gets_visible_ack_without_status_lookup():
    runner = make_runner()
    adapter = DummyAdapter()
    session_key = "weixin:dm:chat-1"
    runner.adapters[Platform.WEIXIN] = adapter
    runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL

    event = make_event("新任务", "m3")

    assert await runner._handle_active_session_busy_message(event, session_key) is True

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["reply_to"] == "m3"
    assert "Interrupting current task" in adapter.sent[0]["content"]


@pytest.mark.asyncio
async def test_busy_interrupt_ack_includes_status_without_name_error(monkeypatch):
    runner = make_runner()
    adapter = DummyAdapter()
    agent = DummyAgent()
    session_key = "weixin:dm:chat-1"
    runner.adapters[Platform.WEIXIN] = adapter
    runner._running_agents[session_key] = agent
    runner._running_agents_ts[session_key] = 100.0

    monkeypatch.setattr("gateway.run.time.time", lambda: 220.0)

    event = make_event("看到了吗", "m4")

    assert await runner._handle_active_session_busy_message(event, session_key) is True

    assert len(adapter.sent) == 1
    content = adapter.sent[0]["content"]
    assert "2 min elapsed" in content
    assert "iteration 2/10" in content
    assert "running: terminal" in content
    assert agent.interrupts == ["看到了吗"]

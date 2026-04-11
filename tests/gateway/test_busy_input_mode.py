import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    def __init__(self, *, busy_input_mode: str = "interrupt"):
        super().__init__(
            PlatformConfig(
                enabled=True,
                token="test-token",
                extra={"busy_input_mode": busy_input_mode},
            ),
            Platform.TELEGRAM,
        )

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _make_source(chat_id="12345"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id="179033731",
    )


def _make_event(text: str, chat_id="12345"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(chat_id),
        message_id="m1",
    )


class TestBaseAdapterBusyInputMode:
    @pytest.mark.asyncio
    async def test_queue_mode_queues_without_interrupt(self):
        adapter = _StubAdapter(busy_input_mode="queue")
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))

        event = _make_event("first follow-up")
        session_key = build_session_key(event.source)
        interrupt_event = asyncio.Event()
        adapter._active_sessions[session_key] = interrupt_event

        await adapter.handle_message(event)

        assert session_key in adapter._pending_messages
        assert adapter._pending_messages[session_key].text == "first follow-up"
        assert interrupt_event.is_set() is False

    @pytest.mark.asyncio
    async def test_queue_mode_merges_text_messages(self):
        adapter = _StubAdapter(busy_input_mode="queue")
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))

        first = _make_event("first follow-up")
        second = _make_event("second follow-up")
        session_key = build_session_key(first.source)
        interrupt_event = asyncio.Event()
        adapter._active_sessions[session_key] = interrupt_event

        await adapter.handle_message(first)
        await adapter.handle_message(second)

        assert adapter._pending_messages[session_key].text == "first follow-up\nsecond follow-up"
        assert interrupt_event.is_set() is False


def _make_runner(*, busy_input_mode: str):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="***",
                extra={"busy_input_mode": busy_input_mode},
            )
        },
        busy_input_mode=busy_input_mode,
    )
    runner.adapters = {Platform.TELEGRAM: _StubAdapter(busy_input_mode=busy_input_mode)}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._background_tasks = set()
    runner._failed_platforms = {}
    runner._update_prompt_pending = {}
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = MagicMock()
    runner._effective_model = None
    runner._effective_provider = None
    runner._is_user_authorized = lambda _source: True
    runner._get_unauthorized_dm_behavior = lambda _platform: "pair"
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.delivery_router = MagicMock()
    return runner


def _make_qq_source(chat_type="dm", chat_id="179033731"):
    return SessionSource(
        platform=Platform.QQ_NAPCAT,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id="179033731",
        user_name="發發發",
    )


def _make_qq_event(text: str, *, chat_type="dm", chat_id="179033731"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_qq_source(chat_type=chat_type, chat_id=chat_id),
        message_id="qq-m1",
    )


class TestGatewayRunnerBusyInputMode:
    @pytest.mark.asyncio
    async def test_queue_mode_does_not_interrupt_running_agent(self):
        runner = _make_runner(busy_input_mode="queue")
        event = _make_event("keep going")
        session_key = build_session_key(event.source)
        running_agent = MagicMock()
        runner._running_agents[session_key] = running_agent

        result = await runner._handle_message(event)

        assert result is None
        running_agent.interrupt.assert_not_called()
        queued = runner.adapters[Platform.TELEGRAM]._pending_messages[session_key]
        assert queued.text == "keep going"
        assert runner._pending_messages == {}

    @pytest.mark.asyncio
    async def test_interrupt_mode_keeps_interrupt_behavior(self):
        runner = _make_runner(busy_input_mode="interrupt")
        event = _make_event("stop current work")
        session_key = build_session_key(event.source)
        running_agent = MagicMock()
        runner._running_agents[session_key] = running_agent

        result = await runner._handle_message(event)

        assert result is None
        running_agent.interrupt.assert_called_once_with("stop current work")
        assert runner._pending_messages[session_key] == "stop current work"

    @pytest.mark.asyncio
    async def test_queue_mode_returns_visible_ack_for_qq_dm(self):
        runner = _make_runner(busy_input_mode="queue")
        runner.adapters = {Platform.QQ_NAPCAT: _StubAdapter(busy_input_mode="queue")}
        event = _make_qq_event("你先继续，等会儿把进度回我")
        session_key = build_session_key(event.source)
        running_agent = MagicMock()
        runner._running_agents[session_key] = running_agent

        result = await runner._handle_message(event)

        assert "排队" in result
        queued = runner.adapters[Platform.QQ_NAPCAT]._pending_messages[session_key]
        assert queued.text == "你先继续，等会儿把进度回我"

    @pytest.mark.asyncio
    async def test_queue_mode_returns_visible_ack_for_explicit_qq_group_follow_up(self):
        runner = _make_runner(busy_input_mode="queue")
        runner.adapters = {Platform.QQ_NAPCAT: _StubAdapter(busy_input_mode="queue")}
        event = _make_qq_event("@马嘎 前面的事做到哪了", chat_type="group", chat_id="685403987")
        session_key = build_session_key(event.source)
        running_agent = MagicMock()
        runner._running_agents[session_key] = running_agent

        result = await runner._handle_message(event)

        assert "排队" in result
        queued = runner.adapters[Platform.QQ_NAPCAT]._pending_messages[session_key]
        assert queued.text == "@马嘎 前面的事做到哪了"

    @pytest.mark.asyncio
    async def test_queue_mode_keeps_low_signal_qq_group_silent(self):
        runner = _make_runner(busy_input_mode="queue")
        runner.adapters = {Platform.QQ_NAPCAT: _StubAdapter(busy_input_mode="queue")}
        event = _make_qq_event("今天天气不错", chat_type="group", chat_id="685403987")
        session_key = build_session_key(event.source)
        runner._running_agents[session_key] = MagicMock()

        result = await runner._handle_message(event)

        assert result is None


class TestGatewayBusyInputModeConfig:
    def test_platform_override_wins_over_global_default(self):
        config = GatewayConfig.from_dict(
            {
                "busy_input_mode": "queue",
                "platforms": {
                    "telegram": {
                        "enabled": True,
                        "token": "tok",
                        "extra": {"busy_input_mode": "interrupt"},
                    }
                },
            }
        )

        assert config.get_busy_input_mode() == "queue"
        assert config.get_busy_input_mode(Platform.TELEGRAM) == "interrupt"

    def test_load_gateway_config_reads_display_busy_input_mode(self, monkeypatch, tmp_path):
        monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path)
        (tmp_path / "config.yaml").write_text(
            "display:\n  busy_input_mode: queue\n",
            encoding="utf-8",
        )

        config = load_gateway_config()

        assert config.busy_input_mode == "queue"

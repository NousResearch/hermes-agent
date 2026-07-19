import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.background_jobs import BackgroundJobStore
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.platforms.qq_napcat import QqNapCatAdapter
from gateway.run import GatewayRunner
from gateway.runtime_shortcuts_service import build_long_running_status_detail
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    def __init__(self, *, busy_input_mode: str = "interrupt", platform: Platform = Platform.TELEGRAM):
        super().__init__(
            PlatformConfig(
                enabled=True,
                token="test-token",
                extra={"busy_input_mode": busy_input_mode},
            ),
            platform,
        )
        # Unit tests assert immediate pending visibility; skip the real
        # queue-mode text debounce window.
        self._busy_text_debounce_seconds = 0.0

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


class _StubQqAdapter(_StubAdapter):
    def __init__(self, *, busy_input_mode: str = "interrupt"):
        super().__init__(busy_input_mode=busy_input_mode, platform=Platform.QQ_NAPCAT)

    def _should_inline_active_session_message(self, event: MessageEvent) -> bool:
        text = str(event.text or "").strip()
        return any(
            marker in text
            for marker in (
                "你现在忙什么",
                "后台任务",
                "好友申请",
                "自动通过",
                "禁言",
                "监听状态",
            )
        )

    def _is_explicit_busy_followup(self, event: MessageEvent) -> bool:
        text = event.text or ""
        return (
            getattr(event.source, "chat_type", "") == "dm"
            or "@马嘎" in text
            or "马嘎" in text
        )

    def _busy_followup_ack(self, event: MessageEvent, *, interrupting: bool = False) -> str:
        if getattr(event.source, "chat_type", "") == "dm":
            return "收到，上一轮有点久，我先切到你这条，马上接着回你。" if interrupting else "收到，这条我先排队，上一轮忙完马上接着回你。"
        if self._is_explicit_busy_followup(event):
            return "收到，上一轮有点久，我先切到这条，马上接着回。" if interrupting else "收到，这条我先排队，上一轮忙完接着回你。"
        return ""


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

    @pytest.mark.asyncio
    async def test_queue_mode_merges_text_messages_upgrading_to_explicit_reply_metadata(self):
        adapter = _StubQqAdapter(busy_input_mode="queue")
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))

        first = _make_qq_event("先记一条", chat_type="group", chat_id="685403987")
        first.metadata = {
            "group_trigger_reason": "require_mention_disabled",
            "explicit_group_trigger": False,
            "explicit_addressed": False,
            "requires_reply": False,
        }
        second = _make_qq_event("@马嘎 接着处理", chat_type="group", chat_id="685403987")
        second.message_id = "qq-m2"
        second.raw_message = {"id": "latest"}
        second.reply_to_message_id = "bot-msg-7"
        second.metadata = {
            "group_trigger_reason": "require_mention_disabled",
            "explicit_group_trigger": True,
            "explicit_group_trigger_reason": "bot_mention",
            "explicit_addressed": True,
            "address_reason": "bot_mention",
            "requires_reply": True,
        }

        session_key = build_session_key(first.source)
        adapter._active_sessions[session_key] = asyncio.Event()

        await adapter.handle_message(first)
        await adapter.handle_message(second)

        queued = adapter._pending_messages[session_key]
        assert queued.text == "先记一条\n@马嘎 接着处理"
        assert queued.message_id == "qq-m2"
        assert queued.raw_message == {"id": "latest"}
        assert queued.reply_to_message_id == "bot-msg-7"
        assert queued.metadata["explicit_addressed"] is True
        assert queued.metadata["requires_reply"] is True
        assert queued.metadata["explicit_group_trigger_reason"] == "bot_mention"

    @pytest.mark.asyncio
    async def test_queue_mode_merges_text_messages_without_downgrading_explicit_reply_metadata(self):
        adapter = _StubQqAdapter(busy_input_mode="queue")
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))

        first = _make_qq_event("@马嘎 先看这个", chat_type="group", chat_id="685403987")
        first.metadata = {
            "group_trigger_reason": "require_mention_disabled",
            "explicit_group_trigger": True,
            "explicit_group_trigger_reason": "bot_mention",
            "explicit_addressed": True,
            "address_reason": "bot_mention",
            "requires_reply": True,
        }
        second = _make_qq_event("补充一句背景", chat_type="group", chat_id="685403987")
        second.metadata = {
            "group_trigger_reason": "require_mention_disabled",
            "explicit_group_trigger": False,
            "explicit_addressed": False,
            "requires_reply": False,
        }

        session_key = build_session_key(first.source)
        adapter._active_sessions[session_key] = asyncio.Event()

        await adapter.handle_message(first)
        await adapter.handle_message(second)

        queued = adapter._pending_messages[session_key]
        assert queued.text == "@马嘎 先看这个\n补充一句背景"
        assert queued.metadata["explicit_addressed"] is True
        assert queued.metadata["requires_reply"] is True
        assert queued.metadata["address_reason"] == "bot_mention"

    @pytest.mark.asyncio
    async def test_smart_mode_queues_recent_dm_followup_without_interrupt(self):
        adapter = _StubAdapter(busy_input_mode="smart")
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))

        event = _make_event("recent follow-up")
        session_key = build_session_key(event.source)
        interrupt_event = asyncio.Event()
        adapter._active_sessions[session_key] = interrupt_event
        adapter._active_session_started_at[session_key] = time.time()

        await adapter.handle_message(event)

        assert adapter._pending_messages[session_key].text == "recent follow-up"
        assert interrupt_event.is_set() is False

    @pytest.mark.asyncio
    async def test_smart_mode_interrupts_stale_dm_followup(self):
        adapter = _StubAdapter(busy_input_mode="smart")
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))

        event = _make_event("stale follow-up")
        session_key = build_session_key(event.source)
        interrupt_event = asyncio.Event()
        adapter._active_sessions[session_key] = interrupt_event
        adapter._active_session_started_at[session_key] = time.time() - 9.0

        await adapter.handle_message(event)

        assert adapter._pending_messages[session_key].text == "stale follow-up"
        assert interrupt_event.is_set() is True

    @pytest.mark.asyncio
    async def test_active_session_inline_hook_dispatches_qq_message_immediately(self):
        adapter = _StubQqAdapter(busy_input_mode="queue")
        adapter.set_message_handler(AsyncMock(return_value="当前前台这轮还在跑。"))
        adapter._send_with_retry = AsyncMock()

        event = _make_qq_event("你现在忙什么？")
        session_key = build_session_key(event.source)
        adapter._active_sessions[session_key] = asyncio.Event()

        await adapter.handle_message(event)

        assert session_key not in adapter._pending_messages
        adapter._send_with_retry.assert_awaited_once()
        assert adapter._send_with_retry.await_args.kwargs["content"] == "当前前台这轮还在跑。"

    @pytest.mark.asyncio
    async def test_active_session_inline_hook_falls_back_to_queue_when_no_direct_response(self):
        adapter = _StubQqAdapter(busy_input_mode="queue")
        adapter.set_message_handler(AsyncMock(return_value=None))
        adapter._send_with_retry = AsyncMock()

        event = _make_qq_event("看看待处理的好友申请")
        session_key = build_session_key(event.source)
        adapter._active_sessions[session_key] = asyncio.Event()

        await adapter.handle_message(event)

        assert adapter._pending_messages[session_key].text == "看看待处理的好友申请"
        adapter._send_with_retry.assert_awaited_once()
        assert "排队" in adapter._send_with_retry.await_args.kwargs["content"]


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
        # Upstream delivers the interrupting text via agent._interrupt_message /
        # adapter._pending_messages (Level 1). runner._pending_messages is no
        # longer written on the Level 2 interrupt path (write-only dead store).
        assert session_key not in runner._pending_messages

    @pytest.mark.asyncio
    async def test_interrupt_mode_denies_blocking_approval_before_followup(self, monkeypatch):
        runner = _make_runner(busy_input_mode="interrupt")
        runner.adapters = {Platform.QQ_NAPCAT: _StubQqAdapter(busy_input_mode="interrupt")}
        event = _make_qq_event("@马嘎 你他妈要干嘛", chat_type="group", chat_id="685403987")
        session_key = build_session_key(event.source)
        running_agent = MagicMock()
        runner._running_agents[session_key] = running_agent
        resolved = []

        monkeypatch.setattr("tools.approval.has_blocking_approval", lambda _key: True)
        monkeypatch.setattr(
            "tools.approval.peek_blocking_approval",
            lambda _key: {"command": "rm -rf /root/projects/hermes-agent/repos"},
        )
        monkeypatch.setattr(
            "tools.approval.resolve_gateway_approval",
            lambda key, choice, resolve_all=False: resolved.append((key, choice, resolve_all)) or 1,
        )

        result = await runner._handle_message(event)

        assert "危险命令" in result
        assert "拒" in result
        assert "rm -rf /root/projects/hermes-agent/repos" in result
        assert resolved == [(session_key, "deny", True)]
        running_agent.interrupt.assert_called_once_with("@马嘎 你他妈要干嘛")
        assert runner._pending_messages[session_key] == "@马嘎 你他妈要干嘛"

    @pytest.mark.asyncio
    async def test_interrupt_mode_queues_when_qq_group_moderation_tool_is_running(self):
        runner = _make_runner(busy_input_mode="interrupt")
        runner.adapters = {Platform.QQ_NAPCAT: _StubQqAdapter(busy_input_mode="interrupt")}
        event = _make_qq_event("@马嘎 前面的广告处理完没", chat_type="group", chat_id="685403987")
        session_key = build_session_key(event.source)
        running_agent = MagicMock()
        running_agent.get_activity_summary.return_value = {
            "current_tool": "qq_group_moderation",
            "last_activity_desc": "running qq_group_moderation",
        }
        runner._running_agents[session_key] = running_agent

        result = await runner._handle_message(event)

        assert "排队" in result
        running_agent.interrupt.assert_not_called()
        queued = runner.adapters[Platform.QQ_NAPCAT]._pending_messages[session_key]
        assert queued.text == "@马嘎 前面的广告处理完没"

    @pytest.mark.asyncio
    async def test_queue_mode_returns_visible_ack_for_qq_dm(self):
        runner = _make_runner(busy_input_mode="queue")
        runner.adapters = {Platform.QQ_NAPCAT: _StubQqAdapter(busy_input_mode="queue")}
        event = _make_qq_event("你先继续，等会儿把进度回我")
        session_key = build_session_key(event.source)
        running_agent = MagicMock()
        runner._running_agents[session_key] = running_agent

        result = await runner._handle_message(event)

        assert "排队" in result
        queued = runner.adapters[Platform.QQ_NAPCAT]._pending_messages[session_key]
        assert queued.text == "你先继续，等会儿把进度回我"

    @pytest.mark.asyncio
    async def test_queue_mode_returns_visible_reply_for_explicit_qq_group_follow_up(self):
        runner = _make_runner(busy_input_mode="queue")
        runner.adapters = {Platform.QQ_NAPCAT: _StubQqAdapter(busy_input_mode="queue")}
        event = _make_qq_event("@马嘎 前面的事做到哪了", chat_type="group", chat_id="685403987")
        session_key = build_session_key(event.source)
        running_agent = MagicMock()
        runner._running_agents[session_key] = running_agent

        result = await runner._handle_message(event)

        assert result is not None
        assert ("排队" in result) or ("前台" in result)
        queued = runner.adapters[Platform.QQ_NAPCAT]._pending_messages.get(session_key)
        if queued is not None:
            assert queued.text == "@马嘎 前面的事做到哪了"
        running_agent.interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_queue_mode_keeps_low_signal_qq_group_silent(self):
        runner = _make_runner(busy_input_mode="queue")
        runner.adapters = {Platform.QQ_NAPCAT: _StubQqAdapter(busy_input_mode="queue")}
        event = _make_qq_event("今天天气不错", chat_type="group", chat_id="685403987")
        session_key = build_session_key(event.source)
        runner._running_agents[session_key] = MagicMock()

        result = await runner._handle_message(event)

        assert result is None

    @pytest.mark.asyncio
    async def test_smart_mode_queues_recent_qq_dm_followup(self):
        runner = _make_runner(busy_input_mode="smart")
        runner.adapters = {Platform.QQ_NAPCAT: _StubQqAdapter(busy_input_mode="smart")}
        event = _make_qq_event("你先继续，等会回我")
        session_key = build_session_key(event.source)
        running_agent = MagicMock()
        runner._running_agents[session_key] = running_agent
        runner._running_agents_ts[session_key] = time.time()

        result = await runner._handle_message(event)

        assert "排队" in result
        running_agent.interrupt.assert_not_called()
        queued = runner.adapters[Platform.QQ_NAPCAT]._pending_messages[session_key]
        assert queued.text == "你先继续，等会回我"

    @pytest.mark.asyncio
    async def test_smart_mode_returns_visible_reply_for_stale_explicit_qq_group_followup(self):
        runner = _make_runner(busy_input_mode="smart")
        runner.adapters = {Platform.QQ_NAPCAT: _StubQqAdapter(busy_input_mode="smart")}
        event = _make_qq_event("@马嘎 前面的事做到哪了", chat_type="group", chat_id="685403987")
        session_key = build_session_key(event.source)
        running_agent = MagicMock()
        runner._running_agents[session_key] = running_agent
        runner._running_agents_ts[session_key] = time.time() - 9.0

        result = await runner._handle_message(event)

        assert result is not None
        assert ("先切到" in result) or ("前台" in result)
        queued = runner.adapters[Platform.QQ_NAPCAT]._pending_messages.get(session_key)
        if queued is not None:
            assert queued.text == "@马嘎 前面的事做到哪了"
        if "先切到" in result:
            running_agent.interrupt.assert_called_once_with("@马嘎 前面的事做到哪了")
        else:
            running_agent.interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_busy_qq_followup_while_background_job_runs_uses_shortcut_for_plain_hai_zaima(self, tmp_path):
        runner = _make_runner(busy_input_mode="queue")
        runner.adapters = {Platform.QQ_NAPCAT: _StubQqAdapter(busy_input_mode="queue")}
        event = _make_qq_event("@马嘎 还在吗", chat_type="group", chat_id="685403987")
        session_key = build_session_key(event.source)
        running_agent = MagicMock()
        runner._running_agents[session_key] = running_agent
        runner._background_job_store = BackgroundJobStore(db_path=tmp_path / "background_jobs.db")
        runner._background_job_store.create_job(
            task_id="bg_busy_1",
            prompt="继续处理线上问题",
            source=event.source,
            session_key=session_key,
            job_kind="auto",
            worker_name="铁柱",
        )
        runner._background_job_store.mark_job_running("bg_busy_1")

        result = await runner._handle_message(event)

        assert result is not None
        assert "bg_busy_1" in result
        assert session_key not in runner.adapters[Platform.QQ_NAPCAT]._pending_messages
        running_agent.interrupt.assert_not_called()


class TestGatewayBusyInputModeConfig:
    def test_platform_override_wins_over_global_default(self):
        config = GatewayConfig.from_dict(
            {
                "busy_input_mode": "smart",
                "platforms": {
                    "telegram": {
                        "enabled": True,
                        "token": "tok",
                        "extra": {"busy_input_mode": "interrupt"},
                    }
                },
            }
        )

        assert config.get_busy_input_mode() == "smart"
        assert config.get_busy_input_mode(Platform.TELEGRAM) == "interrupt"

    def test_load_gateway_config_reads_display_busy_input_mode(self, monkeypatch, tmp_path):
        monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path)
        (tmp_path / "config.yaml").write_text(
            "display:\n  busy_input_mode: smart\n",
            encoding="utf-8",
        )

        config = load_gateway_config()

        assert config.busy_input_mode == "smart"

    def test_qq_defaults_to_smart_without_platform_override(self):
        config = GatewayConfig.from_dict(
            {
                "busy_input_mode": "queue",
                "platforms": {
                    "qq_napcat": {
                        "enabled": True,
                        "extra": {"ws_url": "ws://127.0.0.1:3001"},
                    }
                },
            }
        )

        assert config.get_busy_input_mode(Platform.QQ_NAPCAT) == "smart"


def test_qq_adapter_inline_busy_shortcuts_cover_plain_presence_and_progress_queries(monkeypatch):
    monkeypatch.setattr(
        "gateway.run._looks_like_qq_active_session_inline_candidate",
        lambda *_args, **_kwargs: False,
    )
    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )

    event_zai_ma = _make_qq_event("@马嘎 在吗", chat_type="group", chat_id="685403987")
    event_progress = _make_qq_event("@马嘎 前面那个任务呢", chat_type="group", chat_id="685403987")
    event_busy = _make_qq_event("@马嘎 现在忙什么", chat_type="group", chat_id="685403987")

    assert adapter._should_inline_active_session_message(event_zai_ma) is True
    assert adapter._should_inline_active_session_message(event_progress) is True
    assert adapter._should_inline_active_session_message(event_busy) is True


def test_long_running_status_detail_mentions_blocking_approval(monkeypatch):
    agent = MagicMock()
    agent.get_activity_summary.return_value = {
        "api_call_count": 2,
        "max_iterations": 60,
        "current_tool": "delegate_task",
        "last_activity_desc": "executing tool: delegate_task",
    }

    monkeypatch.setattr("tools.approval.has_blocking_approval", lambda _key: True)
    monkeypatch.setattr(
        "tools.approval.peek_blocking_approval",
        lambda _key: {"command": "rm -rf /root/projects/hermes-agent/repos"},
    )

    detail = build_long_running_status_detail(agent, "qq:group:685403987")

    assert "iteration 2/60" in detail
    assert "running: delegate_task" in detail
    assert "waiting for approval" in detail
    assert "rm -rf /root/projects/hermes-agent/repos" in detail

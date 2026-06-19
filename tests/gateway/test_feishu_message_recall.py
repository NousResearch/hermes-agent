"""Tests for cancelling Feishu gateway work when a user recalls a message."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key


def _feishu_source() -> SessionSource:
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_test_chat",
        chat_type="dm",
        user_id="ou_test_user",
    )


def _feishu_event(message_id: str = "om_recalled") -> MessageEvent:
    return MessageEvent(
        text="please ignore this if recalled",
        source=_feishu_source(),
        message_id=message_id,
    )


def _bootstrap_runner(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    config = GatewayConfig(
        platforms={Platform.FEISHU: PlatformConfig(enabled=True, token="fake")}
    )
    runner = gateway_run.GatewayRunner(config)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: []
    runner._clear_session_env = lambda _tokens: None
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda event: getattr(event, "message_id", None)
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._deliver_platform_notice = AsyncMock()
    runner._sync_telegram_topic_binding = MagicMock()
    runner._record_telegram_topic_binding = MagicMock()
    runner._format_session_info = lambda: ""
    runner._resolve_session_agent_runtime = MagicMock(
        return_value=("test-model", {"api_key": "fake", "provider": "test"})
    )
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner._session_db = None

    session_key = build_session_key(_feishu_source())
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=session_key,
        session_id="sess-recall",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.FEISHU,
        chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    return runner


def test_feishu_recall_event_routes_to_registered_handler():
    from gateway.platforms.feishu import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())
    adapter._loop = object()
    adapter._message_recall_handler = AsyncMock()
    payload = SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(message_id="om_recalled"),
        ),
    )

    with patch.object(adapter, "_submit_on_loop", return_value=True) as submit:
        adapter._on_message_recalled(payload)

    submit.assert_called_once()
    coro = submit.call_args.args[1]
    try:
        import asyncio

        asyncio.run(coro)
    finally:
        close = getattr(coro, "close", None)
        if close:
            close()
    adapter._message_recall_handler.assert_awaited_once_with(
        Platform.FEISHU,
        "om_recalled",
    )


@pytest.mark.asyncio
async def test_feishu_recall_removes_recalled_text_from_pending_batch():
    from gateway.platforms.feishu import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())
    adapter._handle_message_with_guards = AsyncMock()
    first = _feishu_event("om_first")
    first.text = "recalled text"
    second = _feishu_event("om_second")
    second.text = "kept text"

    await adapter._enqueue_text_event(first)
    await adapter._enqueue_text_event(second)
    adapter._on_message_recalled(
        SimpleNamespace(event=SimpleNamespace(message=SimpleNamespace(message_id="om_first")))
    )
    await adapter._flush_text_batch_now(adapter._text_batch_key(first))

    adapter._handle_message_with_guards.assert_awaited_once()
    flushed = adapter._handle_message_with_guards.await_args.args[0]
    assert flushed.text == "kept text"
    assert flushed.message_id == "om_second"

    for task in list(adapter._pending_text_batch_tasks.values()):
        task.cancel()


@pytest.mark.asyncio
async def test_feishu_recall_removes_recalled_media_from_pending_batch():
    from gateway.platforms.feishu import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())
    adapter._handle_message_with_guards = AsyncMock()
    first = MessageEvent(
        text="recalled caption",
        message_type=MessageType.PHOTO,
        source=_feishu_source(),
        message_id="om_media_first",
        media_urls=["/tmp/recalled.png"],
        media_types=["image/png"],
    )
    second = MessageEvent(
        text="kept caption",
        message_type=MessageType.PHOTO,
        source=_feishu_source(),
        message_id="om_media_second",
        media_urls=["/tmp/kept.png"],
        media_types=["image/png"],
    )

    await adapter._enqueue_media_event(first)
    await adapter._enqueue_media_event(second)
    adapter._on_message_recalled(
        SimpleNamespace(event=SimpleNamespace(message=SimpleNamespace(message_id="om_media_first")))
    )
    await adapter._flush_media_batch_now(adapter._media_batch_key(first))

    adapter._handle_message_with_guards.assert_awaited_once()
    flushed = adapter._handle_message_with_guards.await_args.args[0]
    assert flushed.text == "kept caption"
    assert flushed.message_id == "om_media_second"
    assert flushed.media_urls == ["/tmp/kept.png"]

    for task in list(adapter._pending_media_batch_tasks.values()):
        task.cancel()


@pytest.mark.asyncio
async def test_recalled_feishu_message_does_not_start_agent(monkeypatch, tmp_path):
    runner = _bootstrap_runner(monkeypatch, tmp_path)
    event = _feishu_event("om_before_run")
    session_key = build_session_key(event.source)
    runner._mark_inbound_message_recalled(Platform.FEISHU, "om_before_run")
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "this should never be generated",
            "messages": [],
            "history_offset": 0,
        }
    )

    result = await runner._handle_message_with_agent(event, event.source, session_key, 1)

    assert result is None
    runner._run_agent.assert_not_called()
    runner.session_store.append_to_transcript.assert_not_called()


@pytest.mark.asyncio
async def test_recalled_feishu_batch_item_does_not_start_agent(monkeypatch, tmp_path):
    runner = _bootstrap_runner(monkeypatch, tmp_path)
    event = _feishu_event("om_batch_second")
    event._feishu_batch_items = [  # type: ignore[attr-defined]
        {"message_id": "om_batch_first", "text": "recalled"},
        {"message_id": "om_batch_second", "text": "kept"},
    ]
    session_key = build_session_key(event.source)
    runner._mark_inbound_message_recalled(Platform.FEISHU, "om_batch_first")
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "this should never be generated",
            "messages": [],
            "history_offset": 0,
        }
    )

    result = await runner._handle_message_with_agent(event, event.source, session_key, 1)

    assert result is None
    runner._run_agent.assert_not_called()
    runner.session_store.append_to_transcript.assert_not_called()


@pytest.mark.asyncio
async def test_feishu_recall_interrupts_inflight_agent(monkeypatch, tmp_path):
    runner = _bootstrap_runner(monkeypatch, tmp_path)
    event = _feishu_event("om_live")
    session_key = build_session_key(event.source)
    runner._register_inbound_message_for_run(event, session_key)
    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent

    handled = await runner._handle_message_recalled(Platform.FEISHU, "om_live")

    assert handled is True
    running_agent.interrupt.assert_called_once()
    assert runner._is_inbound_message_recalled(Platform.FEISHU, "om_live") is True


@pytest.mark.asyncio
async def test_feishu_recall_interrupts_inflight_batched_agent(monkeypatch, tmp_path):
    runner = _bootstrap_runner(monkeypatch, tmp_path)
    event = _feishu_event("om_batch_live_second")
    event._feishu_batch_items = [  # type: ignore[attr-defined]
        {"message_id": "om_batch_live_first", "text": "recalled"},
        {"message_id": "om_batch_live_second", "text": "kept"},
    ]
    session_key = build_session_key(event.source)
    runner._register_inbound_message_for_run(event, session_key)
    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent

    handled = await runner._handle_message_recalled(Platform.FEISHU, "om_batch_live_first")

    assert handled is True
    running_agent.interrupt.assert_called_once()
    assert runner._is_inbound_message_recalled(Platform.FEISHU, "om_batch_live_first") is True


@pytest.mark.asyncio
async def test_feishu_recall_suppresses_result_that_finishes_late(monkeypatch, tmp_path):
    runner = _bootstrap_runner(monkeypatch, tmp_path)
    event = _feishu_event("om_late")
    session_key = build_session_key(event.source)

    async def fake_run_agent(**_kwargs):
        runner._mark_inbound_message_recalled(Platform.FEISHU, "om_late")
        return {
            "final_response": "late answer should be suppressed",
            "messages": [
                {"role": "user", "content": "please ignore this if recalled"},
                {"role": "assistant", "content": "late answer should be suppressed"},
            ],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }

    runner._run_agent = AsyncMock(side_effect=fake_run_agent)

    result = await runner._handle_message_with_agent(event, event.source, session_key, 1)

    assert result is None
    runner.session_store.append_to_transcript.assert_not_called()

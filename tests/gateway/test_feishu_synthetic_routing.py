"""Synthetic events keep Feishu topic routes without impersonating inbound messages."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base_thread_metadata import _reply_anchor_for_event, _thread_metadata_for_event, _thread_metadata_for_source
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from plugins.platforms.feishu.adapter import FeishuAdapter
from tests.gateway.restart_test_helpers import make_restart_runner


def _transport():
    pytest.importorskip("lark_oapi")
    from plugins.platforms.feishu.adapter import _load_lark_oapi
    assert _load_lark_oapi()
    adapter = FeishuAdapter(PlatformConfig(enabled=True, typing_indicator=False))
    response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_sent"))
    wire = SimpleNamespace(
        reply=Mock(return_value=response), create=Mock(return_value=response),
        list=Mock(return_value=SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(items=[SimpleNamespace(message_id="om_recovered", thread_id="omt_topic")]),
        )),
    )
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=wire)))

    async def run_blocking(func, *args):
        return func(*args)

    adapter._run_blocking = run_blocking
    adapter._add_reaction = AsyncMock()
    adapter._remove_reaction = AsyncMock()
    adapter._keep_typing = AsyncMock()
    adapter._stop_typing_refresh = AsyncMock()
    return adapter, wire


@pytest.mark.asyncio
@pytest.mark.parametrize("origin_anchor", ["om_persisted", None])
async def test_startup_resume_dispatches_progress_and_final_to_persisted_topic(tmp_path, origin_anchor):
    """Persist/reload → startup scheduler → real runner/base dispatch → Feishu wire API."""
    if origin_anchor is None:
        pytest.importorskip("lark_oapi")  # recovery builds the optional SDK's thread-list request
    adapter, wire = _transport()
    runner, _ = make_restart_runner(adapter)
    runner.config = GatewayConfig(platforms={Platform.FEISHU: adapter.config})
    runner.adapters = {Platform.FEISHU: adapter}
    source = SessionSource(
        platform=Platform.FEISHU, chat_id="oc_chat", chat_type="group", user_id="ou_user",
        thread_id="omt_topic", message_id=origin_anchor,
    )
    sessions_dir = tmp_path / "sessions"
    store = SessionStore(sessions_dir, runner.config)
    entry = store.get_or_create_session(source)
    assert store.mark_resume_pending(entry.session_key, "restart_interrupted")
    runner.session_store = SessionStore(sessions_dir, runner.config)
    runner.session_store._ensure_loaded()

    # Exercise the real inbound sentinel/queue guards. Only model execution and unrelated
    # slash/session lease plumbing are replaced; send/retry/reaction hooks remain real.
    runner._check_slash_access = lambda *args, **kwargs: None
    runner._begin_session_run_generation = lambda key: 1
    runner._is_session_run_current = lambda key, generation: True
    runner._invalidate_session_run_generation = lambda *args, **kwargs: 0
    runner._claim_active_session_slot = lambda key, source: (object(), None)
    runner._active_session_leases = {}
    runner._busy_ack_ts = {}
    runner._post_turn_goal_continuation = AsyncMock()
    received = []

    async def agent_turn(event, source, session_key, run_generation):
        received.append(event)
        # The progress/status lane only sends metadata, without an explicit reply_to.
        await adapter.send(source.chat_id, "Working", metadata=runner._event_thread_metadata(event, source))
        return "RESUMED OK"

    runner._handle_message_with_agent = agent_turn
    adapter.set_message_handler(runner._handle_message)
    assert runner._schedule_resume_pending_sessions() == 1
    await asyncio.wait_for(asyncio.gather(*runner._background_tasks), timeout=10)
    if adapter._background_tasks:
        await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), timeout=10)

    assert len(received) == 1
    event = received[0]
    assert event.internal and event.message_id is None
    assert event.reply_anchor_override == origin_anchor
    assert event.source.thread_id == source.thread_id
    assert wire.reply.call_count == 2
    assert all(call.args[0].message_id == (origin_anchor or "om_recovered") for call in wire.reply.call_args_list)
    assert all(call.args[0].request_body.reply_in_thread is True for call in wire.reply.call_args_list)
    assert wire.list.call_count == (0 if origin_anchor else 1)
    wire.create.assert_not_called()
    adapter._add_reaction.assert_not_awaited()
    adapter._remove_reaction.assert_not_awaited()
    assert entry.session_key not in runner._running_agents
    assert entry.session_key not in adapter._pending_messages


@pytest.mark.asyncio
@pytest.mark.parametrize("platform,thread_id", [
    (Platform.FEISHU, "omt_topic"), (Platform.FEISHU, None),
    (Platform.TELEGRAM, "42"), (Platform.DISCORD, None), (Platform.SLACK, "123.456"),
])
async def test_synthetic_prompt_routes_only_feishu_topics_without_stale_lifecycle_identity(platform, thread_id):
    """Goal/heartbeat and metadata-only sends preserve required anchors, not stale quotes."""
    source = SessionSource(
        platform=platform, chat_id="oc_chat", chat_type="dm", user_id="user",
        thread_id=thread_id, message_id="om_original",
    )
    transport_marker = object()
    source._authorization_profile_home = transport_marker
    event = GatewayRunner._synthetic_prompt_event(source, "Check the goal", internal=True)
    topic_anchor = "om_original" if platform == Platform.FEISHU and thread_id else None
    assert _reply_anchor_for_event(MessageEvent(text="", source=source)) == topic_anchor
    assert event.message_id is None
    assert event.reply_anchor_override == topic_anchor
    assert _reply_anchor_for_event(event) == topic_anchor
    assert event.source.message_id == topic_anchor
    assert event.source._authorization_profile_home is transport_marker
    assert source.message_id == "om_original"

    runner = object.__new__(GatewayRunner)
    base_metadata = _thread_metadata_for_event(event)
    runner_metadata = runner._event_thread_metadata(event, event.source)
    if topic_anchor:
        adapter, wire = _transport()
        state = base_metadata["_feishu_topic_delivery"]
        assert runner_metadata["_feishu_topic_delivery"] is state
        assert _thread_metadata_for_event(event)["_feishu_topic_delivery"] is state
        next_event = GatewayRunner._synthetic_prompt_event(source, "Next check", internal=True)
        assert _thread_metadata_for_event(next_event)["_feishu_topic_delivery"] is not state
        assert "_feishu_topic_delivery" not in event.metadata
        assert "_feishu_topic_delivery" not in event.source.to_dict()
        # Both independent metadata builders support source-only sends too.
        for metadata in (
            base_metadata, runner_metadata,
            _thread_metadata_for_source(source), runner._thread_metadata_for_source(source),
        ):
            assert metadata["reply_to_message_id"] == topic_anchor
            assert (await adapter.send(source.chat_id, "Metadata-only notice", metadata=metadata)).success
        event.reply_anchor_override = "om_redirected"
        assert _thread_metadata_for_event(event)["reply_to_message_id"] == "om_redirected"
        assert runner._event_thread_metadata(event, event.source)["reply_to_message_id"] == "om_redirected"
        assert all(call.args[0].message_id == topic_anchor for call in wire.reply.call_args_list)
        wire.create.assert_not_called()
    else:
        assert "reply_to_message_id" not in (base_metadata or {})
        assert "reply_to_message_id" not in (runner_metadata or {})

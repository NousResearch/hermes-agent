"""Delegated replies retain their account identity or make no transport call.

Exercise the real metadata builders, Telegram egress and durable recovery with an
isolated Hermes home. A synthetic source must never become a normal bot DM.
"""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import _thread_metadata_for_source
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner, _parse_session_key
from gateway.session import SessionSource, SessionStore, build_session_key
from plugins.platforms.telegram.adapter import TelegramAdapter


def _gateway():
    config = PlatformConfig(enabled=True, token="fake", typing_indicator=False, extra={
        "business": {
            "enabled": True,
            "allow_business_send_as_account": True,
            "trigger_words": ["Help"],
            "allowed_owner_ids": ["owner"],
        },
    })
    adapter = TelegramAdapter(config)
    adapter._bot = SimpleNamespace(
        id=999,
        username="test_bot",
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=77)),
        edit_message_text=AsyncMock(return_value=SimpleNamespace(message_id=77)),
        get_business_connection=AsyncMock(return_value=SimpleNamespace(
            id="connection", user=SimpleNamespace(id="owner"), is_enabled=True,
            rights=SimpleNamespace(can_reply=True),
        )),
    )
    runner = GatewayRunner(GatewayConfig(platforms={Platform.TELEGRAM: config}))
    runner.adapters[Platform.TELEGRAM] = adapter
    source = adapter.build_source(chat_id="456", user_id="customer", chat_type="dm")
    source.scope_id = "telegram-business:connection"
    source.authorized_via_telegram_business = True
    source.telegram_business_owner_id = "owner"
    return runner, adapter, source


@pytest.mark.asyncio
@pytest.mark.parametrize("authority", ["immediate", "missing", "mismatched", "restored"])
@pytest.mark.parametrize("lane", [
    "base_final", "runner_stream", "progress", "retargeted_progress", "notice", "context_warning", "heartbeat",
    "ephemeral_inline", "ephemeral_final",
])
async def test_business_delivery_never_falls_back_to_bot(authority, lane):
    runner, adapter, source = _gateway()
    event_metadata = {"allow_business_send_as_account": True, "business_connection_id": "connection"}
    if authority == "missing":
        event_metadata = None
    elif authority == "mismatched":
        event_metadata["business_connection_id"] = "another-connection"
    elif authority == "restored":
        source = SessionSource.from_dict(source.to_dict())

    # Intake authority is bound to the receiving transport, never a deliverable
    # fallback owned by the runtime profile.
    if authority == "immediate":
        with patch.object(runner, "_delivery_adapter_for", return_value=None):
            assert runner._is_user_authorized_for_source(source)

    try:
        if lane == "notice":
            await runner._deliver_platform_notice(source, "notice", event_metadata=event_metadata)
        elif lane == "context_warning":
            with (
                patch.object(runner, "_inbound_model_context_length", AsyncMock(return_value=4096)),
                patch("agent.context_references.preprocess_context_references_async", AsyncMock(
                    return_value=SimpleNamespace(blocked=True, warnings=["Refused reference"]))),
            ):
                assert await runner._expand_inbound_context_references(
                    source, build_session_key(source), "@file:blocked", event_metadata) is None
        elif lane == "heartbeat":
            turn = SimpleNamespace(
                source=source, session_key=build_session_key(source), agent_holder=[None],
                _status_thread_metadata=runner._thread_metadata_for_source(source, event_metadata=event_metadata),
                _cleanup_progress=False,
            )
            display = SimpleNamespace(
                _display_surface_mode=lambda *a, **k: "generic", _generic_status_phrase=lambda *a: "Working",
                resolve_display_setting=lambda *a: False, user_config={}, platform_key="telegram",
            )
            with (
                patch("gateway.run_turn.asyncio.sleep", AsyncMock()),
                patch.object(runner, "_should_emit_long_running_notification", side_effect=[True, True, True, False]),
                patch.object(runner, "_agent_activity_summary", return_value=None),
            ):
                await runner._run_agent_notify_long_running(display, turn, [None])
        elif lane.startswith("ephemeral_"):
            from gateway.platforms.base import EphemeralReply
            adapter._schedule_ephemeral_delete = MagicMock()
            event = MessageEvent(text="help", source=source, metadata=event_metadata or {})
            if lane == "ephemeral_inline":
                adapter._message_handler = AsyncMock(return_value=EphemeralReply("reply", ttl_seconds=1))
                await adapter._dispatch_inline_reply(event)
            else:
                await adapter._send_final_text(
                    event, build_session_key(source), "reply",
                    runner._thread_metadata_for_source(source, event_metadata=event_metadata),
                    True, 1, lambda result: None,
                )
            adapter._schedule_ephemeral_delete.assert_not_called()
        else:
            if lane == "base_final":
                metadata = _thread_metadata_for_source(source, event_metadata=event_metadata)
            elif lane == "runner_stream":
                metadata = runner._thread_metadata_for_source(source, event_metadata=event_metadata)
            else:
                metadata = runner._thread_metadata_for_progress(
                    source, None, "42" if lane == "retargeted_progress" else None,
                    None, event_metadata,
                )
            assert metadata["business_connection_id"] == "connection"
            if authority == "immediate":
                assert metadata["_telegram_business_source"] is source
            else:
                assert metadata["_delivery_route_blocked"] is True
                assert "_telegram_business_source" not in metadata
            if lane == "runner_stream":
                result = await adapter.edit_message("456", "77", "stream text", metadata=metadata)
            else:
                result = await adapter.send("456", "reply", metadata={**metadata, "notify": True})
            assert result.success is (authority == "immediate")

        calls = adapter._bot.send_message.await_args_list + adapter._bot.edit_message_text.await_args_list
        if authority == "immediate":
            assert calls
            assert all(call.kwargs["business_connection_id"] == "connection" for call in calls)
        else:
            assert calls == []
    finally:
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["persisted", "cached", "key_only", "explicit_scope"])
async def test_business_async_and_restart_routes_cannot_acquire_authority(origin):
    from gateway import delivery_ledger
    from tools import async_delegation

    runner, adapter, source = _gateway()
    key = build_session_key(source)
    event = {
        "type": "completion", "session_key": key,
        "platform": "telegram", "chat_id": source.chat_id, "chat_type": "dm",
        "session_id": "process", "started_at": time.time(),
    }
    if origin == "persisted":
        runner.session_store.get_or_create_session(source)
        old_store = runner.session_store
        runner.session_store = SessionStore(old_store.sessions_dir, runner.config)
        old_store.close_all_db_handles()
    elif origin == "cached":
        runner._session_sources[key] = source
    elif origin == "explicit_scope":
        event["session_key"] = ""
        event["scope_id"] = source.scope_id
    else:
        parsed = _parse_session_key(key)
        assert parsed["chat_id"] == source.chat_id
        assert parsed["scope_id"] == source.scope_id

    restored = runner._build_process_event_source(event)
    assert restored.scope_id == source.scope_id
    assert restored.chat_id == source.chat_id
    # The cached source may retain an in-process proof, but no new event-bound
    # send opt-in is created for a completion, progress notice or resumed turn.
    assert runner._thread_metadata_for_source(restored)["_delivery_route_blocked"] is True
    adapter.handle_message = AsyncMock()
    try:
        assert await runner._inject_watch_notification("completed", event) is None
        await runner._send_watcher_message("telegram", source.chat_id, None, "raw completion", event)
        await runner._send_watcher_message("telegram", source.chat_id, None, "raw completion", {
            "scope_id": source.scope_id,
        })
        assert await runner._hm_admit_event(MessageEvent(text="resume", source=restored, internal=True)) is None
        assert not runner._resume_owner_authorized(key, restored)
        assert await runner._shutdown_notification_target(key) is None
        await runner._hm_offer_pairing_code(restored)
        await runner._hm_send_unauthorized_decline(restored)
        marker_metadata = runner._pending_marker_metadata(Platform.TELEGRAM, source.chat_id, event, adapter)
        assert marker_metadata["_delivery_route_blocked"] is True
        from gateway.stream_consumer import GatewayStreamConsumer
        adapter.delete_message = AsyncMock(return_value=True)
        consumer = GatewayStreamConsumer(
            adapter=adapter, chat_id=source.chat_id,
            metadata=runner._thread_metadata_for_source(source, event_metadata={
                "allow_business_send_as_account": True, "business_connection_id": "connection",
            }),
        )
        await consumer._delete_previews({"77"}, label="business preview")
        adapter.delete_message.assert_not_awaited()

        # Route-only APIs cannot gain account authority. Legacy cron origin
        # capture retains intent even when async capability is disabled.
        from gateway.delivery import DeliveryRouter, DeliveryTarget
        from gateway.session_context import async_delivery_supported, clear_session_vars, set_session_vars
        from tools.cronjob_job_args import _origin_from_env
        from tools.cronjob_tools import _action_create
        from tools.send_message_tool import _send_to_platform, send_message_tool
        from cron.scheduler_delivery import _prepare_target_delivery

        tokens = set_session_vars(platform="telegram", chat_id=source.chat_id, scope_id=source.scope_id)
        try:
            assert not async_delivery_supported()
            captured_origin = _origin_from_env("1h")
            assert captured_origin["scope_id"] == source.scope_id
            assert "unavailable" in json.loads(_action_create({}))["error"]
            assert "unavailable" in json.loads(send_message_tool({"action": "send"}))["error"]
            assert "unavailable" in (await _send_to_platform(
                Platform.TELEGRAM, adapter.config, source.chat_id, "separate send"))["error"]
        finally:
            clear_session_vars(tokens)
        assert async_delivery_supported()
        target = DeliveryTarget.parse("origin", source)
        assert target.scope_id == source.scope_id
        routed = await DeliveryRouter(runner.config, runner.adapters).deliver("cron result", [target])
        assert routed["origin"]["success"] is False
        with patch("cron.jobs.get_job", return_value={
            "id": "job", "deliver": "origin", "origin": captured_origin,
        }):
            assert await runner._notify_interrupted_cron_jobs(["job"]) == 0
        for loop in (None, asyncio.get_running_loop()):
            errors = []
            result = _prepare_target_delivery(
                {"id": "job", "deliver": "origin", "origin": captured_origin},
                {"platform": "telegram", "chat_id": source.chat_id, "thread_id": "changed-thread"},
                adapters=runner.adapters, loop=loop, config=runner.config,
                notify_delivery=True, mirror_enabled=True, mirror_text="cron result", delivery_errors=errors,
            )
            assert result is None
            assert errors and "event-bound" in errors[0]

        # Durable async results get an honest terminal disposition, including
        # grouped completions; they must not replay on every gateway restart.
        group = []
        for index in range(2):
            completion = {**event, "type": "async_delegation", "delegation_id": f"delegation-{index}",
                          "dispatched_at": time.time(), "status": "completed", "summary": "finished"}
            async_delegation._persist_dispatch(completion)
            async_delegation._persist_completion(completion, {"result": "finished"})
            group.append(completion)
        assert await runner._deliver_async_delegation_group(group) is None
        for completion in group:
            stored = async_delegation.get_durable_delegation(completion["delegation_id"])
            assert stored["delivery_state"] == "dropped"
            assert stored["result"] == {"result": "finished"}

        # A crash-left transcript must not mint a new ordinary delivery row.
        entry = runner.session_store.get_or_create_session(source)
        runner.session_store.mark_turn_active(entry.session_key)
        runner.session_store.append_to_transcript(entry.session_id, {
            "role": "user", "content": "Help with this", "timestamp": time.time(),
        })
        runner.session_store.append_to_transcript(entry.session_id, {
            "role": "assistant", "content": "Saved reply", "timestamp": time.time(),
        })
        assert await runner._ledger_crash_left_replies(3600) == 0
        assert entry.active_turn_token is None

        # Also refuse an old ledger row written before the fail-closed policy.
        delivery_ledger.record_obligation(
            obligation_id="old-business-final", session_key=key, platform="telegram",
            chat_id=source.chat_id, thread_id=None, content="old reply",
        )
        assert await runner._redeliver_claimed_obligations([{
            "obligation_id": "old-business-final", "session_key": key,
            "platform": "telegram", "chat_id": source.chat_id, "content": "old reply", "attempts": 1,
        }]) == 0
        with delivery_ledger._connect() as conn:
            rows = conn.execute("SELECT obligation_id, state FROM delivery_obligations").fetchall()
        assert rows == [("old-business-final", "failed")]
        adapter.handle_message.assert_not_awaited()
        adapter._bot.send_message.assert_not_awaited()
        adapter._bot.edit_message_text.assert_not_awaited()
    finally:
        runner.session_store.close_all_db_handles()

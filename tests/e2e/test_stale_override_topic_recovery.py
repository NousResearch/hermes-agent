"""Real-ingress regression for stale notices in Telegram DM topic mode."""

import dataclasses
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.run import GatewayRunner
from gateway.session import (
    AsyncSessionStore,
    SessionSource,
    SessionStore,
    build_session_key,
)
from gateway.stale_override_notice import (
    OverrideNoticeDecision,
    StaleOverrideNoticeConfig,
)
from plugins.platforms.telegram.adapter import TelegramAdapter


def _runner_with_store(config, store, db):
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = 0.0
    runner._stop_task = None
    runner._busy_input_mode = "interrupt"
    runner._running_agents_ts = {}
    runner._pending_model_notes = {}
    runner._update_prompt_pending = {}
    runner._session_db = SimpleNamespace(_db=db)
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._stale_override_pending = {}
    runner._is_user_authorized = lambda _source: True
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_topic_recovered_ingress_uses_canonical_clock_entry(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="e2e-test-token")
        },
        stale_override_notice=StaleOverrideNoticeConfig(
            mode="info_only",
            idle_minutes=1,
            channels=("*",),
        ),
        sessions_dir=hermes_home / "sessions",
    )
    store = SessionStore(config.sessions_dir, config)

    canonical_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="topic-chat",
        chat_type="dm",
        user_id="topic-user",
        user_name="topic user",
        thread_id="42",
    )
    entry = store.get_or_create_session(canonical_source)
    canonical_key = entry.session_key
    raw_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="topic-chat",
        chat_type="dm",
        user_id="topic-user",
        user_name="topic user",
        thread_id=None,
    )
    raw_key = build_session_key(raw_source)
    assert raw_key != canonical_key
    store.set_session_metadata(
        canonical_key,
        "stale_override_last_completed_at",
        time.time() - 3600,
    )

    db = store._db
    db.enable_telegram_topic_mode(chat_id="topic-chat", user_id="topic-user")
    db.bind_telegram_topic(
        chat_id="topic-chat",
        thread_id="42",
        user_id="topic-user",
        session_key=canonical_key,
        session_id=entry.session_id,
    )

    runner = _runner_with_store(config, store, db)
    runner._stale_override_decision = MagicMock(
        return_value=OverrideNoticeDecision(
            model_stale=True,
            current_route="provider/custom",
            default_route="provider/default",
        )
    )

    seen = {}

    async def _agent(event, source, session_key, generation):
        seen.update(
            source=source,
            session_key=session_key,
            generation=generation,
        )
        return "agent response"

    runner._handle_message_with_agent = _agent
    adapter = TelegramAdapter(config.platforms[Platform.TELEGRAM])
    adapter.send = AsyncMock(
        return_value=SendResult(success=True, message_id="e2e-response")
    )
    adapter.send_typing = AsyncMock()
    adapter.set_message_handler(runner._handle_message)
    runner.adapters[Platform.TELEGRAM] = adapter

    event = MessageEvent(
        text="ordinary message",
        message_id="inbound-message",
        source=raw_source,
    )
    result = await runner._handle_message(event)
    assert result == "agent response"

    completion_callback = adapter.pop_post_delivery_callback(
        canonical_key,
        generation=seen["generation"],
    )
    assert completion_callback is not None
    await completion_callback()

    assert event.source.thread_id == "42"
    assert seen["source"].thread_id == "42"
    assert seen["session_key"] == canonical_key
    runner._stale_override_decision.assert_called_once()
    assert (
        runner._stale_override_decision.call_args.kwargs["session_key"] == canonical_key
    )
    assert (
        store.get_session_metadata(raw_key, "stale_override_last_completed_at") is None
    )

    # Completion metadata uses the single-entry DB UPSERT and survives a fresh
    # SessionStore load without requiring a full sessions.json rewrite.
    restarted = SessionStore(config.sessions_dir, config)
    completed_at = restarted.get_session_metadata(
        canonical_key, "stale_override_last_completed_at"
    )
    assert completed_at > time.time() - 10


@pytest.mark.asyncio
async def test_topic_recovered_busy_ingress_preserves_original_active_agent(
    tmp_path, monkeypatch
):
    """A stripped topic follow-up must enter the canonical busy path."""
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "0")
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="e2e-test-token")
        },
        sessions_dir=hermes_home / "sessions",
    )
    store = SessionStore(config.sessions_dir, config)
    canonical_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="topic-chat",
        chat_type="dm",
        user_id="topic-user",
        user_name="topic user",
        thread_id="42",
    )
    canonical_entry = store.get_or_create_session(canonical_source)
    canonical_key = canonical_entry.session_key
    raw_source = dataclasses.replace(canonical_source, thread_id=None)

    db = store._db
    db.enable_telegram_topic_mode(chat_id="topic-chat", user_id="topic-user")
    db.bind_telegram_topic(
        chat_id="topic-chat",
        thread_id="42",
        user_id="topic-user",
        session_key=canonical_key,
        session_id=canonical_entry.session_id,
    )

    class ActiveAgent:
        def __init__(self):
            self.steered = []

        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

        def steer(self, text):
            self.steered.append(text)
            return True

    runner = _runner_with_store(config, store, db)
    runner._busy_input_mode = "steer"
    active_agent = ActiveAgent()
    active_state = runner._session_state(canonical_key)
    active_state.turn.agent = active_agent
    active_state.turn.started_ts = time.time() - 10
    runner._handle_message_with_agent = AsyncMock(
        side_effect=AssertionError("canonical busy follow-up dispatched a second turn")
    )
    original_claim = runner._claim_active_session_slot
    runner._claim_active_session_slot = MagicMock(wraps=original_claim)

    event = MessageEvent(
        text="steer the active topic turn",
        message_id="busy-follow-up",
        source=raw_source,
    )
    result = await runner._handle_message(event)

    assert result is None
    assert event.source.thread_id == "42"
    assert active_agent.steered == ["steer the active topic turn"]
    runner._claim_active_session_slot.assert_not_called()
    runner._handle_message_with_agent.assert_not_awaited()
    assert runner._peek_session_state(canonical_key).turn.agent is active_agent

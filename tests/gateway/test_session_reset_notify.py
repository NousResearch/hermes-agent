"""Tests for session auto-reset notifications.

Verifies that:
- _should_reset() returns a reason string ("idle" or "daily") instead of bool
- SessionEntry captures auto_reset_reason
- SessionResetPolicy.notify controls whether notifications are sent
- notify_exclude_platforms skips notifications for excluded platforms
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import (
    GatewayConfig,
    Platform,
    PlatformConfig,
    SessionResetPolicy,
)
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(platform=Platform.TELEGRAM, chat_id="123", user_id="u1"):
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
    )


def _make_store(policy=None, tmp_path=None):
    config = GatewayConfig()
    if policy:
        config.default_reset_policy = policy
    store = SessionStore(sessions_dir=tmp_path or "/tmp/test-sessions", config=config)
    return store


def _make_runner_for_suspended_notice(tmp_path, policy):
    from gateway.run import GatewayRunner

    source = _make_source()
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token"),
        },
        default_reset_policy=policy,
        sessions_dir=tmp_path / "sessions",
    )
    store = SessionStore(sessions_dir=config.sessions_dir, config=config)
    store._db = None

    original = store.get_or_create_session(source)
    assert store.suspend_session(original.session_key)

    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.stop_typing = AsyncMock()
    adapter.supports_async_delivery = True
    adapter._active_sessions = {}

    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = store
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_db = None
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._last_resolved_model = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_native_image_paths_by_session = {}
    runner._session_run_generation = {original.session_key: 1}
    runner._show_reasoning = False
    runner._should_send_voice_reply = lambda *args, **kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._reset_notice_session_info = lambda _source: ""
    runner._deliver_platform_notice = AsyncMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._is_telegram_topic_lane = lambda _source: False
    runner._record_telegram_topic_binding = lambda *args, **kwargs: None
    runner._sync_telegram_topic_binding = lambda *args, **kwargs: None

    async def _run_agent(**kwargs):
        message = kwargs["message"]
        return {
            "final_response": "stubbed response",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "stubbed response"},
            ],
            "history_offset": 0,
            "api_calls": 0,
            "last_prompt_tokens": 0,
            "context_length": 1000,
            "model": "test-model",
            "tools": [],
        }

    runner._run_agent = AsyncMock(side_effect=_run_agent)
    event = MessageEvent(text="hello", source=source, message_id="m1")
    return runner, adapter, source, original.session_key, event


async def _run_suspended_notice_case(tmp_path, policy):
    runner, adapter, source, session_key, event = _make_runner_for_suspended_notice(
        tmp_path,
        policy,
    )
    result = await runner._handle_message_with_agent(
        event,
        source,
        session_key,
        1,
    )
    return adapter, result


# ---------------------------------------------------------------------------
# _should_reset returns reason string
# ---------------------------------------------------------------------------

class TestShouldResetReason:
    def test_returns_none_when_not_expired(self, tmp_path):
        store = _make_store(
            SessionResetPolicy(mode="both", idle_minutes=60, at_hour=4),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),  # just updated
        )
        source = _make_source()
        assert store._should_reset(entry, source) is None

    def test_returns_idle_when_idle_expired(self, tmp_path):
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=30),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now() - timedelta(hours=2),
            updated_at=datetime.now() - timedelta(hours=1),  # 60min ago > 30min threshold
        )
        source = _make_source()
        assert store._should_reset(entry, source) == "idle"

    def test_returns_daily_when_daily_boundary_crossed(self, tmp_path):
        now = datetime.now()
        store = _make_store(
            SessionResetPolicy(mode="daily", at_hour=now.hour),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=now - timedelta(days=2),
            updated_at=now - timedelta(days=1),  # last active yesterday
        )
        source = _make_source()
        assert store._should_reset(entry, source) == "daily"

    def test_returns_none_when_mode_is_none(self, tmp_path):
        store = _make_store(
            SessionResetPolicy(mode="none"),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now() - timedelta(days=30),
            updated_at=datetime.now() - timedelta(days=30),
        )
        source = _make_source()
        assert store._should_reset(entry, source) is None


# ---------------------------------------------------------------------------
# SessionEntry captures reason
# ---------------------------------------------------------------------------

class TestSessionEntryReason:
    def test_auto_reset_reason_stored(self, tmp_path):
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=1),
            tmp_path,
        )
        source = _make_source()

        # Create initial session
        entry1 = store.get_or_create_session(source)
        assert not entry1.was_auto_reset

        # Age it past the idle threshold
        entry1.updated_at = datetime.now() - timedelta(minutes=5)
        store._save()

        # Next call should create a new session with reason
        entry2 = store.get_or_create_session(source)
        assert entry2.was_auto_reset is True
        assert entry2.auto_reset_reason == "idle"
        assert entry2.session_id != entry1.session_id

    def test_reset_had_activity_false_when_no_tokens(self, tmp_path):
        """Expired session with no tokens → reset_had_activity=False."""
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=1),
            tmp_path,
        )
        source = _make_source()

        entry1 = store.get_or_create_session(source)
        # No tokens used — session was idle with no conversation
        entry1.updated_at = datetime.now() - timedelta(minutes=5)
        store._save()

        entry2 = store.get_or_create_session(source)
        assert entry2.was_auto_reset is True
        assert entry2.reset_had_activity is False

    def test_reset_had_activity_true_when_tokens_used(self, tmp_path):
        """Expired session with tokens → reset_had_activity=True."""
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=1),
            tmp_path,
        )
        source = _make_source()

        entry1 = store.get_or_create_session(source)
        # Simulate some conversation happened (last_prompt_tokens is the field
        # written on every turn; total_tokens is never persisted).
        entry1.last_prompt_tokens = 5000
        entry1.updated_at = datetime.now() - timedelta(minutes=5)
        store._save()

        entry2 = store.get_or_create_session(source)
        assert entry2.was_auto_reset is True
        assert entry2.reset_had_activity is True


# ---------------------------------------------------------------------------
# SessionResetPolicy notify config
# ---------------------------------------------------------------------------

class TestResetPolicyNotify:
    def test_notify_defaults_true(self):
        policy = SessionResetPolicy()
        assert policy.notify is True

    def test_notify_exclude_defaults(self):
        policy = SessionResetPolicy()
        assert "api_server" in policy.notify_exclude_platforms
        assert "webhook" in policy.notify_exclude_platforms

    def test_from_dict_with_notify_false(self):
        policy = SessionResetPolicy.from_dict({"notify": False})
        assert policy.notify is False

    def test_from_dict_with_custom_excludes(self):
        policy = SessionResetPolicy.from_dict({
            "notify_exclude_platforms": ["api_server", "webhook", "homeassistant"],
        })
        assert "homeassistant" in policy.notify_exclude_platforms

    def test_from_dict_preserves_defaults_on_missing_keys(self):
        policy = SessionResetPolicy.from_dict({})
        assert policy.notify is True
        assert "api_server" in policy.notify_exclude_platforms

    def test_to_dict_roundtrip(self):
        original = SessionResetPolicy(
            mode="idle",
            notify=False,
            notify_exclude_platforms=("api_server",),
        )
        restored = SessionResetPolicy.from_dict(original.to_dict())
        assert restored.notify == original.notify
        assert restored.notify_exclude_platforms == original.notify_exclude_platforms
        assert restored.mode == original.mode


# ---------------------------------------------------------------------------
# GatewayRunner notification policy for suspended auto-resets
# ---------------------------------------------------------------------------

class TestGatewayRunnerSuspendedResetNotify:
    @pytest.mark.asyncio
    async def test_suspended_auto_reset_notifies_when_configured(self, tmp_path):
        policy = SessionResetPolicy(mode="none", notify=True)

        adapter, result = await _run_suspended_notice_case(tmp_path, policy)

        assert result == "stubbed response"
        adapter.send.assert_awaited_once()
        args, kwargs = adapter.send.await_args
        assert args[0] == "123"
        assert "previous session was stopped or interrupted" in args[1]
        assert kwargs.get("metadata") is None

    @pytest.mark.asyncio
    async def test_suspended_auto_reset_notify_false_does_not_send(self, tmp_path):
        policy = SessionResetPolicy(mode="none", notify=False)

        adapter, result = await _run_suspended_notice_case(tmp_path, policy)

        assert result == "stubbed response"
        adapter.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_suspended_auto_reset_excluded_platform_does_not_send(self, tmp_path):
        policy = SessionResetPolicy(
            mode="none",
            notify=True,
            notify_exclude_platforms=("telegram",),
        )

        adapter, result = await _run_suspended_notice_case(tmp_path, policy)

        assert result == "stubbed response"
        adapter.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# SessionEntry to_dict / from_dict roundtrip for auto-reset fields
# ---------------------------------------------------------------------------

class TestSessionEntryAutoResetRoundtrip:
    def test_was_auto_reset_persists_across_roundtrip(self, tmp_path):
        """was_auto_reset=True survives to_dict() → from_dict() (gateway restart)."""
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=1),
            tmp_path,
        )
        source = _make_source()

        entry = store.get_or_create_session(source)
        entry.updated_at = datetime.now() - timedelta(minutes=5)
        store._save()

        entry2 = store.get_or_create_session(source)
        assert entry2.was_auto_reset is True
        assert entry2.auto_reset_reason == "idle"
        assert entry2.session_id != entry.session_id

        # Simulate gateway restart: reload from disk
        store._loaded = False
        store._entries.clear()
        store._ensure_loaded()

        reloaded = store._entries.get(entry2.session_key)
        assert reloaded is not None
        assert reloaded.was_auto_reset is True
        assert reloaded.auto_reset_reason == "idle"

    def test_reset_had_activity_persists_across_roundtrip(self, tmp_path):
        """reset_had_activity survives to_dict() → from_dict() (gateway restart)."""
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=1),
            tmp_path,
        )
        source = _make_source()

        entry = store.get_or_create_session(source)
        entry.last_prompt_tokens = 1000
        entry.updated_at = datetime.now() - timedelta(minutes=5)
        store._save()

        entry2 = store.get_or_create_session(source)
        assert entry2.reset_had_activity is True

        store._loaded = False
        store._entries.clear()
        store._ensure_loaded()

        reloaded = store._entries.get(entry2.session_key)
        assert reloaded is not None
        assert reloaded.reset_had_activity is True

    def test_auto_reset_reason_none_roundtrip(self, tmp_path):
        """auto_reset_reason=None (no reset) survives roundtrip cleanly."""
        store = _make_store(tmp_path=tmp_path)
        source = _make_source()

        entry = store.get_or_create_session(source)
        assert entry.was_auto_reset is False

        store._loaded = False
        store._entries.clear()
        store._ensure_loaded()

        reloaded = store._entries.get(entry.session_key)
        assert reloaded is not None
        assert reloaded.was_auto_reset is False
        assert reloaded.auto_reset_reason is None
        assert reloaded.reset_had_activity is False

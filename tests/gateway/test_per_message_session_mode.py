"""Per-message session boundaries are scoped to a Telegram chat/topic."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner, _get_session_mode
from gateway.session import SessionSource, SessionStore


def source(chat="100", thread=None):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat, chat_type="dm",
                         thread_id=thread, user_id="owner")


def config():
    return GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(channel_overrides={
        "100": ChannelOverride(session_mode="per_message"),
        "100:8": ChannelOverride(session_mode="conversational"),
        "100:9": ChannelOverride(session_mode="per_message"),
        "100:10": ChannelOverride(model="other/model"),
    })})


def test_mode_topic_first_then_chat_then_default():
    cfg = config()
    assert _get_session_mode(cfg, source("100", "8")) == "conversational"
    assert _get_session_mode(cfg, source("100", "9")) == "per_message"
    assert _get_session_mode(cfg, source("100", "10")) == "per_message"
    assert _get_session_mode(cfg, source("100")) == "per_message"
    assert _get_session_mode(cfg, source("200", "9")) == "conversational"


def test_config_roundtrip_and_invalid_mode():
    pc = PlatformConfig.from_dict({"channel_overrides": {
        "100:9": {"session_mode": "per_message"}, "100": {"model": "m"},
    }})
    assert pc.channel_overrides["100:9"].session_mode == "per_message"
    assert pc.channel_overrides["100"].session_mode is None
    assert PlatformConfig.from_dict(pc.to_dict()).channel_overrides["100:9"].session_mode == "per_message"
    with pytest.raises(ValueError, match="session_mode"):
        ChannelOverride.from_dict({"session_mode": "stateless"})


def test_busy_mode_queues_distinct_messages_in_per_message_topic():
    runner = object.__new__(GatewayRunner)
    runner.config = config()
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "queue"
    assert runner._effective_busy_input_mode(source("100", "9")) == "queue"
    assert runner._effective_busy_text_mode(source("100", "9")) == "interrupt"
    assert runner._effective_busy_input_mode(source("100", "8")) == "interrupt"


@pytest.mark.asyncio
async def test_topic_turn_rotates_after_binding_heal_without_old_history(tmp_path):
    cfg = config()
    store = SessionStore(sessions_dir=tmp_path, config=cfg)
    runner = object.__new__(GatewayRunner)
    runner.config = cfg
    runner.session_store = store
    runner._session_db = None
    runner._recover_telegram_topic_thread_id = Mock(return_value=None)
    runner._session_key_for_source = store._generate_session_key
    runner._cache_session_source = Mock()
    runner._is_telegram_topic_lane = Mock(return_value=True)
    runner._record_telegram_topic_binding = Mock()
    runner._clear_conversation_scope = Mock()
    runner._evict_cached_agent = Mock()
    src = source("100", "9")
    event = SimpleNamespace(metadata={}, internal=False, source=src)
    with patch("gateway.run_heartbeat_acceptance.resolve_heartbeat_owner", new_callable=AsyncMock, return_value=True):
        _, first, key = await runner._hmwa_resolve_session(event, src)
        store.append_to_transcript(first.session_id, {"role": "user", "content": "lights off"})
        _, second, _ = await runner._hmwa_resolve_session(event, src)
    assert first.session_id != second.session_id
    assert store.load_transcript(second.session_id) == []
    assert runner._record_telegram_topic_binding.call_args.args == (src, second)
    assert runner._clear_conversation_scope.call_count == 2
    assert key == second.session_key


@pytest.mark.asyncio
async def test_internal_turn_keeps_its_session(tmp_path):
    cfg = config()
    store = SessionStore(sessions_dir=tmp_path, config=cfg)
    runner = object.__new__(GatewayRunner)
    runner.config = cfg
    runner.session_store = store
    runner._session_db = None
    runner._recover_telegram_topic_thread_id = Mock(return_value=None)
    runner._session_key_for_source = store._generate_session_key
    runner._cache_session_source = Mock()
    runner._is_telegram_topic_lane = Mock(return_value=False)
    runner._clear_conversation_scope = Mock()
    runner._evict_cached_agent = Mock()
    src = source()
    event = SimpleNamespace(metadata={}, internal=True, source=src)
    with patch("gateway.run_heartbeat_acceptance.resolve_heartbeat_owner", new_callable=AsyncMock, return_value=True):
        _, first, _ = await runner._hmwa_resolve_session(event, src)
        _, second, _ = await runner._hmwa_resolve_session(event, src)
    assert first.session_id == second.session_id
    runner._clear_conversation_scope.assert_not_called()

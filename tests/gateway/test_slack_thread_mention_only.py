"""Regression tests for Slack per-thread mention-only routing.

When a user tells a bot to stay out of a Slack thread until it is called,
the adapter should set a thread-local flag before the message reaches the
agent. While the flag is active, normal active-session / mentioned-thread /
free-response heuristics must not wake the agent; only an explicit Slack
@mention may pass.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter, _MentionOnlyThreadState


CHANNEL = "C_THREAD"
THREAD = "1700000000.000001"
BOT = "U_BOT"
USER = "U_USER"
TEAM = "T_TEAM"


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-test")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = BOT
    a._running = True
    a.handle_message = AsyncMock()
    return a


def _thread_event(text: str, *, ts: str = "1700000000.000010") -> dict:
    return {
        "channel": CHANNEL,
        "channel_type": "channel",
        "team": TEAM,
        "user": USER,
        "text": text,
        "ts": ts,
        "thread_ts": THREAD,
    }


async def _handle(adapter: SlackAdapter, event: dict) -> None:
    with (
        patch.object(
            adapter, "_resolve_user_name", new=AsyncMock(return_value="tester")
        ),
        patch.object(adapter, "_fetch_thread_context", new=AsyncMock(return_value="")),
        patch.object(
            adapter, "_fetch_thread_parent_text", new=AsyncMock(return_value=None)
        ),
    ):
        await adapter._handle_slack_message(event)


@pytest.mark.asyncio
async def test_enable_phrase_sets_thread_flag_and_consumes_message(adapter):
    await _handle(adapter, _thread_event(f"<@{BOT}> 이 쓰레드는 이제 태그할 때만 나와"))

    assert (CHANNEL, THREAD) in adapter._mention_only_threads
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_mention_only_thread_blocks_active_session_followups(adapter):
    adapter._set_slack_thread_mention_only(CHANNEL, THREAD, enabled=True)
    # These legacy wake paths must not override the explicit per-thread mute.
    adapter._mentioned_threads.add(THREAD)
    adapter._bot_message_ts.add(THREAD)

    await _handle(adapter, _thread_event("네 그렇게 진행하면 됩니다"))

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_mention_only_thread_blocks_free_response_followups(adapter):
    adapter.config.extra["free_response_channels"] = CHANNEL
    adapter._set_slack_thread_mention_only(CHANNEL, THREAD, enabled=True)

    await _handle(adapter, _thread_event("free-response 채널이어도 그냥 끼어들지 마"))

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_mention_only_thread_blocks_configured_name_patterns(adapter):
    adapter.config.extra["mention_patterns"] = [r"오시야"]
    adapter._set_slack_thread_mention_only(CHANNEL, THREAD, enabled=True)

    await _handle(adapter, _thread_event("오시야 이건 다시 확인해줘"))

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_mention_only_thread_allows_explicit_slack_mention(adapter):
    adapter._set_slack_thread_mention_only(CHANNEL, THREAD, enabled=True)

    await _handle(adapter, _thread_event(f"<@{BOT}> 이건 다시 확인해줘"))

    adapter.handle_message.assert_awaited_once()
    assert (CHANNEL, THREAD) in adapter._mention_only_threads
    assert THREAD not in adapter._mentioned_threads


@pytest.mark.asyncio
async def test_disable_phrase_clears_thread_flag(adapter):
    adapter._set_slack_thread_mention_only(CHANNEL, THREAD, enabled=True)

    await _handle(adapter, _thread_event(f"<@{BOT}> 침묵 해제하고 다시 답해도 돼"))

    assert (CHANNEL, THREAD) not in adapter._mention_only_threads
    adapter.handle_message.assert_awaited_once()


def test_mention_only_thread_cleanup_prunes_stale_entries(adapter):
    adapter.config.extra["mention_only_thread_ttl_minutes"] = 0.01
    adapter._mention_only_threads[(CHANNEL, THREAD)] = _MentionOnlyThreadState(
        enabled_at=0.0,
        updated_at=0.0,
    )

    assert not adapter._slack_thread_is_mention_only(CHANNEL, THREAD)
    assert (CHANNEL, THREAD) not in adapter._mention_only_threads

"""Slack thread watermark on the cold-start turn, and thread context on long threads (#157082).

The first message in an existing thread has no session yet, so its watermark can't be written
when the thread is hydrated. It must still reach the session, so that the next turn (in the same
process or after a restart) injects what other apps posted in between. And thread context must
be the newest replies of a long thread, not the oldest page.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_slack import SlackAdapter  # installs the slack SDK stand-in

CH, ROOT, TEAM, USER = "C123", "100.000", "T1", "U1"
WATERMARK_KEY = f"slack_thread_watermark:{CH}:{ROOT}"


def _thread(n_replies, bot_reply_ts=()):
    msgs = [{"ts": ROOT, "user": USER, "text": "root"}]
    for i in range(1, n_replies + 1):
        ts = f"100.{i:03d}"
        if ts in bot_reply_ts:
            msgs.append({"ts": ts, "bot_id": "B_OTHER", "subtype": "bot_message",
                         "username": "other-app", "text": f"bot update {i}"})
        else:
            msgs.append({"ts": ts, "user": USER, "text": f"reply {i}"})
    return msgs


def _slack_replies(messages):
    """Emulate conversations.replies: oldest-first pages of ``limit``, with a cursor."""
    async def _replies(channel, ts, limit, inclusive=True, cursor=None, **kw):
        start = int(cursor) if cursor else 0
        end = start + limit
        more = end < len(messages)
        return {"ok": True, "messages": messages[start:end], "has_more": more,
                "response_metadata": {"next_cursor": str(end) if more else ""}}
    return _replies


@pytest.fixture
def slack():
    a = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(return_value={"user": {
        "is_bot": False, "profile": {"display_name": "Test User"}, "real_name": "Test User"}})
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.fixture
def store(slack, tmp_path):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    s = SessionStore(tmp_path / "sessions", GatewayConfig())
    slack.set_session_store(s)
    slack._PENDING_WATERMARK_RETRY_DELAYS = (0.01, 0.05, 0.1)
    yield s
    for task in list(slack._pending_watermark_tasks):
        task.cancel()
    s.close_all_db_handles()


async def _turn(slack, ts, mentioned=False):
    return await slack._hydrate_thread_context(
        channel_id=CH, event_thread_ts=ROOT, ts=ts, user_id=USER, team_id=TEAM,
        is_thread_reply=True, is_mentioned=mentioned, is_dm=False)


def _create_session(slack, store):
    """What the gateway does after the adapter hands over the cold-start message."""
    return store.get_or_create_session(slack.build_source(
        chat_id=CH, chat_type="group", user_id=USER, thread_id=ROOT, scope_id=TEAM))


@pytest.mark.asyncio
@pytest.mark.parametrize("restart_between_turns", [False, True])
async def test_app_post_between_first_and_second_turn_is_injected(slack, store, restart_between_turns):
    slack._app.client.conversations_replies.side_effect = _slack_replies(_thread(2))
    ctx1, _, _ = await _turn(slack, "100.002")  # cold start: no session yet
    assert ctx1 and "reply 1" in ctx1
    entry = _create_session(slack, store)

    # The pending watermark reaches the session in the background, before any restart.
    await asyncio.gather(*slack._pending_watermark_tasks)
    assert store.get_session_metadata(entry.session_key, WATERMARK_KEY, "") == "100.002"

    # Another app posts 100.003 (dropped under allow_bots=none), then a human replies 100.004.
    slack._app.client.conversations_replies.side_effect = _slack_replies(
        _thread(4, bot_reply_ts=("100.003",)))
    slack._thread_context_cache.clear()
    if restart_between_turns:
        slack._thread_rehydration_checked.clear()
        slack._pending_thread_watermarks.clear()
    ctx2, _, _ = await _turn(slack, "100.004")
    assert ctx2 and "bot update 3" in ctx2
    assert "reply 1" not in ctx2  # only the delta, not the whole thread again
    assert store.get_session_metadata(entry.session_key, WATERMARK_KEY, "") == "100.004"


@pytest.mark.asyncio
async def test_second_turn_reads_pending_watermark_before_background_write(slack, store):
    slack._PENDING_WATERMARK_RETRY_DELAYS = (60.0,)  # the background write hasn't run yet
    slack._app.client.conversations_replies.side_effect = _slack_replies(_thread(2))
    await _turn(slack, "100.002")
    entry = _create_session(slack, store)

    slack._app.client.conversations_replies.side_effect = _slack_replies(
        _thread(4, bot_reply_ts=("100.003",)))
    slack._thread_context_cache.clear()
    ctx2, _, _ = await _turn(slack, "100.004")
    assert ctx2 and "bot update 3" in ctx2
    assert slack._pending_thread_watermarks == {}
    assert store.get_session_metadata(entry.session_key, WATERMARK_KEY, "") == "100.004"


@pytest.mark.asyncio
async def test_later_turns_skip_the_fetch_once_the_watermark_is_stored(slack, store):
    slack._app.client.conversations_replies.side_effect = _slack_replies(_thread(2))
    await _turn(slack, "100.002")
    _create_session(slack, store)
    await asyncio.gather(*slack._pending_watermark_tasks)
    await _turn(slack, "100.003")
    calls = slack._app.client.conversations_replies.call_count
    await _turn(slack, "100.004")
    assert slack._app.client.conversations_replies.call_count == calls


@pytest.mark.asyncio
async def test_thread_context_keeps_the_newest_replies(slack, store):
    slack._app.client.conversations_replies.side_effect = _slack_replies(_thread(50))
    ctx = await slack._fetch_thread_context(
        channel_id=CH, thread_ts=ROOT, current_ts="100.050", team_id=TEAM)
    assert "root" in ctx
    assert "reply 49" in ctx
    assert "reply 19" not in ctx  # a window of the newest 30, plus the root


@pytest.mark.asyncio
async def test_watermark_delta_on_a_long_thread(slack, store):
    slack._app.client.conversations_replies.side_effect = _slack_replies(_thread(50))
    delta = await slack._fetch_thread_context(
        channel_id=CH, thread_ts=ROOT, current_ts="100.050", team_id=TEAM,
        after_ts="100.040", force_refresh=True)
    assert "reply 41" in delta and "reply 49" in delta
    assert "reply 40" not in delta


@pytest.mark.asyncio
async def test_short_thread_is_one_call(slack, store):
    slack._app.client.conversations_replies.side_effect = _slack_replies(_thread(5))
    await slack._fetch_thread_context(
        channel_id=CH, thread_ts=ROOT, current_ts="100.005", team_id=TEAM)
    assert slack._app.client.conversations_replies.call_count == 1

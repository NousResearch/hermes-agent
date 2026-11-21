"""Root-mentioned conversations use native handling without enrolling every mentioned thread."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_slack import SlackAdapter


@pytest.fixture
def adapter():
    result = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    result.config.extra.update(thread_followup_mode="root_mention", allowed_channels=[],
                               free_response_channels=[], ignore_other_user_mentions=False)
    result._bot_user_id = "BOT"
    result._team_bot_user_ids = {"TEAM": "BOT"}
    result._client = AsyncMock()
    result._get_client = MagicMock(return_value=result._client)
    return result


async def admit(adapter, **changes):
    arguments = dict(channel_id="CHAN", routing_text="hello", bot_uid="BOT", is_mentioned=False,
                     is_thread_reply=True, event_thread_ts="1.0", user_id="ALICE", team_id="TEAM",
                     is_dm=False, force_process=False)
    arguments.update(changes)
    return await adapter._channel_gate_allows(**arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("head,allowed", [("<@BOT> design", True), ("<@BOT|Dyno> design", True),
                                        ("normal RFR", False), ("<@BOT_OTHER> design", False)])
async def test_only_actual_root_mention_admits_unmentioned_followups(adapter, head, allowed):
    adapter._client.conversations_replies.return_value = {"messages": [{"ts": "1.0", "text": head}]}
    adapter._register_mentioned_thread("1.0", team_id="TEAM")
    adapter._has_active_session_for_thread = MagicMock(return_value=True)
    adapter._bot_authored_thread_root = AsyncMock(return_value=True)
    assert await admit(adapter) is allowed
    assert await admit(adapter, user_id="BOB") is allowed
    adapter._client.conversations_replies.assert_awaited_once_with(
        channel="CHAN", ts="1.0", limit=1, inclusive=True)
    adapter._has_active_session_for_thread.assert_not_called()
    adapter._bot_authored_thread_root.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeated_explicit_mentions_do_not_enroll_a_normal_thread(adapter):
    adapter._client.conversations_replies.return_value = {"messages": [{"ts": "1.0", "text": "RFR"}]}
    for _ in range(2):
        assert await admit(adapter, routing_text="<@BOT> reviewers", is_mentioned=True)
        adapter._apply_bot_mention("<@BOT> reviewers", "<@BOT> reviewers", "", False, "BOT", "1.0", "TEAM")
        assert not await admit(adapter)
    assert not adapter._mentioned_threads


@pytest.mark.asyncio
async def test_root_failure_strict_mention_and_channel_boundaries_stay_quiet(adapter):
    adapter._client.conversations_replies.side_effect = RuntimeError("unavailable")
    assert not await admit(adapter)
    adapter._client.conversations_replies.reset_mock()
    adapter.config.extra["strict_mention"] = True
    assert not await admit(adapter)
    adapter._client.conversations_replies.assert_not_awaited()
    adapter.config.extra["allowed_channels"] = ["ELSEWHERE"]
    assert not await admit(adapter, is_mentioned=True, routing_text="<@BOT> hello")


@pytest.mark.asyncio
async def test_legacy_default_keeps_existing_active_thread_wakeups(adapter):
    adapter.config.extra.pop("thread_followup_mode")
    adapter._has_active_session_for_thread = MagicMock(return_value=True)
    assert await admit(adapter)
    adapter._client.conversations_replies.assert_not_awaited()


@pytest.mark.asyncio
async def test_root_mode_restricts_free_channel_threads_and_scopes_cache(adapter):
    adapter.config.extra["free_response_channels"] = ["CHAN"]
    adapter._client.conversations_replies.side_effect = [
        {"messages": [{"ts": "1.0", "text": "<@BOT> design"}]},
        {"messages": [{"ts": "1.0", "text": "ordinary thread"}]},
    ]
    assert await admit(adapter)
    adapter._team_bot_user_ids["OTHERTEAM"] = "BOT"
    assert not await admit(adapter, team_id="OTHERTEAM")
    assert adapter._client.conversations_replies.await_count == 2

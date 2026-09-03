"""Regression tests for #63530 — Slack adapter drops human replies in
threads whose root was posted by the bot via direct chat.postMessage
(outside the gateway's send() path).

Background: the adapter's wake-decision at the un-mentioned branch in
_handle_slack_message uses three checks:

  1. thread_ts ∈ _bot_message_ts          (only populated by send() / files_upload_v2)
  2. thread_ts ∈ _mentioned_threads        (only populated on @mention)
  3. _has_active_session_for_thread(...)   (survives restarts)

When a skill posts a triage message into a Slack thread via the Web API
directly (chat.postMessage, no gateway run), the bot's own ts is NOT
recorded in _bot_message_ts. A human reply in that thread, without an
@-mention and without an existing session, falls through all three
checks and is silently dropped. The same gap opens after a gateway
restart: _bot_message_ts is process memory, so threads the bot started
before the restart no longer wake it.

Fix: a 4th check — was the thread root authored by the bot? Root
authorship is derived from the Slack API (conversations.replies), so it
survives restarts, unlike the in-memory ts set. The wake decision is
extracted into _should_wake_on_unmentioned_message so it's directly
testable without spinning up Slack.
"""

import logging
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


# Mock slack-bolt / slack-sdk the same way test_slack_mention.py does.
def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock
    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock
    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        (
            "slack_bolt.adapter.socket_mode.async_handler",
            slack_bolt.adapter.socket_mode.async_handler,
        ),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import (  # noqa: E402
    SlackAdapter,
    _ThreadContextCache,
)

from gateway.config import Platform, PlatformConfig  # noqa: E402


BOT_USER_ID = "U_BOT_OWN"
CHANNEL_ID = "C_incident"
USER_ID = "U_engineer"
THREAD_TS = "1700000000.000100"


def _make_adapter(bot_authored_root: bool = False):
    """Build a bare SlackAdapter with the wake-decision state controlled.

    None of the 3 legacy in-memory checks pass by default: the bot didn't
    send via gateway, the thread wasn't @-mentioned, and there is no active
    session — exactly the post-restart / outside-send state.
    """
    adapter = object.__new__(SlackAdapter)
    adapter.platform = Platform.SLACK
    adapter.config = PlatformConfig(
        enabled=True,
        extra={"require_mention": True, "strict_mention": False},
    )
    adapter._bot_user_id = BOT_USER_ID
    adapter._team_bot_user_ids = {}
    adapter._bot_message_ts = set()
    adapter._mentioned_threads = set()
    adapter._MENTIONED_THREADS_MAX = 5000
    adapter._THREAD_CACHE_TTL = 300
    adapter._thread_context_cache = {}
    if bot_authored_root:
        adapter._thread_context_cache[f"{CHANNEL_ID}:{THREAD_TS}:"] = (
            _ThreadContextCache(
                content="ctx",
                parent_user_id=BOT_USER_ID,
                parent_text="bot-posted root",
                messages=[
                    {
                        "ts": THREAD_TS,
                        "user": BOT_USER_ID,
                        "text": "bot-posted root",
                        "bot_id": "B_SELF",
                    }
                ],
            )
        )

    adapter._has_active_session_for_thread = lambda **kw: False
    # Mock _fetch_thread_context so the miss-path doesn't make a real
    # Slack API call. Tests that need a populated cache pre-populate
    # _thread_context_cache directly.
    adapter._fetch_thread_context = AsyncMock(return_value="")
    adapter._resolve_user_is_bot = AsyncMock(return_value=False)
    adapter._slack_strict_mention = lambda: False

    return adapter


def _cache_root(adapter, *, user_id: str, bot_id: str = "", text: str = "root"):
    adapter._thread_context_cache[f"{CHANNEL_ID}:{THREAD_TS}:"] = _ThreadContextCache(
        content="ctx",
        parent_user_id=user_id,
        parent_text=text,
        messages=[
            {
                "ts": THREAD_TS,
                "user": user_id,
                "text": text,
                "bot_id": bot_id,
                "subtype": "bot_message" if bot_id else None,
            }
        ],
    )


# ---------------------------------------------------------------------------
# _should_wake_on_unmentioned_message — composes all 4 checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wake_decision_returns_false_when_not_thread_reply():
    """A top-level channel message (no thread_ts) should never wake the bot
    when require_mention is true — unchanged by this fix."""
    adapter = _make_adapter(bot_authored_root=True)
    wake = await adapter._should_wake_on_unmentioned_message(
        event_thread_ts=None,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=False,
    )
    assert wake is False


@pytest.mark.asyncio
async def test_wake_decision_returns_true_when_bot_authored_thread_root():
    """The new behavior (#63530): a human reply in a thread whose root was
    authored by the bot via direct chat.postMessage (outside gateway send)
    wakes the bot even though none of the legacy 3 checks pass — including
    after a restart, when _bot_message_ts is empty."""
    adapter = _make_adapter(bot_authored_root=True)
    wake = await adapter._should_wake_on_unmentioned_message(
        event_thread_ts=THREAD_TS,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=True,
    )
    assert wake is True, (
        "human reply in a thread whose root was bot-posted (not via gateway "
        "send) should wake the bot — #63530"
    )


@pytest.mark.asyncio
async def test_foreign_bot_root_rejects_stale_local_send_marker():
    """Peer-root metadata wins over contradictory process-local state."""
    adapter = _make_adapter()
    _cache_root(adapter, user_id="U_PEER_BOT", bot_id="B_PEER")
    adapter._bot_message_ts = {THREAD_TS}

    decision = await adapter._decide_unmentioned_message_wake(
        event_thread_ts=THREAD_TS,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=True,
    )

    assert decision.wake is False
    assert decision.reason == "foreign_bot_root_without_ownership"
    assert decision.root_owner == "foreign_bot"


@pytest.mark.asyncio
async def test_foreign_bot_root_rejects_other_workspace_bare_mention_marker():
    adapter = _make_adapter()
    adapter._team_bot_user_ids = {"T_CURRENT": BOT_USER_ID}
    adapter._thread_context_cache[
        f"{CHANNEL_ID}:{THREAD_TS}:T_CURRENT"
    ] = _ThreadContextCache(
        content="ctx",
        parent_user_id="U_PEER_BOT",
        parent_text="peer root",
        messages=[
            {
                "ts": THREAD_TS,
                "user": "U_PEER_BOT",
                "text": "peer root",
                "bot_id": "B_PEER",
                "subtype": "bot_message",
            }
        ],
    )
    adapter._mentioned_threads = {THREAD_TS}

    decision = await adapter._decide_unmentioned_message_wake(
        event_thread_ts=THREAD_TS,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=True,
        team_id="T_CURRENT",
    )

    assert decision.wake is False
    assert decision.reason == "foreign_bot_root_without_ownership"


@pytest.mark.asyncio
async def test_foreign_bot_root_rejects_after_cold_cache_fetch():
    """A cold process classifies the Slack API root before admitting."""
    adapter = _make_adapter()

    async def _fake_fetch(channel_id, thread_ts, current_ts, team_id=""):
        _cache_root(adapter, user_id="U_PEER_BOT", bot_id="B_PEER")
        return "ctx"

    adapter._fetch_thread_context = AsyncMock(side_effect=_fake_fetch)

    decision = await adapter._decide_unmentioned_message_wake(
        event_thread_ts=THREAD_TS,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=True,
    )

    assert decision.reason == "foreign_bot_root_without_ownership"
    adapter._fetch_thread_context.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ownership", "reason"),
    [
        ("mention", "prior_bot_specific_mention"),
        ("session", "owned_active_session"),
    ],
)
async def test_foreign_bot_root_allows_current_bot_owned_continuation(
    ownership, reason
):
    adapter = _make_adapter()
    _cache_root(adapter, user_id="U_PEER_BOT", bot_id="B_PEER")
    if ownership == "mention":
        adapter._mentioned_threads = {THREAD_TS}
    else:
        adapter._has_active_session_for_thread = lambda **kw: True

    decision = await adapter._decide_unmentioned_message_wake(
        event_thread_ts=THREAD_TS,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=True,
    )

    assert decision.wake is True
    assert decision.reason == reason


@pytest.mark.asyncio
async def test_foreign_bot_root_mention_survives_cold_process_start():
    """A root that explicitly summons this bot remains a valid handoff."""
    adapter = _make_adapter()
    _cache_root(
        adapter,
        user_id="U_PEER_BOT",
        bot_id="B_PEER",
        text=f"<@{BOT_USER_ID}> please take this",
    )

    decision = await adapter._decide_unmentioned_message_wake(
        event_thread_ts=THREAD_TS,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=True,
    )

    assert decision.wake is True
    assert decision.reason == "root_bot_specific_mention"
    assert THREAD_TS in adapter._mentioned_threads


@pytest.mark.asyncio
async def test_human_root_rejects_contradictory_local_send_marker():
    adapter = _make_adapter()
    _cache_root(adapter, user_id="U_HUMAN")
    adapter._bot_message_ts = {THREAD_TS}

    decision = await adapter._decide_unmentioned_message_wake(
        event_thread_ts=THREAD_TS,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=True,
    )

    assert decision.reason == "local_send_marker_owner_conflict"
    assert decision.wake is False


@pytest.mark.asyncio
async def test_unknown_root_preserves_authoritative_local_send_marker():
    """A transient Slack lookup failure does not orphan a root sent by us."""
    adapter = _make_adapter()
    adapter._bot_message_ts = {THREAD_TS}

    decision = await adapter._decide_unmentioned_message_wake(
        event_thread_ts=THREAD_TS,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=True,
    )

    assert decision.wake is True
    assert decision.reason == "local_self_send_marker"
    assert decision.root_owner == "unknown"


@pytest.mark.asyncio
async def test_safe_reason_coded_diagnostic_omits_message_text(caplog):
    adapter = _make_adapter()
    _cache_root(adapter, user_id="U_PEER_BOT", bot_id="B_PEER")

    with caplog.at_level(logging.DEBUG):
        wake = await adapter._should_wake_on_unmentioned_message(
            event_thread_ts=THREAD_TS,
            channel_id=CHANNEL_ID,
            user_id=USER_ID,
            is_thread_reply=True,
        )

    assert wake is False
    message = next(
        record.getMessage()
        for record in caplog.records
        if "unmentioned_thread_admission" in record.getMessage()
    )
    assert "reason=foreign_bot_root_without_ownership" in message
    assert "root_owner=foreign_bot" in message
    assert "please take this" not in message


# ---------------------------------------------------------------------------
# _bot_authored_thread_root — the API-derived, restart-surviving check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bot_authored_thread_root_fetches_on_cache_miss():
    """Cache miss → _fetch_thread_context runs; a successful fetch that
    populates parent_user_id with the bot's id yields True. This is the
    restart path: fresh process, empty caches, root authorship recovered
    from the Slack API."""
    adapter = _make_adapter()

    async def _fake_fetch(channel_id, thread_ts, current_ts, team_id=""):
        adapter._thread_context_cache[f"{channel_id}:{thread_ts}:{team_id}"] = (
            _ThreadContextCache(
                content="ctx",
                fetched_at=0,
                message_count=1,
                parent_text="bot-posted root",
                parent_user_id=BOT_USER_ID,
            )
        )
        return "ctx"

    adapter._fetch_thread_context = AsyncMock(side_effect=_fake_fetch)

    result = await SlackAdapter._bot_authored_thread_root(
        adapter, CHANNEL_ID, THREAD_TS
    )
    assert result is True
    adapter._fetch_thread_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_bot_authored_thread_root_uses_per_team_bot_id():
    """Multi-workspace: the comparison must use the team's bot user id,
    not the primary workspace's."""
    adapter = _make_adapter()
    adapter._team_bot_user_ids = {"T2": "U_BOT_T2"}
    adapter._thread_context_cache = {
        f"{CHANNEL_ID}:{THREAD_TS}:T2": _ThreadContextCache(
            content="ctx",
            fetched_at=0,
            message_count=1,
            parent_text="root",
            parent_user_id="U_BOT_T2",
        ),
    }

    result = await SlackAdapter._bot_authored_thread_root(
        adapter, CHANNEL_ID, THREAD_TS, team_id="T2"
    )
    assert result is True
    # And the primary bot id must NOT match in that workspace.
    adapter._thread_context_cache[f"{CHANNEL_ID}:{THREAD_TS}:T2"].parent_user_id = (
        BOT_USER_ID
    )
    result = await SlackAdapter._bot_authored_thread_root(
        adapter, CHANNEL_ID, THREAD_TS, team_id="T2"
    )
    assert result is False

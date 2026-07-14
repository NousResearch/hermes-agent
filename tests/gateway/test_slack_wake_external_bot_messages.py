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
checks and is silently dropped.

Fix (this PR): add a 4th check — was the thread root authored by the bot?
This uses the existing _thread_context_cache (already populated when
the bot is later pulled into a thread) plus an on-demand lookup. We
extract the wake-decision into a pure helper so it's directly testable
without spinning up a Slack server.
"""

import asyncio
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
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402
_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402

from gateway.config import Platform, PlatformConfig  # noqa: E402


BOT_USER_ID = "U_BOT_OWN"
CHANNEL_ID = "C_incident"
USER_ID = "U_engineer"
THREAD_TS = "1700000000.100200"
PARENT_TS = "1700000000.000100"  # different — the actual root, posted by the bot via chat.postMessage


def _make_adapter(bot_authored_root: bool = False):
    """Build a SlackAdapter with the wake-decision state controlled for tests.

    Note: we deliberately do NOT mock _bot_authored_thread_root here. The
    helper-test (`test_bot_authored_thread_root_returns_*`) populates the
    _thread_context_cache directly and exercises the real helper; the
    wake-decision tests pass `bot_authored_root=True` to influence the
    AsyncMock that IS set on _bot_authored_thread_root, but only when
    needed. To avoid surprise: tests that rely on the real cache path
    should not rely on the _make_adapter mock of this method.
    """
    adapter = object.__new__(SlackAdapter)
    adapter.platform = Platform.SLACK
    adapter.config = PlatformConfig(enabled=True, extra={
        "require_mention": True,
        "strict_mention": False,
    })
    adapter._bot_user_id = BOT_USER_ID
    adapter._team_bot_user_ids = {}
    # None of the 3 in-memory checks pass: bot didn't send via gateway,
    # thread wasn't @-mentioned, no active session.
    adapter._bot_message_ts = set()
    adapter._mentioned_threads = set()
    adapter._sessions = {}

    # _has_active_session_for_thread returns False (no sessions)
    adapter._has_active_session_for_thread = lambda **kw: False

    # Mock _fetch_thread_context so the miss-path doesn't make a real
    # Slack API call. Returns empty string by default; tests that need a
    # populated cache should pre-populate _thread_context_cache directly
    # (and bypass this mock by clearing it).
    adapter._fetch_thread_context = AsyncMock(return_value="")

    # The 4th-check helper is mocked so wake-decision tests can control
    # its result without setting up the full cache path. Helper-specific
    # tests clear this mock before calling the real helper.
    adapter._bot_authored_thread_root = AsyncMock(return_value=bot_authored_root)

    adapter._slack_require_mention = lambda: True
    adapter._slack_strict_mention = lambda: False
    adapter._slack_free_response_channels = lambda: set()
    adapter._slack_allowed_channels = lambda: None

    return adapter


# ---------------------------------------------------------------------------
# Tests for the new pure helper _should_wake_on_unmentioned_message
# (async — composes all 4 checks including the new bot-authored-root one)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wake_decision_returns_false_when_not_thread_reply():
    """A top-level channel message (no thread_ts) should never wake the bot
    when require_mention is true — unchanged by this PR."""
    adapter = _make_adapter()
    wake = await adapter._should_wake_on_unmentioned_message(
        event_thread_ts=None,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=False,
    )
    assert wake is False


@pytest.mark.asyncio
async def test_wake_decision_returns_false_when_all_four_checks_miss():
    """All four checks miss (no bot-message, no mention, no session, no
    bot-authored root) → wake decision is False."""
    adapter = _make_adapter(bot_authored_root=False)
    wake = await adapter._should_wake_on_unmentioned_message(
        event_thread_ts=THREAD_TS,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=True,
    )
    assert wake is False


@pytest.mark.asyncio
async def test_wake_decision_returns_true_when_bot_authored_thread_root():
    """The new behavior (#63530): a human reply in a thread whose root was
    authored by the bot via direct chat.postMessage (outside gateway send)
    should wake the bot, even though none of the legacy 3 checks pass.
    This test relies on the AsyncMock set in _make_adapter(bot_authored_root=True)."""
    adapter = _make_adapter(bot_authored_root=True)
    wake = await adapter._should_wake_on_unmentioned_message(
        event_thread_ts=THREAD_TS,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=True,
    )
    assert wake is True, (
        "human reply in a thread whose root was bot-posted (not via gateway send) "
        "should wake the bot — #63530"
    )


@pytest.mark.asyncio
async def test_wake_decision_returns_true_when_legacy_check_1_hits():
    """Regression guard: when _bot_message_ts contains the thread_ts
    (the legacy 'reply to bot_thread' path), the wake decision must still
    be True. The new check is additive, not replacing."""
    adapter = _make_adapter(bot_authored_root=False)
    adapter._bot_message_ts = {THREAD_TS}
    wake = await adapter._should_wake_on_unmentioned_message(
        event_thread_ts=THREAD_TS,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=True,
    )
    assert wake is True


@pytest.mark.asyncio
async def test_wake_decision_returns_true_when_legacy_check_2_hits():
    """Regression guard: when _mentioned_threads contains the thread_ts,
    wake. The new check is additive."""
    adapter = _make_adapter(bot_authored_root=False)
    adapter._mentioned_threads = {THREAD_TS}
    wake = await adapter._should_wake_on_unmentioned_message(
        event_thread_ts=THREAD_TS,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=True,
    )
    assert wake is True


@pytest.mark.asyncio
async def test_wake_decision_returns_true_when_legacy_check_3_hits():
    """Regression guard: when there's an active session for the thread,
    wake. The new check is additive."""
    adapter = _make_adapter(bot_authored_root=False)
    adapter._has_active_session_for_thread = lambda **kw: True
    wake = await adapter._should_wake_on_unmentioned_message(
        event_thread_ts=THREAD_TS,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=True,
    )
    assert wake is True


# ---------------------------------------------------------------------------
# Test for the new _bot_authored_thread_root helper
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bot_authored_thread_root_returns_true_for_bot_authored_root():
    """The new helper calls _fetch_thread_context (or its cache) and returns
    True when the thread root's user_id matches the bot's user_id."""
    from plugins.platforms.slack.adapter import _ThreadContextCache

    adapter = _make_adapter()
    # _make_adapter mocks _bot_authored_thread_root with AsyncMock;
    # clear it so we exercise the real helper.
    adapter._bot_authored_thread_root = lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("real helper should have run; mock was not cleared")
    )

    # Pre-populate the cache with a bot-authored root.
    cache_key = f"{CHANNEL_ID}:{THREAD_TS}:"
    adapter._thread_context_cache = {
        cache_key: _ThreadContextCache(
            content="[Thread context — prior messages...]",
            fetched_at=0,
            message_count=1,
            parent_text="triage analysis",
            parent_user_id=BOT_USER_ID,  # ← the new field added by this PR
        ),
    }

    # Call the real helper directly.
    from plugins.platforms.slack.adapter import SlackAdapter
    result = await SlackAdapter._bot_authored_thread_root(adapter, CHANNEL_ID, THREAD_TS)
    assert result is True


@pytest.mark.asyncio
async def test_bot_authored_thread_root_returns_false_for_human_authored_root():
    """If a human posted the thread root, the helper must return False
    even on a cache hit. This guards against false positives where the
    helper wakes the bot on any thread reply with cache warmth."""
    from plugins.platforms.slack.adapter import _ThreadContextCache

    adapter = _make_adapter()
    adapter._bot_authored_thread_root = lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("real helper should have run; mock was not cleared")
    )

    cache_key = f"{CHANNEL_ID}:{THREAD_TS}:"
    adapter._thread_context_cache = {
        cache_key: _ThreadContextCache(
            content="[Thread context — prior messages...]",
            fetched_at=0,
            message_count=1,
            parent_text="someone else's message",
            parent_user_id="U_other_user",
        ),
    }

    from plugins.platforms.slack.adapter import SlackAdapter
    result = await SlackAdapter._bot_authored_thread_root(adapter, CHANNEL_ID, THREAD_TS)
    assert result is False


@pytest.mark.asyncio
async def test_bot_authored_thread_root_returns_false_on_empty_thread_ts():
    """Defensive: empty thread_ts should short-circuit to False without
    any cache lookup or network call."""
    adapter = _make_adapter()
    result = await adapter._bot_authored_thread_root(CHANNEL_ID, "")
    assert result is False

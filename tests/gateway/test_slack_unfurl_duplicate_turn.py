"""Regression: a link unfurl must not become a second user turn.

Reproduction (Snow/adminbot, 2026-08-22 02:23:30-31Z, channel C0BF1EYUA9H):

    02:23:30.718  listener=message      ts=...409.908499  dedup_hit=False
    02:23:30.737  listener=app_mention  ts=...409.908499  dedup_hit=True
    02:23:31.675  listener=message      ts=...411.012100  dedup_hit=False
                  subtype=message_changed          <-- LEAKED

The user posted ONE message containing two Slack permalinks. While the first
copy was still resolving those permalinks (several awaits against the Slack
API), Slack's link unfurl fired `message_changed` for the same message with a
*different* event ts. That ts legitimately misses `_dedup`, so the
`_processed_message_ts` guard is the only thing standing between it and a
duplicate turn — but that guard was only populated at the very END of
`_handle_slack_message`, i.e. after the in-flight work had finished.

Result: spurious "Interrupting current task" plus the same answer twice.

The fix claims the message ts immediately after the dedup check. These tests
pin the behaviour at that seam.
"""

import asyncio
import importlib
import sys
import time
from importlib.machinery import PathFinder
from types import ModuleType
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig


def _load_installed_package(name):
    if PathFinder.find_spec(name) is None:
        return None
    prefix = f"{name}."
    displaced = {
        m: sys.modules.pop(m)
        for m in tuple(sys.modules)
        if (m == name or m.startswith(prefix)) and not isinstance(sys.modules[m], ModuleType)
    }
    try:
        return importlib.import_module(name)
    except ImportError:
        sys.modules.update(displaced)
        return None


_load_installed_package("slack_bolt")
_load_installed_package("slack_sdk")

_slack_mod = importlib.import_module("plugins.platforms.slack.adapter")
SlackAdapter = _slack_mod.SlackAdapter

CHANNEL = "C0BF1EYUA9H"
TEAM = "T025KND0E"
ORIGINAL_TS = "1787365409.908499"
UNFURL_EVENT_TS = "1787365411.012100"
USER = "U0374GH838U"


def _make_adapter(delivered):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._bot_user_id = "U0BCLP7DB7B"

    async def _capture(event):
        delivered.append(event)

    adapter.handle_message = _capture
    return adapter


def _original_event():
    return {
        "type": "message",
        "user": USER,
        "text": "<@U0BCLP7DB7B> see https://example.com/a and https://example.com/b",
        "channel": CHANNEL,
        "channel_type": "channel",
        "ts": ORIGINAL_TS,
        "team": TEAM,
        "client_msg_id": "cmid-1",
    }


def _unfurl_event():
    """Slack's link-unfurl edit: same message, different event ts."""
    return {
        "type": "message",
        "subtype": "message_changed",
        "channel": CHANNEL,
        "channel_type": "channel",
        "team": TEAM,
        "ts": UNFURL_EVENT_TS,
        "event_ts": UNFURL_EVENT_TS,
        "message": {
            "type": "message",
            "user": USER,
            "text": "<@U0BCLP7DB7B> see https://example.com/a and https://example.com/b",
            "ts": ORIGINAL_TS,
            "client_msg_id": "cmid-1",
            "attachments": [{"title": "unfurled"}],
        },
    }


def _body():
    return {"team_id": TEAM, "event_id": "Ev0BRUTU4GP7"}


class TestUnfurlDuringInflightMessage:
    def test_unfurl_arriving_mid_flight_does_not_create_second_turn(self):
        """THE regression: unfurl lands while the original is still awaiting."""
        delivered = []
        adapter = _make_adapter(delivered)

        release = asyncio.Event()
        entered = asyncio.Event()
        calls = {"n": 0}

        async def _slow_user_name_resolve(*a, **k):
            # Stand in for the real in-flight work. In the production
            # reproduction the first copy was resolving TWO Slack permalinks
            # (~950ms) when the unfurl arrived. Any await AFTER the point
            # where the handler has committed to delivering the event
            # reproduces the same window; `_resolve_user_name` is such an
            # await and, unlike permalink resolution, it is reached for this
            # event shape on a plain channel message.
            #
            # Only the FIRST copy parks. Otherwise a leaked second copy would
            # block here too and the test would deadlock instead of reporting
            # the duplicate.
            calls["n"] += 1
            if calls["n"] == 1:
                entered.set()
                await release.wait()
            return "richard"

        adapter._resolve_user_name = _slow_user_name_resolve

        async def scenario():
            first = asyncio.create_task(
                adapter._handle_slack_message(_original_event(), _body())
            )
            # The test is only meaningful if the first copy really is parked
            # mid-handler when the unfurl lands.
            await asyncio.wait_for(entered.wait(), timeout=2.0)

            await asyncio.wait_for(
                adapter._handle_slack_message(_unfurl_event(), _body()),
                timeout=5.0,
            )

            release.set()
            await asyncio.wait_for(first, timeout=5.0)

        asyncio.run(scenario())

        assert len(delivered) == 1, (
            f"unfurl became a duplicate user turn ({len(delivered)} deliveries); "
            "this is the spurious-interrupt bug"
        )

    def test_unfurl_after_completion_still_suppressed(self):
        """Sequential case must keep working (the pre-existing guarantee)."""
        delivered = []
        adapter = _make_adapter(delivered)
        adapter._resolve_user_name = AsyncMock(return_value="richard")

        async def scenario():
            await adapter._handle_slack_message(_original_event(), _body())
            await adapter._handle_slack_message(_unfurl_event(), _body())

        asyncio.run(scenario())
        assert len(delivered) == 1


class TestClaimDoesNotSwallowRealMessages:
    def test_a_different_message_is_unaffected(self):
        """Claiming one ts must not suppress an unrelated message."""
        delivered = []
        adapter = _make_adapter(delivered)
        adapter._resolve_user_name = AsyncMock(return_value="richard")

        other = _original_event()
        other["ts"] = "1787365500.000100"
        other["client_msg_id"] = "cmid-2"

        async def scenario():
            await adapter._handle_slack_message(_original_event(), _body())
            await adapter._handle_slack_message(other, _body())

        asyncio.run(scenario())
        assert len(delivered) == 2

    def test_a_genuine_user_edit_is_still_suppressed(self):
        """A real edit of an already-answered message must not re-trigger.

        Same contract as before the fix — re-answering an edited message the
        bot already replied to was never wanted.
        """
        delivered = []
        adapter = _make_adapter(delivered)
        adapter._resolve_user_name = AsyncMock(return_value="richard")

        edit = _unfurl_event()
        edit["message"]["text"] = "<@U0BCLP7DB7B> edited text"
        edit["message"]["edited"] = {"user": USER, "ts": "1787365412.000000"}

        async def scenario():
            await adapter._handle_slack_message(_original_event(), _body())
            await adapter._handle_slack_message(edit, _body())

        asyncio.run(scenario())
        assert len(delivered) == 1


class TestClaimHelperBounded:
    def test_claim_map_is_evicted_oldest_first(self):
        delivered = []
        adapter = _make_adapter(delivered)
        adapter._PROCESSED_MESSAGE_TS_MAX = 10

        for i in range(25):
            adapter._remember_processed_message_ts(f"{i}.0")
            time.sleep(0.001)

        assert len(adapter._processed_message_ts) <= 10
        # newest survive, oldest evicted
        assert "24.0" in adapter._processed_message_ts
        assert "0.0" not in adapter._processed_message_ts

    def test_empty_ts_is_ignored(self):
        delivered = []
        adapter = _make_adapter(delivered)
        adapter._remember_processed_message_ts("")
        assert adapter._processed_message_ts == {}


class TestClaimReleasedOnFailure:
    """A claim made by an invocation that raises must not swallow the turn.

    Trade-off pinned here: the entry-claim closes the unfurl race, but if the
    handler dies mid-enrichment while holding a fresh claim, a Slack retry or
    a user edit must still be able to re-drive the message. The wrapper in
    ``_handle_slack_message`` releases only claims taken by the failed
    invocation itself.
    """

    def test_failed_first_copy_releases_claim_so_edit_can_redrive(self):
        delivered = []
        adapter = _make_adapter(delivered)
        calls = {"n": 0}

        async def _flaky_user_name_resolve(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("slack api died mid-enrichment")
            return "richard"

        adapter._resolve_user_name = _flaky_user_name_resolve

        edit = _unfurl_event()
        edit["message"]["text"] = "<@U0BCLP7DB7B> edited text"
        edit["message"]["edited"] = {"user": USER, "ts": "1787365412.000000"}

        async def scenario():
            with pytest.raises(RuntimeError):
                await adapter._handle_slack_message(_original_event(), _body())
            # The failed invocation must not leave the ts claimed...
            assert ORIGINAL_TS not in adapter._processed_message_ts
            # ...so a user edit of the unanswered message still summons the bot.
            await adapter._handle_slack_message(edit, _body())

        asyncio.run(scenario())
        assert len(delivered) == 1

    def test_failure_does_not_release_a_preexisting_claim(self):
        """Sequential suppression survives a later failing duplicate."""
        delivered = []
        adapter = _make_adapter(delivered)
        adapter._resolve_user_name = AsyncMock(return_value="richard")

        async def scenario():
            await adapter._handle_slack_message(_original_event(), _body())
            assert ORIGINAL_TS in adapter._processed_message_ts
            # A later invocation for the same ts that fails must not strip
            # the claim the successful turn already holds.
            adapter._resolve_user_name = AsyncMock(
                side_effect=RuntimeError("boom")
            )
            second = _original_event()
            second["client_msg_id"] = "cmid-replay"
            try:
                await adapter._handle_slack_message(second, _body())
            except RuntimeError:
                pass
            assert ORIGINAL_TS in adapter._processed_message_ts

        asyncio.run(scenario())
        assert len(delivered) == 1

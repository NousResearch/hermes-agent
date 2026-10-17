"""Regression: a Slack ``message_changed`` must not replay an old message as a new turn (#118349).

Shapes pinned here, all of them reported on the issue:

1. thread-parent reply-metadata update (``reply_count`` / ``latest_reply``) with the
   text unchanged — Slack emits this on every thread reply, and after a gateway
   restart the in-memory ``_processed_message_ts`` cache is empty, so the parent
   was replayed as a fresh user turn (idle: a duplicate answer; busy: a
   "Redirected current run" interrupt);
2. the same metadata update on a parent that carries an old ``edited`` block
   (an edit from the past must not mask the change as a "real edit");
3. a ``message_changed`` with no ``previous_message`` at all (file-state /
   agent-session / unfurl updates arrive that way) for a message that predates
   this gateway process — a message that predates the process cannot be a new
   inbound, so the restart window must not widen;
4. the same shape after the bounded claim map evicted the message's claim.

Genuine user actions must keep routing: a text change against
``previous_message``, and a fresh edit (``edited.ts`` at/after this change) even
without ``previous_message``.
"""

import asyncio
import importlib
import sys
import time
from importlib.machinery import PathFinder
from types import ModuleType
from unittest.mock import AsyncMock

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
USER = "U0374GH838U"
BOT = "U0BCLP7DB7B"
# Old Slack ts (2026-08-20): predates any adapter built during this test run.
PARENT_TS = "1787365409.908499"
CHANGE_EVENT_TS = "1787365411.012100"
PARENT_TEXT = "<@U0BCLP7DB7B> ping"


def _make_adapter(delivered):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._bot_user_id = BOT
    adapter._resolve_user_name = AsyncMock(return_value="richard")

    async def _capture(event):
        delivered.append(event)

    adapter.handle_message = _capture
    return adapter


def _parent_message(**overrides):
    message = {
        "type": "message",
        "user": USER,
        "text": PARENT_TEXT,
        "channel": CHANNEL,
        "channel_type": "channel",
        "ts": PARENT_TS,
        "team": TEAM,
    }
    message.update(overrides)
    return message


def _message_changed(message, *, previous_message=None, event_ts=CHANGE_EVENT_TS,
                     outer_ts=None):
    event = {
        "type": "message",
        "subtype": "message_changed",
        "channel": CHANNEL,
        "channel_type": "channel",
        "team": TEAM,
        "ts": outer_ts if outer_ts is not None else event_ts,
        "event_ts": event_ts,
        "message": message,
    }
    if previous_message is not None:
        event["previous_message"] = previous_message
    return event


def _thread_parent_metadata_event(**message_overrides):
    """The reported event: reply metadata updated, text identical."""
    original = _parent_message()
    updated = _parent_message(**message_overrides)
    updated.update({"reply_count": 1, "latest_reply": "1787365411.012100"})
    return _message_changed(updated, previous_message=original)


def _body():
    return {"team_id": TEAM, "event_id": "Ev0BRUTU4GP7"}


def _run(adapter, *events):
    async def scenario():
        for event in events:
            await adapter._handle_slack_message(event, _body())
    asyncio.run(scenario())


class TestMetadataOnlyChangesNeverRoute:
    """Slack generates these on its own; they are never a person speaking."""

    def test_parent_reply_metadata_update_with_empty_cache_is_ignored(self):
        """THE report: restart empties the claim map, the next thread reply
        re-delivers the parent as a new user turn."""
        delivered = []
        adapter = _make_adapter(delivered)
        assert adapter._processed_message_ts == {}  # post-restart state

        _run(adapter, _thread_parent_metadata_event())

        assert delivered == [], "thread-parent metadata update became a user turn"

    def test_parent_update_with_old_edited_block_is_ignored(self):
        """A past edit riding along must not mask a metadata-only change."""
        delivered = []
        adapter = _make_adapter(delivered)
        event = _thread_parent_metadata_event(
            edited={"user": USER, "ts": "1787365410.500000"})

        _run(adapter, event)

        assert delivered == [], "old edited block let the metadata update through"

    def test_change_without_previous_message_predating_process_is_ignored(self):
        """No ``previous_message`` (file-state/unfurl/agent-session stamp) for a
        message from before this process — it cannot be a new inbound."""
        delivered = []
        adapter = _make_adapter(delivered)

        _run(adapter, _message_changed(_parent_message()))

        assert delivered == [], "pre-restart message replayed without previous_message"

    def test_evicted_claim_lets_floor_absorb_the_change(self):
        """Bounded map eviction is the same blind spot as a restart."""
        delivered = []
        adapter = _make_adapter(delivered)
        adapter._PROCESSED_MESSAGE_TS_MAX = 3

        base = time.time()  # >= adapter start, like a message posted at runtime
        for i in range(4):
            adapter._remember_processed_message_ts(f"{base + 10 + i:.6f}")
            time.sleep(0.001)
        evicted_ts = f"{base + 10:.6f}"
        assert evicted_ts not in adapter._processed_message_ts

        event = _message_changed(
            _parent_message(ts=evicted_ts),
            event_ts=f"{base + 50:.6f}",
        )
        _run(adapter, event)

        assert delivered == [], "evicted claim replayed its message as a turn"


class TestGenuineUserActionsStillRoute:
    def test_text_change_is_still_processed(self):
        """A real text edit of an old, unclaimed message must route."""
        delivered = []
        adapter = _make_adapter(delivered)
        updated = _parent_message(text="<@U0BCLP7DB7B> corrected question")
        event = _message_changed(updated, previous_message=_parent_message())

        _run(adapter, event)

        assert len(delivered) == 1

    def test_fresh_edit_without_previous_message_is_still_processed(self):
        """``edited.ts`` equal to this change = a user edit; wake once."""
        delivered = []
        adapter = _make_adapter(delivered)
        base = time.time()
        fresh_ts = f"{base + 1:.6f}"
        message = _parent_message(
            ts=fresh_ts, text="<@U0BCLP7DB7B> edited mention",
            edited={"user": USER, "ts": f"{base + 2:.6f}"})
        event = _message_changed(message, event_ts=f"{base + 2:.6f}")

        _run(adapter, event)

        assert len(delivered) == 1

    def test_change_to_message_from_after_start_without_previous_message_routes(self):
        """Shape unknown and the message is from this process's lifetime —
        keep today's behaviour (claim map still decides)."""
        delivered = []
        adapter = _make_adapter(delivered)
        base = time.time()
        fresh_ts = f"{base + 1:.6f}"
        event = _message_changed(
            _parent_message(ts=fresh_ts), event_ts=f"{base + 3:.6f}")

        _run(adapter, event)

        assert len(delivered) == 1

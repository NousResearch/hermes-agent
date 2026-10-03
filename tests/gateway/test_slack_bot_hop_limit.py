"""Slack bot->bot hop limit (local patch 227), re-cut onto the refactored adapter.

Upstream v0.21.1 moved the inline peer-bot policy into ``_peer_bot_drop``, so the
hop check now lives there rather than in ``_handle_slack_message_impl``. These
tests pin behaviour on the helper AND on the drop path, so a future refactor that
moves the code again fails loudly instead of silently dropping the guard.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.slack.adapter import SlackAdapter


def _adapter(**extra):
    a = object.__new__(SlackAdapter)
    a.config = MagicMock()
    a.config.extra = dict(extra)
    return a


def _bot(text="hi <@U1>"):
    return {"bot_id": "B1", "text": text, "ts": "1.0"}


def _human(text="oi"):
    return {"user": "U9", "text": text, "ts": "2.0"}


class TestHopCounting:
    def test_bot_mentions_count_as_hops(self):
        a = _adapter()
        assert a._slack_count_bot_hops([_bot(), _bot()]) == 2

    def test_bot_message_without_mention_is_not_a_hop(self):
        a = _adapter()
        assert a._slack_count_bot_hops([_bot("just talking")]) == 0

    def test_human_message_resets_the_budget(self):
        a = _adapter()
        assert a._slack_count_bot_hops([_bot(), _bot(), _human(), _bot()]) == 1

    def test_limit_defaults_to_one(self, monkeypatch):
        monkeypatch.delenv("SLACK_BOT_HOP_LIMIT", raising=False)
        assert _adapter()._slack_bot_hop_limit() == 1

    def test_garbage_limit_falls_back_to_one(self):
        assert _adapter(bot_hop_limit="banana")._slack_bot_hop_limit() == 1

    def test_negative_limit_clamps_to_zero(self):
        assert _adapter(bot_hop_limit="-5")._slack_bot_hop_limit() == 0


class TestHopExhaustion:
    def _with_replies(self, messages, **extra):
        a = _adapter(**extra)
        client = MagicMock()
        client.conversations_replies = AsyncMock(return_value={"messages": messages})
        a._get_client = MagicMock(return_value=client)
        return a

    def test_first_hop_is_allowed(self):
        a = self._with_replies([_human()])
        assert asyncio.run(a._slack_bot_hop_exhausted(channel_id="C", thread_ts="1")) is False

    def test_second_hop_is_blocked(self):
        a = self._with_replies([_bot()])
        assert asyncio.run(a._slack_bot_hop_exhausted(channel_id="C", thread_ts="1")) is True

    def test_incoming_message_does_not_count_itself(self):
        """An unthreaded bot message falls back to its own ts; counting it blocked hop one."""
        a = self._with_replies([{"bot_id": "B1", "text": "hi <@U1>", "ts": "7.7"}])
        assert asyncio.run(a._slack_bot_hop_exhausted(
            channel_id="C", thread_ts="7.7", incoming_ts="7.7")) is False

    def test_limit_zero_fails_closed_without_an_api_call(self):
        a = _adapter(bot_hop_limit="0")
        a._get_client = MagicMock(side_effect=AssertionError("must not call Slack"))
        assert asyncio.run(a._slack_bot_hop_exhausted(channel_id="C", thread_ts="1")) is True

    def test_api_error_fails_closed(self):
        """A thread whose history cannot be counted is the loop this guard stops.

        Humans never reach this gate, so a Slack outage cannot silence Sam.
        """
        a = _adapter()
        client = MagicMock()
        client.conversations_replies = AsyncMock(side_effect=RuntimeError("slack down"))
        a._get_client = MagicMock(return_value=client)
        assert asyncio.run(a._slack_bot_hop_exhausted(channel_id="C", thread_ts="1")) is True


class TestPeerBotDropWiring:
    """The guard must be reachable from the real inbound path, not just present."""

    def _drop_adapter(self, replies, **extra):
        a = _adapter(allow_bots="mentions", **extra)
        a._event_declares_bot_sender = MagicMock(return_value=True)
        a._resolve_user_is_bot = AsyncMock(return_value=True)
        client = MagicMock()
        client.conversations_replies = AsyncMock(return_value={"messages": replies})
        a._get_client = MagicMock(return_value=client)
        return a

    def _call(self, a, **kw):
        return asyncio.run(a._peer_bot_drop(
            {"bot_id": "B1", "text": "<@U1> ping"}, "U2", "U1", "C", "T", True, **kw))

    def test_peer_bot_drop_accepts_thread_context(self):
        import inspect
        params = inspect.signature(SlackAdapter._peer_bot_drop).parameters
        assert "thread_ts" in params and "incoming_ts" in params

    def test_drop_path_consults_the_hop_limit(self):
        a = self._drop_adapter([_bot()])
        assert self._call(a, thread_ts="1.0", incoming_ts="3.0") is True

    def test_drop_path_allows_the_first_hop(self):
        a = self._drop_adapter([_human()])
        assert self._call(a, thread_ts="1.0", incoming_ts="3.0") is False

    def test_allow_bots_none_still_drops_before_any_hop_lookup(self):
        a = self._drop_adapter([_human()])
        a.config.extra["allow_bots"] = "none"
        a._get_client = MagicMock(side_effect=AssertionError("must not call Slack"))
        assert self._call(a, thread_ts="1.0") is True

    def test_inbound_handler_passes_thread_and_incoming_ts(self):
        """Source-level: the call site must forward the thread context."""
        import inspect
        src = inspect.getsource(SlackAdapter._handle_slack_message_impl)
        assert "_peer_bot_drop(" in src
        assert "thread_ts=event_thread_ts or ts" in src
        assert "incoming_ts=ts" in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

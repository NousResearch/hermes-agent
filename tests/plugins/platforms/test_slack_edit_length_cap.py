"""Slack ``chat.update`` caps ``text`` at 4,000 chars; ``chat.postMessage`` allows ~40,000.

The adapter used one constant (``MAX_MESSAGE_LENGTH`` = 39000) for both, so every reply between
4,000 and 39,000 chars made an edit that Slack rejected outright with ``msg_too_long``: 33 such
failures across nine days in production (five of them on 2026-09-14, the day it was found). An
edit cannot be split, so above the cap the adapter must DECLINE the edit (without burning a Slack
call) and let the caller re-deliver the whole reply through the chunked send path.

Written without ``pytest.mark.asyncio`` on purpose: this checkout has no ``pytest-asyncio``.
"""
import asyncio

from plugins.platforms.slack.adapter import SlackAdapter


class _RecordingAdapter(SlackAdapter):
    """Minimal stand-in that records Slack calls instead of making them."""

    def __init__(self):  # noqa: D107 - deliberately skips real construction
        self.calls = []

    def _outbound_blocked(self, chat_id, what):
        return None

    def format_message(self, content):
        return content

    def truncate_message(self, text, limit):
        return [text[:limit]]

    def _client_for(self, chat_id, metadata):
        return object()

    def _maybe_blocks(self, content):
        return None

    async def _clear_thread_status_quietly(self, chat_id, metadata):
        return None

    async def _call_with_block_fallback(self, client_factory, method, kwargs, kind):
        self.calls.append((method, len(kwargs.get("text", ""))))
        return {"ok": True}


def test_oversized_edit_is_declined_without_calling_slack():
    """4,342 chars: a production reply size (logged 2026-09-14), above Slack's edit cap."""
    adapter = _RecordingAdapter()
    result = asyncio.run(
        adapter.edit_message("C123", "1.0", "x" * 4342, finalize=True))
    assert result.success is False
    assert adapter.calls == []          # no wasted chat.update
    assert "chat.update" in (result.error or "")


def test_oversized_edit_is_not_an_egress_decline():
    """The caller must read it as 'editing unavailable' and fall back to a send."""
    from gateway.relay.egress import declined_send

    adapter = _RecordingAdapter()
    result = asyncio.run(
        adapter.edit_message("C123", "1.0", "x" * 4342, finalize=True))
    assert declined_send(result) is False


def test_edit_at_or_below_cap_still_updates():
    adapter = _RecordingAdapter()
    for size in (3500, SlackAdapter.MAX_EDIT_LENGTH):
        adapter.calls.clear()
        result = asyncio.run(
            adapter.edit_message("C123", "1.0", "y" * size, finalize=False))
        assert result.success is True, size
        assert [m for m, _ in adapter.calls] == ["chat_update"], size


def test_edit_cap_is_below_slacks_documented_limit():
    assert SlackAdapter.MAX_EDIT_LENGTH < 4000
    assert SlackAdapter.MAX_EDIT_LENGTH < SlackAdapter.MAX_MESSAGE_LENGTH

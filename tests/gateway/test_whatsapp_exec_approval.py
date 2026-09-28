"""WhatsApp (Baileys) exec-approval polls: render, vote, resolve, expire.

The Baileys adapter used to have no native approval rendering, so the gateway's
plain-text ``/approve`` fallback carried the prompt — and a failed or silently
dropped send left the agent wedged for ``approvals.timeout`` with nothing in
the chat (t_f6d13263). The adapter now renders the approval as a native
WhatsApp poll; a vote arrives as a ``poll_update`` bridge event that the
adapter intercepts and routes to ``tools.approval.resolve_gateway_approval``
(the same seam whatsapp_cloud's approval buttons use).

These tests drive the REAL adapter methods with a stubbed bridge send, and the
vote path through the REAL pending-approval queue.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import ExecApprovalPrompt, SendResult
from tools import approval as _approval
from tools.approval_gateway_wait import _ApprovalEntry

SESSION = "agent:main:whatsapp:dm:31629030434"
CHAT = "82631362355453@lid"


def _make_adapter():
    """WhatsAppAdapter with test attributes (bypass __init__, like test_whatsapp_connect)."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter._reply_prefix = None
    adapter._dm_policy = "open"
    adapter._group_policy = "allowlist"
    adapter._allow_from = set()
    adapter._group_allow_from = set()
    adapter._dm_allowlist_source = "config"
    adapter._typing_paused = set()
    adapter._background_tasks = set()
    adapter._dm_policy = "allowlist"
    adapter._allow_from = {"31629030434@s.whatsapp.net"}  # the voter's JID (OWNER_DM)
    adapter._exec_approval_polls = OrderedDict()
    return adapter


def _prompt() -> ExecApprovalPrompt:
    return ExecApprovalPrompt(
        chat_id=CHAT, session_key=SESSION, metadata=None,
        command="himalaya message send 123 --send", description="mail",
        smart_denied=False, text="⚠️ Hermes wants to run a command that needs your OK\n```\nhimalaya message send 123 --send\n```\n",
        actions=[("Allow Once", "once", "primary"), ("Allow Session", "session", ""),
                 ("Always Allow", "always", ""), ("Deny", "deny", "danger")],
    )


def _vote_data(poll_id: str, label: str, sender: str = "31629030434@s.whatsapp.net") -> Dict[str, Any]:
    """A bridge poll_update event shaped exactly like enqueuePollUpdateEvent emits."""
    return {
        "messageId": f"{poll_id}:update:1",
        "chatId": CHAT,
        "senderId": sender,
        "isGroup": False,
        "body": label,
        "hasMedia": False,
        "nativeType": "pollUpdateMessage",
        "nativeMetadata": {"pollUpdate": {"pollId": poll_id, "selectedOptions": [label], "aggregation": {}}},
        "quotedMessageId": poll_id,
        "hasQuotedMessage": True,
    }


class _PollRecorder:
    """Stub send_poll/send capturing payloads; returns canned results in order."""

    def __init__(self, results: List[SendResult]):
        self.results = list(results)
        self.polls: List[Dict[str, Any]] = []
        self.sends: List[str] = []

    async def send_poll(self, chat_id, question, options, *, selectable_count=1):
        self.polls.append({"chat_id": chat_id, "question": question, "options": list(options),
                           "selectable_count": selectable_count})
        return self.results.pop(0)

    async def send(self, chat_id, content, **k):
        self.sends.append(content)
        return SendResult(success=True, message_id="ack-1")


# ── rendering ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_exec_approval_prompt_renders_a_poll_and_registers_state():
    adapter = _make_adapter()
    rec = _PollRecorder([SendResult(success=True, message_id="poll-7")])
    adapter.send_poll = rec.send_poll
    adapter.send = rec.send

    result = await adapter._send_exec_approval_prompt(_prompt())

    assert result.success and result.message_id == "poll-7"
    assert len(rec.polls) == 1
    poll = rec.polls[0]
    assert poll["chat_id"] == CHAT
    assert poll["options"] == ["Allow Once", "Allow Session", "Always Allow", "Deny"]
    assert poll["selectable_count"] == 1
    assert "himalaya" in poll["question"]
    # State maps every option label back to its choice for the vote interception.
    assert adapter._exec_approval_polls["poll-7"] == {
        "session_key": SESSION,
        "chat_id": CHAT,
        "choices": {"Allow Once": "once", "Allow Session": "session",
                    "Always Allow": "always", "Deny": "deny"},
    }


@pytest.mark.asyncio
async def test_send_exec_approval_prompt_retries_once_on_not_connected():
    """The WhatsApp socket reconnects in seconds while the bridge HTTP stays up; the
    tappable poll is worth one bounded retry before the gateway's text fallback."""
    adapter = _make_adapter()
    rec = _PollRecorder([
        SendResult(success=False, error="Not connected to WhatsApp"),
        SendResult(success=True, message_id="poll-9"),
    ])
    adapter.send_poll = rec.send_poll
    adapter.send = rec.send

    real_sleep = asyncio.sleep

    async def _fast_sleep(_s):
        await real_sleep(0)

    import plugins.platforms.whatsapp.adapter as wa_mod
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(wa_mod.asyncio, "sleep", _fast_sleep)
        result = await adapter._send_exec_approval_prompt(_prompt())

    assert result.success and result.message_id == "poll-9"
    assert len(rec.polls) == 2, "exactly one retry"
    assert "poll-9" in adapter._exec_approval_polls


@pytest.mark.asyncio
async def test_send_exec_approval_prompt_does_not_retry_a_delivered_failure():
    """A non-'Not connected' failure (e.g. a 500 with a message id) must not re-send —
    the runner classifies it and falls back to text."""
    adapter = _make_adapter()
    rec = _PollRecorder([SendResult(success=False, error="sendMessage timed out after 60s")])
    adapter.send_poll = rec.send_poll
    adapter.send = rec.send

    result = await adapter._send_exec_approval_prompt(_prompt())

    assert not result.success
    assert len(rec.polls) == 1, "no retry on non-connection failures"
    assert not adapter._exec_approval_polls


# ── vote interception ────────────────────────────────────────────────────────


@pytest.fixture
def pending_entry():
    """A queued approval entry exactly as ``_await_gateway_decision`` registers it."""
    entry = _ApprovalEntry({"command": "himalaya message send 123 --send",
                            "description": "mail", "pattern_key": "k"})
    with _approval._lock:
        _approval._gateway_queues[SESSION] = [entry]
    yield entry
    with _approval._lock:
        _approval._gateway_queues.pop(SESSION, None)


@pytest.mark.asyncio
async def test_vote_resolves_the_blocked_approval_and_is_consumed(pending_entry):
    adapter = _make_adapter()
    rec = _PollRecorder([SendResult(success=True, message_id="poll-7")])
    adapter.send_poll = rec.send_poll
    adapter.send = rec.send
    await adapter._send_exec_approval_prompt(_prompt())
    adapter._typing_paused.add(CHAT)  # the notify path pauses typing for the wait

    event = await adapter._build_message_event(_vote_data("poll-7", "Allow Once"))
    await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)  # the ack is fire-and-forget

    assert event is None, "a consumed vote must not become an agent-visible message"
    assert pending_entry.event.is_set(), "the blocked agent thread must be woken"
    assert pending_entry.result == "once"
    assert CHAT not in adapter._typing_paused, "typing resumes once the wait ends"
    assert rec.sends == ["✅ Approved."]
    assert "poll-7" not in adapter._exec_approval_polls, "resolved polls release their state"


@pytest.mark.asyncio
async def test_deny_vote_maps_to_deny(pending_entry):
    adapter = _make_adapter()
    rec = _PollRecorder([SendResult(success=True, message_id="poll-7")])
    adapter.send_poll = rec.send_poll
    adapter.send = rec.send
    await adapter._send_exec_approval_prompt(_prompt())

    event = await adapter._build_message_event(_vote_data("poll-7", "Deny"))
    await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)

    assert event is None
    assert pending_entry.result == "deny"
    assert rec.sends == ["❌ Denied."]


@pytest.mark.asyncio
async def test_vote_after_timeout_posts_expired_notice_and_claims_the_event():
    """A vote on an already-timed-out prompt must not look like an approval."""
    adapter = _make_adapter()
    rec = _PollRecorder([SendResult(success=True, message_id="poll-7")])
    adapter.send_poll = rec.send_poll
    adapter.send = rec.send
    await adapter._send_exec_approval_prompt(_prompt())
    # No pending entry: the wait already timed out / resolved elsewhere.

    event = await adapter._build_message_event(_vote_data("poll-7", "Allow Once"))
    await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)

    assert event is None, "claimed, so the vote is not re-dispatched as plain text"
    assert any("Approval expired" in s for s in rec.sends)


@pytest.mark.asyncio
async def test_vote_from_unauthorized_sender_is_rejected_and_claimed():
    adapter = _make_adapter()
    adapter._dm_policy = "allowlist"  # sender not in the allowlist
    rec = _PollRecorder([SendResult(success=True, message_id="poll-7")])
    adapter.send_poll = rec.send_poll
    adapter.send = rec.send
    await adapter._send_exec_approval_prompt(_prompt())

    event = await adapter._build_message_event(
        _vote_data("poll-7", "Allow Once", sender="999@s.whatsapp.net"))

    assert event is None, "claimed so the unauthorized vote cannot flow on as text"
    assert rec.sends == [], "no ack to an unauthorized voter"


@pytest.mark.asyncio
async def test_vote_on_untracked_poll_flows_through_untouched():
    """Clarify polls (and foreign polls the bridge surfaces) must keep their normal path."""
    adapter = _make_adapter()
    rec = _PollRecorder([])
    adapter.send_poll = rec.send_poll
    adapter.send = rec.send
    adapter._should_process_message = lambda data: True
    adapter._message_queue = asyncio.Queue()
    adapter._auto_tts_disabled_chats = set()
    adapter._clean_bot_mention_text = lambda text, data: text

    class _Ctx:
        def __init__(self):
            self.events = []
        def __call__(self, data):
            self.events.append(data)
            return None

    event = await adapter._build_message_event(_vote_data("poll-other", "Some choice"))

    assert event is not None, "an untracked poll vote stays a normal message event"
    assert event.text == "Some choice"


@pytest.mark.asyncio
async def test_state_is_lru_capped():
    adapter = _make_adapter()
    adapter._EXEC_APPROVAL_POLL_CACHE = 2
    results = [SendResult(success=True, message_id=f"poll-{i}") for i in range(4)]
    rec = _PollRecorder(results)
    adapter.send_poll = rec.send_poll
    adapter.send = rec.send
    for _ in range(4):
        await adapter._send_exec_approval_prompt(_prompt())

    assert list(adapter._exec_approval_polls) == ["poll-2", "poll-3"]

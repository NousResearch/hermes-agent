"""Regression guard: clarify/approval cards must survive a dead primary bot.

Incident (hub, 2026-08-28): the primary Telegram bot was flood-banned on Sam's
chat (retry_after ~41000s) AND polling was dead on repeated TimeoutError, so
`adapter._bot` was None. Every clarify card returned "Not connected", the
gateway logged "Clarify send failed definitively; clearing registration" four
times, and a real A/B/C decision degraded into closing prose instead of being
surfaced as a card.

Two distinct defects, both guarded here:
  1. send_clarify had NO failover path at all (its sibling send_exec_approval
     got one, but clarify cards do not take that code path).
  2. Both card senders returned early on `not self._bot` BEFORE reaching any
     fallback -- so even the sibling's flood fallback was unreachable in the
     exact outage it was written for.

Run: python3 -m pytest tests/test_telegram_clarify_failover.py -q
"""

import asyncio

import pytest

from plugins.platforms.telegram import adapter as adapter_mod


def _load_adapter_module():
    return adapter_mod


@pytest.fixture()
def adapter():
    """A TelegramAdapter with _bot=None and a stubbed failover bot."""
    mod = _load_adapter_module()

    # `name` is a read-only property on the real class (it derives from
    # `self.platform`), and the logger calls in the paths under test read it.
    # Subclass rather than instantiate a live bot, so the guard exercises the
    # real send_clarify / send_exec_approval / send_slash_confirm code.
    class _Probe(mod.TelegramAdapter):
        @property
        def name(self) -> str:
            return "Telegram"

    obj = _Probe.__new__(_Probe)
    obj._bot = None
    obj._clarify_state = {}
    obj._approval_state = {}
    obj._slash_confirm_state = {}
    obj._fallback_bot_token = "stub-token"
    obj._fallback_topic_names = {}
    obj._failover_dead_threads = set()
    obj._dm_topics = {}
    # Real instance attrs the shared prompt shell reads. __new__ skips __init__,
    # so anything the live path touches must be set here or the probe dies with
    # AttributeError and hides the behaviour under test.
    obj._reply_to_mode = None
    obj.sent = []

    # Stub the transport, not the guard: stub the outermost real seam -- the
    # httpx POST -- so every line of _send_via_fallback_bot actually runs.
    async def _fake_post(payload):
        obj.sent.append({
            "chat_id": payload.get("chat_id"),
            "text": payload.get("text"),
        })
        return {"ok": True, "result": {"message_id": 9001}}

    obj._failover_post_for_test = _fake_post
    obj._metadata_thread_id = lambda md: None
    return obj


def test_clarify_delivers_via_failover_when_bot_is_none(adapter):
    """The core guard: _bot=None must NOT mean the decision is dropped."""
    res = asyncio.run(
        adapter.send_clarify(
            chat_id="1335137548",
            question="Which promotion cadence?",
            choices=["Fri/Mon cadence", "Keep 24h rule"],
            clarify_id="cl-1",
            session_key="sess-1",
        )
    )
    assert res.success is True, "clarify card was dropped when primary bot was None"
    assert res.message_id == "9001"
    assert len(adapter.sent) == 1
    body = adapter.sent[0]["text"]
    assert "Which promotion cadence?" in body
    assert "1. Fri/Mon cadence" in body
    assert "2. Keep 24h rule" in body
    assert "Reply with the number" in body


def test_clarify_registration_survives_the_failover_path(adapter):
    """A delivered fallback card must still be answerable by typed reply."""
    asyncio.run(
        adapter.send_clarify(
            chat_id="1335137548",
            question="Ship it?",
            choices=["Yes", "No"],
            clarify_id="cl-2",
            session_key="sess-2",
        )
    )
    assert adapter._clarify_state.get("cl-2") == "sess-2", (
        "clarify registration was not recorded, so a typed answer cannot resolve it"
    )


def test_clarify_open_ended_has_no_numbered_list(adapter):
    res = asyncio.run(
        adapter.send_clarify(
            chat_id="1335137548",
            question="What should I tell Viktor?",
            choices=[],
            clarify_id="cl-3",
            session_key="sess-3",
        )
    )
    assert res.success is True
    body = adapter.sent[0]["text"]
    assert "Reply with your answer." in body
    assert "1." not in body


def test_clarify_reports_failure_when_failover_also_dead(adapter):
    """Fail closed honestly: no failover token means a real failure, not a lie."""

    async def _dead(payload):
        return {"ok": False, "description": "Forbidden: bot was blocked by the user"}

    adapter._failover_post_for_test = _dead
    res = asyncio.run(
        adapter.send_clarify(
            chat_id="1335137548",
            question="anything",
            choices=["a"],
            clarify_id="cl-4",
            session_key="sess-4",
        )
    )
    assert res.success is False
    assert "cl-4" not in adapter._clarify_state


def test_exec_approval_delivers_via_failover_when_bot_is_none(adapter):
    """The sibling path had the same early-return defect."""
    res = asyncio.run(
        adapter.send_exec_approval(
            chat_id="1335137548",
            command="rm -rf /tmp/thing",
            session_key="sess-5",
            description="dangerous command",
        )
    )
    assert res.success is True, "approval card was dropped when primary bot was None"
    assert res.message_id == "9001"
    assert "rm -rf /tmp/thing" in adapter.sent[0]["text"]
    assert "sess-5" in adapter._approval_state.values()


def test_slash_confirm_delivers_via_failover_when_bot_is_none(adapter):
    """Third card sender in the same class -- also must not drop a decision."""
    res = asyncio.run(
        adapter.send_slash_confirm(
            chat_id="1335137548",
            title="Run /deploy?",
            message="This will deploy production.",
            session_key="sess-6",
            confirm_id="sc-1",
        )
    )
    assert res.success is True, "slash-confirm card was dropped when primary bot was None"
    body = adapter.sent[0]["text"]
    assert "Run /deploy?" in body
    assert "This will deploy production" in body.replace("\\", "")
    assert "approve / always / cancel" in body
    assert adapter._slash_confirm_state.get("sc-1") == "sess-6"


def test_interim_bubble_is_not_relayed_while_flood_window_is_open(adapter):
    """A tool-progress bubble must not spend a reserve-bot send during a ban.

    Real answers have no _interim_send mark and still go through the reserve bot.
    """
    adapter._bot = object()
    adapter._send_path_degraded = False
    adapter._telegram_send_cooldown_until = {"1335137548": 10**12}

    res = asyncio.run(
        adapter.send(
            chat_id="1335137548",
            content="🔧 looking at the logs",
            metadata={"_interim_send": True},
        )
    )
    assert res.success is False
    assert adapter.sent == []

    real = asyncio.run(
        adapter.send(
            chat_id="1335137548",
            content="De hoofdbot is weer vrij.",
            metadata={"thread_id": "449834"},
        )
    )
    assert real.success is True
    assert real.message_id == "9001"
    assert adapter.sent[-1]["text"] == "De hoofdbot is weer vrij."

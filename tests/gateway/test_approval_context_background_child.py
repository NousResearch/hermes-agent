"""Optional approval context on a background child's prompt (the #133514 transport).

A background ``delegate_task`` child asks through the transport its dispatching turn published
(``gateway/run_turn_runner_approval_transport.py``): a presenter that renders through the turn's own
chat context and never binds the parent's canonical owner. The model-provided context must reach that
prompt exactly as it reaches a foreground one (display-only, fitted to the card's budget), must never
become the queued request's or the decision's description, and must never reach the fail-closed
result the lease returns when no chat can show the prompt. A background prompt that was not delivered
fails closed whatever the presenter is attached to.

Each case drives the real chain: ``check_all_command_guards`` inside the child's lease
(``approval_notify_lease.run``), after the parent's notifier was unregistered, into a fake transport.
"""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run  # bootstrap before the per-test home I/O guard
import tools.approval as approval_module
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.relay.egress import EGRESS_DECLINE_CODE
from gateway.run_turn_runner import TurnRunner
from gateway.run_turn_runner_approval_delivery import ApprovalDeliveryError, send_text_approval_prompt
from gateway.run_turn_runner_approval_transport import approval_transport
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from plugins.platforms.discord.adapter import DiscordAdapter
from tools import approval_context, approval_notify_lease
from tools.approval import check_all_command_guards
from tools.approval_context import reset_current_session_key, set_current_session_key

_HEAD = "—— Model-provided context (unverified) ——"
_TAIL = "—— End unverified context ——"
_SESSION = "agent:main:discord:dm:background-child"
_COMMAND = "rm -rf /tmp/background-child-example"
_SHORT = {"purpose": "clear the build cache", "effect": "removes one temp directory", "risk": "a rebuild takes longer"}
_LONG = {"purpose": "p" * 900, "effect": "e" * 900, "risk": "r" * 900}


def _schedule(self, coro, *_args):
    loop = asyncio.new_event_loop()
    try:
        value = loop.run_until_complete(coro)
    finally:
        loop.close()
    fut = Future()
    fut.set_result(value)
    return fut


@pytest.fixture(autouse=True)
def _gateway_session(monkeypatch):
    for key in ("HERMES_INTERACTIVE", "HERMES_EXEC_ASK", "HERMES_YOLO_MODE", "DISCORD_APPROVAL_MENTIONS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(TurnRunner, "_schedule", _schedule)
    token = set_current_session_key(_SESSION)
    yield
    reset_current_session_key(token)
    approval_module._gateway_queues.clear()
    approval_module._gateway_notify_cbs.clear()
    approval_module._session_approved.clear()
    approval_module._pending.pop(_SESSION, None)


def _turn(adapter, platform=Platform.DISCORD):
    source = SessionSource(platform=platform, chat_id="555", chat_type="dm", user_id="111111111111111111")
    return TurnRunner(None, TurnContext(session_key=_SESSION, source=source, _status_adapter_pinned=adapter,
                                        _status_chat_id="555"))


def _child_asks(turn, context):
    """The parent turn registers its notifier, dispatches the child (which takes its lease) and ends;
    only then does the child hit the command gate."""
    approval_module.register_gateway_notify(_SESSION, approval_transport(turn))
    lease = approval_notify_lease.acquire(_SESSION)
    approval_module.unregister_gateway_notify(_SESSION)
    assert approval_module._gateway_notify_cb(_SESSION) is None, "the parent turn ended: its notifier is gone"
    return approval_notify_lease.run(
        lease, lambda: check_all_command_guards(_COMMAND, "local", approval_context=context))


def _discord(cards, queued):
    """A Discord card transport; posting the card snapshots the queued request, then denies it."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._allowed_user_ids = {"111111111111111111"}

    async def channel_send(**kwargs):
        if kwargs.get("view") is None:  # the full-command message ahead of a shortened preview
            return SimpleNamespace(id=1)
        cards.append(kwargs["content"])
        entry = approval_module._gateway_queues[_SESSION][0]
        queued.append(dict(entry.data))
        assert approval_module.resolve_gateway_approval_card(_SESSION, "deny", entry.data["request_id"]) == 1
        return SimpleNamespace(id=1234)

    channel = SimpleNamespace(send=AsyncMock(side_effect=channel_send))
    adapter._client = SimpleNamespace(get_channel=lambda _chat_id: channel, fetch_channel=AsyncMock())
    return adapter


def _plain_card():
    cards, queued = [], []
    _child_asks(_turn(_discord(cards, queued)), None)
    return cards[0]


@pytest.mark.parametrize("context", [_SHORT, _LONG], ids=["short", "long"])
def test_background_child_card_shows_the_context_fitted_to_the_card(context):
    cards, queued = [], []
    adapter = _discord(cards, queued)
    result = _child_asks(_turn(adapter), context)

    assert result["approved"] is False and result["outcome"] == "denied"
    [card] = cards
    # Shown, marked unverified, with both delimiters, inside Discord's own card budget.
    assert _HEAD in card and _TAIL in card
    assert "Purpose: " + context["purpose"][:12] in card
    assert len(card) <= DiscordAdapter._EA_CARD_CAP
    # The card keeps the plain card's command block: context never costs the preview anything.
    plain = _plain_card()
    label = adapter._EA_REASON_LABEL
    assert label in card and card.split(label)[0] == plain.split(label)[0]
    if context is _SHORT:
        assert all(value in card for value in _SHORT.values())
    # Display-only: the queued request keeps the scanner text and carries the context separately;
    # the decision result never sees it.
    assert _HEAD not in queued[0]["description"] and "Purpose:" not in queued[0]["description"]
    assert set(queued[0]["explanation"]) == {"purpose", "effect", "risk"}
    assert _HEAD not in str(result) and context["purpose"][:12] not in str(result)


class _TextAdapter:
    """No native card; ``send`` returns ``result`` and records the prompt."""

    typed_command_prefix = "/"

    def __init__(self, result):
        self.result, self.sent = result, []

    def pause_typing_for_chat(self, chat_id):
        raise AssertionError("a background prompt must not touch the finished turn's typing state")

    async def send(self, chat_id, content, metadata=None, **_kw):
        self.sent.append(content)
        return self.result


def test_background_child_text_prompt_shows_the_context_and_keeps_the_scanner_decision(monkeypatch):
    adapter = _TextAdapter(SendResult(success=True, message_id="m1"))
    asked = []

    def answer_after_send(runner_self, coro, *args):
        fut = _schedule(runner_self, coro, *args)
        asked.extend(dict(e.data) for e in approval_module._gateway_queues.get(_SESSION, []))
        approval_module.resolve_gateway_approval(_SESSION, "deny")
        return fut

    monkeypatch.setattr(TurnRunner, "_schedule", answer_after_send)
    result = _child_asks(_turn(adapter), _SHORT)
    [prompt] = adapter.sent
    assert _HEAD in prompt and _TAIL in prompt and all(value in prompt for value in _SHORT.values())
    assert asked and _HEAD not in asked[0]["description"]
    assert result["approved"] is False and _HEAD not in str(result)


@pytest.mark.parametrize("outcome", ["failed", "declined"])
def test_undelivered_background_text_prompt_fails_closed_even_beside_an_owner(outcome):
    """A foreground turn owned by a session authority only logs a failed chat send (its viewers still
    get the request). A background child's prompt has no other surface: undelivered, it raises."""
    result = (SendResult(success=False, error="chat unavailable") if outcome == "failed" else
              SendResult(success=False, error="declined", raw_response={"success": False, "code": EGRESS_DECLINE_CODE}))
    adapter = _TextAdapter(result)
    runner = _turn(adapter)
    runner._approval_owner = (SimpleNamespace(), "sess-1", 1)
    data = {"command": _COMMAND, "request_id": "r-bg", "explanation": {"purpose": "clean"},
            "description": f"recursive delete\n\n{_HEAD}\nPurpose: clean\n{_TAIL}"}
    with pytest.raises(ApprovalDeliveryError):
        runner._render_approval_sync(dict(data), background=True)
    assert len(adapter.sent) == 1
    flags = {"allow_permanent": True, "allow_session": True, "smart_denied": False}
    if outcome == "failed":  # the same send on the owned foreground path is logged, not raised
        send_text_approval_prompt(runner, dict(data), desc=data["description"], command=_COMMAND, flags=flags)
    else:  # a decline is definitive on every path
        with pytest.raises(ApprovalDeliveryError):
            send_text_approval_prompt(runner, dict(data), desc=data["description"], command=_COMMAND, flags=flags)


def test_stranded_background_child_fails_closed_with_scanner_text_only():
    """An API-server parent publishes no background transport: once it ended, the child's request is
    refused (never parked as pending), and the refusal carries the scanner text, never the context."""
    adapter = _TextAdapter(SendResult(success=True, message_id="never"))
    result = _child_asks(_turn(adapter, Platform.API_SERVER), _SHORT)

    assert result["approved"] is False and result["outcome"] == "notify_failed"
    assert result.get("status") != "pending_approval" and _SESSION not in approval_module._pending
    assert adapter.sent == []
    assert "explanation" not in result
    assert all(value not in str(result) for value in _SHORT.values()) and _HEAD not in str(result)

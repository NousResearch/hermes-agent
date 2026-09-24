"""Regression for #121976: busy-session admission runs once before steering/queueing."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


def _event(user, text="incoming"):
    return MessageEvent(
        text=text,
        message_id=f"msg-{user}",
        source=SessionSource(platform=Platform.WHATSAPP, chat_id="group", chat_type="group", user_id=user),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["queue", "steer", "interrupt"])
async def test_busy_hook_skips_before_auth_or_turn_mutation(monkeypatch, mode):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._busy_input_mode = mode
    runner._draining = False
    runner._admit_bot_message_for_source = lambda source: pytest.fail("bot admission after skip")
    runner._is_user_authorized_for_source = lambda source: pytest.fail("auth after skip")
    runner._queue_or_replace_pending_event = lambda *args: pytest.fail("queued after skip")
    event = _event("other-member")
    seen = []

    def hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            seen.append((kwargs["event"].source.user_id, kwargs["event"]._gateway_busy_followup))
            return [{"action": "skip"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)
    assert await runner._handle_active_session_busy_message(event, "shared-session") is True
    assert seen == [("other-member", True)]


@pytest.mark.asyncio
async def test_busy_rewrite_survives_adapter_fallback_and_is_not_reapplied(monkeypatch):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "queue"
    runner._draining = False
    runner._is_user_authorized_for_source = lambda source: True
    runner._admit_bot_message_for_source = lambda source: True
    event = _event("sender-a")
    seen = []

    def hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            seen.append(kwargs["event"].source.user_id)
            return [{"action": "rewrite", "text": "screened"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)
    # Returning False is the normal queue-text fallback: the adapter keeps `event`.
    assert await runner._handle_active_session_busy_message(event, "session") is False
    assert event.text == "screened"
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_user_authorized_for_source = lambda source: True
    runner._admit_bot_message_for_source = lambda source: True
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._startup_restore_in_progress = False
    runner._intake_adapter_for = lambda source: None
    # The cold admission of the drained event must not execute the hook twice.
    admitted = await runner._hm_admit_event(event)
    assert admitted is not None and admitted[0].text == "screened"
    assert seen == ["sender-a"]


@pytest.mark.asyncio
async def test_adapter_only_pending_is_screened_before_next_turn(monkeypatch):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._draining = False
    runner._promote_queued_event = lambda key, adapter, event: event
    event = _event("sender-b", "unaddressed")
    pending_slot = {"session": event}
    adapter = SimpleNamespace(
        _pending_messages=pending_slot, get_pending_message=lambda key: pending_slot.pop(key, None)
    )
    seen = []

    def hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            seen.append(kwargs["event"]._gateway_busy_followup)
            return [{"action": "skip"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)
    pending_event, pending = await runner._run_agent_drain_pending(
        {"interrupted": True, "interrupt_message": "unaddressed"}, adapter, event.source, "session"
    )
    assert (pending_event, pending) == (None, None)
    assert seen == [True]


@pytest.mark.asyncio
async def test_skipped_fallback_head_does_not_strand_next_sender(monkeypatch):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._draining = False
    first, second = _event("sender-a"), _event("sender-b", "allowed")
    pending_slot = {"session": first}
    overflow = [second]
    runner._overflow_queue = lambda key: overflow
    adapter = SimpleNamespace(
        _pending_messages=pending_slot, get_pending_message=lambda key: pending_slot.pop(key, None)
    )
    seen = []

    def hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            seen.append(kwargs["event"].source.user_id)
            return [{"action": "skip"}] if len(seen) == 1 else []
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)
    pending_event, pending = await runner._run_agent_drain_pending(
        {"final_response": "done"}, adapter, first.source, "session"
    )
    assert pending_event is second and pending == "allowed"
    assert seen == ["sender-a", "sender-b"]

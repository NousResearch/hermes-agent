"""Busy-mode steer reaches the subagent the parent is blocked on (#112095, Telegram surface).

A parent inside ``delegate_task`` drains its own steer queue only after the tool returns — i.e.
after the child finishes — so a steer that lands on the parent alone is never read by a looping
child. Every gateway steer entry point must fan the text out to ``_active_children`` and the ack
must say so.
"""

import threading

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

class _Agent:
    def __init__(self, children=()):
        self.payload = None
        self._active_children = list(children)
        self._active_children_lock = threading.Lock()

    def steer(self, text):
        self.payload = text
        return True

def _event(text="focus on rows 10-20"):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c", user_id="u", chat_type="dm")
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id="m")

@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["busy_steer_mode", "priority", "explicit_command"])
async def test_busy_steer_fans_out_to_active_subagents(route, tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    runner = GatewayRunner(config=GatewayConfig())
    child_a, child_b = _Agent(), _Agent()
    parent = _Agent(children=[child_a, child_b])
    runner._session_state("key").turn.agent = parent
    event = _event("/steer focus on rows 10-20" if route == "explicit_command" else "focus on rows 10-20")

    if route == "busy_steer_mode":
        outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "steer", parent)
        assert outcome.steered is True
    elif route == "priority":
        runner._hm_busy_steer(event, parent, "key")
    else:
        await runner._busy_steer_command(event, "key", event.source)

    assert parent.payload and parent.payload.endswith("focus on rows 10-20")
    # The looping child — the one actually doing the work — gets the same text, not just the parent.
    assert child_a.payload == parent.payload
    assert child_b.payload == parent.payload

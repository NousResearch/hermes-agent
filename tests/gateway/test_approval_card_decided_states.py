"""A delivered exec-approval button card never stays looking pending, and never claims an outcome
nobody knows (gateway/run_turn_runner_approval_settle.py).

Drives the real wait (``tools.approval_gateway_wait._await_gateway_decision``), the real settle
hook (``register_timeout_notice``) and the real typed ``/approve`` / ``/deny`` resolver; only the
platform adapter is a recorder of in-place edits.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway import run_turn_runner_approval_settle as settle_mod
from gateway.platforms.base import ApprovalCardOutcome, ApprovalCardState, approval_card_notice
from tools import approval
from tools import approval_gateway_wait as wait_mod
from tools.approval_context import reset_current_observability_context, set_current_observability_context

SESSION = "agent:main:telegram:dm:42"


class _Adapter:
    def __init__(self):
        self.edits: list[tuple[str, str]] = []

    def max_message_length_for_chat(self, chat_id):
        return 4096

    def message_len_fn_for_chat(self, chat_id):
        return len

    async def edit_message(self, chat_id, message_id, content):
        self.edits.append((message_id, content))
        return SimpleNamespace(success=True, raw_response=None)


class _Runner:
    """The two things the settle hook reads from a TurnRunner: its context and its loop hop."""

    def __init__(self, agent):
        self.pending = []
        self._ctx = SimpleNamespace(
            session_key=SESSION, _run_still_current=lambda: True, _status_adapter=_Adapter(), _status_chat_id="42",
            _status_thread_metadata=None, source=None, agent_holder=[agent], mute_notification_reply=False,
            open_approval_cards={})

    def _schedule(self, coro, label):
        self.pending.append(coro)

    def drain(self):
        async def run():
            while self.pending:
                await self.pending.pop(0)
        asyncio.run(run())
        return [content for _, content in self._ctx._status_adapter.edits]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setattr(wait_mod._ctx, "_get_approval_timeout", lambda: 30)
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda *a, **k: None)
    yield
    approval.unregister_gateway_notify(SESSION)
    with approval._lock:
        approval._gateway_queues.pop(SESSION, None)


def _ask(runner, *, tool_call_id: str, session_id: str):
    """Raise one approval on an agent thread (inside tool call ``tool_call_id``); its button card is
    delivered as message ``card-1``. Returns (thread, request id)."""
    published, out = threading.Event(), {}

    def notify(data):
        out["rid"] = data["request_id"]
        settle_mod.register_timeout_notice(runner, data, command="rm -rf build", card_message_id="card-1")
        published.set()

    def wait():
        tokens = set_current_observability_context(tool_call_id=tool_call_id, session_id=session_id)
        try:
            out["decision"] = wait_mod._await_gateway_decision(
                SESSION, notify, {"command": "rm -rf build", "description": "recursive delete",
                                  "pattern_key": "rm", "pattern_keys": ["rm"]})
        finally:
            reset_current_observability_context(tokens)

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    assert published.wait(5)
    return thread, out


def test_typed_deny_with_reason_retires_the_button_card_as_denied(gate):
    """Typed /deny answered a card nobody tapped: the card itself must stop offering buttons and say
    "denied" with the reason (before, it stayed pending and a later tap said "expired")."""
    runner = _Runner(SimpleNamespace(session_id="s1", tool_complete_callback=None))
    thread, out = _ask(runner, tool_call_id="call-1", session_id="s1")

    assert approval.resolve_gateway_approval(SESSION, "deny", reason="use git clean", request_id=out["rid"]) == 1
    thread.join(5)

    (content,) = runner.drain()
    assert content.startswith(approval_card_notice(ApprovalCardOutcome(ApprovalCardState.DENIED, reason="use git clean")))
    assert "rm -rf build" in content


@pytest.mark.parametrize("result, expected", [
    ('{"output": "", "exit_code": 1, "error": "rm: cannot remove"}',
     ApprovalCardOutcome(ApprovalCardState.FAILED, reason="rm: cannot remove")),
    ('{"output": "", "exit_code": 0}', ApprovalCardOutcome(ApprovalCardState.APPROVED)),
    (None, ApprovalCardOutcome(ApprovalCardState.UNKNOWN)),
], ids=["failed", "finished", "never-reported"])
def test_approved_card_follows_the_command_to_its_real_outcome(gate, result, expected):
    """An approval is not the outcome: the card says "running", then what the asking tool call
    reported — failed, finished — or "unknown" when the turn ended before it reported back."""
    agent = SimpleNamespace(session_id="s1", tool_complete_callback=None)
    runner = _Runner(agent)
    thread, out = _ask(runner, tool_call_id="call-7", session_id="s1")

    assert approval.resolve_gateway_approval(SESSION, "once", request_id=out["rid"]) == 1
    thread.join(5)
    if result is not None:
        agent.tool_complete_callback("call-7", "terminal", {"command": "rm -rf build"}, result)
    else:
        settle_mod.settle_unreported_cards(runner)  # the turn's finally

    applying, final = runner.drain()
    assert applying.startswith(approval_card_notice(ApprovalCardOutcome(ApprovalCardState.APPLYING)))
    assert final.startswith(approval_card_notice(expected))
    assert runner._ctx.open_approval_cards == {}

"""When a messaging-platform approval prompt times out, the chat must say so.

``_await_gateway_decision`` fires ``entry.settle("timeout")`` when nobody answers within
``approvals.timeout``; only the TUI server registered a settle hook, so on Telegram / Slack /
WhatsApp the card kept live buttons and the user never learned the command did NOT run.
These tests drive the real ``TurnRunner._approval_notify_sync`` with a fake adapter.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List

import pytest

from gateway.platforms.base import SendResult
from tools import approval as _approval
from tools.approval_gateway_wait import _ApprovalEntry

SESSION = "agent:main:telegram:dm:1"
APPROVAL = {"command": "rm -rf /tmp/x", "description": "recursive delete", "pattern_key": "k"}


class _ButtonAdapter:
    """Renders native buttons and remembers the posted card id; records edits and sends."""

    typed_command_prefix = "/"

    def __init__(self, *, editable: bool = True) -> None:
        self.sends: List[str] = []
        self.edits: List[tuple] = []
        self._editable = editable

    def pause_typing_for_chat(self, chat_id: str) -> None:
        return None

    async def send_exec_approval(self, *a: Any, **k: Any) -> SendResult:
        return SendResult(success=True, message_id="card-1")

    async def send(self, chat_id: str, message: str, **k: Any) -> SendResult:
        self.sends.append(message)
        return SendResult(success=True, message_id="m2")

    async def edit_message(self, chat_id: str, message_id: str, content: str, **k: Any) -> SendResult:
        self.edits.append((message_id, content))
        return SendResult(success=self._editable, error=None if self._editable else "cannot edit")


def _runner(adapter):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="C1", _status_thread_metadata={"thread_id": "t1"},
        session_key=SESSION, source=SimpleNamespace(chat_id="C1", platform="telegram", session_key=SESSION),
    )

    class _Fut:
        def __init__(self, result): self._r = result
        def result(self, timeout=None): return self._r

    runner._schedule = lambda coro, _label: _Fut(asyncio.run(coro))
    runner._close_native_stream_boundary = lambda _why: None
    return runner


@pytest.fixture
def pending_entry(monkeypatch):
    """A queued approval entry exactly as ``_await_gateway_decision`` registers it."""
    monkeypatch.setattr("gateway.platforms.base_exec_approval.approval_timeout_seconds", lambda: 300)
    entry = _ApprovalEntry(dict(APPROVAL))
    with _approval._lock:
        _approval._gateway_queues[SESSION] = [entry]
    yield entry
    with _approval._lock:
        _approval._gateway_queues.pop(SESSION, None)


@pytest.mark.parametrize("warning_notifications", [True, False])
def test_timeout_edits_the_card_to_say_the_command_did_not_run(pending_entry, tmp_path, monkeypatch, warning_notifications):
    import gateway.run as gateway_run
    (tmp_path / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {str(not warning_notifications).lower()}}}")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    adapter = _ButtonAdapter()
    _runner(adapter)._approval_notify_sync(dict(pending_entry.data))  # what notify_cb receives

    assert pending_entry.settle is not None, "settle hook must be registered for the request_id"
    pending_entry.settle("timeout")

    assert len(adapter.edits) == 1
    message_id, content = adapter.edits[0]
    assert message_id == "card-1"
    assert "NOT run" in content and "5 minutes" in content
    assert adapter.sends == [], "an editable card needs no extra message"


def test_timeout_falls_back_to_a_new_message_when_the_card_cannot_be_edited(pending_entry):
    adapter = _ButtonAdapter(editable=False)
    _runner(adapter)._approval_notify_sync(dict(pending_entry.data))  # what notify_cb receives

    pending_entry.settle("timeout")

    assert len(adapter.sends) == 1
    assert "NOT run" in adapter.sends[0]


@pytest.mark.parametrize("reason", ["answered", "interrupted", "notify_failed"])
def test_other_settle_reasons_post_nothing(pending_entry, reason):
    adapter = _ButtonAdapter()
    _runner(adapter)._approval_notify_sync(dict(pending_entry.data))  # what notify_cb receives

    pending_entry.settle(reason)

    assert adapter.edits == [] and adapter.sends == []

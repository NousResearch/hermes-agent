"""Native button approvals must resolve the request the user tapped, not the FIFO-oldest.

``TurnRunner._approval_notify_sync`` used to drop the queued approval's ``request_id`` on the
floor, so every bundled button surface could only call
``resolve_gateway_approval(session_key, choice)`` — which pops the OLDEST pending entry in the
session's queue. With two pending exec approvals, tapping the newest card answered the oldest
request and the tapped one blocked until ``approvals.timeout`` (#124974). The queue already
stamped every entry with a ``request_id``; these tests pin the plumbing that finally carries it
to the adapters and back.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict

import pytest

from gateway.platforms.base import ExecApprovalPrompt, SendResult
from tools import approval as _approval
from tools.approval_gateway_wait import _ApprovalEntry, _await_gateway_decision

SESSION = "agent:main:discord:dm:1"


def _entry(**extra) -> _ApprovalEntry:
    return _ApprovalEntry({"command": "rm -rf /tmp/x", "description": "d", **extra})


# ── 1. The notify path forwards request_id to the adapter ──────────────────────────────────


class _ButtonAdapter:
    """Native-buttons adapter that records the kwargs it was called with."""

    typed_command_prefix = "/"

    def __init__(self) -> None:
        self.kwargs: Dict[str, Any] = {}

    def pause_typing_for_chat(self, chat_id: str) -> None:
        return None

    async def send_exec_approval(self, *a: Any, **k: Any) -> SendResult:
        self.kwargs = k
        return SendResult(success=True, message_id="card-1")

    async def send(self, chat_id: str, message: str, **k: Any) -> SendResult:
        return SendResult(success=True, message_id="m2")

    async def edit_message(self, chat_id: str, message_id: str, content: str, **k: Any) -> SendResult:
        return SendResult(success=True)


def _runner(adapter):
    from types import SimpleNamespace

    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="C1", _status_thread_metadata={"thread_id": "t1"},
        session_key=SESSION, source=SimpleNamespace(chat_id="C1", platform="discord", session_key=SESSION),
    )

    class _Fut:
        def __init__(self, result):
            self._r = result

        def result(self, timeout=None):
            return self._r

    runner._schedule = lambda coro, _label: _Fut(asyncio.run(coro))
    runner._close_native_stream_boundary = lambda _why: None
    return runner


def test_notify_forwards_request_id_to_the_adapter(monkeypatch, tmp_path):
    import gateway.run as gateway_run
    import gateway.platforms.base_exec_approval as _bea

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(_bea, "approval_timeout_seconds", lambda: 300)
    entry = _entry(request_id="rid-abc")
    adapter = _ButtonAdapter()
    _runner(adapter)._approval_notify_sync(dict(entry.data))
    assert adapter.kwargs.get("request_id") == "rid-abc", (
        "the queued approval's request_id must reach adapter.send_exec_approval")


# ── 2. ExecApprovalPrompt carries request_id end-to-end ────────────────────────────────────


@pytest.mark.asyncio
async def test_send_exec_approval_stamps_the_prompt_with_request_id():
    captured: list = []

    class _Capture:
        async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
            captured.append(prompt)
            return SendResult(success=True, message_id="m1")

    from gateway.config import Platform, PlatformConfig

    class _Adapter(_Capture):
        pass

    # Build through the template method on a minimal real adapter.
    from gateway.platforms.base import BasePlatformAdapter

    class _Real(BasePlatformAdapter):
        async def connect(self, *, is_reconnect: bool = False) -> bool:
            return True

        async def disconnect(self) -> None:
            pass

        async def send(self, *a: Any, **k: Any) -> SendResult:
            return SendResult(success=True)

        async def get_chat_info(self, chat_id: str) -> dict:
            return {}

        async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
            captured.append(prompt)
            return SendResult(success=True, message_id="m1")

    adapter = _Real(PlatformConfig(enabled=True), Platform.TELEGRAM)
    await adapter.send_exec_approval("chat", "cmd", SESSION, request_id="rid-1")
    assert captured[-1].request_id == "rid-1"


# ── 3. Resolution contract: request_id picks ITS entry, not the FIFO-oldest ────────────────


def test_request_id_resolves_the_tapped_request_not_the_oldest():
    q = [_entry(request_id="rid-old"), _entry(request_id="rid-new")]
    with _approval._lock:
        _approval._gateway_queues[SESSION] = q
    try:
        resolved = _approval.resolve_gateway_approval(SESSION, "once", request_id="rid-new")
        assert resolved == 1
        with _approval._lock:
            remaining = [e.data.get("request_id") for e in _approval._gateway_queues.get(SESSION, [])]
        assert remaining == ["rid-old"], "the tapped (newest) entry resolved; the oldest must stay pending"
    finally:
        with _approval._lock:
            _approval._gateway_queues.pop(SESSION, None)


def test_blocking_wait_wakes_only_for_its_own_request_id():
    results: dict = {}

    def _wait(tag: str, payload: dict) -> None:
        results[tag] = _await_gateway_decision(SESSION, lambda data: None, payload)

    threads = [
        threading.Thread(target=_wait, args=("old", {"command": "c1", "request_id": "rid-old"}), daemon=True),
        threading.Thread(target=_wait, args=("new", {"command": "c2", "request_id": "rid-new"}), daemon=True),
    ]
    for t in threads:
        t.start()
    time.sleep(0.3)
    try:
        assert _approval.resolve_gateway_approval(SESSION, "once", request_id="rid-new") == 1
        threads[1].join(timeout=5)
        assert results["new"]["resolved"] and results["new"]["choice"] == "once"
        assert "old" not in results, "the FIFO-oldest waiter must stay blocked after a targeted resolve"
    finally:
        # Unblock the loser thread so the test process drains cleanly.
        _approval.resolve_gateway_approval(SESSION, "deny", resolve_all=True)
        threads[0].join(timeout=5)

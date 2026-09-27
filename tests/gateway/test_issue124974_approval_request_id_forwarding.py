"""A button approval must reach its adapter bound to ONE queued request (#124974).

``TurnRunner._approval_notify_sync`` used to pass only
``ctx._status_thread_metadata`` to ``adapter.send_exec_approval``, so the
queued approval's ``request_id`` never crossed into the adapter. A button
handler then had nothing to bind its card to one specific request and could
only call ``resolve_gateway_approval(session_key, choice)`` — which resolves
the FIFO-oldest entry in that session's queue. With two approvals pending,
tapping the NEWEST card answered the OLDEST request, and the newest one kept
blocking for the full ``approvals.timeout``.

The id is already the canonical targeting mechanism: ``api_server_runs``
passes and enforces it for room-scoped approvals, and
``_ApprovalEntry.__init__`` assigns one to every queued entry, so it is
present in the ``entry.data`` copy the notify callback receives.

Contract: the card metadata carries ``approval_request_id`` (the id of the
request that card belongs to) and ``requester_user_id`` (who triggered the
turn), on top of the existing thread/chat keys. Purely additive: an adapter
that ignores the keys behaves exactly as before, and
``resolve_gateway_approval`` still falls back to FIFO when no id is given.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List

from gateway.platforms.base import SendResult
from tools.approval_gateway_wait import _ApprovalEntry

SESSION = "agent:main:telegram:dm:1"
APPROVAL = {"command": "rm -rf /tmp/x", "description": "recursive delete", "pattern_key": "k"}


class _ButtonAdapter:
    """Renders native buttons and records the metadata each card was built with."""

    typed_command_prefix = "/"

    def __init__(self) -> None:
        self.metadata: List[dict] = []
        self.sends: List[str] = []

    def pause_typing_for_chat(self, chat_id: str) -> None:
        return None

    async def send_exec_approval(self, *a: Any, **k: Any) -> SendResult:
        self.metadata.append(k.get("metadata"))
        return SendResult(success=True, message_id="card-1")

    async def send(self, chat_id: str, message: str, **k: Any) -> SendResult:
        self.sends.append(message)
        return SendResult(success=True, message_id="m2")


def _runner(adapter, *, source_user_id="42", thread_metadata=None):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter,
        _status_chat_id="C1",
        _status_thread_metadata=thread_metadata if thread_metadata is not None else {"thread_id": "t1"},
        session_key=SESSION,
        source=SimpleNamespace(
            chat_id="C1", platform="telegram", session_key=SESSION,
            user_id=source_user_id,
        ),
    )

    class _Fut:
        def __init__(self, result):
            self._r = result

        def result(self, timeout=None):
            return self._r

    runner._schedule = lambda coro, _label: _Fut(asyncio.run(coro))
    runner._close_native_stream_boundary = lambda _why: None
    return runner


def test_card_metadata_carries_the_queued_request_id():
    """The card must name the request it belongs to, or a handler can only guess."""
    entry = _ApprovalEntry(dict(APPROVAL))
    request_id = entry.data["request_id"]
    assert request_id, "every queued approval carries a request_id"

    adapter = _ButtonAdapter()
    _runner(adapter)._approval_notify_sync(dict(entry.data))  # what notify_cb receives

    assert len(adapter.metadata) == 1
    metadata = adapter.metadata[0]
    assert metadata is not None
    assert metadata.get("approval_request_id") == request_id


def test_card_metadata_carries_the_requesting_user_id():
    """Adapters bind the requester end to end only if the id is forwarded too."""
    entry = _ApprovalEntry(dict(APPROVAL))

    adapter = _ButtonAdapter()
    _runner(adapter)._approval_notify_sync(dict(entry.data))

    assert adapter.metadata[0].get("requester_user_id") == "42"


def test_existing_thread_metadata_survives_the_forwarding():
    """Forwarding is additive: the chat/thread keys the adapter already uses stay."""
    entry = _ApprovalEntry(dict(APPROVAL))

    adapter = _ButtonAdapter()
    _runner(adapter, thread_metadata={"thread_id": "t1", "other": "keep"})._approval_notify_sync(
        dict(entry.data)
    )

    metadata = adapter.metadata[0]
    assert metadata.get("thread_id") == "t1"
    assert metadata.get("other") == "keep"
    assert metadata.get("approval_request_id") == entry.data["request_id"]


def test_forwarding_does_not_mutate_the_callers_thread_metadata():
    """``ctx._status_thread_metadata`` is shared state; the copy must be local."""
    entry = _ApprovalEntry(dict(APPROVAL))
    thread_metadata = {"thread_id": "t1"}

    adapter = _ButtonAdapter()
    _runner(adapter, thread_metadata=thread_metadata)._approval_notify_sync(dict(entry.data))

    assert thread_metadata == {"thread_id": "t1"}, (
        "the runner's shared thread metadata was mutated — a second approval "
        "in the same turn would inherit the first one's request id"
    )


def test_a_missing_request_id_still_posts_the_card():
    """An older queue entry without an id must not stop the prompt rendering."""
    entry = _ApprovalEntry(dict(APPROVAL))
    approval_data = {k: v for k, v in entry.data.items() if k != "request_id"}

    adapter = _ButtonAdapter()
    _runner(adapter)._approval_notify_sync(approval_data)

    assert len(adapter.metadata) == 1
    assert adapter.metadata[0].get("approval_request_id") is None
    assert "thread_id" in adapter.metadata[0]


def test_a_missing_source_user_id_keeps_the_card_sendable():
    """A session with no addressable user still posts the button card."""
    entry = _ApprovalEntry(dict(APPROVAL))

    adapter = _ButtonAdapter()
    _runner(adapter, source_user_id=None)._approval_notify_sync(dict(entry.data))

    assert len(adapter.metadata) == 1
    assert adapter.metadata[0].get("approval_request_id") == entry.data["request_id"]

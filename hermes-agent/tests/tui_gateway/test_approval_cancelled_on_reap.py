"""Pending gateway approvals dropped by interrupt/reap/teardown must be loud, not silent (#106678).

``_interrupt_session_turn`` deny-resolves ``tools.approval._gateway_queues`` so the agent thread
unblocks (silence is not consent), and ``_teardown_session`` pops the queue via
``unregister_gateway_notify``. Before the ``approval.cancelled`` broadcast the drop was invisible:
a reconnecting client saw its parked prompt card answer ``resolved: 0`` (or a bare 4001 on
``approval.pending``), so the prompt looked lost rather than cancelled and the user had no signal
the command would never run.

Contracts here: interrupt/reap with a pending entry broadcasts ``approval.cancelled`` (session ids,
reason, request ids/count) BEFORE the deny-resolve drains the queue, an empty queue stays silent, an
explicit user deny does not emit this event, a broadcast failure still deny-resolves, and teardown
of a session still holding a pending approval broadcasts too.
"""

from __future__ import annotations

import pytest

from tui_gateway import server
from tools.approval import _gateway_queues, list_gateway_approvals, resolve_gateway_approval
from tools.approval_gateway_wait import _ApprovalEntry


SESSION_KEY = "20260909_120000_aaaaaa"
SID = "live-loud-deny"
REQUEST_ID = "req-loud-deny-1"


@pytest.fixture()
def captured(monkeypatch):
    events = []
    monkeypatch.setattr(
        server, "_broadcast_global_event", lambda ev, payload=None: events.append((ev, payload))
    )
    # Teardown's real work (finalize, agent close) is out of scope — this is
    # about what reaches the client.
    monkeypatch.setattr(server, "_finalize_session", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_a, **_k: False)
    monkeypatch.setattr(server, "_clear_pending", lambda *_a, **_k: None)
    return events


@pytest.fixture()
def session():
    return {
        "_sid": SID,
        "session_key": SESSION_KEY,
        "history": [],
        "history_lock": __import__("threading").Lock(),
        "running": False,
        "queued_prompt": None,
        "agent": None,
    }


@pytest.fixture()
def pending_entry():
    entry = _ApprovalEntry({"command": "rm -rf /tmp/x", "request_id": REQUEST_ID})
    _gateway_queues.setdefault(SESSION_KEY, []).append(entry)
    try:
        yield entry
    finally:
        _gateway_queues.pop(SESSION_KEY, None)


def _cancelled_events(captured):
    return [(ev, payload) for ev, payload in captured if ev == "approval.cancelled"]


def test_interrupt_with_pending_broadcasts_approval_cancelled(captured, session, pending_entry):
    """RED on main: the deny-resolve happens, but nothing tells the client the prompt died."""
    server._interrupt_session_turn(SID, session, request_id=f"client-gone-{SID}", orphan=True)

    cancelled = _cancelled_events(captured)
    assert cancelled, "interrupt/reap must broadcast approval.cancelled when it drops a pending prompt"
    _event, payload = cancelled[0]
    assert payload["session_id"] == SID
    assert payload["stored_session_id"] == SESSION_KEY
    assert payload["reason"] == "ws_orphan_reap"
    assert payload["cancelled_count"] == 1
    assert REQUEST_ID in payload["request_ids"]
    assert pending_entry.result == "deny"
    assert pending_entry.event.is_set()
    assert list_gateway_approvals(SESSION_KEY) == []


def test_interrupt_empty_queue_emits_nothing(captured, session):
    """CONTROL: no pending approvals -> keep the historical silent interrupt."""
    _gateway_queues.pop(SESSION_KEY, None)

    server._interrupt_session_turn(SID, session)

    assert _cancelled_events(captured) == []


def test_explicit_user_deny_does_not_emit_cancelled(captured, pending_entry):
    """CONTROL: /deny and approval.respond stay on the existing resolve path."""
    count = resolve_gateway_approval(SESSION_KEY, "deny")

    assert count == 1
    assert pending_entry.result == "deny"
    assert _cancelled_events(captured) == []


def test_broadcast_failure_still_deny_resolves(monkeypatch, session, pending_entry):
    """Fail-open: a wedged peer must not leave the agent thread blocked on the queue."""

    def _boom(*_a, **_k):
        raise RuntimeError("transport gone")

    monkeypatch.setattr(server, "_broadcast_global_event", _boom)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_a, **_k: False)
    monkeypatch.setattr(server, "_clear_pending", lambda *_a, **_k: None)

    server._interrupt_session_turn(SID, session, request_id=f"client-gone-{SID}", orphan=True)

    assert pending_entry.result == "deny"
    assert pending_entry.event.is_set()


def test_user_interrupt_reason_is_interrupt(captured, session, pending_entry):
    """session.interrupt (orphan=False, even with a client-gone request_id) labels interrupt, not reap."""
    server._interrupt_session_turn(SID, session, request_id=f"client-gone-{SID}")

    cancelled = _cancelled_events(captured)
    assert cancelled
    assert cancelled[0][1]["reason"] == "interrupt"
    assert pending_entry.result == "deny"


def test_orphan_kwarg_labels_ws_orphan_reap_without_client_gone_prefix(captured, session, pending_entry):
    """The reason comes from orphan=, not request_id prefix sniffing."""
    server._interrupt_session_turn(SID, session, request_id="interrupt-manual", orphan=True)

    cancelled = _cancelled_events(captured)
    assert cancelled
    assert cancelled[0][1]["reason"] == "ws_orphan_reap"
    assert pending_entry.result == "deny"


def test_teardown_with_pending_broadcasts_approval_cancelled(captured, session, pending_entry):
    """close_on_disconnect / idle reclaim can drop the queue via unregister without an interrupt."""
    server._teardown_session(session, end_reason="ws_orphan_reap")

    cancelled = _cancelled_events(captured)
    assert cancelled, "teardown must broadcast approval.cancelled when it drops a pending prompt"
    _event, payload = cancelled[0]
    assert payload["session_id"] == SID
    assert payload["stored_session_id"] == SESSION_KEY
    assert payload["reason"] == "ws_orphan_reap"
    assert payload["cancelled_count"] == 1
    assert REQUEST_ID in payload["request_ids"]
    assert list_gateway_approvals(SESSION_KEY) == []


def test_teardown_without_pending_stays_silent(captured, session):
    server._teardown_session(session, end_reason="ws_orphan_reap")

    assert _cancelled_events(captured) == []


def test_interrupt_missing_session_key_does_not_crash(captured):
    """Fail-open: a session dict without session_key must not raise out of interrupt."""
    session = {"history_lock": __import__("threading").Lock(), "running": False}
    server._interrupt_session_turn(SID, session)
    assert _cancelled_events(captured) == []

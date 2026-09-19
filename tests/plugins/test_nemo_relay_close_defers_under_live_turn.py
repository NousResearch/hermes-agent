"""close_session must defer under a live turn scope, not violate LIFO.

Production failure (gateway.error.log, 2026-08-15): during gateway
shutdown a session close raced a still-live turn. close_session popped
the session scope under the turn scope ("scope handle is not at the top
of the stack") and set ``closing``; end_turn's finalization was then
refused with "session is closing" — one root cause, both log lines
(repeated 34+ times across Aug 2–15).

Contract: when the coordinator has a live turn for the session,
close_session must NOT close. It defers by setting ``close_pending`` —
the same deferral contract notify_session_compacted established for
rotating compaction — and the live turn's end_turn consumes it via
_consume_deferred_close after its own scope pops, preserving LIFO order.
"""

from __future__ import annotations

import contextvars
import sys
from types import SimpleNamespace

import pytest

from agent import relay_runtime


class _DeferralFakeRelay:
    """Minimal Relay double: push/pop recording, one scope stack."""

    def __init__(self):
        self.events = []
        self._stack = []
        self._serial = 0
        self._scope_context = contextvars.ContextVar(
            "deferral_fake_scope", default=None
        )
        self.ScopeType = SimpleNamespace(Agent="agent", Function="function")
        self.scope = SimpleNamespace(
            push=self._push, pop=self._pop, event=lambda *a, **k: None
        )
        self.subscribers = SimpleNamespace(
            register=lambda *a, **k: None,
            deregister=lambda *a, **k: None,
            flush=lambda: None,
        )
        self.get_scope_stack = lambda: self._scope_context.get()

    def _push(self, name, scope_type, **kwargs):
        self._serial += 1
        handle = ("scope", name, self._serial)
        self._stack.append(handle)
        self.events.append(("push", name, self._serial))
        return handle

    def _pop(self, handle, **kwargs):
        if not self._stack or self._stack[-1] is not handle:
            raise ValueError(
                "invalid argument: scope handle is not at the top of the stack"
            )
        self._stack.pop()
        self.events.append(("pop", name_of(handle), handle[2]))


def name_of(handle):
    return handle[1]


@pytest.fixture()
def coordinator_with_live_turn(monkeypatch):
    from agent import relay_runtime

    fake = _DeferralFakeRelay()
    monkeypatch.setattr(relay_runtime, "_load_nemo_relay", lambda: fake)
    relay_runtime._reset_for_tests()
    coordinator = relay_runtime.SESSION_COORDINATOR

    lease = coordinator.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="s-shutdown-race",
        platform="cli",
        model="test-model",
    )
    assert lease.session is not None
    turn = coordinator.begin_turn(lease, turn_id="t-1", task_id="task-1")
    assert coordinator.has_active_turn(
        profile_key=lease.profile_key, session_id="s-shutdown-race"
    )
    return coordinator, lease, turn, fake


def test_close_session_defers_under_live_turn(coordinator_with_live_turn):
    """The shutdown race: close during a live turn must defer, not pop."""
    coordinator, lease, turn, fake = coordinator_with_live_turn

    coordinator.host_close = None  # not used; keep attribute tidy

    # Production sequence: shutdown fires close_session while the turn is live.
    host = lease.host
    host.close_session({"session_id": "s-shutdown-race"})

    # The session scope must still be on the stack — deferred, not popped.
    assert any(h[1] == relay_runtime.SESSION_SCOPE for h in fake._stack), (
        "close_session must not pop the session scope under a live turn"
    )
    assert lease.session.close_pending is True, (
        "close_session must set close_pending when deferring"
    )
    assert lease.session.closing is False

    # end_turn then finalizes cleanly and consumes the deferred close.
    coordinator.end_turn(turn, outcome="success")
    assert fake._stack == [], "turn and session scopes must both close LIFO"
    assert relay_runtime.get_session_handle("s-shutdown-race") is None


def test_close_session_closes_when_no_live_turn(coordinator_with_live_turn):
    """No live turn: close_session closes immediately (historical path)."""
    coordinator, lease, turn, fake = coordinator_with_live_turn
    coordinator.end_turn(turn, outcome="success")

    host = lease.host
    host.close_session({"session_id": "s-shutdown-race"})
    assert fake._stack == []
    assert relay_runtime.get_session_handle("s-shutdown-race") is None


def test_session_close_pop_ordering_is_lifo(coordinator_with_live_turn):
    """Full order audit: pushes and pops interleave strictly LIFO."""
    coordinator, lease, turn, fake = coordinator_with_live_turn
    relay_runtime = pytest.importorskip("agent.relay_runtime")

    host = lease.host
    host.close_session({"session_id": "s-shutdown-race"})  # defers
    coordinator.end_turn(turn, outcome="success")  # pops turn, consumes close

    ops = fake.events
    pushes = [e for e in ops if e[0] == "push"]
    pops = [e for e in ops if e[0] == "pop"]
    assert len(pushes) == 2 and len(pops) == 2, f"events={ops}"
    # Turn pushed after session; turn popped before session.
    assert pops[0][2] == pushes[1][2], "turn must pop before session"
    assert pops[1][2] == pushes[0][2], "session must pop last"

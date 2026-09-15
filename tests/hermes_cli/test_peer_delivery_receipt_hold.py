"""P12 (§5.2): a DELIVERY turn reports a live-owner hold as a receipt, not a failure.

The refusal vocabulary stays exactly as it was for every other caller: the delivery
layer is told *why* the turn did not run and maps that reason onto a receipt.
"""
from __future__ import annotations

from hermes_cli import active_sessions
from hermes_cli.active_sessions import (
    DELIVERY_TURN_ENV,
    SESSION_NOT_OWNED,
    ActiveSessionRefusal,
    DeliveryHold,
    format_refusal_stderr,
    is_delivery_turn,
    try_acquire_active_session,
)


def _acquire(tmp_path, *, surface, live_id, delivery=None):
    return try_acquire_active_session(
        session_id="sess-1",
        surface=surface,
        config={},
        metadata={"live_session_id": live_id},
        registry_home=tmp_path,
        delivery=delivery,
    )


def test_is_delivery_turn_flag_wins_over_env(monkeypatch):
    monkeypatch.delenv(DELIVERY_TURN_ENV, raising=False)
    assert is_delivery_turn() is False
    assert is_delivery_turn(True) is True
    assert is_delivery_turn(False) is False
    monkeypatch.setenv(DELIVERY_TURN_ENV, "1")
    assert is_delivery_turn() is True, "the env marker is how a spawned delivery turn declares itself"
    monkeypatch.setenv(DELIVERY_TURN_ENV, "off")
    assert is_delivery_turn() is False


def test_delivery_hold_keeps_the_refusal_vocabulary(tmp_path):
    hold = DeliveryHold("held", reason=SESSION_NOT_OWNED, session_id="sess-1",
                        owner_surface="desktop", owner_pid=42)
    assert hold.reason == SESSION_NOT_OWNED, "the CLI contract is unchanged"
    assert hold.delivery is True
    assert not isinstance(hold, ActiveSessionRefusal)
    assert "hermes-refusal-reason: SESSION_NOT_OWNED" in format_refusal_stderr(hold)
    assert format_refusal_stderr(hold).splitlines()[-1] == "held"


def test_ordinary_turn_keeps_the_session_not_owned_refusal(tmp_path):
    assert _acquire(tmp_path, surface="tui", live_id="live-a")[1] is None
    lease, message = _acquire(tmp_path, surface="cli", live_id="live-b")
    assert lease is None
    assert isinstance(message, ActiveSessionRefusal)
    assert not isinstance(message, DeliveryHold)
    assert message.reason == SESSION_NOT_OWNED
    assert "already has a live owner" in message


def test_delivery_turn_gets_a_hold_not_a_refusal(tmp_path):
    assert _acquire(tmp_path, surface="tui", live_id="live-a")[1] is None
    lease, hold = _acquire(tmp_path, surface="cli", live_id="live-b", delivery=True)
    assert lease is None
    assert isinstance(hold, DeliveryHold)
    assert hold.reason == SESSION_NOT_OWNED
    assert hold.session_id == "sess-1"
    assert hold.owner_surface == "tui"
    assert hold.owner_pid is not None
    assert "ACCEPTED" in hold and "Do not resend" in hold
    assert "was NOT delivered" not in hold


def test_env_marker_alone_selects_the_hold(tmp_path, monkeypatch):
    """The spawned delivery turn declares itself through the env, no explicit flag."""
    assert _acquire(tmp_path, surface="tui", live_id="live-a")[1] is None
    monkeypatch.setenv(DELIVERY_TURN_ENV, "1")
    lease, hold = _acquire(tmp_path, surface="cli", live_id="live-b")
    assert lease is None and isinstance(hold, DeliveryHold)
    monkeypatch.setenv(DELIVERY_TURN_ENV, "0")
    lease, message = _acquire(tmp_path, surface="cli", live_id="live-c")
    assert lease is None and isinstance(message, ActiveSessionRefusal)

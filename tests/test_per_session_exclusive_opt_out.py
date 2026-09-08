"""Configurable per-session exclusivity: queue-instead-of-refuse (#101279).

Default behavior is untouched: one live owner per stored session, a second surface
is refused with SESSION_NOT_OWNED (see tests/test_active_session_exclusivity.py).

When the operator opts out via ``gateway.per_session_exclusive: false`` (shared-brain
deployments: several humans deliberately working the same sessions from two desktop
backends over one central gateway), a SESSION_NOT_OWNED refusal becomes a bounded
WAIT for the current owner to finish, then a re-acquire — the serialization the
messaging gateway already gives Telegram ("Another Hermes process is using this
session; waiting..."), instead of pushing a mid-day failure onto the second user.

The opt-out NEVER weakens the other fences:
- registry-unreadable refusals (SESSION_COORDINATION_UNAVAILABLE) still fail closed;
- capacity refusals (MAX_CONCURRENT_SESSIONS) are untouched;
- the wait is bounded, and its timeout falls back to the original refusal.
"""

import itertools
import threading
import time

import pytest

from hermes_cli.active_sessions import (
    SESSION_NOT_OWNED,
    per_session_exclusive,
    release_active_session,
    try_acquire_active_session,
    wait_for_session_ownership,
)


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))


_owner_seq = itertools.count()


def acquire(session_id, config=None, surface="tui", live_id=None):
    return try_acquire_active_session(
        session_id=session_id,
        surface=surface,
        config=config if config is not None else {},
        metadata={"live_session_id": live_id or f"live-{next(_owner_seq)}"},
    )


class TestPerSessionExclusiveResolution:
    def test_default_is_exclusive(self):
        assert per_session_exclusive({}) is True
        assert per_session_exclusive(None) is True

    def test_gateway_false_opts_out(self):
        assert per_session_exclusive({"gateway": {"per_session_exclusive": False}}) is False

    def test_gateway_true_stays_exclusive(self):
        assert per_session_exclusive({"gateway": {"per_session_exclusive": True}}) is True

    def test_top_level_fallback(self):
        assert per_session_exclusive({"per_session_exclusive": False}) is False

    def test_string_values_coerce(self):
        assert per_session_exclusive({"gateway": {"per_session_exclusive": "false"}}) is False
        assert per_session_exclusive({"gateway": {"per_session_exclusive": "true"}}) is True

    def test_invalid_value_warns_and_keeps_default(self):
        assert per_session_exclusive({"gateway": {"per_session_exclusive": "sometimes"}}) is True
        assert per_session_exclusive({"gateway": {"per_session_exclusive": 3}}) is True

    def test_attribute_style_config(self):
        class Cfg:
            per_session_exclusive = False
        assert per_session_exclusive(Cfg()) is False

        class Gateway:
            per_session_exclusive = False

        class Cfg2:
            gateway = Gateway()
        assert per_session_exclusive(Cfg2()) is False


class TestExclusivityStillEnforcedByDefault:
    def test_refusal_unchanged_when_opted_in(self):
        lease_a, _ = acquire("S")
        assert lease_a is not None
        lease_b, refusal = acquire("S", config={"gateway": {"per_session_exclusive": True}})
        assert lease_b is None
        assert refusal.reason == SESSION_NOT_OWNED


class TestWaitForSessionOwnership:
    def test_free_session_returns_immediately(self):
        assert wait_for_session_ownership(session_id="S") is True

    def test_held_session_blocks_until_released(self):
        lease_a, _ = acquire("S")
        assert lease_a is not None

        def _release_after(delay: float) -> None:
            time.sleep(delay)
            release_active_session(lease_a)

        releaser = threading.Thread(target=_release_after, args=(0.3,), daemon=True)
        releaser.start()
        try:
            # poll fast so the test waits ~0.3s, not the default poll interval
            assert wait_for_session_ownership(
                session_id="S", wait_seconds=5.0, poll_seconds=0.05
            ) is True
        finally:
            releaser.join(timeout=2)
        # After the wait, the second surface can acquire.
        lease_b, refusal = acquire("S", config={"gateway": {"per_session_exclusive": False}})
        assert lease_b is not None and refusal is None

    def test_timeout_returns_false_then_refusal_still_applies(self):
        lease_a, _ = acquire("S")
        assert lease_a is not None
        assert wait_for_session_ownership(
            session_id="S", wait_seconds=0.15, poll_seconds=0.05
        ) is False
        # The bounded-timeout caller falls back to the original refusal.
        lease_b, refusal = acquire("S", config={"gateway": {"per_session_exclusive": False}})
        assert lease_b is None
        assert refusal.reason == SESSION_NOT_OWNED

    def test_on_wait_notifies_exactly_once(self):
        lease_a, _ = acquire("S")
        assert lease_a is not None
        calls: list[float] = []
        releaser_done = threading.Event()

        def _release() -> None:
            time.sleep(0.2)
            release_active_session(lease_a)
            releaser_done.set()

        threading.Thread(target=_release, daemon=True).start()
        try:
            assert wait_for_session_ownership(
                session_id="S", wait_seconds=5.0, poll_seconds=0.05, on_wait=calls.append
            ) is True
        finally:
            releaser_done.wait(timeout=2)
        assert len(calls) == 1, "the waiting notice must fire once, not per poll"

    def test_other_sessions_are_not_waited_on(self):
        lease_a, _ = acquire("OTHER")
        assert lease_a is not None
        # Waiting for S while OTHER is held must return immediately.
        assert wait_for_session_ownership(session_id="S", wait_seconds=0.1, poll_seconds=0.05) is True

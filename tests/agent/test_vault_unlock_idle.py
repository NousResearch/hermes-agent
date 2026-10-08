"""The vault unlock's 30-minute idle TTL must count time the machine spent asleep.

``time.monotonic()`` does not advance during system sleep on macOS, so an idle check on that clock
alone kept a password-manager session unlocked across a night with the lid closed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.vault_backends import unlock as unlock_mod

_MIN = 60
_HOUR = 60 * _MIN


@pytest.fixture
def clock(tmp_path, monkeypatch):
    """A controllable (monotonic, wall) clock pair for the unlock module only."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    now = SimpleNamespace(mono=1_000.0, wall=1_790_000_000.0)
    monkeypatch.setattr(unlock_mod, "time", SimpleNamespace(monotonic=lambda: now.mono, time=lambda: now.wall))
    unlock_mod.lock()
    yield now
    unlock_mod.lock()


def _awake(now, seconds):
    now.mono += seconds
    now.wall += seconds


def test_system_sleep_counts_toward_the_idle_timeout(clock):
    assert unlock_mod.store_session_token("bitwarden", "TOKEN")
    _awake(clock, 5 * _MIN)
    assert unlock_mod.get_session_token("bitwarden") == "TOKEN"

    clock.wall += 8 * _HOUR  # asleep: the monotonic clock is frozen, the wall clock is not
    _awake(clock, 2 * _MIN)

    assert unlock_mod.get_session_token("bitwarden") is None
    assert not unlock_mod.is_unlocked("bitwarden")


def test_wall_clock_set_backwards_does_not_extend_the_unlock(clock):
    assert unlock_mod.store_session_token("bitwarden", "TOKEN")
    clock.mono += 31 * _MIN
    clock.wall -= 2 * _HOUR

    assert unlock_mod.get_session_token("bitwarden") is None


def test_use_within_the_timeout_keeps_the_unlock_alive(clock):
    assert unlock_mod.store_session_token("bitwarden", "TOKEN")
    for _ in range(3):
        _awake(clock, 29 * _MIN)
        assert unlock_mod.get_session_token("bitwarden") == "TOKEN"

    _awake(clock, 29 * _MIN)
    assert unlock_mod.is_unlocked("bitwarden")  # a status probe does not refresh the timer
    _awake(clock, 2 * _MIN)
    assert not unlock_mod.is_unlocked("bitwarden")

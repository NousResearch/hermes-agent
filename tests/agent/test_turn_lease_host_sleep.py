"""Session turn lease across a host sleep (#144503).

Reproduces a macOS incident: the host slept 922 s mid-turn (pmset Sleep 03:42:31 -> DarkWake
03:57:53), and the first refresh tick after wake logged "stayed locked through its lifetime" and
hard-interrupted a turn whose lease row nobody had taken. Real SessionDB; the wall clock (row
expiry stamps, reclaim) jumps 922 s while the awake clock (``time.monotonic`` on macOS, which
stops during sleep) moves only 5 s.
"""
import time
from types import SimpleNamespace

import pytest

import hermes_state
import hermes_state_compression
from agent import sleep_clock, turn_facade_lease
from agent.turn_facade_lease import LEASE_TTL_SECONDS, DurableTurnLease

SLEEP_WALL_S = 922.0
SLEEP_AWAKE_S = 5.0


class _Clocks:
    def __init__(self):
        self.wall = 1000.0
        self.awake = time.monotonic()

    def sleep_host(self):
        self.wall += SLEEP_WALL_S
        self.awake += SLEEP_AWAKE_S


@pytest.fixture
def leased(tmp_path, monkeypatch):
    clocks = _Clocks()
    fake_time = SimpleNamespace(time=lambda: clocks.wall, monotonic=time.monotonic, sleep=time.sleep)
    monkeypatch.setattr(turn_facade_lease, "time", fake_time)
    monkeypatch.setattr(hermes_state_compression, "time", fake_time)
    monkeypatch.setattr(sleep_clock, "time", fake_time)
    # raising=False: the pre-fix module has no awake clock; it must then fail on behaviour.
    monkeypatch.setattr(turn_facade_lease, "awake_seconds", lambda: clocks.awake, raising=False)

    db = hermes_state.SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="test")
    assert db.try_acquire_session_turn_lease("s1", "h", ttl_seconds=LEASE_TTL_SECONDS)
    interrupts = []
    agent = SimpleNamespace(session_id="s1", interrupt=lambda msg, **kw: interrupts.append(msg))
    lease = DurableTurnLease(agent, db, "s1", "h", expires_at=db.session_turn_lease_expires_at("s1", "h"))
    lease.turn_active = True
    yield clocks, db, lease, interrupts
    db.close()


def test_first_tick_after_922s_sleep_renews_an_untaken_lease(leased):
    clocks, db, lease, interrupts = leased
    clocks.sleep_host()
    assert db.session_turn_lease_expires_at("s1", "h") < clocks.wall, "the row expired on the wall clock"

    assert lease.refresh_tick() is None
    assert interrupts == [] and lease.interrupt_message is None
    assert lease.stop.is_set() is False
    # The holder-fenced renewal landed: the row is ours again, a full TTL past the wake-up.
    assert db.session_turn_lease_expires_at("s1", "h") >= clocks.wall + LEASE_TTL_SECONDS - 1.0


def test_first_tick_after_922s_sleep_still_stops_when_a_successor_took_the_row(leased):
    clocks, db, lease, interrupts = leased
    clocks.sleep_host()
    # The row expired on the wall clock while the host slept; another turn reclaims it.
    assert db.try_acquire_session_turn_lease("s1", "h2", ttl_seconds=LEASE_TTL_SECONDS)

    assert lease.refresh_tick() is False
    assert len(interrupts) == 1 and lease.interrupt_message == interrupts[0]
    assert db.session_turn_lease_expires_at("s1", "h") is None
    assert db.session_turn_lease_expires_at("s1", "h2") is not None

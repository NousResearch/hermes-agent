"""F4 — stale durable turn leases must be reclaimable while the owning PID is alive.

The 2026-10-23 gateway outage left three ``session_turn_leases`` rows on Telegram
sessions, all held by ``pid=3860160`` — the LIVE gateway process.  Each row's turn had
already ended (or wedged and never unwound), but the PID probe reads "alive" because the
*gateway* is alive, so the rows were unreclaimable until their TTL and every later
inbound on those sessions waited it out.

These tests pin the fix: a turn whose token this process minted and retired is provably
dead even when its PID lives, so the next turn can reclaim the row immediately.
"""
from __future__ import annotations

import os
import threading

import pytest

from agent import turn_liveness_registry as tlr
from agent.turn_liveness_registry import (
    holder_turn_is_dead,
    holder_turn_token,
    register_turn_token,
    retire_turn_token,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    tlr._REGISTRY.reset()
    yield
    tlr._REGISTRY.reset()


def _holder(turn: str, pid: int | None = None) -> str:
    pid = os.getpid() if pid is None else pid
    return f"pid={pid}:pidns=4026531836:turn={turn}:platform=telegram"


class TestHolderTurnTokenExtraction:
    """The holder's turn token must be delimited by the next structured component."""

    def test_extracts_turn_token_before_platform(self):
        holder = _holder("20260803_200414_c8f5f818:20260803_200414_c8f5f818:258c0bde")
        assert holder_turn_token(holder) == "20260803_200414_c8f5f818:20260803_200414_c8f5f818:258c0bde"

    def test_session_id_may_contain_colons(self):
        """A fingerprint-derived session id contains ':' — the token is not colon-delimited."""
        holder = _holder("agent:main:telegram:dm:8943042074:8f2a1b3c:9ab7c8d1")
        assert holder_turn_token(holder) == "agent:main:telegram:dm:8943042074:8f2a1b3c:9ab7c8d1"

    def test_cron_holder_turn_token_stops_at_cron_marker(self):
        holder = (
            "pid=123:pidns=4026531836:turn=cron_d03b9c4f77ff_20261024_005211"
            ":cron:d03b9c4f77ff:9a3a061752524d429f7806e98ad357a3:2b058db2:platform=cron"
        )
        assert holder_turn_token(holder) == "cron_d03b9c4f77ff_20261024_005211"

    def test_unstructured_holder_has_no_token(self):
        assert holder_turn_token("pid=123:legacy") is None
        assert holder_turn_token("") is None
        assert holder_turn_token(None) is None


class TestTurnLivenessRegistry:
    def test_registered_then_retired_token_is_dead(self):
        token = "sess:task:abc12345"
        register_turn_token(token)
        assert not holder_turn_is_dead(_holder(token))
        retire_turn_token(token)
        assert holder_turn_is_dead(_holder(token))

    def test_unknown_token_is_never_dead(self):
        """An unknown token defers to TTL: a false reclaim ends a live turn."""
        assert not holder_turn_is_dead(_holder("someone-elses:turn:9999"))

    def test_active_token_is_not_dead(self):
        token = "sess:task:live"
        register_turn_token(token)
        assert not holder_turn_is_dead(_holder(token))
        assert tlr.turn_token_is_active(token)

    def test_reregistering_clears_retirement(self):
        token = "sess:task:reuse"
        register_turn_token(token)
        retire_turn_token(token)
        assert holder_turn_is_dead(_holder(token))
        register_turn_token(token)
        assert not holder_turn_is_dead(_holder(token))


class TestStateReclaimPredicate:
    """The state layer must reclaim a dead TURN even when its PID is alive."""

    def test_retired_turn_is_stale_despite_live_pid(self):
        from hermes_state import _compression_lock_holder_process_is_dead

        token = "sess:task:dead"
        holder = _holder(token, pid=os.getpid())
        register_turn_token(token)
        # The PID is THIS process — alive — so the PID probe alone cannot reclaim it.
        assert not _compression_lock_holder_process_is_dead(holder)
        retire_turn_token(token)
        assert _compression_lock_holder_process_is_dead(holder)

    def test_live_turn_is_not_stale(self):
        from hermes_state import _compression_lock_holder_process_is_dead

        token = "sess:task:still-running"
        holder = _holder(token, pid=os.getpid())
        register_turn_token(token)
        assert not _compression_lock_holder_process_is_dead(holder)

    def test_foreign_holder_defers_to_ttl(self):
        """A sibling process's holder is never probed against our registry."""
        from hermes_state import _compression_lock_holder_process_is_dead

        holder = _holder("other:process:turn:aaaa", pid=os.getpid())
        assert not _compression_lock_holder_process_is_dead(holder)


class TestRefresherReleasesStaleLease:
    """The refresher must stop renewing a lease whose turn thread is gone."""

    def _make_lease(self, db, session_id, holder):
        from agent.turn_facade_lease import DurableTurnLease

        class _Agent:
            _execution_thread_id = None
            _session_turn_lease_refresh_interval = 60.0
            _last_activity_ts = None
            _turn_liveness_activity_generation = 0

            def _touch_activity(self, *_args, **_kwargs):
                self._last_activity_ts = 1.0

            def _liveness_activity_lock(self):
                import contextlib

                return contextlib.nullcontext()

        agent = _Agent()
        agent._execution_thread_id = threading.current_thread().ident
        return DurableTurnLease(agent, db, session_id, holder)

    def test_live_turn_thread_keeps_renewing(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(tmp_path / "state.db")
        holder = _holder("sess:task:live")
        lease = self._make_lease(db, "sess", holder)
        lease.start()
        try:
            assert lease._turn_thread_is_gone() is False
        finally:
            lease.stop_refresher()
            lease.join_threads()

    def test_dead_turn_thread_releases_lease(self, tmp_path):
        """Once the turn thread is gone the refresher must release the row, not renew it."""
        import sqlite3

        from hermes_state import SessionDB

        db = SessionDB(tmp_path / "state.db")
        holder = _holder("sess:task:wedged")
        lease = self._make_lease(db, "sess", holder)
        lease.start()
        try:
            # Simulate a turn thread that died without unwinding.
            lease.agent._execution_thread_id = 999_999_999
            assert lease._turn_thread_is_gone() is True
            # The refresh tick must release the row rather than renew it.
            lease.refresh_tick()
            with sqlite3.connect(tmp_path / "state.db") as raw:
                n = raw.execute("SELECT COUNT(*) FROM session_turn_leases").fetchone()[0]
            assert n == 0
        finally:
            lease.stop_refresher()
            lease.join_threads()

    def test_unset_thread_id_does_not_claim_death(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(tmp_path / "state.db")
        holder = _holder("sess:task:not-started")
        lease = self._make_lease(db, "sess", holder)
        lease.agent._execution_thread_id = None
        lease.start()
        try:
            assert lease._turn_thread_is_gone() is False
        finally:
            lease.stop_refresher()
            lease.join_threads()

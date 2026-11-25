"""Idle compaction on the resume path: measuring the gap from durable state.

Three causes kept idle compaction from ever firing on the gateway/desktop resume
path:

(1) ``DurableTurnLease.start()`` refreshed ``_last_activity_ts`` to "now" at
    turn entry BEFORE ``_idle_compaction`` measured the gap, so the gap was ~0;
(2) a freshly built agent defaulted ``_last_activity_ts = time.time()``, and the
    gateway evicts cached agents after 1h idle, so every resume started with a
    brand-new clock;
(3) the gateway resets ``_last_activity_ts`` again for every cached agent
    (``GatewayRunner._init_cached_agent_for_turn``), so even a cached agent
    could not express a long idle gap.

The fix keeps the session's last durable activity in its OWN field
(``_durable_last_activity_at``), snapshots the gap in the lease BEFORE the
activity clock is refreshed, and consumes-and-clears that snapshot in the
idle-compaction path. ``_last_activity_ts`` is deliberately NOT seeded: it is
the liveness watchdog's clock, and an old stamp there makes the gateway abandon
the first turn of a resumed session as if it had stalled.

These tests use the real ``AIAgent`` and a real storage handle (no network) and
fail on the unfixed code.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from unittest.mock import patch

from persistence_assembly import open_persistence, profile_scope

from agent.turn_facade_lease import DurableTurnLease

from tests.agent.test_compression_concurrent_fork import _build_agent_with_db
from tests.agent.test_idle_compaction_lock_and_guards import _history, _run_prologue
from tests.fixtures.storage_seed import seed_session

THREE_HOURS = 3 * 3600


def _stale_store(tmp_path: Path, sid: str, *, stale_for: float = THREE_HOURS):
    """A real store whose session row has been idle for ``stale_for`` seconds."""
    store = open_persistence(profile_scope(home=tmp_path))
    seed_session(store, sid, "cli")
    store.sessions.touch_activity(sid, time.time() - stale_for)
    return store


def _prep_seeded_agent(store, sid: str, *, idle_after: int = 7200):
    """Real AIAgent (mock compressor) WITHOUT manually touching the activity clock.

    The idle gap must come from the durable row, so this variant deliberately
    never assigns ``_last_activity_ts`` or ``_durable_last_activity_at``.
    """
    agent = _build_agent_with_db(store, sid)
    agent.compression_enabled = True
    agent.compression_idle_compact_after_seconds = idle_after
    agent.context_compressor.threshold_tokens = 100_000
    agent.context_compressor.summary_target_ratio = 0.20
    agent.context_compressor.protect_first_n = 2
    agent.context_compressor.protect_last_n = 2
    agent.context_compressor.awaiting_real_usage_after_compression = False
    agent.context_compressor.get_active_compression_failure_cooldown = (
        lambda *a, **k: None
    )
    agent._cached_system_prompt = "SYSTEM"
    return agent


def test_durable_activity_stamp_is_seeded_from_the_session_row(tmp_path: Path) -> None:
    """A resumed session remembers when it was last active.

    Without the seed, the idle-compaction path can never observe a gap: the
    agent's own clock starts at "now" on every rebuild.
    """
    sid = "SEED_STALE"
    with _stale_store(tmp_path, sid) as store:
        agent = _build_agent_with_db(store, sid)

        assert agent._durable_last_activity_at is not None
        gap = time.time() - agent._durable_last_activity_at
        assert abs(gap - THREE_HOURS) < 5.0, f"expected ~3h idle gap, got {gap}s"


def test_the_activity_clock_is_never_backdated(tmp_path: Path) -> None:
    """``_last_activity_ts`` must stay fresh: it is the liveness watchdog's clock.

    Backdating it makes the gateway abandon the first turn of a resumed session
    as if it had stalled (the watchdog reads ``seconds_since_activity``).
    """
    sid = "SEED_CLOCK"
    with _stale_store(tmp_path, sid) as store:
        agent = _build_agent_with_db(store, sid)

        assert abs(agent._last_activity_ts - time.time()) < 5.0, (
            "the watchdog clock must NOT be backdated"
        )


def test_the_gateway_watchdog_does_not_abandon_a_resumed_turn(tmp_path: Path) -> None:
    """A resumed session must not trip the gateway's turn-inactivity watchdog.

    Regression for the backdated-clock defect: with ``_last_activity_ts`` seeded
    from the durable row, the watchdog read a 3-hour idle and abandoned the
    first turn of the resumed session.
    """
    from gateway.run import _watch_gateway_turn_inactivity

    sid = "SEED_WATCHDOG"
    with _stale_store(tmp_path, sid) as store:
        agent = _build_agent_with_db(store, sid)

        holder = [agent]
        worker_done = threading.Event()
        timeout_fired = threading.Event()
        cleanup_lock = threading.Lock()

        watcher = threading.Thread(
            target=_watch_gateway_turn_inactivity,
            kwargs={
                "agent_holder": holder,
                "task_id": sid,
                "process_baseline": None,
                # A timeout well above the observation window: the watchdog must not
                # fire because the agent reports a 3-hour idle, while a genuinely
                # stalled turn still reaches its own timeout.
                "timeout": 30.0,
                "worker_done": worker_done,
                "timeout_fired": timeout_fired,
                "cleanup_lock": cleanup_lock,
                "poll_interval": 0.2,
            },
            daemon=True,
        )
        watcher.start()
        try:
            time.sleep(7.0)
        finally:
            worker_done.set()
            watcher.join(timeout=5.0)

        reported = agent.get_activity_summary().get("seconds_since_activity")
        assert reported is not None and reported < 30.0, (
            f"the resumed agent reports {reported}s idle — its watchdog clock was backdated"
        )
        assert not timeout_fired.is_set(), (
            f"the watchdog abandoned a resumed turn: seconds_since_activity={reported}"
        )


def test_durable_seed_tolerates_a_missing_session_row(tmp_path: Path) -> None:
    """A session id with no durable row must not break construction."""
    with open_persistence(profile_scope(home=tmp_path)) as store:
        agent = _build_agent_with_db(store, "NO_SUCH_SESSION")

        assert agent._durable_last_activity_at is None
        assert abs(agent._last_activity_ts - time.time()) < 5.0


def test_idle_compaction_fires_from_the_durable_gap(tmp_path: Path, caplog) -> None:
    """The 3h durable gap alone must make the idle trigger fire.

    Neither ``_last_activity_ts`` nor ``_durable_last_activity_at`` is set by
    hand here: the session row seeded at construction is what makes the idle
    compaction eligible.
    """
    sid = "SEED_FIRES"
    with _stale_store(tmp_path, sid) as store:
        agent = _prep_seeded_agent(store, sid)

        with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None), \
             patch("agent.turn_context._should_run_preflight_estimate",
                   return_value=False), \
             patch("agent.turn_context.estimate_request_tokens_rough",
                   return_value=999_999), \
             patch("agent.turn_context.estimate_messages_tokens_rough",
                   return_value=999_999):
            with caplog.at_level(logging.INFO, logger="agent.turn_context"):
                _run_prologue(agent, _history(), rough_tokens=999_999)

        _idle_records = [
            r for r in caplog.records if r.getMessage().startswith("Idle compaction:")
        ]
        assert _idle_records, (
            f"no 'Idle compaction:' record emitted: "
            f"{[r.getMessage() for r in caplog.records]}"
        )
        agent.context_compressor.compress.assert_called()


def test_lease_snapshots_the_pre_stamp_idle_gap(tmp_path: Path) -> None:
    """The lease captures the gap BEFORE refreshing the activity clock."""
    sid = "SEED_LEASE"
    with _stale_store(tmp_path, sid) as store:
        agent = _build_agent_with_db(store, sid)

        lease = DurableTurnLease(agent, store, sid, "holder")
        try:
            lease.start()
            assert abs(agent._pending_idle_gap_seconds - THREE_HOURS) < 5.0, (
                f"expected ~3h snapshot, got {agent._pending_idle_gap_seconds}"
            )
            # The activity stamp still happened (the liveness watchdog depends on it).
            assert abs(agent._last_activity_ts - time.time()) < 5.0
        finally:
            lease.stop_refresher()
            lease.join_threads()


def test_idle_compaction_consumes_and_clears_the_gap_snapshot(tmp_path: Path) -> None:
    """The snapshot is consumed exactly once and never leaks into a later turn."""
    sid = "SEED_CONSUME"
    with _stale_store(tmp_path, sid) as store:
        agent = _prep_seeded_agent(store, sid)

        with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None), \
             patch("agent.turn_context._should_run_preflight_estimate",
                   return_value=False), \
             patch("agent.turn_context.estimate_request_tokens_rough",
                   return_value=999_999):
            _run_prologue(agent, _history(), rough_tokens=999_999)

        assert agent._pending_idle_gap_seconds is None


def test_a_disabled_trigger_still_clears_the_gap_snapshot(tmp_path: Path) -> None:
    """With idle compaction disabled, the snapshot is consumed, never leaked.

    The consume-and-clear must happen before the trigger guard, or a turn that
    skips idle compaction leaves a stale gap for the next turn.
    """
    sid = "SEED_DISABLED"
    with _stale_store(tmp_path, sid) as store:
        agent = _prep_seeded_agent(store, sid)
        agent.compression_idle_compact_after_seconds = 0
        agent._pending_idle_gap_seconds = 900.0

        with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None), \
             patch("agent.turn_context._should_run_preflight_estimate",
                   return_value=False), \
             patch("agent.turn_context.estimate_request_tokens_rough",
                   return_value=999_999):
            _run_prologue(agent, _history(), rough_tokens=999_999)

        assert agent._pending_idle_gap_seconds is None
        agent.context_compressor.compress.assert_not_called()


def test_a_below_floor_snapshot_is_cleared_and_does_not_fire(tmp_path: Path, caplog) -> None:
    """A snapshot below the floor must be cleared, and must not fire compaction."""
    sid = "SEED_BELOW"
    with _stale_store(tmp_path, sid) as store:
        agent = _prep_seeded_agent(store, sid)
        agent._pending_idle_gap_seconds = 5.0

        with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None), \
             patch("agent.turn_context._should_run_preflight_estimate",
                   return_value=False), \
             patch("agent.turn_context.estimate_request_tokens_rough",
                   return_value=999_999):
            with caplog.at_level(logging.INFO, logger="agent.turn_context"):
                _run_prologue(agent, _history(), rough_tokens=999_999)

        assert agent._pending_idle_gap_seconds is None
        assert not [
            r for r in caplog.records if r.getMessage().startswith("Idle compaction:")
        ], "a below-floor snapshot must not fire idle compaction"
        agent.context_compressor.compress.assert_not_called()

"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect

from gateway.kanban_watchers import GatewayKanbanWatchersMixin

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_gateway_runner_inherits_mixin():
    # Import here so a heavy gateway import only happens if the first test passed.
    from gateway.run import GatewayRunner

    assert issubclass(GatewayRunner, GatewayKanbanWatchersMixin)
    # Each kanban method resolves to the mixin's implementation via the MRO.
    for m in KANBAN_METHODS:
        owner = next(c for c in GatewayRunner.__mro__ if m in c.__dict__)
        assert owner is GatewayKanbanWatchersMixin, (
            f"{m} resolved to {owner.__name__}, expected the mixin"
        )


def test_watcher_loops_are_coroutines():
    # The two long-running watchers are async loops.
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)


def test_singleton_dispatcher_lock_is_exclusive(tmp_path):
    """Only one holder of the dispatcher lock at a time — the backstop that
    stops concurrent dispatchers double reclaiming and corrupting shared
    kanban SQLite index pages under wal_autocheckpoint=0."""
    import os

    from gateway.kanban_watchers import _acquire_singleton_lock, _release_singleton_lock

    lock = tmp_path / "kanban" / ".dispatcher.lock"

    h1, st1 = _acquire_singleton_lock(lock)
    assert st1 == "held" and h1 is not None

    # A second acquire while the first is held must be refused, not granted.
    h2, st2 = _acquire_singleton_lock(lock)
    assert st2 == "contended" and h2 is None

    # Releasing the first lets a fresh acquire succeed (lock is reusable).
    _release_singleton_lock(h1)
    h3, st3 = _acquire_singleton_lock(lock)
    assert st3 == "held" and h3 is not None
    _release_singleton_lock(h3)


# --- stuck-detector: benign decline vs genuine fault ---------------------------
#
# The dispatcher warns "check profile health (venv, PATH, credentials)" when the
# ready queue is non-empty but nothing spawns for N consecutive ticks. That
# condition is ALSO the healthy steady state when the dispatcher DECLINES to
# spawn (every profile at its concurrency cap, an assignee 429-rate-limited, the
# board lock held elsewhere). The detector must only count a tick as "bad" when
# the zero-spawn reflects a real fault — otherwise a large fan-out or a provider
# 429 window mis-fires the warning for hours (observed 2026-07-11, ~2h).

from dataclasses import dataclass, field  # noqa: E402

from gateway.kanban_watchers import _stall_streak_is_bad  # noqa: E402


@dataclass
class _FakeResult:
    """Minimal stand-in for kanban_db.DispatchResult (only the buckets the
    stuck-detector consults)."""

    spawned: list = field(default_factory=list)
    skipped_per_profile_capped: list = field(default_factory=list)
    rate_limited: list = field(default_factory=list)
    respawn_guarded: list = field(default_factory=list)
    skipped_locked: bool = False
    spawn_failed: list = field(default_factory=list)
    auto_blocked: list = field(default_factory=list)


def test_stall_idle_queue_is_not_bad():
    # No spawnable work → never a stall regardless of results.
    assert _stall_streak_is_bad(False, False, [("b", _FakeResult())]) is False


def test_stall_something_spawned_is_not_bad():
    # We spawned this tick → not a stall even with a full queue.
    assert _stall_streak_is_bad(True, True, [("b", _FakeResult(spawned=[("t", "p", "w")]))]) is False


def test_stall_per_profile_cap_is_benign_not_bad():
    # Ready work, zero spawns, but every eligible profile is at its cap → healthy.
    res = _FakeResult(skipped_per_profile_capped=[("t1", "daedalus", 3)])
    assert _stall_streak_is_bad(True, False, [("b", res)]) is False


def test_stall_rate_limited_is_benign_not_bad():
    # Assignee bounced off a provider 429 and the task was released to ready → healthy.
    res = _FakeResult(rate_limited=["t1"])
    assert _stall_streak_is_bad(True, False, [("b", res)]) is False


def test_stall_lock_held_is_benign_not_bad():
    # Another dispatcher process holds the board lock this tick → healthy.
    res = _FakeResult(skipped_locked=True)
    assert _stall_streak_is_bad(True, False, [("b", res)]) is False


def test_stall_respawn_guard_is_benign_not_bad():
    res = _FakeResult(respawn_guarded=[("t1", "recent_success")])
    assert _stall_streak_is_bad(True, False, [("b", res)]) is False


def test_stall_bare_zero_spawn_with_no_reason_IS_bad():
    # Ready work, zero spawns, and NO benign decline to explain it → genuine
    # stall suspect (the true-positive the warning exists to catch: broken
    # venv/PATH/creds that fails silently before the circuit breaker trips).
    assert _stall_streak_is_bad(True, False, [("b", _FakeResult())]) is True


def test_stall_auto_blocked_fault_IS_bad_even_with_benign_sibling():
    # A circuit-breaker auto_block is a real fault and must count even if
    # ANOTHER board this tick declined benignly (cap saturated).
    faulted = _FakeResult(auto_blocked=["t1"])
    capped = _FakeResult(skipped_per_profile_capped=[("t2", "athena", 2)])
    assert _stall_streak_is_bad(True, False, [("b1", faulted), ("b2", capped)]) is True


def test_stall_early_spawn_failure_IS_bad_even_with_benign_sibling():
    # Cross-board masking regression (Greptile #304 P2): board A has an EARLY,
    # pre-circuit-breaker spawn failure (spawn_failed populated, but not yet
    # auto_blocked), while board B is benignly rate-limited the same tick. The
    # benign decline on B must NOT mask the genuine fault on A — spawn_failed is
    # a fault immediately (failure #1), so the tick counts.
    early_fail = _FakeResult(spawn_failed=["t1"])  # not yet auto_blocked
    rate_limited = _FakeResult(rate_limited=["t2"])
    assert _stall_streak_is_bad(True, False, [("A", early_fail), ("B", rate_limited)]) is True


def test_stall_none_results_bare_stall_is_bad():
    # Defensive: a None board result contributes nothing; a bare stall still counts.
    assert _stall_streak_is_bad(True, False, [("b", None)]) is True


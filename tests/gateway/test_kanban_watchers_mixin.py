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
# board lock held elsewhere, a respawn guard cooling a task down). The detector
# must only count a tick as "bad" when the zero-spawn reflects a real fault --
# otherwise a large fan-out or a provider 429 window mis-fires the warning for
# as long as the throttle lasts.

from gateway.kanban_watchers import (  # noqa: E402
    _BENIGN_DECLINE_FIELDS,
    _FAULT_FIELDS,
    _stall_streak_is_bad,
)


def _result(**kw):
    """A real ``DispatchResult`` so the detector is pinned to the real field
    names -- a rename in kanban_db must break these tests, not silently turn
    the detector into a no-op via ``getattr(..., default=None)``."""
    from hermes_cli.kanban_db import DispatchResult

    return DispatchResult(**kw)


def test_stall_detector_consults_only_real_dispatchresult_fields():
    """Invariant: every bucket the detector reads must exist on DispatchResult.

    ``_stall_streak_is_bad`` reads its buckets with ``getattr(res, name, None)``
    so a stale name degrades silently to "never fires". Pin the contract.
    """
    from hermes_cli.kanban_db import DispatchResult

    fields = set(DispatchResult().__dataclass_fields__)
    missing = [n for n in (_BENIGN_DECLINE_FIELDS + _FAULT_FIELDS) if n not in fields]
    assert not missing, f"stuck-detector reads fields absent from DispatchResult: {missing}"


def test_stall_idle_queue_is_not_bad():
    # No spawnable work -> never a stall regardless of results.
    assert _stall_streak_is_bad(False, False, [("b", _result())]) is False


def test_stall_something_spawned_is_not_bad():
    # We spawned this tick -> not a stall even with a full queue.
    res = _result(spawned=[("t", "p", "/w")])
    assert _stall_streak_is_bad(True, True, [("b", res)]) is False


def test_stall_per_profile_cap_is_benign_not_bad():
    # Ready work, zero spawns, but every eligible profile is at its
    # kanban.max_in_progress_per_profile cap -> healthy, self-clearing.
    res = _result(skipped_per_profile_capped=[("t1", "worker-a", 3)])
    assert _stall_streak_is_bad(True, False, [("b", res)]) is False


def test_stall_rate_limited_is_benign_not_bad():
    # Assignee bounced off a provider 429/quota wall and the task was released
    # back to ready WITHOUT counting a failure -> healthy, self-clearing.
    res = _result(rate_limited=["t1"])
    assert _stall_streak_is_bad(True, False, [("b", res)]) is False


def test_stall_lock_held_is_benign_not_bad():
    # Another dispatcher process holds the board's dispatch lock this tick;
    # the lock holder is making progress on the same board -> healthy.
    res = _result(skipped_locked=True)
    assert _stall_streak_is_bad(True, False, [("b", res)]) is False


def test_stall_respawn_guard_is_benign_not_bad():
    # A respawn guard is cooling a task down -> healthy, self-clearing.
    res = _result(respawn_guarded=[("t1", "recent_success")])
    assert _stall_streak_is_bad(True, False, [("b", res)]) is False


def test_stall_bare_zero_spawn_with_no_reason_IS_bad():
    # Ready work, zero spawns, and NO benign decline to explain it -> genuine
    # stall suspect. This is the true positive the warning exists to catch, and
    # it must keep firing.
    assert _stall_streak_is_bad(True, False, [("b", _result())]) is True


def test_stall_auto_blocked_fault_IS_bad_even_with_benign_sibling():
    # A circuit-breaker auto_block is a real fault and must count even when
    # ANOTHER board declined benignly (cap saturated) the same tick.
    faulted = _result(auto_blocked=["t1"])
    capped = _result(skipped_per_profile_capped=[("t2", "worker-b", 2)])
    assert _stall_streak_is_bad(True, False, [("b1", faulted), ("b2", capped)]) is True


def test_stall_early_spawn_failure_IS_bad_even_with_benign_sibling():
    # Cross-board masking: board A has an EARLY, pre-circuit-breaker spawn
    # failure (spawn_failed populated but failure_limit not yet reached, so
    # auto_blocked is still empty) while board B is benignly rate-limited the
    # same tick. The benign decline on B must NOT mask the genuine fault on A.
    early_fail = _result(spawn_failed=["t1"])  # not yet auto_blocked
    rate_limited = _result(rate_limited=["t2"])
    assert _stall_streak_is_bad(True, False, [("A", early_fail), ("B", rate_limited)]) is True


def test_stall_none_results_bare_stall_is_bad():
    # Defensive: a None board result contributes nothing; a bare stall counts.
    assert _stall_streak_is_bad(True, False, [("b", None)]) is True

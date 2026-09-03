"""The dispatcher's "stuck" alarm must not fire on a fleet that is merely busy.

`has_spawnable_ready()` answers "is there a ready+assigned+unclaimed task whose
assignee is a real profile?". The dispatcher applies three further gates before
it spawns anything -- the per-profile cap, the host-level `max_in_progress` cap,
and the respawn guard. A fleet running at capacity with a queue behind it is the
design goal, and it satisfies the probe on every tick forever: 591 warnings on
one fleet and 291 on another over ten days, none of them a fault.

The alarm exists to catch a dispatcher that *cannot* spawn -- broken PATH,
missing venv, credential loss. These tests pin the difference.
"""

from __future__ import annotations

from gateway.kanban_watchers import _dispatch_tick_is_stuck


class _Result:
    """Stand-in for DispatchResult: only the buckets the health check reads."""

    def __init__(self, **kw):
        self.skipped_unassigned = kw.get("skipped_unassigned", [])
        self.skipped_nonspawnable = kw.get("skipped_nonspawnable", [])
        self.skipped_per_profile_capped = kw.get("skipped_per_profile_capped", [])
        self.respawn_guarded = kw.get("respawn_guarded", [])
        self.skipped_locked = kw.get("skipped_locked", False)
        self.memory_pressure = kw.get("memory_pressure", None)


def _call(**over):
    kw = dict(
        ready=True,
        spawned_any=False,
        results=[("b", _Result())],
        running_total=0,
        running_by_assignee={},
        ready_assignees={"dev"},
        max_in_progress=None,
        max_in_progress_per_profile=None,
    )
    kw.update(over)
    return _dispatch_tick_is_stuck(**kw)


def test_ready_work_and_no_spawn_with_no_explanation_is_stuck():
    """The case the alarm exists for: nothing spawned, nothing explains it."""
    assert _call() is True


def test_nothing_ready_is_not_stuck():
    assert _call(ready=False) is False


def test_a_spawn_this_tick_is_not_stuck():
    assert _call(spawned_any=True) is False


def test_per_profile_cap_bucket_is_not_stuck():
    """The dispatcher said why it deferred; that is a working dispatcher."""
    res = _Result(skipped_per_profile_capped=[("t_1", "dev", 2)])
    assert _call(results=[("b", res)]) is False


def test_respawn_guard_is_not_stuck():
    """`active_pr`: work is in flight, re-spawning would duplicate it."""
    res = _Result(respawn_guarded=[("t_1", "active_pr")])
    assert _call(results=[("b", res)]) is False


def test_control_plane_lane_is_not_stuck():
    res = _Result(skipped_nonspawnable=["t_1"])
    assert _call(results=[("b", res)]) is False


def test_host_cap_saturation_is_not_stuck():
    """`max_in_progress` makes dispatch_once return before it buckets anything,
    so saturation has to be recognised from the running count."""
    assert _call(running_total=2, max_in_progress=2) is False


def test_host_cap_with_headroom_is_still_stuck():
    assert _call(running_total=1, max_in_progress=2) is True


def test_every_ready_assignee_at_its_per_profile_cap_is_not_stuck():
    assert _call(
        ready_assignees={"dev", "ava-dev-ic"},
        running_by_assignee={"dev": 2, "ava-dev-ic": 2},
        max_in_progress_per_profile=2,
    ) is False


def test_one_ready_assignee_with_headroom_is_stuck():
    """Do not suppress the alarm because *some* profile is busy."""
    assert _call(
        ready_assignees={"dev", "designer-ic"},
        running_by_assignee={"dev": 2, "designer-ic": 0},
        max_in_progress_per_profile=2,
    ) is True


def test_unassigned_ready_work_is_still_stuck():
    """An unassigned ready task is operator-actionable, not a benign deferral."""
    res = _Result(skipped_unassigned=["t_1"])
    assert _call(results=[("b", res)]) is True


def test_a_board_that_failed_to_tick_is_ignored():
    """_tick_once_for_board returns None for a corrupt/quarantined board."""
    assert _call(results=[("b", None)]) is True


def test_spawn_gate_pause_is_not_stuck():
    """Forward-compatible with the in-flight kanban.spawn_gate field: a gate
    that says PAUSE has explained the zero, so read it if it is there."""
    res = _Result()
    res.spawn_gated = "pace gate: hold dev until usage drops"
    assert _call(results=[("b", res)]) is False

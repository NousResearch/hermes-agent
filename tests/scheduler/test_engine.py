"""Engine registration, pre/post semantics, logical-clock advance, bounded dispatch."""

import threading

import pytest

from scheduler.engine import SchedulingEngine
from scheduler.runner import BoundedCallbackRunner
from scheduler.tasks import TaskSpec


def _engine(**kwargs):
    return SchedulingEngine(register_builtins=False, **kwargs)


def _spec(name="core.t", period="step", every=1, phase="post", handler=None, **kwargs):
    return TaskSpec(plugin_id="core", name=name, period=period, every=every, phase=phase,
                    handler=handler or (lambda ctx: None), **kwargs)


def test_no_registrations_short_circuits_before_any_session_work():
    engine = _engine()
    assert engine.active is False
    assert engine.turn_pre("s", turn_seq=1) == []
    assert engine.step_pre("s", step_index=1) == []
    assert engine.step_committed("s", step_index=1) == []
    assert engine.has_session("s") is False


def test_post_step_fires_at_the_stride_and_reschedules():
    fired = []
    engine = _engine()
    engine.register(_spec(every=3, handler=lambda ctx: fired.append((ctx.step_seq, ctx.phase))))
    engine.ensure_session("s")
    for _ in range(2):
        engine.step_committed("s", step_index=1)
    assert fired == []
    engine.step_committed("s", step_index=3)
    assert fired == [(3, "post")]
    for _ in range(2):
        engine.step_committed("s", step_index=4)
    assert fired == [(3, "post")]


def test_post_step_phase_never_fires_before_the_commit():
    fired = []
    engine = _engine()
    engine.register(_spec(every=1, handler=lambda ctx: fired.append(ctx.phase)))
    engine.ensure_session("s")
    engine.step_pre("s", step_index=1)
    assert fired == []  # a post task is not a pre task
    engine.step_committed("s", step_index=1)
    assert fired == ["post"]


def test_pre_step_refires_on_retry_and_reschedules_only_on_commit():
    fired = []
    engine = _engine()
    engine.register(_spec(every=2, phase="pre", handler=lambda ctx: fired.append(ctx.step_seq)))
    engine.ensure_session("s")
    engine.step_committed("s", step_index=1)
    engine.step_pre("s", step_index=2)
    engine.step_pre("s", step_index=2)  # refund/retry: same logical step, fires again
    assert fired == [2, 2]
    engine.step_committed("s", step_index=2)
    engine.step_committed("s", step_index=3)
    engine.step_pre("s", step_index=4)
    engine.step_pre("s", step_index=4)
    assert fired == [2, 2, 4, 4]
    engine.step_committed("s", step_index=4)
    engine.step_pre("s", step_index=5)
    assert fired == [2, 2, 4, 4]


def test_turn_clock_is_independent_of_the_step_clock():
    fired = []
    engine = _engine()
    engine.register(_spec(name="core.turn", period="turn", every=2, handler=lambda ctx: fired.append(("turn", ctx.turn_seq))))
    engine.register(_spec(name="core.step", period="step", every=2, handler=lambda ctx: fired.append(("step", ctx.step_seq))))
    engine.ensure_session("s", turn_seq=0, step_seq=0)
    engine.step_committed("s", step_index=1)
    engine.step_committed("s", step_index=2)
    assert fired == [("step", 2)]
    engine.turn_committed("s", turn_seq=1)
    assert fired == [("step", 2)]
    engine.turn_committed("s", turn_seq=2)
    assert fired == [("step", 2), ("turn", 2)]


def test_one_shot_pre_and_post_fire_once():
    fired = []
    engine = _engine()
    engine.register(_spec(every=2, phase="pre", once=True, handler=lambda ctx: fired.append("pre")))
    engine.register(_spec(name="core.p", every=2, once=True, handler=lambda ctx: fired.append("post")))
    engine.ensure_session("s")
    engine.step_committed("s", step_index=1)
    engine.step_pre("s", step_index=2)
    engine.step_pre("s", step_index=2)
    assert fired == ["pre"]
    engine.step_committed("s", step_index=2)
    assert fired == ["pre", "post"]
    for _ in range(4):
        engine.step_committed("s", step_index=3)
    engine.step_pre("s", step_index=8)
    assert fired == ["pre", "post"]


def test_re_registration_replaces_the_period_without_duplicate_firing():
    fired = []
    engine = _engine()
    engine.register(_spec(every=5, handler=lambda ctx: fired.append(ctx.step_seq)))
    engine.register(_spec(every=2, handler=lambda ctx: fired.append(ctx.step_seq)))
    engine.ensure_session("s")
    engine.step_committed("s", step_index=1)
    engine.step_committed("s", step_index=2)
    assert fired == [2]


def test_resumed_session_is_seeded_from_the_durable_counter():
    fired = []
    engine = _engine()
    engine.register(_spec(every=3, handler=lambda ctx: fired.append(ctx.step_seq)))
    engine.ensure_session("s", step_seq=5)
    engine.step_committed("s", step_index=6)
    engine.step_committed("s", step_index=7)
    engine.step_committed("s", step_index=8)
    assert fired == [8]


def test_registration_validation_rejects_bad_specs():
    engine = _engine()
    with pytest.raises(ValueError, match="prefixed"):
        engine.register(TaskSpec(plugin_id="core", name="nope", period="step", every=1, phase="post", handler=lambda c: None))
    with pytest.raises(ValueError, match="every"):
        engine.register(_spec(every=0))
    with pytest.raises(ValueError, match="phase"):
        engine.register(_spec(phase="during"))
    with pytest.raises(ValueError, match="callable"):
        engine.register(TaskSpec(plugin_id="core", name="core.x", period="step", every=1,
                                 phase="post", handler=1))  # type: ignore[arg-type]


def test_registration_caps_tasks_per_plugin():
    engine = _engine(max_tasks_per_plugin=2)
    engine.register(_spec(name="core.a"))
    engine.register(_spec(name="core.b"))
    with pytest.raises(ValueError, match="already registered"):
        engine.register(_spec(name="core.c"))


def test_timeout_abandons_the_handler_and_the_tick_returns():
    gate = threading.Event()
    engine = _engine(handler_timeout_ms=10)
    engine.register(_spec(every=1, handler=lambda ctx: gate.wait(5)))
    engine.ensure_session("s")
    outcomes = engine.step_committed("s", step_index=1)
    assert [o.status for o in outcomes] == ["timeout"]
    gate.set()


def test_sync_handler_error_is_an_outcome_not_an_exception():
    def boom(ctx):
        raise RuntimeError("handler exploded")

    engine = _engine()
    engine.register(_spec(handler=boom))
    engine.ensure_session("s")
    [outcome] = engine.step_committed("s", step_index=1)
    assert (outcome.status, outcome.reason) == ("error", "RuntimeError")


def test_context_is_narrow_and_frozen():
    seen = []
    engine = _engine()
    engine.register(_spec(handler=seen.append))
    engine.ensure_session("s", step_seq=3, profile_name="work", agent_name="planner")
    engine.step_committed("s", step_index=4, total_tokens=12)
    [ctx] = seen
    assert ctx.session_id == "s"
    assert ctx.profile_name == "work" and ctx.agent_name == "planner"
    assert (ctx.step.step_seq, ctx.step.step_index, ctx.step.total_tokens) == (4, 4, 12)
    assert not hasattr(ctx, "agent")


def test_registration_does_not_disturb_an_existing_session():
    fired = []
    engine = _engine()
    engine.ensure_session("s")
    engine.register(_spec(every=1, handler=lambda ctx: fired.append(ctx.step_seq)))
    engine.step_committed("s", step_index=1)
    assert fired == [1]


def test_bounded_runner_suppresses_after_a_timeout():
    gate = threading.Event()
    runner = BoundedCallbackRunner(timeout_s=0.01, suppression_s=60.0)
    key = ("core.t", "s")
    assert runner.run(task_key=key, session_id="s", seq=1, fn=lambda: gate.wait(5)).status == "timeout"
    second = runner.run(task_key=key, session_id="s", seq=2, fn=lambda: 1)
    assert (second.status, second.reason) == ("skipped", "suppressed")
    gate.set()


def test_bounded_runner_caps_abandoned_workers():
    gate = threading.Event()
    runner = BoundedCallbackRunner(timeout_s=0.01, suppression_s=0.0, max_abandoned=1)
    key = ("core.t", "s")
    assert runner.run(task_key=key, session_id="s", seq=1, fn=lambda: gate.wait(5)).status == "timeout"
    capped = runner.run(task_key=key, session_id="s", seq=2, fn=lambda: 1)
    assert (capped.status, capped.reason) == ("skipped", "abandoned_cap")
    gate.set()


def test_bounded_runner_cancels_async_handlers():
    import asyncio

    gate = asyncio.Event()

    async def hang():
        await gate.wait()

    runner = BoundedCallbackRunner(timeout_s=0.01)
    outcome = runner.run(task_key=("core.a", "s"), session_id="s", seq=1, fn=hang)
    assert (outcome.status, outcome.reason) == ("timeout", "cancelled")

    async def ok():
        return 5

    assert runner.run(task_key=("core.a", "s"), session_id="s", seq=2, fn=ok).value == 5

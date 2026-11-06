"""A stale cron idle verdict must not hard-interrupt resumed genuine activity."""

import threading
from types import SimpleNamespace

import pytest

from agent.activity_tracking import ActivityTrackingMixin
from agent.interrupt_control import InterruptControlMixin


class _RealAgent(ActivityTrackingMixin, InterruptControlMixin):
    def __init__(self):
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_interrupt_reason = None
        self._turn_user_intervened = False
        self._hard_interrupt_requested = threading.Event()
        self._execution_thread_id = None
        self._active_children_lock = threading.Lock()
        self._active_children = set()
        self.quiet_mode = True
        self._touch_activity("initial progress")

    def get_activity_summary(self):
        return {"seconds_since_activity": 600.0}


class _LegacyAgent(_RealAgent):
    def interrupt(self, message=None):
        return super().interrupt(message)


@pytest.mark.parametrize("agent_type", [_RealAgent, _LegacyAgent])
def test_progress_before_first_publication_declines_real_interrupt(monkeypatch, agent_type):
    from agent import interrupt_control
    from agent.conversation_compression import CompressionCommitFence
    from cron.scheduler_inactivity import InactivityObservation, raise_inactivity_timeout

    agent = agent_type()
    fence = CompressionCommitFence()
    agent._active_compression_commit_fence = fence
    observation = InactivityObservation(agent)
    assert observation.read_idle() == 600.0
    reserved = threading.Event()
    proceed = threading.Event()
    original_fence = interrupt_control._fence_cancel_before_commit
    results = {}

    def before_publication(target, *, when_in_flight, failure_log):
        if when_in_flight:
            # Real interrupt already reserved the successfully validated generation.
            assert agent._turn_liveness_abort_claim == observation.token[0]
            reserved.set()
            assert proceed.wait(2), "progress was never released"
        return original_fence(target, when_in_flight=when_in_flight, failure_log=failure_log)

    def stop():
        try:
            results["committed"] = raise_inactivity_timeout(agent, "race", 600, observation)
        except BaseException as exc:
            results["error"] = exc
            reserved.set()

    monkeypatch.setattr(interrupt_control, "_fence_cancel_before_commit", before_publication)
    thread = threading.Thread(target=stop)
    thread.start()
    try:
        assert reserved.wait(2), "real interrupt did not reach publication seam"
        if "error" in results:
            raise results["error"]
        agent._touch_activity("progress after reservation")
    finally:
        proceed.set()
        thread.join(2)
    assert not thread.is_alive()
    assert "error" not in results
    assert results["committed"] is False
    assert agent._interrupt_requested is False
    assert agent._interrupt_message is None
    assert agent._tool_interrupt_reason is None
    assert agent._turn_user_intervened is False
    assert not agent._hard_interrupt_requested.is_set()
    assert not getattr(agent, "_interrupt_thread_signal_pending", False)
    assert fence.begin_commit() is True


def test_inherited_tracking_with_legacy_interrupt_stops_genuine_idle():
    from cron.scheduler_inactivity import InactivityObservation, raise_inactivity_timeout

    agent = _LegacyAgent()
    observation = InactivityObservation(agent)
    observation.read_idle()
    with pytest.raises(TimeoutError, match="idle for 600s"):
        raise_inactivity_timeout(agent, "legacy", 600, observation)
    assert agent._interrupt_requested is True
    assert agent._hard_interrupt_requested.is_set()
    assert agent._tool_interrupt_reason == "cron inactivity watchdog"


def test_tracking_only_legacy_agent_uses_compatibility_stop():
    from cron.scheduler_inactivity import InactivityObservation, raise_inactivity_timeout

    class LegacyAgent(ActivityTrackingMixin):
        stopped = False

        def get_activity_summary(self):
            return {"seconds_since_activity": 600.0}

        def interrupt(self, message=None):
            self.stopped = True

    agent = LegacyAgent()
    agent._touch_activity("initial progress")
    observation = InactivityObservation(agent)
    observation.read_idle()
    with pytest.raises(TimeoutError, match="idle for 600s"):
        raise_inactivity_timeout(agent, "compatibility", 600, observation)
    assert agent.stopped


@pytest.mark.parametrize("resume_activity", [True, False], ids=["resumed", "true-stall"])
def test_cron_revalidates_idle_observation_before_interrupt(monkeypatch, resume_activity):
    from agent import activity_tracking, session_activity
    from cron import scheduler

    clock = SimpleNamespace(wall=1000.0, mono=1000.0)
    timer = SimpleNamespace(time=lambda: clock.wall, monotonic=lambda: clock.mono)
    started = threading.Event()
    observation_ready = threading.Event()
    consume_observation = threading.Event()
    resume = threading.Event()
    resumed = threading.Event()
    finish = threading.Event()
    sampled_idle = []
    interrupts = []
    worker_state = {}
    watchdog_threads = []
    original_wait = threading.Event.wait
    original_watchdog = scheduler._inactivity_watchdog_loop

    class Agent(activity_tracking.ActivityTrackingMixin):
        def run_conversation(self, *args, **kwargs):
            self._touch_activity("starting new turn")
            started.set()
            assert resume.wait(10), "monitor never released worker"
            if resume_activity:
                self._touch_activity("genuine worker progress")
                resumed.set()
                assert finish.wait(10), "resumed worker was not monitored again"
                return {"final_response": "recovered"}
            assert finish.wait(10), "true stall was not interrupted"
            return {"final_response": "stopped"}

        def get_activity_summary(self):
            return session_activity.build_activity_snapshot(
                last_activity_at=self._last_activity_ts,
                last_activity_description=self._last_activity_desc,
            )

        def interrupt(self, message, **kwargs):
            interrupts.append((message, kwargs))
            finish.set()

    agent = Agent()

    def wait(event, timeout=None):
        if threading.current_thread().name.startswith("cron-inactivity-") and timeout == 5.0:
            assert original_wait(started, 10), "worker never started"
            # Awake time advances equally on both clocks: do not bypass suspend accounting.
            clock.wall += 600.0
            clock.mono += 600.0
            return event.is_set()
        return original_wait(event, timeout)

    def watch(**kwargs):
        watchdog_threads.append(threading.current_thread())
        read_idle = kwargs["get_idle_seconds"]

        def sample():
            idle = read_idle()
            sampled_idle.append(idle)
            return idle

        kwargs["get_idle_seconds"] = sample
        verdict = original_watchdog(**kwargs)
        if len(watchdog_threads) > 1:
            assert interrupts == [], "stale idle observation interrupted resumed activity"
            second_observation.set()
            return verdict
        # Pause the real watchdog's verdict before the scheduler consumes it.
        observation_ready.set()
        assert original_wait(consume_observation, 10), "idle observation was not consumed"
        return verdict

    polls = []
    second_observation = threading.Event()

    def consume(futures, timeout=None, **kwargs):
        polls.append(True)
        if len(polls) > 1:
            assert second_observation.wait(10), "resumed future lost inactivity monitoring"
            for thread in watchdog_threads:
                thread.join(timeout=10)
            return set(), set(futures)
        assert observation_ready.wait(10), "watchdog never observed the idle threshold"
        generation = agent._turn_liveness_activity_generation
        resume.set()
        if resume_activity:
            assert resumed.wait(10), "worker never resumed genuine activity"
            assert agent._turn_liveness_activity_generation > generation
            assert agent.get_activity_summary()["seconds_since_activity"] == 0.0
        consume_observation.set()
        for thread in watchdog_threads:
            thread.join(timeout=10)
            assert not thread.is_alive(), "watchdog did not publish its observation"
        if not resume_activity:
            # Model the scheduler's poll expiring while its worker is still wedged.
            return set(), set(futures)
        return set(), set(futures)

    for module in (activity_tracking, session_activity, scheduler):
        monkeypatch.setattr(module, "time", timer)
    monkeypatch.setattr(threading.Event, "wait", wait)
    monkeypatch.setattr(scheduler, "_cron_inactivity_seconds", lambda: 600.0)
    monkeypatch.setattr(scheduler, "_inactivity_watchdog_loop", watch)
    monkeypatch.setattr(scheduler.concurrent.futures, "wait", consume)
    try:
        args = (agent, "probe", {"schedule": {"kind": "every"}},
                "revalidate", "revalidate", "revalidate", None)
        with pytest.raises(TimeoutError, match=r"idle for 600s \(limit 600s\)"):
            scheduler._run_agent_with_watchdog(*args, worker_state=worker_state)
        assert len(interrupts) == 1
        assert interrupts[0][1]["tool_reason"] == "cron inactivity watchdog"
        if resume_activity:
            assert len(watchdog_threads) == 2, "ongoing future was not monitored again"
        assert sampled_idle and sampled_idle[0] >= 600.0
    finally:
        resume.set()
        finish.set()
        consume_observation.set()
        if "future" in worker_state:
            worker_state["future"].result(timeout=10)
        for thread in watchdog_threads:
            thread.join(timeout=10)

"""Trusted scheduler summaries join the physical Relay scope, not a tool's turn."""

from contextvars import copy_context
from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import pytest

from agent import relay_resource_samples as resources
from agent.relay_resource_cgroup import CgroupSample
from agent.relay_resource_scheduler import SCHEDULER_EVENT, SchedulerToken, SchedulerWindow, project_window
from tests.waiting import wait_until

pytest_plugins = ("tests._fixtures.relay_capture",)
NATIVE_ID = "a" * 64
IDENTITY = ("boot", "container-start", 4, 5, 6, 7)
COUNTS = dict(switch_in=19, switch_out=15, runnable_switch_out=5, nonrunnable_switch_out=10,
              within_container=3, direct_to_other_owned=4, direct_from_other_owned=7,
              to_idle=2, from_idle=1, to_unattributed=6, from_unattributed=8)


def _sample():
    return CgroupSample(
        "available", source_identity=IDENTITY, read_started_ns=999_999_980,
        read_ended_ns=1_000_000_020, cpu_started_ns=999_999_990, cpu_ended_ns=1_000_000_010,
        cpu_usage_us=1000, cpu_user_us=800, cpu_system_us=200,
        memory_started_ns=1_000_000_011, memory_ended_ns=1_000_000_012, memory_current_bytes=4096,
        source_end_unix_ns=101_000_000_020, source_end_monotonic_ns=1_000_000_020,
        clock_mapping_uncertainty_ns=1,
    )


def _token():
    return SchedulerToken("observation", "measurement", "opaque-resource", 1_000_000_010, IDENTITY)


def _window(token=None, **changes):
    token = token or _token()
    start = token.source_start_ns + 1000
    window = SchedulerWindow(
        summary_id="00000000-0000-4000-8000-000000000001", start_monotonic_ns=start, end_monotonic_ns=start + 1_000_000,
        counts=dict(COUNTS), expected_cpus=(0, 1),
        trace_loss={str(cpu): {key: 0 for key in ("overrun", "commit overrun", "dropped events")}
                    for cpu in (0, 1)},
        source_end_unix_ns=start + 101_000_000, source_end_monotonic_ns=start + 1_000_000,
        clock_mapping_uncertainty_ns=100, source_identity_before=token.source_identity,
        source_identity_after=token.source_identity, attribution_verified=True,
    )
    return replace(window, **changes)


class _Owner:
    def __init__(self):
        runtime = SimpleNamespace(_closing=False,
                                  acquire_operation_lease=lambda: SimpleNamespace(release=lambda: None))
        self.scope = SimpleNamespace(closed=False, handle=object(), lock=threading.Lock(),
                                     lease_id="observation", context=copy_context(), runtime=lambda: runtime)


@pytest.fixture
def collector(monkeypatch):
    samplers, observations, owners, gates = [], [], [], []

    def make(reader=lambda *_: _sample(), *, native=False, emit=None):
        sampler = resources.ResourceSampler(reader)
        monkeypatch.setattr(sampler, "_run", lambda: None)
        emitted = []
        if not native:
            monkeypatch.setattr(sampler, "_emit", lambda _owner, _data: True)
            monkeypatch.setattr(sampler, "_emit_named", emit or
                                (lambda _owner, data, name: (emitted.append((name, data)), True)[1]))
        samplers.append(sampler)

        def observe(native_id=NATIVE_ID):
            owner = _Owner()
            owners.append(owner)
            observation = resources.ResourceObservation(owner, 1, sampler)
            observations.append(observation)
            observation.start("docker", native_id)
            return observation

        def tick(native_id=NATIVE_ID):
            sampler._sample(sampler.entries[native_id])
            wait_until(lambda: not sampler.pending, what="sample publication completed")

        return SimpleNamespace(sampler=sampler, observe=observe, emitted=emitted, tick=tick, gates=gates)

    yield make
    for gate in gates:
        gate.set()
    for observation in observations:
        observation.finish()
    for sampler in samplers:
        sampler.publisher.shutdown(wait=True)


def test_scheduler_projection_preserves_actual_partitions_and_clock_without_raw_identity():
    token, window = _token(), _window()
    data = project_window(window, token)
    assert data["counts"] == COUNTS
    assert data["events_involving_container"] == 31
    assert data["window_end_unix_ns"] - data["window_start_unix_ns"] == 1_000_000
    assert data["window_start_unix_ns"] == window.start_monotonic_ns + 100_000_000
    assert data["measurement_id"] == token.measurement_id
    assert data["observation_id"] == token.observation_id
    assert data["resource_id"] == token.resource_id
    assert data["coverage"] == "verified_kernel_trace_window"
    assert data["clock_alignment"] == "source_host_only"
    assert data["schema_version"] == "1" and data["status"] == "available"
    assert "source_identity" not in json.dumps(data) and "container-start" not in json.dumps(data)


@pytest.mark.parametrize("changes", [
    {"counts": {**COUNTS, "switch_in": True}}, {"counts": {**COUNTS, "to_idle": -1}},
    {"counts": {key: value for key, value in COUNTS.items() if key != "switch_out"}},
    {"counts": {**COUNTS, "extra": 0}}, {"counts": {**COUNTS, "runnable_switch_out": 6}},
    {"counts": {**COUNTS, "direct_to_other_owned": 5}},
    {"counts": {**COUNTS, "direct_from_other_owned": 8}},
    {"trace_loss": {}}, {"trace_loss": {"0": {"overrun": 0}}},
    {"expected_cpus": (0,), "trace_loss": {"0": {"overrun": 0}}},
    {"expected_cpus": (0,), "trace_loss": {"0": {"overrun": True, "commit overrun": 0, "dropped events": 0}}},
    {"trace_loss": {"0": {"overrun": 1, "commit overrun": 0, "dropped events": 0},
                    "1": {"overrun": 0, "commit overrun": 0, "dropped events": 0}}},
    {"expected_cpus": ()}, {"expected_cpus": (0, 0)}, {"expected_cpus": (False, 1)},
    {"source_identity_before": ("old",)}, {"source_identity_after": ("restarted",)},
    {"source_clock": "global"}, {"attribution_verified": False}, {"clock_alignment": "guessed"},
    {"start_monotonic_ns": True}, {"start_monotonic_ns": 0}, {"end_monotonic_ns": 1},
    {"source_end_monotonic_ns": 1}, {"source_end_unix_ns": 0}, {"clock_mapping_uncertainty_ns": -1},
    {"summary_id": ""}, {"summary_id": "raw-container-or-cgroup-name"},
])
def test_scheduler_projection_rejects_incomplete_or_inconsistent_evidence(changes):
    with pytest.raises(ValueError):
        project_window(_window(**changes), _token())


def test_scheduler_windows_must_be_disjoint_but_adjacent_windows_are_valid():
    window = _window()
    with pytest.raises(ValueError):
        project_window(window, _token(), previous_end_ns=window.start_monotonic_ns + 1)
    assert project_window(window, _token(), previous_end_ns=window.start_monotonic_ns)["counts"] == COUNTS


def test_native_scheduler_window_joins_physical_scope_and_leaves_tool_under_turn(
    relay_capture, monkeypatch, collector,
):
    from agent import relay_tools
    from tools.environment_observability import capture_observer, environment_acquired

    harness = collector(native=True)
    monkeypatch.setattr(resources, "_SAMPLER", harness.sampler)
    monkeypatch.setattr(resources, "configured_interval", lambda: 1)
    lease = relay_capture.acquire()
    turn = relay_capture.begin(lease)
    environment = SimpleNamespace(env_type="docker", _container_id=NATIVE_ID, _docker_exe="docker")
    owners = []

    def execute(_args):
        environment_acquired(environment)
        owners.append(capture_observer(environment))
        observation = owners[0].resources
        assert observation.scheduler_token() is None
        harness.tick()
        token = observation.scheduler_token()
        assert observation.submit_scheduler_window(token, _window(token))
        wait_until(lambda: not harness.sampler.pending, what="native scheduler publication completed")
        return {"output": "ready", "returncode": 0}

    result, _ = relay_tools.execute("terminal", {"command": "true"}, execute, session_id=lease.session_id)
    assert result["returncode"] == 0
    owners[0].finish(reason="test_complete")
    relay_capture.coordinator.end_turn(turn, outcome="success")
    events = relay_capture.events()
    [physical] = [e for e in events if e.get("name") == "hermes.execution_environment.lifecycle_scope"
                  and e.get("scope_category") == "start"]
    [tool] = [e for e in events if e.get("name") == "terminal" and e.get("scope_category") == "start"]
    [sample] = [e for e in events if e.get("name") == resources.RESOURCE_EVENT]
    [scheduler] = [e for e in events if e.get("name") == SCHEDULER_EVENT]
    assert scheduler["parent_uuid"] == sample["parent_uuid"] == physical["uuid"]
    assert tool["parent_uuid"] == str(turn.handle.uuid)
    assert scheduler["data"]["measurement_id"] == sample["data"]["measurement_id"]
    assert scheduler["data"]["resource_id"] == sample["data"]["resource_id"]
    assert scheduler["data"]["counts"] == COUNTS
    [closed] = [e for e in events if e.get("name") == physical["name"] and e.get("scope_category") == "end"]
    summary = closed["data"]["resource_summary"]
    assert summary["scheduler_counts"] == COUNTS and summary["scheduler_windows"] == 1
    assert summary["scheduler_switches_status"] == "available"
    assert NATIVE_ID not in json.dumps(events) and "container-start" not in json.dumps(events)


def test_unavailable_baseline_never_issues_scheduler_token(collector):
    harness = collector(lambda *_: CgroupSample("unverified_target"))
    observation = harness.observe()
    assert observation.scheduler_token() is None
    harness.tick()
    assert observation.scheduler_token() is None
    assert not observation.submit_scheduler_window(_token(), _window())
    assert harness.emitted == []


@pytest.mark.parametrize("invalid", ["observation_id", "measurement_id", "resource_id", "source_identity", "source_start_ns",
                                    "generation", "cutoff", "before_start", "window_none", "window_object"])
def test_stale_mismatched_or_cutoff_windows_are_not_published(collector, invalid):
    harness = collector()
    observation = harness.observe()
    harness.tick()
    token = observation.scheduler_token()
    window = _window(token)

    def replace_generation():
        observation.stop_resource()
        observation.start("docker", NATIVE_ID)
        harness.tick()
        return token, window

    mutators = {
        **{key: lambda key=key: (replace(token, **{key: "wrong"}), window)
           for key in ("observation_id", "measurement_id", "resource_id")},
        "source_identity": lambda: (replace(token, source_identity=("wrong",)), window),
        "source_start_ns": lambda: (replace(token, source_start_ns=token.source_start_ns - 1), window),
        "generation": replace_generation,
        "cutoff": lambda: (observation.finish(), token, window)[1:],
        "before_start": lambda: (token, replace(window, start_monotonic_ns=token.source_start_ns - 1)),
        "window_none": lambda: (token, None), "window_object": lambda: (token, object()),
    }
    token, window = mutators[invalid]()
    assert not observation.submit_scheduler_window(token, window)
    assert harness.emitted == []


def test_concurrent_duplicate_admission_publishes_and_counts_once(collector):
    harness = collector()
    observation = harness.observe()
    harness.tick()
    token = observation.scheduler_token()
    window, gate, accepted = _window(token), threading.Barrier(3), []

    def submit():
        gate.wait(timeout=5)
        accepted.append(observation.submit_scheduler_window(token, window))

    workers = [threading.Thread(target=submit) for _ in range(2)]
    for worker in workers:
        worker.start()
    gate.wait(timeout=5)
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()
    wait_until(lambda: not harness.sampler.pending, what="duplicate admission publication completed")
    assert sorted(accepted) == [False, True]
    assert len(harness.emitted) == 1
    assert not observation.submit_scheduler_window(token, window)
    shifted = replace(window, start_monotonic_ns=window.end_monotonic_ns,
                      end_monotonic_ns=window.end_monotonic_ns + 1000,
                      source_end_monotonic_ns=window.end_monotonic_ns + 1000)
    assert not observation.submit_scheduler_window(token, shifted)
    summary = observation.finish()
    assert summary["scheduler_windows"] == 1 and summary["scheduler_counts"] == COUNTS


def test_busy_publication_rejects_without_waiting_and_same_window_can_retry(collector):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    def emit(_owner, _data, _name):
        entered.set()
        assert release.wait(5)
        return True

    harness = collector(emit=emit)
    harness.gates.append(release)
    observation = harness.observe()
    harness.tick()
    token = observation.scheduler_token()
    first = _window(token)
    assert observation.submit_scheduler_window(token, first)
    assert entered.wait(5)
    second = _window(token, summary_id="00000000-0000-4000-8000-000000000002", start_monotonic_ns=first.end_monotonic_ns,
                     end_monotonic_ns=first.end_monotonic_ns + 1000,
                     source_end_monotonic_ns=first.end_monotonic_ns + 1000)
    answers = []
    worker = threading.Thread(target=lambda: (answers.append(observation.submit_scheduler_window(token, second)),
                                              returned.set()))
    worker.start()
    assert returned.wait(5)
    assert answers == [False] and not release.is_set()
    release.set()
    worker.join(timeout=5)
    wait_until(lambda: not harness.sampler.pending, what="blocked publication released")
    assert observation.submit_scheduler_window(token, second)
    wait_until(lambda: not harness.sampler.pending, what="retried publication completed")
    summary = observation.finish()
    assert summary["scheduler_windows"] == 2
    assert summary["scheduler_counts"] == {key: value * 2 for key, value in COUNTS.items()}

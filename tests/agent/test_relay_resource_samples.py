"""Sampling is opt-in, physical-scope attached, and bounded without blocking cleanup."""

from contextvars import copy_context
import json
import threading
from types import SimpleNamespace

import pytest

from agent import relay_resource_samples as resources
from agent.relay_resource_cgroup import CgroupSample

pytest_plugins = ("tests._fixtures.relay_capture",)


def _sample(index=1):
    end = index * 1_000_000_000
    return CgroupSample(
        "available", source_identity=("boot", "start", 4, 5, 6, 7),
        read_started_ns=end - 20, read_ended_ns=end + 20,
        cpu_started_ns=end - 10, cpu_ended_ns=end + 10,
        cpu_usage_us=index * 1000, cpu_user_us=index * 800, cpu_system_us=index * 200,
        cpu_pressure_some_us=index * 30, cpu_throttled_us=index * 40,
        memory_started_ns=end + 11, memory_ended_ns=end + 12, memory_current_bytes=4096,
        source_end_unix_ns=end + 100_000_000_000, source_end_monotonic_ns=end + 20,
        clock_mapping_uncertainty_ns=1,
    )


class _Owner:
    def __init__(self):
        runtime = SimpleNamespace(
            _closing=False,
            acquire_operation_lease=lambda: SimpleNamespace(release=lambda: None),
        )
        self.scope = SimpleNamespace(
            closed=False, handle=object(), context=copy_context(), runtime=lambda: runtime,
        )


@pytest.fixture
def collector(monkeypatch):
    """Drive scheduled ticks explicitly; retain the production publication executor."""
    samplers, observations, owners, gates = [], [], [], []

    def make(reader=lambda *_: _sample(), *, emit=None, native=False):
        sampler = resources.ResourceSampler(reader)
        monkeypatch.setattr(sampler, "_run", lambda: None)
        emitted = []
        if not native:
            monkeypatch.setattr(sampler, "_emit", emit or (lambda _owner, data: (emitted.append(data), True)[1]))
        samplers.append(sampler)

        def observe(native_id):
            owner = _Owner()
            owners.append(owner)
            observation = resources.ResourceObservation(owner, 1, sampler)
            observations.append(observation)
            observation.start("docker", native_id)
            return observation

        return SimpleNamespace(
            sampler=sampler, observe=observe, emitted=emitted, gates=gates,
            tick=lambda native_id: sampler._sample(sampler.entries[native_id]),
        )

    yield make
    for gate in gates:
        gate.set()
    for observation in observations:
        observation.finish()
    for sampler in samplers:
        sampler.publisher.shutdown(wait=True)


@pytest.mark.parametrize("settings, interval", [
    ({}, None), ({"enabled": False}, None), ({"enabled": True}, 2.0),
    ({"enabled": True, "interval_seconds": 1}, 1.0),
    ({"enabled": True, "interval_seconds": 60}, 60.0),
    ({"enabled": True, "interval_seconds": True}, None),
    ({"enabled": True, "interval_seconds": 0}, None),
    ({"enabled": True, "interval_seconds": 61}, None),
    ({"enabled": True, "interval_seconds": "nan"}, None),
])
def test_sampling_uses_explicit_profile_configuration(tmp_path, monkeypatch, settings, interval):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {"gateway": {"telemetry": {"execution_environment": {"resource_sampling": settings}}}}
    (tmp_path / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    assert resources.configured_interval() == interval
    if interval is None:
        assert resources.observe_resources(_Owner()) is None


def test_native_sample_attaches_to_physical_scope_without_changing_tool_parent(
    relay_capture, monkeypatch, collector,
):
    from agent import relay_tools
    from tools.environment_observability import capture_observer, environment_acquired, record_transition

    readings = iter([_sample(1), _sample(2)])
    harness = collector(lambda *_: next(readings), native=True)
    monkeypatch.setattr(resources, "_SAMPLER", harness.sampler)
    monkeypatch.setattr(resources, "configured_interval", lambda: 1)
    lease = relay_capture.acquire()
    turn = relay_capture.begin(lease)
    environment = SimpleNamespace(env_type="docker", _container_id="a" * 64, _docker_exe="docker")
    owners = []

    def execute(_args):
        environment_acquired(environment)
        owners.append(capture_observer(environment))
        record_transition(environment, action="reuse", outcome="succeeded",
                          container_id=environment._container_id, state_after="running", observer=owners[0])
        for _ in range(2):
            harness.tick(environment._container_id)
            harness.sampler.entries[environment._container_id].pending.result(timeout=5)
        return {"output": "ready", "returncode": 0}

    result, _ = relay_tools.execute("terminal", {"command": "true"}, execute, session_id=lease.session_id)
    assert result["returncode"] == 0
    owners[0].finish(reason="test_complete")
    relay_capture.coordinator.end_turn(turn, outcome="success")
    events = relay_capture.events()
    [physical] = [e for e in events if e.get("name") == "hermes.execution_environment.lifecycle_scope"
                  and e.get("scope_category") == "start"]
    [tool] = [e for e in events if e.get("name") == "terminal" and e.get("scope_category") == "start"]
    samples = [e for e in events if e.get("name") == resources.RESOURCE_EVENT]
    assert [sample["parent_uuid"] for sample in samples] == [physical["uuid"], physical["uuid"]]
    assert tool["parent_uuid"] == str(turn.handle.uuid)
    assert [sample["data"]["cpu_interval_status"] for sample in samples] == ["baseline", "available"]
    assert {sample["data"]["scheduler_switches_status"] for sample in samples} == {"not_collected"}
    [closed] = [e for e in events if e.get("name") == "hermes.execution_environment.lifecycle_scope"
                and e.get("scope_category") == "end"]
    summary = closed["data"]["resource_summary"]
    assert summary["cpu_usage_us"] == 1000
    assert summary["samples"] == 2 and summary["omitted_publications"] == 0
    assert environment._container_id not in json.dumps(events)


def test_finish_does_not_wait_for_reader_and_discards_inflight_result(collector):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    def reader(*_args):
        entered.set()
        assert release.wait(5)
        return _sample()

    harness = collector(reader)
    harness.gates.append(release)
    observation = harness.observe("a" * 64)
    worker = threading.Thread(target=lambda: (harness.tick("a" * 64), returned.set()))
    worker.start()
    assert entered.wait(5)
    summary = observation.finish()
    assert summary["samples"] == 0
    assert not release.is_set()
    release.set()
    assert returned.wait(5)
    worker.join(timeout=5)
    assert harness.emitted == []


def test_detached_scheduled_tick_never_reads_removed_resource(collector):
    reads = []
    harness = collector(lambda *_args: (reads.append("read"), _sample())[1])
    observation = harness.observe("a" * 64)
    scheduled = harness.sampler.entries["a" * 64]
    observation.stop_resource()
    harness.sampler._sample(scheduled)
    assert reads == []


def test_blocked_publication_has_one_pending_sample_and_does_not_delay_finish(collector):
    entered, release = threading.Event(), threading.Event()
    emitted = []

    def emit(_owner, data):
        entered.set()
        assert release.wait(5)
        emitted.append(data)
        return True

    harness = collector(emit=emit)
    harness.gates.append(release)
    observation = harness.observe("a" * 64)
    harness.tick("a" * 64)
    assert entered.wait(5)
    harness.tick("a" * 64)
    harness.tick("a" * 64)
    summary = observation.finish()
    assert summary["samples"] == 3 and summary["omitted_publications"] == 2
    assert not release.is_set()
    release.set()
    harness.sampler.publisher.shutdown(wait=True)
    assert [data["sequence"] for data in emitted] == [1]


def test_publications_are_globally_bounded_across_distinct_resources(collector):
    release = threading.Event()
    emitted = []

    def emit(_owner, data):
        assert release.wait(5)
        emitted.append(data)
        return True

    harness = collector(emit=emit)
    harness.gates.append(release)
    observations = [harness.observe(f"{index:064x}") for index in range(1, 18)]
    for observation in observations:
        harness.tick(observation.native_id)
    omitted = observations[-1].finish()
    assert omitted["omitted_publications"] == 1
    release.set()
    harness.sampler.publisher.shutdown(wait=True)
    assert len(emitted) == 16
    assert len({data["resource_id"] for data in emitted}) == 16


def test_resource_limit_refuses_extra_sampling_and_releases_capacity_on_detach(collector):
    reads = []
    harness = collector(lambda _exe, native_id: (reads.append(native_id), _sample())[1])
    observations = [harness.observe(f"{index:064x}") for index in range(1, 129)]
    refused = harness.observe("f" * 64)
    assert refused.finish() == {"status": "resource_limit", "coverage": "unavailable"}
    assert reads == []
    observations[0].finish()
    replacement = harness.observe("e" * 64)
    harness.tick(replacement.native_id)
    harness.sampler.entries[replacement.native_id].pending.result(timeout=5)
    assert replacement.finish()["samples"] == 1
    assert reads == ["e" * 64]


def test_same_container_observers_share_one_stream_without_duplicate_totals(collector):
    readings = iter(_sample(index) for index in range(1, 5))
    harness = collector(lambda *_: next(readings))
    first, second = harness.observe("a" * 64), harness.observe("a" * 64)
    successor = harness.observe("a" * 64)
    for _ in range(2):
        harness.tick("a" * 64)
        harness.sampler.entries["a" * 64].pending.result(timeout=5)
    shared, total = second.finish(), first.finish()
    assert shared["status"] == "shared_measurement"
    assert "cpu_usage_us" not in shared
    assert shared["measurement_id"] == total["measurement_id"]
    assert total["cpu_usage_us"] == 1000
    for _ in range(2):
        harness.tick("a" * 64)
        harness.sampler.entries["a" * 64].pending.result(timeout=5)
    later = successor.finish()
    assert later["measurement_id"] != total["measurement_id"]
    assert later["cpu_usage_us"] == 1000
    assert [data["sequence"] for data in harness.emitted] == [1, 2, 1, 2]


@pytest.mark.parametrize("retire", [False, True])
def test_pending_publication_stays_bounded_across_handoff_or_reattach(collector, retire):
    entered, release = threading.Event(), threading.Event()
    emitted = []

    def emit(_owner, data):
        entered.set()
        assert release.wait(5)
        emitted.append(data)
        return True

    harness = collector(emit=emit)
    harness.gates.append(release)
    first = harness.observe("a" * 64)
    second = None if retire else harness.observe("a" * 64)
    harness.tick("a" * 64)
    assert entered.wait(5)
    pending = harness.sampler.entries["a" * 64].pending
    assert first.finish()["pending_publications"] == 1
    if retire:
        second = harness.observe("a" * 64)
    harness.tick("a" * 64)
    release.set()
    assert pending.result(timeout=5) is True
    harness.tick("a" * 64)
    harness.sampler.entries["a" * 64].pending.result(timeout=5)
    summary = second.finish()
    assert summary["omitted_publications"] == 1
    assert summary["samples"] == 2
    assert [data["sequence"] for data in emitted] == [1, 2]
    assert len({data["measurement_id"] for data in emitted}) == 2

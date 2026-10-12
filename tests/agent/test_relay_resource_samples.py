"""External measurements are source-bound and asynchronously attached to physical scopes."""

from contextvars import ContextVar, copy_context
from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import pytest

from agent import relay_resource_samples as resources
from agent.relay_resource_metrics import ResourceWindow, project_resource_window
from agent.relay_resource_scheduler import SCHEDULER_EVENT
from tests.agent.test_relay_resource_scheduler import COUNTS, IDENTITY, _token, _window
from tests.waiting import wait_until
from tools.daemon_pool import DaemonThreadPoolExecutor
from tools.execution_observability import opaque_docker_resource_id

pytest_plugins = ("tests._fixtures.relay_capture",)
NATIVE_ID = "a" * 64


def _resource(token=None, **changes):
    token = token or _token()
    start = token.source_start_ns + 1000
    return replace(ResourceWindow(
        start_monotonic_ns=start, end_monotonic_ns=start + 1_000_000, cpu_usage_us=1000,
        source_end_unix_ns=start + 101_000_000, source_end_monotonic_ns=start + 1_000_000,
        clock_mapping_uncertainty_ns=100, source_identity_before=token.source_identity,
        source_identity_after=token.source_identity, memory_current_bytes=4096, cpu_pressure_us=30,
    ), **changes)


class _Owner:
    def __init__(self, native_id, emit):
        self.lock, self.resources, self.leases = threading.Lock(), None, 0
        self._measurement_resource_id = opaque_docker_resource_id(native_id)
        self._measurements_closed = False

        def acquire():
            self.leases += 1

            def release():
                self.leases -= 1

            return SimpleNamespace(release=release)

        self.runtime = SimpleNamespace(_closing=False, acquire_operation_lease=acquire,
                                       relay=SimpleNamespace(scope=SimpleNamespace(event=emit)))
        self.scope = SimpleNamespace(closed=False, handle=object(), lock=threading.Lock(),
                                     lease_id="observation", context=copy_context(), runtime=lambda: self.runtime)


@pytest.fixture
def publisher(monkeypatch):
    executor = DaemonThreadPoolExecutor(max_workers=1, thread_name_prefix="test-external-measurements")
    monkeypatch.setattr(resources, "_publisher", lambda: executor)
    monkeypatch.setattr(resources, "_PUBLICATION_SLOTS", threading.BoundedSemaphore(2))
    environments, gates, events = [], [], []

    def observe(native_id=NATIVE_ID, emit=None):
        owner = _Owner(native_id, emit or (lambda name, **fields: events.append((name, fields))))
        environment = SimpleNamespace(_container_id=native_id, _relay_lifecycle_observer=owner)
        environments.append(environment)
        observation = resources.attach_measurements(environment, source_identity=IDENTITY, source_start_ns=1_000_000_010)
        return environment, observation

    yield SimpleNamespace(observe=observe, gates=gates, events=events)
    for gate in gates:
        gate.set()
    for environment in environments:
        environment._relay_lifecycle_observer.resources.finish()
    executor.shutdown(wait=True)


def test_resource_projection_preserves_interval_not_lifetime_totals():
    window = _resource()
    data = project_resource_window(window, _token())
    assert data["cpu_usage_delta_us"] == 1000 and data["average_cpu_cores"] == 1
    assert data["memory_current_bytes"] == 4096 and data["cpu_pressure_us"] == 30
    assert data["window_duration_ns"] == 1_000_000 and data["clock_alignment"] == "source_host_only"
    assert "container-start" not in json.dumps(data)
    with pytest.raises(ValueError):
        project_resource_window(window, _token(), previous_end_ns=window.start_monotonic_ns + 1)


@pytest.mark.parametrize("changes", [
    {"cpu_usage_us": True}, {"cpu_usage_us": -1}, {"memory_current_bytes": -1},
    {"source_identity_before": ("old",)}, {"source_identity_after": ("new",)},
    {"start_monotonic_ns": 0}, {"end_monotonic_ns": 1}, {"source_end_monotonic_ns": 1},
    {"clock_alignment": "guessed"},
])
def test_resource_window_rejects_invalid_measurements(changes):
    with pytest.raises(ValueError):
        project_resource_window(_resource(**changes), _token())


def test_binding_is_immutable_and_never_retargets_a_physical_observation(publisher):
    environment, observation = publisher.observe()
    bind = lambda **kwargs: resources.attach_measurements(environment, **kwargs)
    assert bind(source_identity=IDENTITY, source_start_ns=observation.token.source_start_ns) is observation
    assert bind(source_identity=("other", *IDENTITY[1:]), source_start_ns=observation.token.source_start_ns) is None
    assert bind(source_identity=IDENTITY, source_start_ns=observation.token.source_start_ns - 1) is None
    with pytest.raises(AttributeError):
        observation.token = replace(observation.token, resource_id="other")
    assert not observation.submit_resource_window(replace(observation.token, measurement_id="wrong"), _resource(observation.token))
    assert not observation.submit_resource_window(observation.token, None)
    assert not observation.submit_scheduler_window(observation.token, object())
    environment._container_id = "b" * 64
    assert bind(source_identity=IDENTITY, source_start_ns=observation.token.source_start_ns) is None
    environment._container_id = NATIVE_ID
    environment._relay_lifecycle_observer._measurement_resource_id = None
    assert bind(source_identity=IDENTITY, source_start_ns=observation.token.source_start_ns) is None
    environment._relay_lifecycle_observer._measurement_resource_id = observation.token.resource_id
    observation.finish()
    assert bind(source_identity=IDENTITY, source_start_ns=observation.token.source_start_ns) is observation and not observation.active
    assert bind(source_identity=IDENTITY, source_start_ns=observation.token.source_start_ns + 1) is None


def test_native_measurements_keep_original_physical_parent_and_stop_invalidates(
    relay_capture, publisher,
):
    from agent import relay_tools
    from tools.environment_observability import capture_observer, environment_acquired, record_transition

    lease = relay_capture.acquire()
    turn = relay_capture.begin(lease)
    environment = SimpleNamespace(env_type="docker", _container_id=NATIVE_ID, _docker_exe="docker")
    owners = []

    def execute(_args):
        environment_acquired(environment)
        owners.append(capture_observer(environment))
        assert owners[0].resources is None
        observation = resources.attach_measurements(environment, source_identity=IDENTITY, source_start_ns=1_000_000_010)
        assert observation.submit_resource_window(observation.token, _resource(observation.token))
        wait_until(lambda: observation._pending_publications == 0, what="native resource event")
        assert observation.submit_scheduler_window(observation.token, _window(observation.token))
        wait_until(lambda: observation._pending_publications == 0, what="native scheduler event")
        record_transition(environment, action="stop", outcome="requested", container_id=NATIVE_ID, observer=owners[0])
        assert not observation.submit_scheduler_window(observation.token, _window(observation.token))
        assert resources.attach_measurements(environment, source_identity=IDENTITY, source_start_ns=1_000_000_010) is None
        return {"output": "ready", "returncode": 0}

    result, _ = relay_tools.execute("terminal", {"command": "true"}, execute, session_id=lease.session_id)
    assert result["returncode"] == 0
    owners[0].finish(reason="test_complete")
    assert resources.attach_measurements(environment, source_identity=IDENTITY, source_start_ns=1_000_000_010) is None
    relay_capture.coordinator.end_turn(turn, outcome="success")
    events = relay_capture.events()
    [physical] = [e for e in events if e.get("name") == "hermes.execution_environment.lifecycle_scope" and e.get("scope_category") == "start"]
    [tool] = [e for e in events if e.get("name") == "terminal" and e.get("scope_category") == "start"]
    measurements = [e for e in events if e.get("name") in (resources.RESOURCE_EVENT, SCHEDULER_EVENT)]
    assert len(measurements) == 2 and {e["parent_uuid"] for e in measurements} == {physical["uuid"]}
    assert tool["parent_uuid"] == str(turn.handle.uuid)
    assert len({e["data"]["measurement_id"] for e in measurements}) == 1
    [closed] = [e for e in events if e.get("name") == physical["name"] and e.get("scope_category") == "end"]
    summary = closed["data"]["resource_summary"]
    assert summary["scheduler_counts"] == COUNTS and summary["cpu_usage_us"] == 1000
    assert summary["publication_semantics"] == "admitted_not_export_confirmed"
    assert NATIVE_ID not in json.dumps(events) and "container-start" not in json.dumps(events)


def test_concurrent_scheduler_admission_deduplicates_without_double_counting(publisher):
    _, observation = publisher.observe()
    window, barrier, answers = _window(observation.token), threading.Barrier(3), []

    def submit():
        barrier.wait(timeout=5)
        answers.append(observation.submit_scheduler_window(observation.token, window))

    workers = [threading.Thread(target=submit) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait(timeout=5)
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()
    wait_until(lambda: observation._pending_publications == 0, what="concurrent publication")
    assert sorted(answers) == [False, True]
    shifted = replace(window, start_monotonic_ns=window.end_monotonic_ns,
                      end_monotonic_ns=window.end_monotonic_ns + 1000,
                      source_end_monotonic_ns=window.end_monotonic_ns + 1000)
    assert not observation.submit_scheduler_window(observation.token, shifted)
    summary = observation.finish()
    assert summary["scheduler_windows"] == 1 and summary["scheduler_counts"] == COUNTS
    assert len(publisher.events) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_bounded_publication_finish_close_and_export_failure_release_leases(publisher, fail):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    marker = ContextVar("measurement_owner", default="unrelated_caller")
    publisher.gates.append(release)

    def emit(_name, **_fields):
        assert marker.get() == "physical_owner"
        entered.set()
        assert release.wait(5)
        if fail:
            raise RuntimeError("export failed")

    binding = marker.set("physical_owner")
    first_env, first = publisher.observe(emit=emit)
    marker.reset(binding)
    second_env, second = publisher.observe("b" * 64)
    third_env, third = publisher.observe("c" * 64)
    assert first.submit_resource_window(first.token, _resource(first.token))
    assert entered.wait(5)
    assert not first.submit_scheduler_window(first.token, _window(first.token))
    assert second.submit_resource_window(second.token, _resource(second.token))
    answers = []
    worker = threading.Thread(target=lambda: (answers.append(third.submit_resource_window(third.token, _resource(third.token))), returned.set()))
    worker.start()
    assert returned.wait(5) and answers == [False] and not release.is_set()
    snapshot = first.finish()
    assert snapshot["resource_windows"] == 1 and snapshot["pending_publications"] == 1
    assert snapshot["completed_publications"] == 0
    second_env._relay_lifecycle_observer.scope.closed = True
    release.set()
    worker.join(timeout=5)
    wait_until(lambda: first._pending_publications == second._pending_publications == 0
               and first_env._relay_lifecycle_observer.leases == second_env._relay_lifecycle_observer.leases == 0,
               what="leases released after failure or queued close")
    assert publisher.events == []
    assert first.finish()["failed_publications"] == int(fail)
    assert not first.submit_resource_window(first.token, _resource(first.token))
    assert third.submit_resource_window(third.token, _resource(third.token))
    wait_until(lambda: third._pending_publications == third_env._relay_lifecycle_observer.leases == 0,
               what="capacity reusable")
    assert len(publisher.events) == 1

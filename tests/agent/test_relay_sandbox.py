"""Physical sandbox timelines outlive session leases without changing tool parentage."""

from __future__ import annotations

import gc
import json
import subprocess
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest


pytest_plugins = ("tests._fixtures.relay_capture",)


def _scope_events(events, name, category="start"):
    return [
        event
        for event in events
        if event.get("name") == name and event.get("scope_category") == category
    ]


def _lifecycle_events(events):
    return [event for event in events
            if event.get("name", "").startswith("hermes.execution_environment.lifecycle.")]


def _environment(native_id="a" * 64, *, persistent=False):
    return SimpleNamespace(
        env_type="docker",
        _container_id=native_id,
        _persist_across_processes=persistent,
    )


def _observed_tool(capture, lease, environment, *, action="reuse", outcome="succeeded", gate=None):
    from agent import relay_tools
    from tools.environment_observability import (
        capture_observer,
        environment_acquired,
        record_transition,
    )
    from tools.execution_observability import record_docker_result

    observed = []

    def callback(_args):
        environment_acquired(environment)
        observer = capture_observer(environment)
        observed.append(observer)
        record_transition(
            environment,
            action=action,
            outcome=outcome,
            container_id=environment._container_id,
            state_after="running" if outcome == "succeeded" else None,
            observer=observer,
        )
        record_docker_result(environment._container_id)
        if gate is not None:
            gate.wait(timeout=5)
        return {"output": "ready", "returncode": 0}

    result, _ = relay_tools.execute(
        "terminal", {"command": "true"}, callback,
        session_id=lease.session_id,
    )
    assert result == {"output": "ready", "returncode": 0}
    return observed[0]


def test_lease_and_tools_keep_separate_parents_in_real_atof(relay_capture):
    from tools.execution_observability import opaque_docker_resource_id

    capture = relay_capture
    lease = capture.acquire()
    environment = _environment()
    first = capture.begin(lease)
    observer = _observed_tool(capture, lease, environment, action="create_and_start")
    assert observer is not None
    capture.coordinator.end_turn(first, outcome="success")
    second = capture.begin(lease, "turn-2")
    _observed_tool(capture, lease, environment)
    capture.coordinator.end_turn(second, outcome="success")
    lease.host.close_session({"session_id": lease.session_id})

    events = capture.events()
    [session] = _scope_events(events, "hermes.session")
    [sandbox] = _scope_events(events, "hermes.execution_environment.lease")
    [physical] = _scope_events(events, "hermes.execution_environment.lifecycle_scope")
    turns = _scope_events(events, "hermes.turn")
    tools = _scope_events(events, "terminal")
    assert sandbox["parent_uuid"] == session["uuid"]
    assert physical["parent_uuid"] == session["uuid"]
    assert [tool["parent_uuid"] for tool in tools] == [turn["uuid"] for turn in turns]
    assert len(_scope_events(events, "hermes.execution_environment.lease", "end")) == 1
    lifecycle = _lifecycle_events(events)
    assert lifecycle
    assert {event["parent_uuid"] for event in lifecycle} == {physical["uuid"]}
    assert not _scope_events(events, "hermes.execution_environment.lifecycle_scope", "end")
    assert {event["data"]["resource_id"] for event in lifecycle} == {
        opaque_docker_resource_id(environment._container_id)
    }
    assert environment._container_id not in json.dumps(events)


def test_rotation_ends_old_observation_without_recreating_container(relay_capture):
    capture = relay_capture
    lease = capture.acquire()
    environment = _environment()
    first = capture.begin(lease)
    _observed_tool(capture, lease, environment, action="create_and_start")
    capture.coordinator.end_turn(first, outcome="success")
    lease.host.rotate_session_scope(lease.session, reason="compaction")
    second = capture.begin(lease, "turn-2")
    _observed_tool(capture, lease, environment)
    capture.coordinator.end_turn(second, outcome="success")
    lease.host.close_session({"session_id": lease.session_id})

    events = capture.events()
    sessions = _scope_events(events, "hermes.session")
    sandboxes = _scope_events(events, "hermes.execution_environment.lease")
    assert len(sessions) == len(sandboxes) == 2
    assert [sandbox["parent_uuid"] for sandbox in sandboxes] == [session["uuid"] for session in sessions]
    assert len(_scope_events(events, "hermes.execution_environment.lease", "end")) == 2
    [physical] = _scope_events(events, "hermes.execution_environment.lifecycle_scope")
    assert physical["parent_uuid"] == sessions[0]["uuid"]
    assert not _scope_events(events, "hermes.execution_environment.lifecycle_scope", "end")
    lifecycle = _lifecycle_events(events)
    assert {event["parent_uuid"] for event in lifecycle} == {physical["uuid"]}
    transitions = [event["data"] for event in lifecycle]
    assert [transition["action"] for transition in transitions].count("create_and_start") == 1
    assert len({transition["resource_id"] for transition in transitions}) == 1


def test_shared_container_has_two_session_observations(relay_capture):
    capture = relay_capture
    environment = _environment(persistent=True)
    leases = [capture.acquire("session-a"), capture.acquire("session-b")]
    owners = []
    for index, lease in enumerate(leases):
        turn = capture.begin(lease, f"turn-{index}")
        owners.append(_observed_tool(capture, lease, environment))
        capture.coordinator.end_turn(turn, outcome="success")
    assert owners[0] is owners[1] is environment._relay_lifecycle_observer
    for lease in leases:
        lease.host.close_session({"session_id": lease.session_id})

    events = capture.events()
    sessions = _scope_events(events, "hermes.session")
    sandboxes = _scope_events(events, "hermes.execution_environment.lease")
    assert len(sandboxes) == 2
    assert {sandbox["parent_uuid"] for sandbox in sandboxes} == {session["uuid"] for session in sessions}
    assert len(_scope_events(events, "hermes.execution_environment.lease", "end")) == 2
    [physical] = _scope_events(events, "hermes.execution_environment.lifecycle_scope")
    assert physical["parent_uuid"] == sessions[0]["uuid"]
    transitions = [event["data"] for event in _lifecycle_events(events)]
    assert {transition["action"] for transition in transitions} == {"reuse"}
    assert len({transition["resource_id"] for transition in transitions}) == 1


@pytest.mark.parametrize("scope_name", [
    "hermes.execution_environment.lease", "hermes.execution_environment.lifecycle_scope",
])
def test_sandbox_scope_failure_does_not_change_tool_result(relay_capture, monkeypatch, scope_name):
    capture = relay_capture
    original_push = capture.relay.scope.push

    def failing_push(name, *args, **kwargs):
        if name == scope_name:
            raise RuntimeError("test observer unavailable")
        return original_push(name, *args, **kwargs)

    monkeypatch.setattr(capture.relay.scope, "push", failing_push)
    lease = capture.acquire()
    environment = _environment()
    turn = capture.begin(lease)
    _observed_tool(capture, lease, environment)
    capture.coordinator.end_turn(turn, outcome="success")
    lease.host.close_session({"session_id": lease.session_id})
    events = capture.events()
    assert not _scope_events(events, scope_name)
    assert len(_scope_events(events, "terminal", "end")) == 1
    if scope_name == "hermes.execution_environment.lifecycle_scope":
        assert getattr(environment, "_relay_lifecycle_observer", None) is None
        assert not lease.host._environment_lifecycles
        assert lease.host._active_operations == 0


def test_release_is_idempotent_and_does_not_claim_container_destruction(relay_capture):
    from tools.environment_observability import environment_released

    capture = relay_capture
    lease = capture.acquire()
    environment = _environment()
    turn = capture.begin(lease)
    _observed_tool(capture, lease, environment)
    capture.coordinator.end_turn(turn, outcome="success")
    environment_released(environment, reason="cache_release")
    environment_released(environment, reason="cache_release")
    lease.host.close_session({"session_id": lease.session_id})

    events = capture.events()
    [ended] = _scope_events(events, "hermes.execution_environment.lease", "end")
    assert "deleted" not in json.dumps(ended["data"])
    lifecycle = [event["data"] for event in _lifecycle_events(events)]
    assert {event["action"] for event in lifecycle} == {"reuse"}


def test_physical_lifecycle_accepts_cleanup_after_session_closes(relay_capture):
    from tools.environment_observability import finish_observation, prepare_cleanup, record_transition

    capture = relay_capture
    lease = capture.acquire()
    environment = _environment()
    turn = capture.begin(lease)
    observer = _observed_tool(capture, lease, environment)
    capture.coordinator.end_turn(turn, outcome="success")
    lease.host.close_session({"session_id": lease.session_id})
    before = capture.events()
    [physical] = _scope_events(before, "hermes.execution_environment.lifecycle_scope")
    assert not _scope_events(before, "hermes.execution_environment.lifecycle_scope", "end")
    assert prepare_cleanup(environment) is observer
    record_transition(
        environment,
        action="remove",
        outcome="succeeded",
        container_id=environment._container_id,
        state_after="deleted",
        observer=observer,
    )
    finish_observation(observer, reason="removed")
    events = capture.events()
    cleanup = events[len(before):]
    [mark] = [event for event in cleanup if event.get("kind") == "mark"]
    assert mark["parent_uuid"] == physical["uuid"]
    assert mark["data"]["state_after"] == "deleted"
    [ended] = _scope_events(events, "hermes.execution_environment.lifecycle_scope", "end")
    assert ended["data"]["reason"] == "removed"
    [session_end] = _scope_events(events, "hermes.session", "end")
    assert events.index(session_end) < events.index(mark) < events.index(ended)
    assert observer not in lease.host._environment_lifecycles


def test_concurrent_sessions_keep_independent_sandbox_and_tool_stacks(relay_capture):
    capture = relay_capture
    leases = [capture.acquire("session-a"), capture.acquire("session-b")]
    environment = _environment(persistent=True)
    gate = threading.Barrier(2)

    def exercise(index):
        lease = leases[index]
        turn = capture.begin(lease, f"turn-{index}")
        try:
            _observed_tool(capture, lease, environment, gate=gate)
        finally:
            capture.coordinator.end_turn(turn, outcome="success")

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(exercise, index) for index in range(2)]
        for future in futures:
            future.result(timeout=20)
    for lease in leases:
        lease.host.close_session({"session_id": lease.session_id})

    events = capture.events()
    sessions = _scope_events(events, "hermes.session")
    sandboxes = _scope_events(events, "hermes.execution_environment.lease")
    turns = _scope_events(events, "hermes.turn")
    tools = _scope_events(events, "terminal")
    assert len(sandboxes) == len(tools) == 2
    assert {sandbox["parent_uuid"] for sandbox in sandboxes} == {session["uuid"] for session in sessions}
    assert {tool["parent_uuid"] for tool in tools} == {turn["uuid"] for turn in turns}
    assert len(_scope_events(events, "hermes.execution_environment.lease", "end")) == 2
    assert len(_scope_events(events, "hermes.execution_environment.lifecycle_scope")) == 1


def test_delayed_failed_cleanup_stays_in_original_trace_without_claiming_deleted(relay_capture):
    from tools.environment_observability import (
        environment_released, finish_observation, prepare_cleanup, record_transition,
    )

    capture = relay_capture
    lease = capture.acquire()
    environment = _environment()
    turn = capture.begin(lease)
    observer = _observed_tool(capture, lease, environment)
    capture.coordinator.end_turn(turn, outcome="success")
    environment_released(environment, reason="cache_release")
    assert prepare_cleanup(environment) is observer
    lease.host.close_session({"session_id": lease.session_id})
    for outcome in ("requested", "failed"):
        record_transition(
            None,
            action="remove",
            outcome=outcome,
            container_id=environment._container_id,
            error_class="RuntimeError" if outcome == "failed" else None,
            observer=observer,
        )
    finish_observation(observer, reason="cleanup_failed")

    events = capture.events()
    [physical] = _scope_events(events, "hermes.execution_environment.lifecycle_scope")
    cleanup = [
        event for event in _lifecycle_events(events)
        if event["data"]["action"] == "remove"
    ]
    assert [event["data"]["outcome"] for event in cleanup] == ["requested", "failed"]
    assert {event["parent_uuid"] for event in cleanup} == {physical["uuid"]}
    assert all("state_after" not in event["data"] for event in cleanup)
    assert cleanup[-1]["data"]["error_class"] == "RuntimeError"
    [ended] = _scope_events(events, "hermes.execution_environment.lifecycle_scope", "end")
    assert ended["data"]["reason"] == "cleanup_failed"
    [session_end] = _scope_events(events, "hermes.session", "end")
    assert events.index(session_end) < events.index(cleanup[0])


def test_delegated_sandbox_preserves_child_session_parent_and_propagation_root(relay_capture, monkeypatch):
    capture = relay_capture
    original_stack = capture.relay.create_scope_stack_from_propagation
    propagated_roots = []

    def capture_stack(propagation):
        propagated_roots.append(str(propagation.root_uuid))
        return original_stack(propagation)

    monkeypatch.setattr(capture.relay, "create_scope_stack_from_propagation", capture_stack)
    parent = capture.acquire("parent-session")
    parent_turn = capture.begin(parent, "parent-turn")
    child = capture.acquire("child-session", parent_session_id=parent.session_id)
    child_turn = capture.begin(child, "child-turn")
    child_root = child.host.run_in_session(
        child.session, capture.relay.capture_propagation_context
    ).root_uuid
    _observed_tool(capture, child, _environment())
    capture.coordinator.end_turn(child_turn, outcome="success")
    capture.coordinator.end_turn(parent_turn, outcome="success")
    parent.host.close_session({"session_id": parent.session_id})

    events = capture.events()
    sessions = _scope_events(events, "hermes.session")
    turns = _scope_events(events, "hermes.turn")
    [sandbox] = _scope_events(events, "hermes.execution_environment.lease")
    [tool] = _scope_events(events, "terminal")
    assert len(sessions) == len(turns) == 2
    assert sessions[1]["parent_uuid"] == turns[0]["uuid"]
    assert sandbox["parent_uuid"] == sessions[1]["uuid"]
    assert tool["parent_uuid"] == turns[1]["uuid"]
    assert len(_scope_events(events, "hermes.execution_environment.lease", "end")) == 1
    assert set(propagated_roots) == {str(child_root)}


def test_session_close_before_environment_attachment_leaves_no_closed_lease(relay_capture, monkeypatch):
    from agent import relay_sandbox

    capture = relay_capture
    lease = capture.acquire()
    environment = _environment(persistent=True)
    observer = relay_sandbox.SandboxObserver(lease.host, lease.session)
    entered, resume = threading.Event(), threading.Event()
    original_lock = relay_sandbox._ATTACH_LOCK

    class DelayedAttachment:
        def __enter__(self):
            if threading.current_thread().name.startswith("attachment-race"):
                entered.set()
                assert resume.wait(timeout=5)
            return original_lock.__enter__()

        def __exit__(self, *args):
            return original_lock.__exit__(*args)

    monkeypatch.setattr(relay_sandbox, "_ATTACH_LOCK", DelayedAttachment())
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="attachment-race") as workers:
        pending = workers.submit(observer.acquire, environment)
        try:
            assert entered.wait(timeout=5)
            lease.host.close_session({"session_id": lease.session_id})
        finally:
            resume.set()
        pending.result(timeout=5)

    assert not lease.session._environment_observations
    assert not getattr(environment, "_relay_environment_leases", ())
    assert getattr(environment, "_relay_lifecycle_observer", None) is None
    assert not _scope_events(capture.events(), "hermes.execution_environment.lease")


def test_physical_environment_observer_does_not_retain_closed_session(relay_capture):
    from agent import relay_runtime, relay_sandbox
    from tools.environment_observability import record_transition

    capture = relay_capture
    host_lease = capture.acquire()
    owner = relay_runtime.RelaySession(
        session_id="released-owner", handle=host_lease.session.handle,
        context=host_lease.session.context.copy(),
    )
    environment = _environment(persistent=True)
    observer = relay_sandbox.SandboxObserver(host_lease.host, owner)
    observer.acquire(environment)
    relay_sandbox.close_session_observations(host_lease.host, owner, "session_close")
    owner.closing = True
    reference = weakref.ref(owner)
    del owner
    gc.collect()

    assert reference() is None
    before = capture.events()
    record_transition(environment, "remove", "succeeded", environment._container_id, state_after="deleted")
    events = capture.events()
    [physical] = _scope_events(events, "hermes.execution_environment.lifecycle_scope")
    [mark] = events[len(before):]
    assert mark["parent_uuid"] == physical["uuid"]
    assert mark["data"]["state_after"] == "deleted"


def _docker_handle(native_id="a" * 64, *, persistent=False):
    from tools.environments import docker

    environment = object.__new__(docker.DockerEnvironment)
    environment.env_type = "docker"
    environment._container_id = native_id
    environment._docker_exe = "docker"
    environment._persist_across_processes = persistent
    environment._persistent = True
    return environment


def test_delayed_docker_cleanup_retains_native_activation_across_runtime_shutdown(relay_capture, monkeypatch):
    from agent import relay_runtime
    from tools.environments import docker

    capture = relay_capture
    lease = capture.acquire()
    environment = _docker_handle()
    turn = capture.begin(lease)
    owner = _observed_tool(capture, lease, environment)
    capture.coordinator.end_turn(turn, outcome="success")
    started, resume = threading.Event(), threading.Event()

    def run(argv, **_kwargs):
        if argv[1] == "stop":
            started.set()
            assert resume.wait(timeout=5)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(docker.subprocess, "run", run)
    activation = relay_runtime._PLUGIN_CONFIGURATION._activation
    try:
        environment.cleanup()
        assert started.wait(timeout=5)
        assert environment._container_id is None
        assert owner.operation is not None
        lease.host.close_session({"session_id": lease.session_id})
        lease.host.shutdown()
        assert activation.is_active
        assert not lease.host._shutdown_complete.is_set()
        assert not _scope_events(capture.events(), "hermes.execution_environment.lifecycle_scope", "end")
    finally:
        resume.set()
        assert environment.wait_for_cleanup()
    assert lease.host._shutdown_complete.wait(timeout=5)
    assert not activation.is_active
    assert owner.operation is None
    assert not lease.host._environment_lifecycles
    events = capture.events()
    [physical] = _scope_events(events, "hermes.execution_environment.lifecycle_scope")
    [session_end] = _scope_events(events, "hermes.session", "end")
    cleanup = [event for event in _lifecycle_events(events)
               if event["data"]["action"] in {"stop", "remove"}]
    assert [event["data"]["outcome"] for event in cleanup] == ["requested", "succeeded", "requested", "succeeded"]
    assert {event["parent_uuid"] for event in cleanup} == {physical["uuid"]}
    assert cleanup[-1]["data"]["state_after"] == "deleted"
    [ended] = _scope_events(events, "hermes.execution_environment.lifecycle_scope", "end")
    assert ended["data"]["reason"] == "removed"
    assert events.index(session_end) < events.index(cleanup[-1]) < events.index(ended)


def test_timed_out_native_scope_close_retains_activation_until_pop_finishes(relay_capture, monkeypatch):
    from agent import relay_runtime
    from tools.environment_observability import finish_observation, prepare_cleanup

    capture = relay_capture
    lease = capture.acquire()
    environment = _environment()
    turn = capture.begin(lease)
    owner = _observed_tool(capture, lease, environment)
    capture.coordinator.end_turn(turn, outcome="success")
    assert prepare_cleanup(environment) is owner
    lease.host.close_session({"session_id": lease.session_id})
    entered, resume = threading.Event(), threading.Event()
    original_pop = capture.relay.scope.pop
    physical_uuid = owner.scope.handle.uuid

    def delayed_pop(handle, **kwargs):
        if handle.uuid == physical_uuid:
            entered.set()
            assert resume.wait(timeout=5)
        return original_pop(handle, **kwargs)

    monkeypatch.setattr(capture.relay.scope, "pop", delayed_pop)
    monkeypatch.setattr(relay_runtime, "SCOPE_OP_TIMEOUT", 0.05)
    activation = relay_runtime._PLUGIN_CONFIGURATION._activation
    lease.host.shutdown()
    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(finish_observation, owner, reason="removed")
        try:
            assert entered.wait(timeout=5)
            pending.result(timeout=5)
            assert owner.operation is not None
            assert activation.is_active
            assert not lease.host._shutdown_complete.is_set()
        finally:
            resume.set()
    assert lease.host._shutdown_complete.wait(timeout=5)
    assert owner.operation is None
    assert not activation.is_active
    [ended] = _scope_events(capture.events(), "hermes.execution_environment.lifecycle_scope", "end")
    assert ended["data"]["reason"] == "removed"


@pytest.mark.parametrize("cutoff", ["runtime_shutdown", "persistent_release"])
def test_idle_physical_observation_cutoff_never_invents_container_removal(relay_capture, cutoff):
    from tools.environment_observability import record_transition

    capture = relay_capture
    lease = capture.acquire()
    environment = _docker_handle(persistent=True)
    turn = capture.begin(lease)
    owner = _observed_tool(capture, lease, environment)
    capture.coordinator.end_turn(turn, outcome="success")
    if cutoff == "runtime_shutdown":
        lease.host.shutdown()
        assert lease.host._shutdown_complete.wait(timeout=5)
    else:
        environment.cleanup()
        assert environment._container_id is None
    assert owner.operation is None
    assert owner not in lease.host._environment_lifecycles
    events = capture.events()
    [ended] = _scope_events(events, "hermes.execution_environment.lifecycle_scope", "end")
    assert ended["data"]["reason"] == ("observation_ended" if cutoff == "runtime_shutdown" else "retained")
    lifecycle = [event["data"] for event in _lifecycle_events(events)]
    assert {event["action"] for event in lifecycle} == {"reuse"}
    record_transition(environment, "remove", "succeeded", "a" * 64, state_after="deleted", observer=owner)
    assert capture.events() == events


def test_constructor_failure_closes_native_physical_scope_without_owner_leak(relay_capture, monkeypatch):
    from agent import relay_sandbox
    from tools.environment_observability import observe_environment
    from tools.environments import docker

    capture = relay_capture
    lease = capture.acquire()
    observer = relay_sandbox.SandboxObserver(lease.host, lease.session)
    error = docker.EnvironmentConnectionError("Docker unavailable", retry_hint="retry")

    def unavailable():
        raise error

    monkeypatch.setattr(docker, "_ensure_docker_available", unavailable)
    with observe_environment(observer), pytest.raises(docker.EnvironmentConnectionError) as caught:
        docker.DockerEnvironment(image="test-image")
    assert caught.value is error
    assert not lease.host._environment_lifecycles
    assert lease.host._active_operations == 0
    events = capture.events()
    [physical] = _scope_events(events, "hermes.execution_environment.lifecycle_scope")
    [ended] = _scope_events(events, "hermes.execution_environment.lifecycle_scope", "end")
    assert ended["uuid"] == physical["uuid"]
    assert ended["data"]["reason"] == "provision_failed"
    [failure] = _lifecycle_events(events)
    assert failure["parent_uuid"] == physical["uuid"]
    assert failure["data"]["outcome"] == "failed"
    assert failure["data"]["resource_id_status"] == "unavailable"
    assert "state_after" not in failure["data"]


def test_native_recovery_marks_keep_previous_and_replacement_resource_relationship(relay_capture, monkeypatch):
    from unittest.mock import Mock

    from tools.environments import docker
    from tools.execution_observability import opaque_docker_resource_id

    capture = relay_capture
    lease = capture.acquire()
    environment = _docker_handle()
    turn = capture.begin(lease)
    owner = _observed_tool(capture, lease, environment)
    capture.coordinator.end_turn(turn, outcome="success")
    environment._image = "test-image"
    environment._labels = {}
    environment._run_env_values = {}
    environment.cwd = "/workspace"
    environment._find_reusable_container = Mock(return_value=None)
    environment._run_command = Mock(return_value=["docker", "run", "test-image"])
    environment._docker_client_env = Mock(return_value=None)
    environment.init_session = Mock()
    environment._mark_recreated = Mock()
    monkeypatch.setattr(docker, "run_capture", Mock(return_value=subprocess.CompletedProcess([], 0, "b" * 64 + "\n", "")))
    lease.host.close_session({"session_id": lease.session_id})
    assert environment._recreate_container()
    assert environment._relay_lifecycle_observer is owner
    assert environment._container_id == "b" * 64
    events = capture.events()
    [physical] = _scope_events(events, "hermes.execution_environment.lifecycle_scope")
    recovery = [event for event in _lifecycle_events(events) if event["data"]["action"] == "recover"]
    assert {event["parent_uuid"] for event in recovery} == {physical["uuid"]}
    assert [event["data"]["outcome"] for event in recovery] == ["requested", "succeeded"]
    assert recovery[0]["data"]["resource_id"] == opaque_docker_resource_id("a" * 64)
    assert recovery[-1]["data"]["previous_resource_id"] == opaque_docker_resource_id("a" * 64)
    assert recovery[-1]["data"]["resource_id"] == opaque_docker_resource_id("b" * 64)
    assert "state_after" not in recovery[-1]["data"]
    [started] = [event for event in _lifecycle_events(events)
                 if event["data"]["action"] == "create_and_start" and event["data"]["outcome"] == "succeeded"]
    assert started["data"]["state_after"] == "running"
    assert "a" * 64 not in json.dumps(events)
    assert "b" * 64 not in json.dumps(events)


def test_two_native_profile_runtimes_reopen_only_closed_cached_observation(relay_capture):
    from agent import relay_runtime, relay_sandbox
    from tools.environment_observability import record_transition
    from tools.execution_observability import opaque_docker_resource_id

    capture = relay_capture
    hosts = [relay_runtime.RelayRuntime(relay=capture.relay, profile_key=f"profile-{name}") for name in ("a", "b")]
    environment = _environment(persistent=True)
    try:
        sessions = [host.ensure_session({"session_id": f"session-{index}"}) for index, host in enumerate(hosts)]
        observers = [relay_sandbox.SandboxObserver(host, session) for host, session in zip(hosts, sessions)]
        observers[0].acquire(environment)
        original = environment._relay_lifecycle_observer
        observers[1].acquire(environment)
        assert environment._relay_lifecycle_observer is original
        assert original.scope.parent_uuid == str(sessions[0].handle.uuid)
        hosts[0].shutdown()
        assert hosts[0]._shutdown_complete.wait(timeout=5)
        assert original.scope.closed
        assert relay_runtime._PLUGIN_CONFIGURATION._activation.is_active
        observers[1].acquire(environment)
        resumed = environment._relay_lifecycle_observer
        assert resumed is not original
        assert resumed.scope.parent_uuid == str(sessions[1].handle.uuid)
        record_transition(environment, "reuse", "succeeded", environment._container_id,
                          state_after="running", evidence="docker_ps")
        events = capture.events()
        physical = _scope_events(events, "hermes.execution_environment.lifecycle_scope")
        assert len(physical) == 2
        assert [event["parent_uuid"] for event in physical] == [str(session.handle.uuid) for session in sessions]
        assert {event["data"]["resource_id"] for event in physical} == {opaque_docker_resource_id("a" * 64)}
        assert physical[1]["data"]["observation_source"] == "cached_handle_after_cutoff"
        assert physical[1]["data"]["predecessor_observation_id"] == physical[0]["data"]["observation_id"]
        [old_end] = _scope_events(events, "hermes.execution_environment.lifecycle_scope", "end")
        assert old_end["uuid"] == physical[0]["uuid"]
        assert old_end["data"]["reason"] == "observation_ended"
        [reuse] = _lifecycle_events(events)
        assert reuse["name"] == "hermes.execution_environment.lifecycle.reuse.succeeded"
        assert reuse["parent_uuid"] == physical[1]["uuid"]
        assert not any(event["data"]["action"] in {"create_and_start", "remove"} for event in _lifecycle_events(events))
    finally:
        for host in hosts:
            host.shutdown()
            assert host._shutdown_complete.wait(timeout=5)


@pytest.mark.parametrize("tool_name", ["read_file", "execute_code"])
def test_cached_foreground_docker_tools_reopen_closed_observation(relay_capture, monkeypatch, tool_name):
    """Real managed callbacks/scopes; only Docker's physical execution boundary is stubbed."""
    from agent import relay_runtime, relay_sandbox, relay_tools
    from tools import code_execution_tool, file_tools, terminal_tool
    from tools.environments.base import BaseEnvironment
    from tools.execution_observability import docker_exec_started, opaque_docker_resource_id
    from tools.file_operations import ShellFileOperations

    capture = relay_capture
    lease = capture.acquire("cached-successor")
    original_host = relay_runtime.RelayRuntime(relay=capture.relay, profile_key="cached-original")
    environment = _docker_handle(persistent=True)
    environment.cwd = "/workspace"
    try:
        original_session = original_host.ensure_session({"session_id": "cached-original"})
        relay_sandbox.SandboxObserver(original_host, original_session).acquire(environment)
        original = environment._relay_lifecycle_observer
        original_host.shutdown()
        assert original_host._shutdown_complete.wait(timeout=5)
        assert original.scope.closed
        monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda _task: "default")
        monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "docker", "cwd": "/workspace"})
        monkeypatch.setattr(terminal_tool, "_active_environments", {"default": environment})
        monkeypatch.setattr(terminal_tool, "_last_activity", {})
        monkeypatch.setattr(file_tools, "_file_ops_cache", {"default": ShellFileOperations(environment)})
        commands = []

        def physical_execute(env, command, cwd="", **_kwargs):
            commands.append(command)
            docker_exec_started(env._container_id)
            return {"output": "__hermes_missing__", "returncode": 0}

        monkeypatch.setattr(BaseEnvironment, "execute", physical_execute)
        if tool_name == "read_file":
            args = {"path": "/workspace/missing.txt"}
            callback = lambda final: file_tools.read_file_tool(**final, task_id="default")
        else:
            args = {"code": "pass"}
            callback = lambda final: code_execution_tool.execute_code(**final, task_id="default")
        turn = capture.begin(lease)
        callback_results = []

        def observed_callback(final):
            result = callback(final)
            callback_results.append(result)
            return result

        result, observed_args = relay_tools.execute(tool_name, args, observed_callback, session_id=lease.session_id)
        capture.coordinator.end_turn(turn, outcome="success")
        assert result == callback_results[0]
        assert json.loads(result)["error"]
        assert observed_args == args and commands
        resumed = environment._relay_lifecycle_observer
        assert resumed is not original and not resumed.scope.closed
        events = capture.events()
        first, successor = _scope_events(events, "hermes.execution_environment.lifecycle_scope")
        assert first["parent_uuid"] == str(original_session.handle.uuid)
        assert successor["parent_uuid"] == str(lease.session.handle.uuid)
        assert successor["data"]["observation_source"] == "cached_handle_after_cutoff"
        assert successor["data"]["predecessor_observation_id"] == first["data"]["observation_id"]
        assert successor["data"]["observation_id"] != first["data"]["observation_id"]
        assert first["data"]["resource_id"] == successor["data"]["resource_id"] == opaque_docker_resource_id("a" * 64)
        associations = _scope_events(events, "hermes.execution_environment.lease")
        assert len(associations) == 2 and associations[1]["parent_uuid"] == successor["parent_uuid"]
        [tool], [native_turn] = _scope_events(events, tool_name), _scope_events(events, "hermes.turn")
        assert tool["parent_uuid"] == native_turn["uuid"]
        [tool_end] = _scope_events(events, tool_name, "end")
        annotation = tool_end["category_profile"]["tool_result_annotation"]["hermes.execution_environment"]
        assert annotation["resource_id"] == successor["data"]["resource_id"]
        assert annotation["resource_id_status"] == "available"
    finally:
        original_host.shutdown()
        assert original_host._shutdown_complete.wait(timeout=5)


def test_unbound_environment_cannot_borrow_managed_callback_owner(relay_capture):
    from agent import relay_tools
    from tools.environment_observability import (
        capture_observer, environment_acquired, prepare_cleanup, record_transition,
    )

    capture = relay_capture
    lease = capture.acquire()
    owned, unbound = _environment(), _environment("b" * 64)
    payload = {"output": "unchanged", "returncode": 0}

    def callback(_args):
        environment_acquired(owned)
        assert capture_observer(unbound) is None
        assert prepare_cleanup(unbound) is None
        record_transition(unbound, "remove", "succeeded", unbound._container_id, state_after="deleted")
        return payload

    turn = capture.begin(lease)
    result, _ = relay_tools.execute("terminal", {"command": "true"}, callback, session_id=lease.session_id)
    capture.coordinator.end_turn(turn, outcome="success")
    assert result is payload
    events = capture.events()
    [physical] = _scope_events(events, "hermes.execution_environment.lifecycle_scope")
    assert physical["parent_uuid"] == str(lease.session.handle.uuid)
    assert not _lifecycle_events(events)
    assert not hasattr(unbound, "_relay_lifecycle_observer")


def test_provision_and_concurrent_acquire_share_single_native_physical_owner(relay_capture, monkeypatch):
    from agent import relay_sandbox

    capture = relay_capture
    leases = [capture.acquire("constructor-session"), capture.acquire("acquire-session")]
    observers = [relay_sandbox.SandboxObserver(lease.host, lease.session) for lease in leases]
    environment = _environment()
    entered, contended, resume = threading.Event(), threading.Event(), threading.Event()
    original_open = relay_sandbox._EnvironmentLifecycle.open
    lock = threading.Lock()

    class ContendedProvisioningLock:
        def acquire(self, **kwargs):
            if lock.acquire(blocking=False):
                return True
            contended.set()
            return lock.acquire(**kwargs)

        def release(self):
            lock.release()

    environment._relay_provisioning_lock = ContendedProvisioningLock()

    def delayed_open(owner):
        entered.set()
        assert resume.wait(timeout=5)
        return original_open(owner)

    monkeypatch.setattr(relay_sandbox._EnvironmentLifecycle, "open", delayed_open)
    with ThreadPoolExecutor(max_workers=2) as workers:
        provisioning = workers.submit(observers[0].provision, environment)
        try:
            assert entered.wait(timeout=5)
            acquiring = workers.submit(observers[1].acquire, environment)
            assert contended.wait(timeout=5)
        finally:
            resume.set()
        owner = provisioning.result(timeout=5)
        acquiring.result(timeout=5)
    assert owner is environment._relay_lifecycle_observer
    [physical] = _scope_events(capture.events(), "hermes.execution_environment.lifecycle_scope")
    assert physical["parent_uuid"] == str(leases[0].session.handle.uuid)


def test_busy_provisioning_owner_bounds_wait_and_preserves_tool_result(relay_capture, monkeypatch):
    from agent import relay_runtime

    capture = relay_capture
    lease = capture.acquire()
    environment = _environment()
    environment._relay_provisioning_lock = threading.Lock()
    environment._relay_provisioning_lock.acquire()
    monkeypatch.setattr(relay_runtime, "SCOPE_OP_TIMEOUT", 0.05)
    turn = capture.begin(lease)
    try:
        with ThreadPoolExecutor(max_workers=1) as workers:
            pending = workers.submit(_observed_tool, capture, lease, environment)
            try:
                pending.result(timeout=5)
            finally:
                environment._relay_provisioning_lock.release()
    finally:
        capture.coordinator.end_turn(turn, outcome="success")
    assert not lease.host._environment_lifecycles
    events = capture.events()
    assert not _scope_events(events, "hermes.execution_environment.lifecycle_scope")
    assert len(_scope_events(events, "terminal", "end")) == 1


def test_idle_native_pop_timeout_keeps_plugin_alive_until_final_publication(relay_capture, monkeypatch):
    from agent import relay_runtime

    capture = relay_capture
    lease = capture.acquire()
    environment = _environment(persistent=True)
    turn = capture.begin(lease)
    owner = _observed_tool(capture, lease, environment)
    capture.coordinator.end_turn(turn, outcome="success")
    lease.host.close_session({"session_id": lease.session_id})
    original_pop, physical_uuid = capture.relay.scope.pop, owner.scope.handle.uuid
    entered, resume = threading.Event(), threading.Event()
    finalized = threading.Event()
    original_release = owner._release

    def released():
        try:
            original_release()
        finally:
            finalized.set()

    def delayed_pop(handle, **kwargs):
        if handle.uuid == physical_uuid:
            entered.set()
            assert resume.wait(timeout=5)
        return original_pop(handle, **kwargs)

    monkeypatch.setattr(capture.relay.scope, "pop", delayed_pop)
    monkeypatch.setattr(owner, "_release", released)
    monkeypatch.setattr(relay_runtime, "SCOPE_OP_TIMEOUT", 0.05)
    activation = relay_runtime._PLUGIN_CONFIGURATION._activation
    with ThreadPoolExecutor(max_workers=1) as workers:
        shutdown = workers.submit(lease.host.shutdown)
        try:
            assert entered.wait(timeout=5)
            shutdown.result(timeout=5)
            assert activation.is_active
            # Host release is complete, but the cutoff still owns configuration.
            assert lease.host._shutdown_complete.is_set()
            assert owner._configuration_owned
            assert not finalized.is_set()
            with ThreadPoolExecutor(max_workers=2) as repeated:
                finishers = [repeated.submit(owner.finish, reason="observation_ended") for _ in range(2)]
                for finisher in finishers:
                    finisher.result(timeout=5)
            assert activation.is_active
            assert owner._configuration_owned
        finally:
            resume.set()
    assert finalized.wait(timeout=5)
    assert lease.host._shutdown_complete.is_set()
    assert not owner._configuration_owned
    assert not activation.is_active
    [ended] = _scope_events(capture.events(), "hermes.execution_environment.lifecycle_scope", "end")
    assert ended["data"]["reason"] == "observation_ended"
    owner.finish(reason="observation_ended")
    assert not owner._configuration_owned


def test_concurrent_physical_finish_publishes_once_without_configuration_owner_leak(relay_capture, monkeypatch):
    from agent import relay_runtime, relay_sandbox

    capture = relay_capture
    lease = capture.acquire()
    environment = _environment()
    turn = capture.begin(lease)
    owner = _observed_tool(capture, lease, environment)
    capture.coordinator.end_turn(turn, outcome="success")
    gate = threading.Barrier(2)
    original_close = relay_sandbox._EnvironmentLease.close

    def racing_close(scope, reason, **kwargs):
        if scope is owner.scope:
            gate.wait(timeout=5)
        return original_close(scope, reason, **kwargs)

    monkeypatch.setattr(relay_sandbox._EnvironmentLease, "close", racing_close)
    with ThreadPoolExecutor(max_workers=2) as workers:
        finishers = [workers.submit(owner.finish, reason="observation_ended") for _ in range(2)]
        for finisher in finishers:
            finisher.result(timeout=5)
    assert not owner._configuration_owned
    assert id(owner) not in relay_runtime._PLUGIN_CONFIGURATION._owners
    assert owner not in lease.host._environment_lifecycles
    [ended] = _scope_events(capture.events(), "hermes.execution_environment.lifecycle_scope", "end")
    assert ended["data"]["reason"] == "observation_ended"


def test_physical_close_submit_failure_releases_configuration_and_cleanup_lease(relay_capture, monkeypatch):
    from agent import relay_runtime
    from tools.environment_observability import prepare_cleanup

    capture = relay_capture
    lease = capture.acquire()
    environment = _environment()
    turn = capture.begin(lease)
    owner = _observed_tool(capture, lease, environment)
    capture.coordinator.end_turn(turn, outcome="success")
    assert prepare_cleanup(environment) is owner

    def unavailable_submit(*_args, **_kwargs):
        raise RuntimeError("executor unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(relay_runtime, "_scope_op_executor", lambda: SimpleNamespace(submit=unavailable_submit))
        owner.finish(reason="observation_ended")
    assert owner.scope.closed
    assert owner.operation is None
    assert not owner._configuration_owned
    assert id(owner) not in relay_runtime._PLUGIN_CONFIGURATION._owners
    assert owner not in lease.host._environment_lifecycles
    assert lease.host._active_operations == 0
    lease.host.shutdown()
    assert lease.host._shutdown_complete.wait(timeout=5)


def test_publication_timeout_skips_later_marks_but_keeps_cleanup_and_plugin_lifetime(relay_capture, monkeypatch):
    """Stall the native publication boundary, not a real Docker or Phoenix service."""
    from agent import relay_runtime
    from tools.environment_observability import record_transition
    from tools.environments import docker

    capture = relay_capture
    lease = capture.acquire()
    environment = _docker_handle()
    native_id = environment._container_id
    turn = capture.begin(lease)
    owner = _observed_tool(capture, lease, environment)
    capture.coordinator.end_turn(turn, outcome="success")
    lease.host.close_session({"session_id": lease.session_id})
    activation = relay_runtime._PLUGIN_CONFIGURATION._activation
    entered, resume = threading.Event(), threading.Event()
    original_event = capture.relay.scope.event
    executor = relay_runtime._scope_op_executor()
    submissions, cleanup = [], []

    def submit(callback):
        submissions.append(callback.__name__)
        return executor.submit(callback)

    def stalled_event(name, **kwargs):
        if kwargs.get("handle") is owner.scope.handle:
            entered.set()
            assert resume.wait(timeout=5)
        return original_event(name, **kwargs)

    def docker_run(argv, **_kwargs):
        cleanup.append(argv[1])
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(relay_runtime, "SCOPE_OP_TIMEOUT", 0.05)
    monkeypatch.setattr(relay_runtime, "_scope_op_executor", lambda: SimpleNamespace(submit=submit))
    monkeypatch.setattr(capture.relay.scope, "event", stalled_event)
    monkeypatch.setattr(docker.subprocess, "run", docker_run)
    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(record_transition, environment, "stop", "requested", native_id)
        try:
            assert entered.wait(timeout=5)
            pending.result(timeout=5)
            environment.cleanup()
            assert environment.wait_for_cleanup()
            assert cleanup == ["stop", "rm"]
            assert submissions.count("emit") == 1
            lease.host.shutdown()
            assert activation.is_active
            assert not lease.host._shutdown_complete.is_set()
        finally:
            resume.set()
    assert lease.host._shutdown_complete.wait(timeout=5)
    assert not activation.is_active
    events = capture.events()
    # The already admitted request may finish; suppressed marks must not invent a result.
    assert [event["data"]["outcome"] for event in _lifecycle_events(events)
            if event["data"]["action"] == "stop"] == ["requested"]
    [ended] = _scope_events(events, "hermes.execution_environment.lifecycle_scope", "end")
    assert ended["data"]["reason"] == "removed"
